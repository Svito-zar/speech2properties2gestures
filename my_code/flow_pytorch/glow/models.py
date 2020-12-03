import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

from my_code.flow_pytorch.glow import modules, thops, utils

from my_code.flow_pytorch.glow.modules import DiagGaussian


class f_seq(nn.Module):
    """
    Transformation to be used in a coupling layer
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        sp_cond_dim,
        temporal_cond_dim,
    ):
        """
        input_size:             (glow) channels
        output_size:            output channels (ToDo: should be equal to input_size?)
        hidden_size:            size of the hidden layer of the NN
        sp_cond_dim:            dim. of the speech conditioning info
        temporal_cond_dim:      dim. of the temporal conditioning info (from neighbouring flow steps)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.cond_dim = sp_cond_dim + temporal_cond_dim

        #self.cond_transform = nn.Sequential(
        #    nn.Linear(feature_encoder_dim, cond_dim), nn.LeakyReLU(),
        #)

        self.input_hidden = nn.Sequential(nn.Linear(input_size + self.cond_dim, hidden_size), nn.LeakyReLU(), nn.Dropout(0.2))

        self.final_linear = modules.LinearZeros(hidden_size, output_size)
        self.hidden = None
        self.cell = None

    def forward(self, z, condition):
        self.hidden= self.input_hidden(
            torch.cat((z, condition), dim=1)
            )
        return self.final_linear(self.hidden)


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(
        self,
        in_channels,
        hidden_channels,
        cond_dim,
        hparams,
        actnorm_scale=1.0,
        flow_permutation="shuffle",
        flow_coupling="additive",
        LU_decomposed=False,
        L=1,
        K=1,
        scale_eps=1e-6,
        scale_logging=False,
        glow_rnn_type=None,
    ):
        # check configures
        assert (
            flow_coupling in FlowStep.FlowCoupling
        ), "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)

        assert (
            flow_permutation in FlowStep.FlowPermutation
        ), "float_permutation should be in `{}`".format(FlowStep.FlowPermutation.keys())

        super().__init__()

        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.temp_conf_window = hparams.Glow["conv_seq_len"]

        # Custom
        self.scale = None
        self.scale_logging = scale_logging
        self.scale_eps = scale_eps
        self.L = L  # Which multiscale layer this module is in
        self.K = K  # Which step of flow in self.L

        # About sequence processing
        self.past_context = hparams.Cond["Speech"]["prev_context"]
        self.future_context = hparams.Cond["Speech"]["future_context"]

        # Helpers for conditioning info

        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed
            )
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)



        # 3. coupling
        temp_cond_length = hparams.Glow["distr_dim"] // 2 * (hparams.Glow["conv_seq_len"] * 2 + 1)
        if flow_coupling == "additive":
            self.f = f_seq(
                in_channels // 2,
                in_channels - in_channels // 2,
                hidden_channels,
                cond_dim,
                temp_cond_length,
            )
        elif flow_coupling == "affine":
            if in_channels % 2 == 0:
                self.f = f_seq(
                    in_channels // 2,
                    in_channels,
                    hidden_channels,
                    cond_dim,
                    temp_cond_length,
                )
            else:
                self.f = f_seq(
                    in_channels // 2,
                    in_channels + 1,
                    hidden_channels,
                    cond_dim,
                    temp_cond_length,
                )


    def from_z_sequence_to_cond(self, z_seq, seqlen, time_st):
        """
        Obtain z sequence conditioning info with padding if needed
        Args:
            z_seq:   full sequence of z_low values
            seqlen:  sequence length
            time_st: current time-step

        Returns:
            flow_conv_cond:   feature vector corresponding to the conditioning info

        """

        # For the first few frames - pad with repeating current frame
        if time_st < self.temp_conf_window:
            z_pad = z_seq[:, 0,:]
            future_cond = torch.flatten(z_seq[:, :time_st + self.temp_conf_window + 1, :], start_dim=1, end_dim=2)
            padding = z_pad.tile(1, self.temp_conf_window - time_st)
            flow_conv_cond = torch.cat((padding, future_cond), dim=1)

        # For the last few frames - pad with repeating current frame
        elif time_st >= seqlen - self.temp_conf_window:
            z_pad = z_seq[:, -1, :]
            past_cond = torch.flatten(z_seq[:, time_st - self.temp_conf_window:, :], start_dim=1, end_dim=2)
            padding = z_pad.tile(1, self.temp_conf_window - (seqlen - time_st - 1))
            flow_conv_cond = torch.cat((past_cond, padding), dim=1)

        else:
            flow_conv_cond = torch.flatten(
                z_seq[:, time_st - self.temp_conf_window:time_st + self.temp_conf_window + 1, :],
                start_dim=1,end_dim=2)

        return flow_conv_cond

    def forward(self, input_seq_, speech_cond_seq, logdet=None, reverse=False):

        if not reverse:

            if logdet is None:
                logdet = torch.zeros_like(input_seq_[:, 0, 0])

            return self.normal_flow(input_seq_, speech_cond_seq, logdet)

        else:

            assert input_seq_ is not None

            if logdet is None:
                logdet = torch.zeros_like(input_seq_[:, 0, 0])

            return self.reverse_flow(input_seq_, speech_cond_seq, logdet)

    def normal_flow(self, input_seq, sp_cond_seq, logdet):
        """
        Forward path

        Args:
            input_seq:    input sequence in "X" space
            sp_cond_seq:  sequence with speech conditioning
            logdet:       log determinant of the Jacobian from the previous operations

        Returns:
            z3_seq:       output sequence in "Z" space
            logdet:       new value of log determinant of the Jacobian
        """

        z1_seq = None
        z2_l_seq = None
        z2_u_seq = None
        z3_seq = None
        seq_len = input_seq.shape[1]

        #### Go though each steps of the flow separately for the whole sequence

        # 1. actnorm
        for time_st in range(seq_len):
            curr_output = input_seq[:, time_st, :]

            curr_z1, logdet = self.actnorm(curr_output, logdet=logdet, reverse=False)

            # Add current encoding "z" to the sequence of encodings of the 1st operation
            if z1_seq is None:
                z1_seq = curr_z1.unsqueeze(dim=1)
            else:
                z1_seq = torch.cat((z1_seq, curr_z1.unsqueeze(dim=1)), 1)

        # 2. permute
        for time_st in range(seq_len):

            curr_z1 = z1_seq[:, time_st, :]

            curr_z2, logdet = FlowStep.FlowPermutation[self.flow_permutation](
                self, curr_z1.float(), logdet, False
            )

            # 3.1 split on upper and lower

            z2_l, z2_u = thops.split_feature(curr_z2, "split")

            # Add current encodings "z_l" to the sequence of lower encodings
            if z2_l_seq is None:
                z2_l_seq = z2_l.unsqueeze(dim=1)
            else:
                z2_l_seq = torch.cat((z2_l_seq, z2_l.unsqueeze(dim=1)), 1)

            # Add current encodings "z_u" to the sequence of upper encodings
            if z2_u_seq is None:
                z2_u_seq = z2_u.unsqueeze(dim=1)
            else:
                z2_u_seq = torch.cat((z2_u_seq, z2_u.unsqueeze(dim=1)), 1)

        # 3.2 coupling
        for time_st in range(seq_len):

            curr_z2_l = z2_l_seq[:, time_st, :]
            curr_z2_u = z2_u_seq[:, time_st, :]

            speech_cond = sp_cond_seq[:, time_st, :]

            flow_conv_cond = self.from_z_sequence_to_cond(z2_l_seq, seq_len, time_st)

            curr_cond = torch.cat((speech_cond, flow_conv_cond), dim=1)

            # Require some conditioning
            assert curr_cond is not None

            if self.flow_coupling == "additive":
                curr_z2_u = curr_z2_u + self.f(curr_z2_l, curr_cond)
            elif self.flow_coupling == "affine":
                h = self.f(curr_z2_l, curr_cond)
                shift, scale = thops.split_feature(h, "cross")
                scale = torch.sigmoid(scale + 2.0).clamp(self.scale_eps)

                if self.scale_logging:
                    self.scale = scale
                curr_z2_u = curr_z2_u + shift
                curr_z2_u = curr_z2_u * scale
                logdet = thops.sum(torch.log(scale), dim=[1]) + logdet

            final_z = thops.cat_feature(curr_z2_l, curr_z2_u)

            # Add current encoding "z" to the sequence of encodings
            if z3_seq is None:
                z3_seq = final_z.unsqueeze(dim=1)
            else:
                z3_seq = torch.cat((z3_seq, final_z.unsqueeze(dim=1)), 1)

        return z3_seq, logdet

    def reverse_flow(self, input_seq, cond_seq, logdet):
        """
        Backward path

        Args:
            input_seq:    input sequence in "Z" space
            cond_seq:     sequence with conditioning information
            logdet:       log determinant of the Jacobian from the previous operations

        Returns:
            z1_seq:       output sequence in "X" space
            logdet:       new value of log determinant of the Jacobian
        """

        z1_seq = None
        z2_seq = None
        z3_seq = None
        input_u_seq = None
        input_l_seq = None

        seq_len = input_seq.shape[1]

        #### Go though each steps of the flow separately for the whole sequence

        # 3.1 split on upper and lower

        for time_st in range(seq_len):

            curr_input = input_seq[:, time_st, :]

            in_l, in_u = thops.split_feature(curr_input, "split")

            # Add current encodings "z" to the sequence of encodings
            if input_l_seq is None:
                input_l_seq = in_l.unsqueeze(dim=1)
            else:
                input_l_seq = torch.cat((input_l_seq, in_l.unsqueeze(dim=1)), 1)

            # Add current encodings "z" to the sequence of encodings
            if input_u_seq is None:
                input_u_seq = in_u.unsqueeze(dim=1)
            else:
                input_u_seq = torch.cat((input_u_seq, in_u.unsqueeze(dim=1)), 1)

        # 3.2 coupling
        for time_st in range(seq_len):

            zl = input_l_seq[:, time_st, :]
            zu = input_u_seq[:, time_st, :]

            speech_cond = cond_seq[:, time_st, :]

            flow_conv_cond = self.from_z_sequence_to_cond(input_l_seq, seq_len, time_st)

            curr_cond = torch.cat((speech_cond, flow_conv_cond), dim=1)

            # Require some conditioning
            assert curr_cond is not None

            if self.flow_coupling == "additive":
                zu = zu - self.f(zl, curr_cond)
            elif self.flow_coupling == "affine":
                h = self.f(zl, curr_cond)
                shift, scale = thops.split_feature(h, "cross")
                scale = torch.sigmoid(scale + 2.0).clamp(self.scale_eps)

                if self.scale_logging:
                    self.scale = scale
                zu = zu / scale
                zu = zu - shift
                logdet = -thops.sum(torch.log(scale), dim=[1]) + logdet

            curr_z3 = thops.cat_feature(zl, zu)

            # Add current encoding "z" to the sequence of encodings
            if z3_seq is None:
                z3_seq = curr_z3.unsqueeze(dim=1)
            else:
                z3_seq = torch.cat((z3_seq, curr_z3.unsqueeze(dim=1)), 1)

        # 2. permute
        for time_st in range(seq_len):

            curr_z3 = z3_seq[:, time_st, :]

            curr_z2, logdet = FlowStep.FlowPermutation[self.flow_permutation](
                self, curr_z3.float(), logdet, True
            )

            # Add current encoding "z" to the sequence of encodings
            if z2_seq is None:
                z2_seq = curr_z2.unsqueeze(dim=1)
            else:
                z2_seq = torch.cat((z2_seq, curr_z2.unsqueeze(dim=1)), 1)

        # 1. actnorm
        for time_st in range(seq_len):
            curr_z2 = z2_seq[:, time_st, :]

            z, logdet = self.actnorm(curr_z2, logdet=logdet, reverse=True)

            curr_z1 = z

            # Add current encoding "z" to the sequence of encodings of the 1st operation
            if z1_seq is None:
                z1_seq = curr_z1.unsqueeze(dim=1)
            else:
                z1_seq = torch.cat((z1_seq, curr_z1.unsqueeze(dim=1)), 1)

        return z1_seq, logdet

class SeqFlowNet(nn.Module):
    """
    The whole flow as visualized below
    """

    def __init__(
        self,
        C,
        hidden_channels,
        cond_dim,
        K,
        L,
        hparams,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="additive",
        LU_decomposed=False,
        scale_eps=1e-6,
        scale_logging=False,
        glow_rnn_type=None,
    ):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L

        # About sequence processing
        self.past_context = hparams.Cond["Speech"]["prev_context"]
        self.future_context = hparams.Cond["Speech"]["future_context"]

        # Encode speech features
        self.encode_speech = nn.Sequential(nn.Linear(hparams.Cond["Speech"]["dim"],
                                                     hparams.Cond["Speech"]["fr_enc_dim"]), nn.LeakyReLU(),
                                           nn.Dropout(hparams.dropout))

        # To reduce deminsionality of the speech encoding
        self.reduce_speech_enc = nn.Sequential(nn.Linear(int(hparams.Cond["Speech"]["fr_enc_dim"] * \
                                                             (self.past_context + self.future_context)),
                                                         hparams.Cond["Speech"]["total_enc_dim"]),
                                               nn.LeakyReLU(), nn.Dropout(hparams.dropout))

        # Map conditioning to prior
        self.cond2prior = nn.Linear(hparams.Glow["speech_cond_dim"], hparams.Glow["distr_dim"]*2)

        for l in range(L):

            # 2. K FlowStep
            for k in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        cond_dim=cond_dim,
                        hparams=hparams,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        L=l,
                        K=k,
                        scale_eps=scale_eps,
                        scale_logging=scale_logging,
                        glow_rnn_type=glow_rnn_type,
                    )
                )
                self.output_shapes.append([-1, C])


    def create_conditioning(self, batch, time_st):
        """
        Pre-calculates a sequence of conditioning information for a given batch

        Args:
            batch:     current training batch (input features only)
            time_st:   current time step

        Returns:
            curr_cond: sequence of conditioning information

        """

        # take current audio and text of the speech
        curr_audio = batch["audio"][:, time_st - self.past_context:time_st + self.future_context]
        curr_text = batch["text"][:, time_st - self.past_context:time_st + self.future_context]
        curr_speech = torch.cat((curr_audio, curr_text), 2)

        # encode speech
        speech_encoding_full = self.encode_speech(curr_speech)

        speech_encoding_concat = torch.flatten(speech_encoding_full, start_dim=1)
        speech_cond_info = self.reduce_speech_enc(speech_encoding_concat)

        curr_cond = speech_cond_info

        return curr_cond


    def forward(self, input_seq_, condition_speech_info, logdet=0.0, reverse=False):

        # First create all the conditioning info
        cond_seq = None
        seq_len = condition_speech_info["audio"].shape[1]

        for time_st in range(self.past_context, seq_len - self.future_context):

            curr_cond = self.create_conditioning(condition_speech_info, time_st)

            if cond_seq is None:
                cond_seq = curr_cond.unsqueeze(dim=1)
            else:
                cond_seq = torch.cat((cond_seq, curr_cond.unsqueeze(dim=1)), 1)

        speech_cond_seq = cond_seq

        if not reverse:
            # Remove info which has no conditioning
            input_seq_cropped = input_seq_[:, self.past_context:-self.future_context]

            return self.encode(input_seq_cropped, speech_cond_seq, logdet)
        else:

            return self.decode(input_seq_, speech_cond_seq)

    def encode(self, z_seq, condition_seq, logdet=0.0):
        """
        Forward path
        """

        for layer in self.layers:
            z_seq, logdet = layer(z_seq, condition_seq, logdet, reverse=False)

        # Add prior log likelihood
        prior_nll,  mu, sigma = self.calc_prior_nll(z_seq, condition_seq)

        debug = False
        if debug:
            print("\nEncode\n")
            print("Prior logdet: ", prior_nll)
            print("Logdet: ", logdet)

        return z_seq, logdet, prior_nll,  mu, sigma

    def decode(self, z_seq, condition_seq):
        """
        Backward path
        """

        # Sample z's, if needed
        if z_seq is None:
            z_seq, prior_nll,  mu, sigma = self.sample_n_calc_nll(condition_seq)
        else:
            prior_nll,  mu, sigma = self.calc_prior_nll(z_seq, condition_seq)

        # backward path
        logdet = 0.0
        for layer in reversed(self.layers):
            z_seq, logdet = layer(z_seq, condition_seq, logdet, reverse=True)

        debug = False
        if debug:
            print("\nDecode\n")
            print("Prior logdet: ", prior_nll)
            print("Logdet: ", logdet)


        return z_seq, logdet, prior_nll,  mu, sigma


    def calc_prior_nll(self, z_seq, cond_info_seq):
        """
        Calculate log likelihood for the z_enc accordingly to the prior
        Args:
            z_seq:            sequence of z_enc values
            cond_info_seq:    sequence of conditioning information

        Returns:
            total_nll:        negative log likelihood for the given "z" sequence under prior given by the conditioning
            mu:               mean of the prior distribution
            sigma:            variance of the prior distributuion

        """
        seq_len = cond_info_seq.shape[1]
        eps = 1e-5
        total_nll = 0

        for time_st in range(seq_len):

            curr_z = z_seq[:, time_st]
            curr_cond = cond_info_seq[:, time_st]

            prior_info = self.cond2prior(curr_cond)

            mu, sigma = thops.split_feature(prior_info, "split")

            # Normalize values
            sigma = torch.sigmoid(sigma) + eps
            mu = torch.tanh(mu)

            log_sigma = torch.log(sigma)

            nll = - DiagGaussian.log_likelihood(mu, log_sigma, curr_z)

            total_nll += nll

        return total_nll,  mu, sigma


    def sample_n_calc_nll(self, cond_info_seq):
        """
        Sample z's from the prior given by the conditioning
        and calculate log likelihood for the z_seq accordingly to the prior
        Args:
            cond_info_seq:    sequence of conditioning information

        Returns:
            z_seq:            samples sequence in latent space
            total_nll:        negative log likelihood for the given "z" sequence under prior given by the conditioning

        """
        seq_len = cond_info_seq.shape[1]
        eps = 1e-5
        total_nll = 0
        z_seq = None

        for time_st in range(seq_len):

            curr_cond = cond_info_seq[:, time_st]

            prior_info = self.cond2prior(curr_cond)

            mu, sigma = thops.split_feature(prior_info, "split")

            # Normalize values
            sigma = torch.sigmoid(sigma) + eps
            mu = torch.tanh(mu)

            log_sigma = torch.log(sigma)

            # sample
            curr_z = modules.DiagGaussian.sample(mu, sigma).to(curr_cond.device)

            # get nll
            nll = -DiagGaussian.log_likelihood(mu, log_sigma, curr_z)

            total_nll += nll

            # Add z into the sequence
            if z_seq is None:
                z_seq = curr_z.unsqueeze(dim=1).detach()
            else:
                z_seq = torch.cat((z_seq, curr_z.unsqueeze(dim=1).detach()), 1)

        return z_seq, total_nll, mu, sigma


class Seq_Flow(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.flow = SeqFlowNet(
            C=hparams.Glow["distr_dim"],
            hidden_channels=hparams.Glow["hidden_channels"],
            cond_dim=hparams.Glow["speech_cond_dim"],
            hparams=hparams,
            K=hparams.Glow["K"],
            L=hparams.Glow["L"],
            actnorm_scale=hparams.Glow["actnorm_scale"],
            flow_permutation=hparams.Glow["flow_permutation"],
            flow_coupling=hparams.Glow["flow_coupling"],
            LU_decomposed=hparams.Glow["LU_decomposed"],
            scale_eps=hparams.Glow["scale_eps"],
            scale_logging=hparams.Validation["scale_logging"],
            glow_rnn_type=hparams.Glow["rnn_type"]
        )

        self.hparams = hparams

        self.past_context = hparams.Cond["Speech"]["prev_context"]
        self.future_context = hparams.Cond["Speech"]["future_context"]

    def forward(
        self,
        all_the_batch_data,
        z_seq=None,
        reverse=False,
    ):

        if not reverse:
            return self.normal_flow(all_the_batch_data)
        else:
            return self.reverse_flow(z_seq, all_the_batch_data)

    def normal_flow(self,  all_the_batch_data):

        x_seq = all_the_batch_data["gesture"]
        speech_batch_data = dict(all_the_batch_data)
        del speech_batch_data["gesture"]

        logdet = torch.zeros_like(all_the_batch_data["audio"][:, 0, 0])

        return self.flow(x_seq, speech_batch_data, logdet=logdet, reverse=False)

    def reverse_flow(self, z_seq, all_the_batch_data):
        with torch.no_grad():

            speech_condition = dict(all_the_batch_data)
            del speech_condition["gesture"]

            # Backward flow
            return self.flow(z_seq, speech_condition, logdet=None, reverse=True)


    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if m.__class__.__name__.find("ActNorm") >= 0:
                m.inited = inited
