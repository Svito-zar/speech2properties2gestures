import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
        cond_dim,
        rnn_type,
    ):
        """
        input_size:             (glow) channels
        output_size:            output channels (ToDo: should be equal to input_size?)
        hidden_size:            size of the hidden layer of the NN
        cond_dim:               final dim. of the conditioning info
        rnn_type:               GRU/LSTM
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        #self.cond_transform = nn.Sequential(
        #    nn.Linear(feature_encoder_dim, cond_dim), nn.LeakyReLU(),
        #)

        self.input_hidden = nn.Sequential(nn.Linear(input_size + cond_dim, hidden_size), nn.LeakyReLU(), nn.Dropout(0.2))

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
        if flow_coupling == "additive":
            self.f = f_seq(
                in_channels // 2,
                in_channels - in_channels // 2,
                hidden_channels,
                cond_dim,
                glow_rnn_type,
            )
        elif flow_coupling == "affine":
            if in_channels % 2 == 0:
                self.f = f_seq(
                    in_channels // 2,
                    in_channels,
                    hidden_channels,
                    cond_dim,
                    glow_rnn_type,
                )
            else:
                self.f = f_seq(
                    in_channels // 2,
                    in_channels + 1,
                    hidden_channels,
                    cond_dim,
                    glow_rnn_type,
                )


    def forward(self, input_seq_, speech_cond_seq, logdet=None, reverse=False):

        if not reverse:

            if logdet is None:
                logdet = torch.zeros_like(input_seq_[:, 0, 0])
            z_seq = None
            seq_len = input_seq_.shape[1]

            for time_st in range(seq_len):
                curr_output = input_seq_[:, time_st, :]

                curr_cond = speech_cond_seq[:, time_st, :]

                curr_z, logdet  = self.normal_flow(curr_output, curr_cond, logdet)
                # ToDo: should it be one log_det for the whole sequence?

                # Add current encoding "z" to the sequence of encodings
                if z_seq is None:
                    z_seq = curr_z.unsqueeze(dim=1).detach()
                else:
                    z_seq = torch.cat((z_seq, curr_z.unsqueeze(dim=1).detach()), 1)

            return z_seq, logdet

        else:

            assert input_seq_ is not None

            logdet = torch.zeros_like(input_seq_[:, 0, 0])
            z_seq = None
            seq_len = input_seq_.shape[1]

            for time_st in range(seq_len):
                    curr_output = input_seq_[:, time_st, :]

                    curr_cond = speech_cond_seq[:, time_st, :]

                    curr_z, logdet = self.reverse_flow(curr_output, curr_cond, logdet)
                    # ToDo: should it be one log_det for the whole sequence?

                    # Add z into the sequence
                    if z_seq is None:
                        z_seq = curr_z.unsqueeze(dim=1).detach()
                    else:
                        z_seq = torch.cat((z_seq, curr_z.unsqueeze(dim=1).detach()), 1)

            return z_seq, logdet

    def normal_flow(self, input_, condition, logdet):
        """
        Forward path
        """

        # 1. actnorm
        z, logdet = self.actnorm(input_, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z.float(), logdet, False
        )

        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")

        # Require some conditioning
        assert condition is not None

        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1, condition)
        elif self.flow_coupling == "affine":
            h = self.f(z1, condition)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0).clamp(self.scale_eps)
            if self.scale_logging:
                self.scale = scale
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1]) + logdet
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, input_, condition, logdet):
        """
        Backward path
        """

        # 1.coupling
        z1, z2 = thops.split_feature(input_, "split")

        # Require some conditioning
        assert condition is not None

        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1, condition)
        elif self.flow_coupling == "affine":
            h = self.f(z1, condition)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0).clamp(self.scale_eps)
            z2 = z2 / scale
            z2 = z2 - shift

            logdet = -thops.sum(torch.log(scale), dim=[1]) + logdet
        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True
        )
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


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
        self.cond2prior = nn.Linear(hparams.Glow["cond_dim"], hparams.Glow["distr_dim"]*2)

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
                cond_seq = curr_cond.unsqueeze(dim=1).detach()
            else:
                cond_seq = torch.cat((cond_seq, curr_cond.unsqueeze(dim=1).detach()), 1)

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
        prior_nll = self.calc_prior_nll(z_seq, condition_seq)

        nll = logdet + prior_nll

        # DEBUG
        debug = False
        if debug:
            print("\nEncode\n")
            print("Z_seq shape: ", z_seq.shape)
            print("Z_seq: ", z_seq[:3,:3,:3])
            print("Prior logdet: ", self.calc_prior_nll(z_seq, condition_seq))

        return z_seq, nll

    def decode(self, z_seq, condition_seq):
        """
        Backward path
        """

        # Sample z's, if needed
        if z_seq is None:
            z_seq, prior_nll = self.sample_n_calc_nll(condition_seq)
        else:
            prior_nll = self.calc_prior_nll(z_seq, condition_seq)

        # DEBUG
        debug = False
        if debug:
            print("\nDecode\n")
            print("Z_seq shape: ", z_seq.shape)
            print("Z_seq: ", z_seq[:3, :3, :3])
            print("Prior logdet: ", prior_nll)

        # backward path
        logdet = 0.0
        for layer in reversed(self.layers):
            z_seq, logdet = layer(z_seq, condition_seq, logdet, reverse=True)

        nll = logdet + prior_nll

        return z_seq, nll


    def calc_prior_nll(self, z_seq, cond_info_seq):
        """
        Calculate log likelihood for the z_enc accordingly to the prior
        Args:
            z_seq:            sequence of z_enc values
            cond_info_seq:    sequence of conditioning information

        Returns:
            total_nll:        negative log likelihood for the given "z" sequence under prior given by the conditioning

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

            if time_st == 5:

                print("\nCalc Pior\n")
                print("mu : ", mu[:3, :3])
                print("log_sigma: ", log_sigma[:3, :3])

            nll = -DiagGaussian.log_likelihood(mu, log_sigma, curr_z)

            total_nll += nll

        return total_nll


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

            if time_st == 5:
                print("\nCalc Pior\n")
                print("mu : ", mu[:3, :3])
                print("log_sigma: ", log_sigma[:3, :3])

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

        return z_seq, total_nll


class Seq_Flow(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.flow = SeqFlowNet(
            C=hparams.Glow["distr_dim"],
            hidden_channels=hparams.Glow["hidden_channels"],
            cond_dim=hparams.Glow["cond_dim"],
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