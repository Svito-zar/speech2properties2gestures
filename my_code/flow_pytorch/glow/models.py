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

    def forward(self, input_, audio_features, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input_, audio_features, logdet)
        else:
            return self.reverse_flow(input_, audio_features, logdet)

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


class FlowNet(nn.Module):
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

        for l in range(L):

            # 2. K FlowStep
            for k in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        cond_dim=cond_dim,
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

    def forward(self, input_, condition, logdet=0.0, reverse=False):
        # audio_features = self.conditionNet(audio_features)  # Spectrogram

        if not reverse:
            return self.encode(input_, condition, logdet)
        else:
            return self.decode(input_, condition)

    def encode(self, z, condition, logdet=0.0):
        """
        Forward path
        """

        for layer in self.layers:
            z, logdet = layer(z, condition, logdet, reverse=False)
        return z, logdet

    def decode(self, z, condition):
        """
        Backward path
        """

        logdet = 0.0

        for layer in reversed(self.layers):
            z, logdet = layer(z, condition, logdet, reverse=True)
        return z, logdet


class Seq_Flow(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.flow = FlowNet(
            C=hparams.Glow["distr_dim"],
            hidden_channels=hparams.Glow["hidden_channels"],
            cond_dim=hparams.Glow["cond_dim"],
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
        self.autoregr_hist_length = hparams.Cond["Autoregression"]["history_length"]

        # Encode speech features
        self.encode_speech = nn.Sequential(nn.Linear(hparams.Cond["Speech"]["dim"],
                                                     hparams.Cond["Speech"]["fr_enc_dim"]), nn.LeakyReLU(),
                                           nn.Dropout(hparams.dropout))

        # To reduce deminsionality of the speech encoding
        self.reduce_speech_enc = nn.Sequential(nn.Linear(int(hparams.Cond["Speech"]["fr_enc_dim"] * \
                                                             (self.past_context + self.future_context)),
                                                         hparams.Cond["Speech"]["total_enc_dim"]),
                                               nn.LeakyReLU(), nn.Dropout(hparams.dropout))

        self.cond2prior = nn.Linear(hparams.Glow["cond_dim"], hparams.Glow["distr_dim"]*2)

    def forward(
        self,
        all_the_batch_data,
        z_seq=None,
        mean=None,
        std=None,
        reverse=False,
        output_shape=None,
    ):

        if not reverse:
            return self.normal_flow(all_the_batch_data)
        else:
            return self.reverse_flow(z_seq, all_the_batch_data, mean, std)

    def normal_flow(self,  all_the_batch_data):

        x_seq = all_the_batch_data["gesture"]
        speech_batch_data = dict(all_the_batch_data)
        del speech_batch_data["gesture"]

        logdet = torch.zeros_like(all_the_batch_data["audio"][:, 0, 0])
        z_seq = []
        seq_len = speech_batch_data["audio"].shape[1]
        eps = 1e-4
        total_loss = 0

        for time_st in range(self.past_context, seq_len - self.future_context):

            curr_output = x_seq[:, time_st, :]

            curr_cond = self.create_conditioning(speech_batch_data, time_st)

            z_enc, objective = self.flow(curr_output, curr_cond,  logdet=logdet, reverse=False)

            prior_info = self.cond2prior(curr_cond)

            mu, sigma = thops.split_feature(prior_info, "split")

            # Normalize values
            sigma = torch.sigmoid(sigma) + eps
            mu = torch.tanh(mu)

            log_sigma = torch.log(sigma)

            tmp_loss = self.loss(objective, z_enc, mu, log_sigma)

            total_loss += torch.mean(tmp_loss)

            z_seq.append(z_enc.detach())

        return z_seq, total_loss

    def reverse_flow(self, z_seq, all_the_batch_data, mean, std):
        with torch.no_grad():

            eps = 1e-3
            produced_poses = None
            condition = dict(all_the_batch_data)
            del condition["gesture"]
            seq_len = condition["audio"].shape[1]
            log_det_sum = 0

            for time_st in range(self.past_context, seq_len - self.future_context):

                curr_cond = self.create_conditioning(condition, time_st)

                # Define prior parameters
                if mean is None:

                    # Require some conditioning
                    assert curr_cond is not None

                    prior_info = self.cond2prior(curr_cond)

                    mu, sigma = thops.split_feature(prior_info, "split")

                    sigma = torch.sigmoid(sigma) + eps
                    mu = torch.tanh(mu)

                else:
                    mu = mean
                    sigma = std

                # Sample , if needed
                if z_seq is None:
                    curr_z = modules.DiagGaussian.sample(mu, sigma).to(curr_cond.device)
                else:
                    curr_z = z_seq[:,time_st]

                # Backward flow
                curr_output, curr_log_det = self.flow(curr_z, curr_cond, logdet=0.0, reverse=True) #, std=sigma * self.hparams.Infer["temp"])

                log_det_sum += curr_log_det

                # add current frame to the total motion sequence
                if produced_poses is None:
                    produced_poses = curr_output.unsqueeze(1)
                else:
                    produced_poses = torch.cat((produced_poses, curr_output.unsqueeze(1)), 1)

        return produced_poses, log_det_sum


    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if m.__class__.__name__.find("ActNorm") >= 0:
                m.inited = inited


    def create_conditioning(self, batch, time_st):

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

    def loss(self, objective, z, mu, log_sigma):
        log_likelihood = objective + DiagGaussian.log_likelihood(mu, log_sigma, z)
        nll = (-log_likelihood) / float(np.log(2.0))
        return nll