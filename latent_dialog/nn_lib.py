import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from latent_dialog.utils import cast_type, FLOAT


class Hidden2Gaussian(nn.Module):
    def __init__(self, input_size, output_size, is_lstm=False, has_bias=True):
        super(Hidden2Gaussian, self).__init__()
        if is_lstm:
            self.mu_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar_h = nn.Linear(input_size, output_size, bias=has_bias)

            self.mu_c = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.mu = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar = nn.Linear(input_size, output_size, bias=has_bias)

        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h, c= inputs
            if h.dim() == 3:
                h = h.squeeze(0)
                c = c.squeeze(0)

            mu_h, mu_c = self.mu_h(h), self.mu_c(c)
            logvar_h, logvar_c = self.logvar_h(h), self.logvar_c(c)
            return mu_h+mu_c, logvar_h+logvar_c
        else:
            # if inputs.dim() == 3:
            #    inputs = inputs.squeeze(0)
            mu = self.mu(inputs)
            logvar = self.logvar(inputs)
            return mu, logvar


class GaussianConnector(nn.Module):
    def __init__(self, use_gpu):
        super(GaussianConnector, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, mu, logvar):
        """
        Sample a sample from a multivariate Gaussian distribution with a diagonal covariance matrix using the
        reparametrization trick.
        :param mu: a tensor of size [batch_size, variable_dim]. Batch_size can be None to support dynamic batching
        :param logvar: a tensor of size [batch_size, variable_dim]. Batch_size can be None.
        :return:
        """
        epsilon = th.randn(logvar.size())
        epsilon = cast_type(Variable(epsilon), FLOAT, self.use_gpu)
        std = th.exp(0.5 * logvar)
        z = mu + std * epsilon
        return z
