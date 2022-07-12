import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class NLLEntropy(_Loss):
    def __init__(self, padding_idx, avg_type):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        self.avg_type = avg_type

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        pred = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)

        if self.avg_type is None:
            loss = F.nll_loss(pred, target, size_average=False, ignore_index=self.padding_idx)
        elif self.avg_type == 'seq':
            loss = F.nll_loss(pred, target, size_average=False, ignore_index=self.padding_idx)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = F.nll_loss(pred, target, ignore_index=self.padding_idx, reduce=False)
            loss = loss.view(-1, net_output.size(1))
            loss = th.sum(loss, dim=1)
            word_cnt = th.sum(th.sign(labels), dim=1).float()
            loss = loss / word_cnt
            loss = th.mean(loss)
        elif self.avg_type == 'word':
            loss = F.nll_loss(pred, target, size_average=True, ignore_index=self.padding_idx)
        else:
            raise ValueError('Unknown average type')

        return loss


class NormKLLoss(_Loss):
    def __init__(self, unit_average=False):
        super(NormKLLoss, self).__init__()
        self.unit_average = unit_average

    def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        # find the KL divergence between two Gaussian distribution
        loss = 1.0 + (recog_logvar - prior_logvar)
        loss -= th.div(th.pow(prior_mu - recog_mu, 2), th.exp(prior_logvar))
        loss -= th.div(th.exp(recog_logvar), th.exp(prior_logvar))
        if self.unit_average:
            kl_loss = -0.5 * th.mean(loss, dim=1)
        else:
            kl_loss = -0.5 * th.sum(loss, dim=1)
        avg_kl_loss = th.mean(kl_loss)
        return avg_kl_loss
