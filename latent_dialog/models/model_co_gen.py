import numpy as np
import torch as th
import torch.nn as nn
from latent_dialog.woz_util  import EOS, PAD, BOS
from latent_dialog.utils import FLOAT, LONG, cast_type
from latent_dialog.enc2dec.encdec import GEN, TEACH_FORCE, RnnUttEncoder, RnnRspEncoder, RnnRspDecoder
from latent_dialog.criterions import NLLEntropy, NormKLLoss
from latent_dialog import nn_lib
from latent_dialog.models import BaseModel



class Co_Gen(BaseModel):
    def __init__(self, corpus, config):
        super(Co_Gen, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.act_vocab = corpus.act_vocab
        self.act_vocab_dict = corpus.act_vocab_dict
        self.act_vocab_size = len(self.act_vocab)
        self.act_bos_id = self.act_vocab_dict[BOS]
        self.act_eos_id = self.act_vocab_dict[EOS]
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.dim_act_vec = config.dim_act_vec
        self.act_vec_dropout = config.act_vec_dropout
        self.w_kl = config.w_kl
        self.w_kl_a2r = config.w_kl_a2r
        self.w_rec = config.w_rec
        self.w_rec_a2a = config.w_rec_a2a
        self.w_rec_r2r = config.w_rec_r2r
        self.w_dec = config.w_dec
        self.w_dec_c2a = config.w_dec_c2a
        self.w_dec_c2r = config.w_dec_c2r
        self.embedding_dim = config.embed_size
        self.dim_z = config.dim_z  # size of latent variable z
        self.dim_enc = 2 * self.config.utt_cell_size if self.config.bi_utt_cell else self.config.utt_cell_size  # size of encoded c/a/r
        self.dim_h = config.dim_h # size of decoder hidden layer

        self.init_net()

    def init_net(self):
        # usr utt encoder
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=self.embedding_dim,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=self.config.enc_rnn_cell,  # gru
                                         utt_cell_size=self.config.utt_cell_size,
                                         num_layers=self.config.num_layers,
                                         input_dropout_p=self.config.dropout,
                                         output_dropout_p=self.config.dropout,
                                         bidirectional=self.config.bi_utt_cell,
                                         variable_lengths=False,  # since all inputs have been padded
                                         use_attn=self.config.enc_use_attn,
                                         embedding=self.embedding)
        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.bs_size + self.db_size,
                                          self.dim_z, is_lstm=False)

        # rsp encoder
        self.rsp_encoder_embedding = self.embedding if self.config.utt_rsp_enc_share_emb else nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.rsp_encoder = RnnRspEncoder(embedding_dim=self.embedding_dim,
                                             rnn_cell=self.config.enc_rnn_cell,  # gru
                                             cell_e_size=self.dim_enc,
                                             num_layers=self.config.num_layers,
                                             input_dropout_p=self.config.dropout,
                                             output_dropout_p=self.config.dropout,
                                             bidirectional=self.config.bi_rsp_enc_cell,
                                             variable_lengths=False,  # since all inputs have been padded
                                             use_attn=self.config.enc_use_attn,
                                             embedding=self.rsp_encoder_embedding)

        # act encoder
        self.act_encoder_embedding = nn.Embedding(num_embeddings=self.act_vocab_size, embedding_dim=self.embedding_dim)
        self.act_encoder = RnnRspEncoder(embedding_dim=self.embedding_dim,
                                           rnn_cell=self.config.enc_rnn_cell,  # gru
                                           cell_e_size=self.dim_enc,
                                           num_layers=self.config.num_layers,
                                           input_dropout_p=self.config.dropout,
                                           output_dropout_p=self.config.dropout,
                                           bidirectional=self.config.bi_act_enc_cell,
                                           variable_lengths=False,  # since all inputs have been padded
                                           use_attn=self.config.enc_use_attn,
                                           embedding=self.act_encoder_embedding)

        # rsp generator
        self.z_embedding_x2r = nn.Linear(self.dim_z + self.dim_act_vec, self.dim_h, bias=True)
        self.rsp_decoder_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.rsp_projection = nn.Linear(self.dim_h, self.vocab_size, bias=True)
        self.rsp_generator = RnnRspDecoder(rnn_cell=self.config.dec_rnn_cell,  # gru
                                               input_size=self.embedding_dim,
                                               hidden_size=self.dim_h,
                                               num_layers=self.config.num_layers,
                                               bidirectional=False,
                                               input_dropout_p=self.config.dropout,
                                               output_dropout_p=self.config.dropout,
                                               embedding=self.rsp_decoder_embedding,
                                               project=self.rsp_projection,
                                               vocab_size=self.vocab_size,
                                               sys_id=self.bos_id,
                                               eos_id=self.eos_id,
                                               use_gpu=self.config.use_gpu,
                                               max_dec_len=self.config.max_dec_len)

        # act generator
        self.z_embedding_x2a = nn.Linear(self.dim_z + self.dim_act_vec, self.dim_h, bias=True)
        self.act_decoder_embedding = nn.Embedding(num_embeddings=self.act_vocab_size, embedding_dim=self.embedding_dim)
        self.act_projection = nn.Linear(self.dim_h, self.act_vocab_size, bias=True)
        self.act_generator = RnnRspDecoder(rnn_cell=self.config.dec_rnn_cell,  # gru
                                             input_size=self.embedding_dim,
                                             hidden_size=self.dim_h,
                                             num_layers=self.config.num_layers,
                                             bidirectional=False,
                                             input_dropout_p=self.config.dropout,
                                             output_dropout_p=self.config.dropout,
                                             embedding=self.act_decoder_embedding,
                                             project=self.act_projection,
                                             vocab_size=self.act_vocab_size,
                                             sys_id=self.act_bos_id,
                                             eos_id=self.act_eos_id,
                                             use_gpu=self.config.use_gpu,
                                             max_dec_len=self.config.max_dec_len)

        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)

        self.relu = nn.ReLU()
        self.nll = NLLEntropy(self.pad_id, self.config.avg_type)
        self.gauss_kl = NormKLLoss(unit_average=True)

    def valid_loss(self, losses, batch_cnt=None, add_loss_adv=None):
        # to (weighted) sum up losses
        total_loss = 0
        for key, loss in losses.items():
            if key == 'loss_kl':
                total_loss += self.w_kl * loss
            elif key == 'loss_kl_a2r':
                total_loss += self.w_kl_a2r * loss
            elif key == 'nll_loss_rec_a2a':
                total_loss += self.w_rec * self.w_rec_a2a * loss
            elif key == 'nll_loss_rec_r2r':
                total_loss += self.w_rec * self.w_rec_r2r * loss
            elif key == 'nll_loss_dec_c2a':
                total_loss += self.w_dec * self.w_dec_c2a * loss
            elif key == 'nll_loss_dec_c2r':
                total_loss += self.w_dec * self.w_dec_c2r * loss
            else:
                total_loss += loss
        return total_loss

    def drop_vec(self, act_vec, act_vec_dropout):
        batch, length = act_vec.size()
        probs = cast_type(th.ones(batch, length) * (1-act_vec_dropout), FLOAT, self.use_gpu)
        mask = th.bernoulli(probs)
        act_vec_masked = act_vec * mask
        return act_vec_masked

    def forward(self, data_feed, mode, gen_type='greedy', use_py=None, return_latent=False, epoch=np.inf, act_vec_drop=True):
        # usr_utts, sys_utts
        ctx_lens = data_feed['context_lens']  # (batch_size, ), number of turns, i.e. [#turns_in_context for each pair]
        batch_size = len(ctx_lens)
        usr_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)  # (batch_size, max_utt_len)
        rsps = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        acts = self.np2var(data_feed['out_acts'], LONG)  # (batch_size, max_act_len)
        # add teach_rsps, teach_acts, target_rsps, target_acts
        teach_rsps = self.np2var(data_feed['teach_rsps'], LONG)  # (batch_size, max_out_len)
        target_rsps = self.np2var(data_feed['target_rsps'], LONG)  # (batch_size, max_out_len)
        teach_acts = self.np2var(data_feed['teach_acts'], LONG)  # (batch_size, max_act_len)
        target_acts = self.np2var(data_feed['target_acts'], LONG)  # (batch_size, max_act_len)
        # <s> := 4, pad := 0, <eos> := 6

        # bs, db
        bs = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, 94)
        db = self.np2var(data_feed['db'], FLOAT)  # (batch_size, 30)
        act_vec_in = self.np2var(data_feed['act_vec'], FLOAT)  # (batch_size, 44)
        if mode == GEN:
            act_vec_drop = False
        if act_vec_drop:
            act_vec = self.drop_vec(act_vec_in, self.act_vec_dropout)
        else:
            act_vec = act_vec_in

        # result to store loss
        result = {}

        # encode for x
        # encoder input: (batch_size, 1, max_utt_len), since only use one prev turn as context
        # or (batch_size, 1, max_utt_len+max_act_len) if add usr action in context
        user_utt_summary, _, _ = self.utt_encoder(usr_utts.unsqueeze(1)) # (batch_size, 1，dim_enc)
        # embed bs and db
        x_enc = th.cat([bs, db, user_utt_summary.squeeze(1)], dim=1)
        # compute the prior p(z|c)
        mu_c2z, logvar_c2z = self.c2z(x_enc)

        if mode == GEN:
            z_sample = mu_c2z  # dim_z
        else:
            z_sample = self.gauss_connector(mu_c2z, logvar_c2z)

        # z_sample = th.cat([z_sample, x_enc], dim=1)
        z_sample_x2a = self.relu(self.z_embedding_x2a(th.cat([z_sample, act_vec], dim=1)))
        z_sample_x2r = self.relu(self.z_embedding_x2r(th.cat([z_sample, act_vec], dim=1)))
        generator_init_state_c2a = z_sample_x2a.unsqueeze(0)  # (num_directions*num_layer, batch_size, dim_h)
        generator_init_state_c2r = z_sample_x2r.unsqueeze(0)  # (num_directions*num_layer, batch_size, dim_h)

        # generate system action
        prob_outputs_c2a, hidden_state_c2a, ret_dict_c2a = self.act_generator(batch_size=batch_size,
                                                                       dec_inputs=teach_acts,
                                                                       dec_init_state=generator_init_state_c2a,
                                                                       mode=mode,
                                                                       gen_type=gen_type,
                                                                       beam_size=self.config.beam_size)
        # generate system response
        prob_outputs_c2r, hidden_state_c2r, ret_dict_c2r = self.rsp_generator(batch_size=batch_size,
                                                                       dec_inputs=teach_rsps,
                                                                       dec_init_state=generator_init_state_c2r,
                                                                       mode=mode,
                                                                       gen_type=gen_type,
                                                                       beam_size=self.config.beam_size)
        if mode == GEN:
            ret_dict_c2r['sample_z'] = z_sample
            return ret_dict_c2a, target_acts, ret_dict_c2r, target_rsps

        # else compute loss when mode = TEACH_FORCE

        # encode for branch 1: system action
        z_a = self.act_encoder(acts)  # (batch_size, dim_enc)
        z_a_combine = th.cat([bs, db, z_a], dim=1)
        mu_a2z, logvar_a2z = self.c2z(z_a_combine)
        z_sample_a2z = self.gauss_connector(mu_a2z, logvar_a2z)  # dim_z
        z_sample_a2z_x2y = self.relu(self.z_embedding_x2a(th.cat([z_sample_a2z, act_vec], dim=1)))

        # encode for branch 2: system response
        z_r = self.rsp_encoder(rsps)  # (batch_size, dim_enc)
        z_r_combine = th.cat([bs, db, z_r], dim=1)
        mu_r2z, logvar_r2z = self.c2z(z_r_combine)
        z_sample_r2z = self.gauss_connector(mu_r2z, logvar_r2z) # dim_z
        z_sample_r2z_x2y = self.relu(self.z_embedding_x2r(th.cat([z_sample_r2z, act_vec], dim=1)))

        # reconstruction for branch 1: system action
        generator_init_state_a2a = z_sample_a2z_x2y.unsqueeze(0) # (num_directions*num_layer, batch_size, dim_h)
        prob_outputs_a2a, decoder_hidden_state_a2a, ret_dict_a2a = self.act_generator(batch_size=batch_size,
                                               dec_inputs=teach_acts,
                                               dec_init_state=generator_init_state_a2a,
                                               mode=TEACH_FORCE)

        # reconstruction for branch 2: system response
        generator_init_state_r2r = z_sample_r2z_x2y.unsqueeze(0) # (num_directions*num_layer, batch_size, dim_h)
        prob_outputs_r2r, decoder_hidden_state_r2r, ret_dict_r2r = self.rsp_generator(batch_size=batch_size,
                                               dec_inputs=teach_rsps,
                                               dec_init_state=generator_init_state_r2r,
                                               mode=TEACH_FORCE)

        # loss_kl
        loss_kl_a2c = self.gauss_kl(mu_a2z, logvar_a2z, mu_c2z, logvar_c2z)
        loss_kl_r2c = self.gauss_kl(mu_r2z, logvar_r2z, mu_c2z, logvar_c2z)
        result['loss_kl'] = loss_kl_a2c + loss_kl_r2c

        loss_kl_a2r = self.gauss_kl(mu_a2z, logvar_a2z, mu_r2z, logvar_r2z)
        result['loss_kl_a2r'] = loss_kl_a2r

        # loss_rec
        # prob_outputs: (batch_size, max_dec_len, vocab_size)
        loss_rec_a2a = self.nll(prob_outputs_a2a, target_acts)
        loss_rec_r2r = self.nll(prob_outputs_r2r, target_rsps)
        result['nll_loss_rec_a2a'] = loss_rec_a2a
        result['nll_loss_rec_r2r'] = loss_rec_r2r

        # loss_dec
        # prob_outputs: (batch_size, max_dec_len, vocab_size)
        loss_dec_c2a = self.nll(prob_outputs_c2a, target_acts)
        loss_dec_c2r = self.nll(prob_outputs_c2r, target_rsps)
        result['nll_loss_dec_c2a'] = loss_dec_c2a
        result['nll_loss_dec_c2r'] = loss_dec_c2r

        return result

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2 * np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu - sample_z), 2) / (2.0 * var)
        return logprob

    def forward_rl(self, data_feed, max_words, temp=0.1, args=None):
        # user_utts, sys_utts
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        batch_size = len(ctx_lens)
        usr_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        rsp_labels = self.np2var(data_feed['rsp_labels'], FLOAT)  # (batch_size, 1)
        act_vec_in = self.np2var(data_feed['act_vec'], FLOAT)  # (batch_size, 44)
        act_vec = self.drop_vec(act_vec_in, self.act_vec_dropout)

        # bd, db
        bs = self.np2var(data_feed['bs'], FLOAT)
        db = self.np2var(data_feed['db'], FLOAT)

        # encode for x
        user_utt_summary, _, _ = self.utt_encoder(usr_utts.unsqueeze(1))
        # embed bs and db
        x_enc = th.cat([bs, db, user_utt_summary.squeeze(1)], dim=1)
        # compute the variational posterior
        mu_c2z, logvar_c2z = self.c2z(x_enc)  # q(z|x,db,ds)

        ### Transfer from x to y
        z_sample = th.normal(mu_c2z, th.sqrt(th.exp(logvar_c2z))).detach()
        logprob_sample_z = self.gaussian_logprob(mu_c2z, logvar_c2z, z_sample)
        joint_logpz = th.sum(logprob_sample_z, dim=1)

        # decode
        z_sample_x2r = self.relu(self.z_embedding_x2r(th.cat([z_sample, act_vec], dim=1)))
        generator_init_state_c2r = z_sample_x2r.unsqueeze(0)  # (num_directions*num_layer, batch_size, dim_h)

        logprobs_r, outs_r = self.rsp_generator.forward_rl(batch_size=batch_size,
                                                          dec_init_state=generator_init_state_c2r,
                                                          vocab=self.vocab,
                                                          max_words=max_words,
                                                          temp=temp)

        return logprobs_r, outs_r, joint_logpz, z_sample