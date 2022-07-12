import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from latent_dialog.enc2dec.base_modules import BaseRNN
from latent_dialog.utils import cast_type, LONG
from latent_dialog.woz_util import EOS

TEACH_FORCE = 'teacher_forcing'
GEN = 'gen'
LENGTH_AVERAGE = True
INF = 1e4

class EncoderRNN(BaseRNN):
    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional, variable_lengths):
        super(EncoderRNN, self).__init__(input_dropout_p=input_dropout_p,
                                         rnn_cell=rnn_cell,
                                         input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         output_dropout_p=output_dropout_p,
                                         bidirectional=bidirectional)
        self.variable_lengths = variable_lengths  # False, since all inputs have been padded
        self.output_size = hidden_size*2 if bidirectional else hidden_size

    def forward(self, input_var, init_state=None, input_lengths=None, goals=None):
        embedded = self.input_dropout(input_var)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        return output, hidden


class RnnUttEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feat_size, goal_nhid, rnn_cell,
                 utt_cell_size, num_layers, input_dropout_p, output_dropout_p,
                 bidirectional, variable_lengths, use_attn, embedding=None):
        super(RnnUttEncoder, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p,
                              rnn_cell=rnn_cell,
                              input_size=embedding_dim + feat_size + goal_nhid,
                              hidden_size=utt_cell_size,
                              num_layers=num_layers,
                              output_dropout_p=output_dropout_p,
                              bidirectional=bidirectional,
                              variable_lengths=variable_lengths)

        self.utt_cell_size = utt_cell_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.utt_cell_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, feats=None, init_state=None, goals=None):
        batch_size, max_ctx_len, max_utt_len = utterances.size()
        # get word embeddings
        flat_words = utterances.view(-1, max_utt_len)  # (batch_size*max_ctx_len, max_utt_len)
        word_embeddings = self.embedding(flat_words)  # (batch_size*max_ctx_len, max_utt_len, embedding_dim)
        flat_mask = th.sign(flat_words).float()
        # enc_outs: (batch_size*max_ctx_len, max_utt_len, num_directions*utt_cell_size)
        # enc_last: (num_layers*num_directions, batch_size*max_ctx_len, utt_cell_size)
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)

        if self.use_attn:
            fc1 = th.tanh(self.key_w(enc_outs))  # (batch_size, max_utt_len, utt_cell_size)
            attn = self.query(fc1).squeeze(2)  # (batch_size, max_utt_len)
            attn = F.softmax(attn, attn.dim() - 1)  # (batch_size, max_utt_len, 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True) + 1e-10)).unsqueeze(2)
            utt_embedded = attn * enc_outs  # (batch_size*max_ctx_len, max_utt_len, num_directions*utt_cell_size)
            utt_embedded = th.sum(utt_embedded, dim=1)  # (batch_size*max_ctx_len, num_directions*utt_cell_size)
        else:
            attn = None
            utt_embedded = enc_last.transpose(0,
                                              1).contiguous()  # (batch_size*max_ctx_lens, num_layers*num_directions, utt_cell_size)
            utt_embedded = utt_embedded.view(-1,
                                             self.output_size)  # (batch_size*max_ctx_len*num_layers, num_directions*utt_cell_size)

        utt_embedded = utt_embedded.view(batch_size, max_ctx_len, self.output_size)
        # word_embedding view: (batch_size, max_ctx_len*max_utt_len, embedding_dim)
        # enc_outs view: (batch_size, max_ctx_len*max_utt_len, num_directions*utt_cell_size=output_size)
        return utt_embedded, word_embeddings.contiguous().view(batch_size, max_ctx_len * max_utt_len, -1), \
               enc_outs.contiguous().view(batch_size, max_ctx_len * max_utt_len, -1)


class RnnRspEncoder(nn.Module):
    def __init__(self, embedding_dim, rnn_cell,
                 cell_e_size, num_layers, input_dropout_p, output_dropout_p,
                 bidirectional, variable_lengths, use_attn, embedding):
        super(RnnRspEncoder, self).__init__()
        self.embedding = embedding
        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p,
                              rnn_cell=rnn_cell,
                              input_size=embedding_dim,
                              hidden_size=cell_e_size,
                              num_layers=num_layers,
                              output_dropout_p=output_dropout_p,
                              bidirectional=bidirectional,
                              variable_lengths=variable_lengths)
        self.cell_e_size = cell_e_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.cell_e_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.cell_e_size)
            self.query = nn.Linear(self.cell_e_size, 1)

    def forward(self, sents, init_state=None):
        batch_size, max_utt_len = sents.size()  # (batch_size, max_out_len)
        # get word embeddings
        flat_words = sents.view(-1, max_utt_len)  # (batch_size, max_utt_len)
        flat_mask = th.sign(flat_words).float()
        word_embeddings = self.embedding(sents)  # (batch_size, sent_len, embedding_dim)

        # enc_outs: (batch_size, sent_len, num_directions*dim_h)
        # enc_last: (num_layers*num_directions, batch_size, dim_h)
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)

        if self.use_attn:
            fc1 = th.tanh(self.key_w(enc_outs)) # (batch_size, max_utt_len, utt_cell_size)
            attn = self.query(fc1).squeeze(2)  # (batch_size, max_utt_len)
            attn = F.softmax(attn, attn.dim()-1) # (batch_size, max_utt_len, 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True)+1e-10)).unsqueeze(2)
            utt_embedded = attn * enc_outs # (batch_size, max_utt_len, num_directions*utt_cell_size)
            z = th.sum(utt_embedded, dim=1) # (batch_size, num_directions*utt_cell_size)
        else:
            attn = None
            z = enc_last.transpose(0, 1).contiguous() # (batch_size, num_layers*num_directions, utt_cell_size)
            # (batch_size, num_layers*num_directions, dim_h)
            z = z.view(-1, self.output_size)
            # (batch_size*num_layers, num_directions*dim_h)

        if self.multiplier == 2:
            z = z[:, :self.cell_e_size] + z[:, self.cell_e_size:]

        return z # (batch_size, dim_h)


class RnnRspDecoder(BaseRNN):
    def __init__(self, rnn_cell, input_size, hidden_size, num_layers, bidirectional,
                 input_dropout_p, output_dropout_p, embedding, project,
                 vocab_size, sys_id, eos_id, use_gpu, max_dec_len):

        super(RnnRspDecoder, self).__init__(rnn_cell=rnn_cell,
                                             input_size=input_size, # embedding_size
                                             hidden_size=hidden_size, # cell_g_size = dim_h
                                             num_layers=num_layers,
                                             input_dropout_p=input_dropout_p,
                                             output_dropout_p=output_dropout_p,
                                             bidirectional=bidirectional)  # False
        self.embedding = embedding
        self.project = project
        self.dec_cell_size = hidden_size
        self.output_size = vocab_size
        self.sys_id = sys_id
        self.eos_id = eos_id
        self.use_gpu = use_gpu
        self.max_dec_len = max_dec_len
        self.log_softmax = F.log_softmax

    # greedy and beam together
    def forward(self, batch_size, dec_inputs, dec_init_state, mode, gen_type=None, beam_size=None):
        # mode: TEACH_FORCE/GEN
        # gen_type: greedy/beam
        ret_dict = dict()
        vocab_size = self.output_size

        ##### Create Decoder Inputs #####
        # 1. decoder_input
        if mode != GEN and dec_inputs is not None: # mode = TEACH_FORCE
            decoder_input = dec_inputs
        else:  # mode = GEN
            if gen_type != 'beam':
                beam_size = 1
            # prepare the BOS inputs
            with th.no_grad():
                bos_var = Variable(th.LongTensor([self.sys_id]))
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size * beam_size, 1).clone()  # [b*k,1]

        # 2.decoder_hidden_state
        if mode == GEN and gen_type == 'beam':
            # beam search: repeat the initial states of the RNN
            # dec_init_state: [num_directions, batch, hidden_size], here num_directions = 1
            decoder_hidden_state = th.cat([dec_init_state.squeeze(0)] * beam_size, dim=-1).view(1, batch_size * beam_size, -1)  # [1,b*k,h]

        else: # h_0 should in size: (num_directions*num_layer, batch_size, dim_h)
            decoder_hidden_state = dec_init_state  # dec_init_state: [1, b, h]

        ##### Decode #####
        symbol_outputs, logits = [], []
        # 1. mode=TEACH_FORCE
        if mode == TEACH_FORCE:
            prob_outputs, decoder_hidden_state, logits = self.forward_step(input_var=decoder_input, hidden_state=decoder_hidden_state)

        # 2. mode=GEN
        else:
            stored_scores, stored_symbols, stored_predecessors, stored_logits = [], [], [], []
            sequence_scores = th.zeros_like(decoder_input.squeeze(), dtype=th.float)  # [b*k]
            sequence_scores.fill_(-INF)
            ind = th.LongTensor([i * beam_size for i in range(batch_size)])
            ind = cast_type(ind, LONG, self.use_gpu)
            sequence_scores.index_fill_(0, ind, 0.0)

            for step in range(self.max_dec_len):
                # one step for decoder
                # --- decoder_input: [b*k,1]  decoder_hidden_state: [1,b*k,h]
                # --- decoder_output: [b*k,1,v] step_logit: [b*k,1,v]
                decoder_output, decoder_hidden_state, step_logit = self.forward_step(decoder_input, decoder_hidden_state)

                # greedy search
                if gen_type == "greedy":
                    log_softmax = decoder_output.squeeze() # [b,v]
                    _, symbols = log_softmax.topk(1)  # [b,1], by default rank in last dimension
                    # 1. create next step input
                    decoder_input = symbols  # indices
                    # 2. save
                    stored_symbols.append(symbols.clone())  # [b,1]
                    stored_scores.append(log_softmax)  # [b,v]
                    stored_logits.append(step_logit)  # [b*k,1,v]

                # beam search
                elif gen_type == "beam":
                    log_softmax = decoder_output.squeeze()  # [b*k,v]
                    # update sequence socres
                    sequence_scores = sequence_scores.unsqueeze(1).repeat(1, vocab_size)  # [b*k,v]
                    if LENGTH_AVERAGE:
                        t = step + 2
                        sequence_scores = sequence_scores * (1 - 1 / t) + log_softmax / t
                    else:
                        sequence_scores += log_softmax

                    # diverse beam search: penalize short sequence.
                    # select topk
                    scores, candidates = sequence_scores.view(batch_size, -1).topk(beam_size, dim=1)  # [b,k], [b,k]

                    # 1.1 create  next step decoder_input
                    input_var = (candidates % vocab_size)  # [b,k]
                    decoder_input = input_var.view(batch_size * beam_size, 1)  # [b*k,1] input for next step

                    # 1.2 create next step decoder_hidden_state
                    pp = candidates // vocab_size  # [b,k]
                    predecessors = pp.clone()
                    for b, p in enumerate(pp):
                        predecessors[b] = p + b * beam_size # to be in one line
                    predecessors = predecessors.view(-1)  # [b*k]
                    decoder_hidden_state = decoder_hidden_state.index_select(1, predecessors)

                    # 1.3 create next step scores
                    sequence_scores = scores.view(batch_size * beam_size)  # [b*k]
                    # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
                    stored_scores.append(sequence_scores.clone())  # [b*k]
                    eos_indices = input_var.data.eq(self.eos_id).view(-1)  # [b*k]
                    if eos_indices.nonzero().dim() > 0:
                        sequence_scores.masked_fill_(eos_indices, -INF)

                    # 2. Cache results for backtracking
                    stored_predecessors.append(predecessors)  # [b*k]
                    stored_symbols.append(decoder_input.squeeze())  # [b*k]

                else:
                    raise NotImplementedError

            if gen_type == "greedy":
                symbol_outputs = th.cat(stored_symbols, dim=1).squeeze()  # [b,len]
                prob_outputs = 0  # dontcare
                logits = th.cat(stored_logits, dim=1)  # [b,t,v]

            elif gen_type == "beam":
                # beam search backtrack
                predicts, lengths, scores = self._backtrack(
                    stored_predecessors, stored_symbols, stored_scores, batch_size, beam_size)
                # predicts: (b, k, len)
                # only select top1 for beam search
                symbol_outputs = predicts[:, 0, :]  # [b,len]
                prob_outputs = 0  # dontcare logits for beam search generation
                logits = 0  # dontcare logits for beam search generation
            else:
                raise NotImplementedError

        # : store logits
        ret_dict['logits'] = logits
        ret_dict['sequence'] = symbol_outputs

        # prob_outputs: (batch_size, max_dec_len, vocab_size)
        # decoder_hidden_state:(1, b, h)
        # ret_dict['logits']: (batch_size, sent_len, vocab_size)
        # ret_dict['sequence']: max_dec_len*(batch_size)
        return prob_outputs, decoder_hidden_state, ret_dict

    # beam search backtrack
    def _backtrack(self, predecessors, symbols, scores, batch_size, beam_size):
        p = list()
        l = [[self.max_dec_len] * beam_size for _ in range(batch_size)]  # length of each seq

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(batch_size, beam_size).topk(beam_size, dim=-1)

        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()  # [b,k]

        # the number of EOS found in the backward loop below for each batch
        batch_eos_found = [0] * batch_size

        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = sorted_idx.clone()
        for b, idx in enumerate(sorted_idx):
            t_predecessors[b] = idx + b * beam_size
        t_predecessors = t_predecessors.view(-1)  # [b*k]

        t = self.max_dec_len - 1
        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = symbols[t].index_select(0, t_predecessors)  # [b*k]
            # Re-order the back pointer of the previous step with the back pointer of the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors)  # [b*k]

            # Deal with EOS
            eos_indices = symbols[t].data.eq(self.eos_id).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = idx[0].item() // beam_size
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = beam_size - (batch_eos_found[b_idx] % beam_size) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * beam_size + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_symbol[res_idx] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]]
                    l[b_idx][res_k_idx] = t + 1

            # save current_symbol
            p.append(current_symbol)  # [b*k]

            t -= 1

        # Sort and re-order again as the added ended sequences may change the order (very unlikely)
        s, re_sorted_idx = s.topk(beam_size)
        for b_idx in range(batch_size):
            l[b_idx] = [l[b_idx][k_idx.item()]
                        for k_idx in re_sorted_idx[b_idx, :]]
        rr = re_sorted_idx.clone()
        for b, idx in enumerate(re_sorted_idx):
            rr[b] = idx + b * beam_size
        re_sorted_idx = rr.view(-1)  # [b*k]

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        predicts = th.stack(p[::-1]).t()  # [b*k,t]
        predicts = predicts[re_sorted_idx].contiguous().view(batch_size, beam_size, -1)
        scores = s
        lengths = l
        return predicts, scores, lengths

    def forward_step(self, input_var, hidden_state):
        # input_var=decoder_input, hidden_state=decoder_hidden_state
        batch_size, output_seq_len = input_var.size()
        embedded = self.embedding(input_var)  # (batch_size, output_seq_len, embedding_dim)
        embedded = self.input_dropout(embedded)

        # output: (batch_size, output_seq_len, dec_cell_size), hidden_s: (1, b, h)
        output, hidden_s = self.rnn(embedded, hidden_state)

        logits = self.project(output.contiguous().view(-1, self.dec_cell_size))  # (batch_size*output_seq_len, vocab_size)
        prediction = self.log_softmax(logits, dim=logits.dim() - 1).view(batch_size, output_seq_len, -1)  # (batch_size, output_seq_len, vocab_size)
        return prediction, hidden_s, logits.view(batch_size, output_seq_len, -1)

    # special for rl
    def _step(self, input_var, hidden_state):
        # input_var: (b, 1)
        # hidden_state: tuple: (h, c)
        batch_size, output_seq_len = input_var.size()
        embedded = self.embedding(input_var)  # (b, 1, embedding_dim)
        embedded = self.input_dropout(embedded)

        # output: (b, 1, h)
        # hidden: tuple: (h, c)
        output, hidden_s = self.rnn(embedded, hidden_state)

        logits = self.project(output.view(-1, self.dec_cell_size))  # (b, vocab_size)
        prediction = logits.view(batch_size, output_seq_len, -1)  # (b, 1, vocab_size)
        # prediction = self.log_softmax(logits, dim=logits.dim()-1).view(batch_size, output_seq_len, -1) # (batch_size, output_seq_len, vocab_size)
        return prediction, hidden_s

    # special for rl
    def write(self, input_var, hidden_state, max_words, vocab, stop_tokens):
        # input_var: (1, 1)
        # hidden_state: tuple: (h, c)
        # encoder_outputs: max_dlg_len*(1, 1, dlg_cell_size)
        # goal_hid: (1, goal_nhid)
        logprob_outputs = []  # list of logprob | max_dec_len*(1, )
        symbol_outputs = []  # list of word ids | max_dec_len*(1, )
        decoder_input = input_var
        decoder_hidden_state = hidden_state

        def _sample(dec_output, num_i):
            # dec_output: (1, 1, vocab_size), need to softmax and log_softmax
            dec_output = dec_output.view(-1)  # (vocab_size, )
            # TODO temperature
            prob = F.softmax(dec_output / 0.6, dim=0)  # (vocab_size, )
            logprob = F.log_softmax(dec_output, dim=0)  # (vocab_size, )
            symbol = prob.multinomial(num_samples=1).detach()  # (1, )
            logprob = logprob.gather(0, symbol)  # (1, )
            return logprob, symbol

        for i in range(max_words):
            decoder_output, decoder_hidden_state = self._step(decoder_input, decoder_hidden_state)
            # disable special tokens from being generated in a normal turn
            logprob, symbol = _sample(decoder_output, i)
            logprob_outputs.append(logprob)
            symbol_outputs.append(symbol)
            decoder_input = symbol.view(1, -1)

            if vocab[symbol.item()] in stop_tokens:
                break

        assert len(logprob_outputs) == len(symbol_outputs)
        # logprob_list = [t.item() for t in logprob_outputs]
        logprob_list = logprob_outputs
        symbol_list = [t.item() for t in symbol_outputs]
        return logprob_list, symbol_list

    # For MultiWoz RL
    def forward_rl(self, batch_size, dec_init_state, vocab, max_words, temp=0.1):
        # prepare the BOS inputs
        with th.no_grad():
            bos_var = Variable(th.LongTensor([self.sys_id]))
        bos_var = cast_type(bos_var, LONG, self.use_gpu)
        decoder_input = bos_var.expand(batch_size, 1)  # (batch, 1)
        decoder_hidden_state = dec_init_state # (1, b, h)

        logprob_outputs = []  # list of logprob | max_dec_len*(1, )
        symbol_outputs = []  # list of word ids | max_dec_len*(1, )

        def _sample(dec_output, num_i):
            # dec_output: (b, 1, vocab_size), need to softmax and log_softmax
            dec_output = dec_output.view(batch_size, -1)  # (batch_size, vocab_size, )
            prob = F.softmax(dec_output / temp, dim=1)  # (batch_size, vocab_size, )
            logprob = F.log_softmax(dec_output, dim=1)  # (batch_size, vocab_size, )
            symbol = prob.multinomial(num_samples=1).detach()  # (batch_size, 1)
            logprob = logprob.gather(1, symbol)  # (b, 1)
            return logprob, symbol # (b, 1)

        stopped_samples = set()
        for i in range(max_words):
            # output: (b, 1, vocab_size)
            decoder_output, decoder_hidden_state = self._step(decoder_input, decoder_hidden_state)
            logprob, symbol = _sample(decoder_output, i) # (b, 1)
            logprob_outputs.append(logprob)
            symbol_outputs.append(symbol)
            decoder_input = symbol.view(batch_size, -1)
            for b_id in range(batch_size):
                if vocab[symbol[b_id].item()] == EOS:
                    stopped_samples.add(b_id)

            if len(stopped_samples) == batch_size:
                break

        assert len(logprob_outputs) == len(symbol_outputs)  # max sent length in this batch
        symbol_outputs = th.cat(symbol_outputs, dim=1).cpu().data.numpy().tolist()
        logprob_outputs = th.cat(logprob_outputs, dim=1) # b, max_sent_len
        logprob_list = [] # [batch_size of [sent_len of tokens' logprob] ]
        symbol_list = [] # b, each_sent_len
        for b_id in range(batch_size):
            b_logprob = [] # each_sent_len
            b_symbol = []  # each_sent_len
            for t_id in range(logprob_outputs.shape[1]): # max_sent_len
                symbol = symbol_outputs[b_id][t_id]
                if vocab[symbol] == EOS and t_id != 0:
                    break

                b_symbol.append(symbol_outputs[b_id][t_id])
                b_logprob.append(logprob_outputs[b_id][t_id])

            logprob_list.append(b_logprob)
            symbol_list.append(b_symbol)

        if batch_size == 1:
            logprob_list = logprob_list[0]
            symbol_list = symbol_list[0]

        return logprob_list, symbol_list
