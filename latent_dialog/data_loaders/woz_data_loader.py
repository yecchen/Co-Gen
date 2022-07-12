"""
    This is used to load multiwoz type data, include: <utterance, db, bs, response>
"""

import json
import logging
import numpy as np
from latent_dialog.utils import Pack
from latent_dialog.data_loaders import BaseDataLoader
from latent_dialog.data_loaders.woz_corpora import USR, SYS

logger = logging.getLogger()


class MultiWozDataLoader(BaseDataLoader):

    def __init__(self, mode, data, config):
        super(MultiWozDataLoader, self).__init__(mode)
        self.mode = mode  # 'Train'/'Val'/'Test'
        self.raw_data = data # xxx_dial = [Pack: dlg=[Pack: id_utt, speaker, db, bs], goal={domain: one-hot for a-s-v}, key=filename]
        self.config = config
        self.max_utt_len = config.max_utt_len
        self.max_act_len = config.max_act_len
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data)
        # data: [Pack: context=[turns=[pad_id_utt]], response=Pack: pad_id_utt, speaker=SYS, db, bs, goal={domain: one-hot for a-s-v}, key=filename]
        self.data_size = len(self.data)  # number of <c,r> pairs
        self.domains = ['hotel', 'restaurant', 'train', 'attraction', 'hospital', 'police', 'taxi']

    def flatten_dialog(self, data):
        """
            Reorganize data in <context, response> pairs.
            Args:
                - data: [Pack: dlg=[Pack: id_utt, speaker, db, bs, act_vec], goal={domain: one-hot for a-s-v}, key=filename]
        """
        results = []  # [Pack: context=[turns=Pack(id_utt, speaker, db, bs)], response=Pack(id_utt, speaker=SYS, db, bs, act_vec), goal={domain: one-hot for a-s-v}, key=filename]
        indexes = []  # [0,1,...,num_<context,response>_pairs-1]
        batch_indexes = []   # [ [all <c,r> pair id for each dialogue] ]
        resp_set = set()  # set of JSON object { [id_utt] }
        for dlg in data:
            goal = dlg.goal  # goal={domain: one-hot for a-s-v}
            key = dlg.key  # key=filename
            batch_index = []  # all <c,r> pair id in this dialogue
            for i in range(1, len(dlg.dlg)):
                if dlg.dlg[i].speaker == USR:  # include user utt and begin/end of a dialogue
                    continue
                e_idx = i  # current turn id
                s_idx = max(0, e_idx - 1)   # start turn id
                response = dlg.dlg[i].copy()  # Pack: id_utt, speaker=SYS, db, bs, act
                response['utt'] = self.pad_to(self.max_utt_len, response.utt, do_pad=False)
                response['act_seq'] = self.pad_to(self.max_act_len, response.act_seq, do_pad=False)
                resp_set.add(json.dumps(response.utt))
                context = []  # [ turn=Pack(id_utt, speaker, db, bs, id_act) ]
                for turn in dlg.dlg[s_idx: e_idx]:  # include begin of a dialogue
                    turn['utt'] = self.pad_to(self.max_utt_len, turn.utt, do_pad=False)
                    context.append(turn)
                results.append(Pack(context=context, response=response, goal=goal, key=key))
                indexes.append(len(indexes))
                batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)

        logger.info("Unique/Total Response {}/{} - {}".format(len(resp_set), len(results), self.mode))
        # indexes: [0,1,2,...]   batch_indexes: [[0,1,2,3,4],[5,6,7,8,9],...]
        return results, indexes, batch_indexes

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        # fix_batch: False for supervised training, True for validation, generation & reinforcement learning
        self.ptr = 0
        if fix_batch:
            self.batch_size = None
            self.num_batch = len(self.batch_indexes)
            if shuffle:
                self._shuffle_batch_indexes()
        else:
            if shuffle:
                self._shuffle_indexes()
            self.batch_size = config.batch_size
            self.num_batch = self.data_size // config.batch_size
            self.batch_indexes = []  # updated by batch_size
            for i in range(self.num_batch):
                self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
            if verbose:
                print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
        # if shuffle:
        #     if fix_batch:
        #         self._shuffle_batch_indexes()
        #     else:
        #         self._shuffle_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def _prepare_batch(self, selected_index):
        # add in system action
        # selected_index = [pair_id in a batch]
        rows = [self.data[idx] for idx in selected_index]
        # rows = [Pack: context=[turns=Pack(id_utt, speaker, db, bs)], response=Pack(id_utt, speaker=SYS, db, bs, act_vec), goal={domain: one-hot for a-s-v}, key=filename]

        ctx_utts, ctx_lens = [], []  # [context for each pair] [#turns_in_context for each pair]
        out_utts, out_lens = [], []  # [pad_id_sys_utt for each pair] [#token_in_rsp for each pair]
        out_acts, out_act_lens = [], []  # [pad_id_sys_act for each pair] [#token_in_sys_act for each pair]

        # for teach force decoder
        teach_rsps, teach_acts, target_rsps, target_acts = [], [], [], []

        out_act_vector = []
        out_bs, out_db = [] , []
        goals, goal_lens = [], [[] for _ in range(len(self.domains))]  # [goal for each pair] [[one_hot_length in this domain for each pair] for each goal_domain]
        keys = []  # [filenames in this batch]

        # bs_{t-1}, bs_{t+1}, db_{t-1}, db_{t+1}
        out_bs_prev, out_bs_next, out_db_prev, out_db_next = [] , [], [], []

        for j, row in enumerate(rows):
            in_row, out_row, goal_row = row.context, row.response, row.goal

            # source context
            keys.append(row.key)
            batch_ctx = []
            for turn in in_row:  # all context=[turns=Pack(id_utt, speaker, db, bs, id_act)]
                batch_ctx.append(self.pad_to(self.max_utt_len, turn.utt, do_pad=True))
            ctx_utts.append(batch_ctx)  # [context for each pair]
            ctx_lens.append(len(batch_ctx))   # [#turns_in_context for each pair]

            # add teach and output version
            # response=Pack(id_utt, speaker=SYS, db, bs, act_vec)
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            out_utts.append(out_utt)  # [id_sys_utt for each pair]
            out_lens.append(len(out_utt))  # [#token_in_rsp for each pair]
            teach_rsps.append(out_utt[:-1])
            target_rsps.append(out_utt[1:])
            # system act
            out_act = [t for idx, t in enumerate(out_row.act_seq)]
            out_acts.append(out_act)  # [id_sys_act for each pair]
            out_act_lens.append(len(out_act))  # [#token_in_sys_act for each pair]
            teach_acts.append(out_act[:-1])
            target_acts.append(out_act[1:])

            out_act_vector.append(out_row.act)
            out_bs.append(out_row.bs)
            out_db.append(out_row.db)

            # :
            if j==0:  # first sample in this batch
                out_bs_prev.append(np.zeros_like(out_bs[0]))
                out_db_prev.append(np.zeros_like(out_db[0]))
            else:
                out_bs_prev.append(out_bs[-2])
                out_db_prev.append(out_db[-2])
            if j==0:
                pass
            else:
                out_bs_next.append(out_bs[-1])
                out_db_next.append(out_db[-1])

            # goal
            goals.append(goal_row)  # goal_row={domain: one-hot for a-s-v}
            for i, d in enumerate(self.domains):
                goal_lens[i].append(len(goal_row[d]))  # [[one_hot_length in this domain for each pair] for each goal_domain]

        # :
        out_bs_next.append(np.zeros_like(out_bs[0]))  # for the final sample
        out_db_next.append(np.zeros_like(out_db[0]))
        vec_out_bs_next = np.array(out_bs_next) # (batch_size, 94)
        vec_out_db_next = np.array(out_db_next) # (batch_size, 30)
        vec_out_bs_prev = np.array(out_bs_prev) # (batch_size, 94)
        vec_out_db_prev = np.array(out_db_prev) # (batch_size, 30)

        batch_size = len(ctx_lens)  # number of pairs in this batch
        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns, i.e. [#turns_in_context for each pair]
        max_ctx_len = np.max(vec_ctx_lens)  # max # turns_in_context
        vec_ctx_utts = np.zeros((batch_size, max_ctx_len, self.max_utt_len), dtype=np.int32)

        vec_out_bs = np.array(out_bs) # (batch_size, 94)
        vec_out_db = np.array(out_db) # (batch_size, 30)
        vec_out_act_vector = np.array(out_act_vector) # (b, 44)

        vec_out_lens = np.array(out_lens)  # (batch_size, ), number of tokens, i.e. [#token_in_rsp for each pair]
        max_out_len = np.max(vec_out_lens)  # max # token_in_rsp
        vec_out_utts = np.zeros((batch_size, max_out_len), dtype=np.int32)
        vec_utt_labels = np.ones((batch_size, 1), dtype=np.int32)

        vec_out_act_lens = np.array(out_act_lens)  # (batch_size, ), number of tokens, i.e. [#token_in_sys_act for each pair]
        max_out_act_len = np.max(vec_out_act_lens)  # max # of token_in_sys_act
        vec_out_acts = np.zeros((batch_size, max_out_act_len), dtype=np.int32)
        vec_act_labels = np.zeros((batch_size, 1), dtype=np.int32)

        # add teach and output version
        vec_teach_rsps = np.zeros((batch_size, max_out_len), dtype=np.int32)
        vec_target_rsps = np.zeros((batch_size, max_out_len), dtype=np.int32)
        vec_teach_acts = np.zeros((batch_size, max_out_act_len), dtype=np.int32)
        vec_target_acts = np.zeros((batch_size, max_out_act_len), dtype=np.int32)

        # goal_lens: [[one_hot_length in this domain for each pair] for each goal_domain]
        max_goal_lens, min_goal_lens = [max(ls) for ls in goal_lens], [min(ls) for ls in goal_lens]
        # [max/min one_hot_length for each domain]
        if max_goal_lens != min_goal_lens:
            print('Fatal Error!')
            exit(-1)
        self.goal_lens = max_goal_lens  # [max one_hot_length for each domain]
        vec_goals_list = [np.zeros((batch_size, l), dtype=np.float32) for l in self.goal_lens]  # [#domain, (batch_size, max_one_hot_length)]

        # fill in values
        for b_id in range(batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]
            vec_out_acts[b_id, :vec_out_act_lens[b_id]] = out_acts[b_id]
            # add teach and output version
            vec_teach_rsps[b_id, :vec_out_lens[b_id]-1] = teach_rsps[b_id]
            vec_target_rsps[b_id, :vec_out_lens[b_id]-1] = target_rsps[b_id]
            vec_teach_acts[b_id, :vec_out_act_lens[b_id] - 1] = teach_acts[b_id]
            vec_target_acts[b_id, :vec_out_act_lens[b_id] - 1] = target_acts[b_id]
            for i, d in enumerate(self.domains):
                vec_goals_list[i][b_id, :] = goals[b_id][d]

        return Pack(context_lens=vec_ctx_lens, # (batch_size, ), number of turns, i.e. [#turns_in_context for each pair]
                    contexts=vec_ctx_utts, # (batch_size, max_ctx_len, max_utt_len)
                    output_lens=vec_out_lens, # (batch_size, ), number of tokens, i.e. [#token_in_rsp for each pair]
                    outputs=vec_out_utts, # (batch_size, max_out_len)
                    rsp_labels=vec_utt_labels, # (batch_size, 1), all 1s
                    teach_rsps=vec_teach_rsps,
                    target_rsps=vec_target_rsps,
                    out_act_lens=vec_out_act_lens, # (batch_size, ), number of tokens, i.e. [#token_in_sys_act for each pair]
                    out_acts=vec_out_acts,  # (batch_size, max_act_len), all 0s
                    act_labels=vec_act_labels,  # (batch_size, 1)
                    teach_acts=vec_teach_acts,
                    target_acts=vec_target_acts,
                    bs=vec_out_bs, # (batch_size, 94)
                    db=vec_out_db, # (batch_size, 30)
                    act_vec=vec_out_act_vector, # (batch_size, 44)
                    goals_list=vec_goals_list, # 7*(batch_size, bow_len), bow_len differs w.r.t. domain, max_onehot_len for each domain
                    keys=keys, # [filenames in this batch]
                    bs_next=vec_out_bs_next,
                    bs_prev=vec_out_bs_prev,
                    db_next=vec_out_db_next,
                    db_prev=vec_out_db_prev
                    )

    def clone(self):
        return MultiWozDataLoader(self.mode, self.raw_data, self.config)
