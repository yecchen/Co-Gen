import os, sys
import csv
import json
import numpy as np
from collections import Counter
import json
import logging

from latent_dialog.utils import Pack, DATA_DIR
from latent_dialog.woz_util import *

logger = logging.getLogger()

class MultiWozCorpus(object):

    def __init__(self, config):
        self.special_d_a = {'taxi-inform': {'dest', 'depart'},
                       'booking-inform': {'name', 'ref'},
                       'booking-nobook': {'name', 'ref'},
                       'booking-book': {'name', 'ref'}}

        self.bs_size = 94 # this is true for multiwoz_2.0
        self.db_size = 30 # this is true for multiwoz_2.0
        self.bs_types =['b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'c', 'c', 'c', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'b']
        self.domains = ['hotel', 'restaurant', 'train', 'attraction', 'hospital', 'police', 'taxi']
        self.info_types = ['book', 'fail_book', 'fail_info', 'info', 'reqt']
        self.config = config
        self.tokenize = lambda x: x.split()
        self.train_corpus, self.val_corpus, self.test_corpus, self.train_act_vectors, self.val_act_vectors, self.test_act_vectors, self.sys_act_seq = self._read_file(self.config)
        # Pack: dlg=norm_dlg, goal=processed_goal, key=key

        vocab_file = DATA_DIR + config.data_name + '/vocab.json'

        if os.path.exists(vocab_file):
            with open(vocab_file, "r") as f:
                self.vocab_dict = json.load(f)
                self.vocab = list(self.vocab_dict.keys())
            self.unk_id = self.vocab_dict[UNK]
        else:
            self._extract_vocab()
            with open(vocab_file, "w") as f:
                json.dump(self.vocab_dict, f, ensure_ascii=False)

        act_vocab = DATA_DIR  + 'act/act_vocab.json'
        with open(act_vocab, "r") as f:
            self.act_vocab_dict = json.load(f)
            self.act_vocab = list(self.act_vocab_dict.keys())

        self._extract_goal_vocab()
        logger.info('Loading corpus finished.')
        logger.info('-'*20)

    def _read_file(self, config):

        train_path = DATA_DIR + config.data_name + '/train_dials.json'
        val_path = DATA_DIR + config.data_name + '/val_dials.json'
        test_path = DATA_DIR + config.data_name + '/test_dials.json'
        train_data = json.load(open(train_path))
        valid_data = json.load(open(val_path))
        test_data = json.load(open(test_path))

        train_data = self._process_dialogue(train_data, "Train")  # Pack: dlg=norm_dlg, goal=processed_goal, key=key
        valid_data = self._process_dialogue(valid_data, "Val")
        test_data = self._process_dialogue(test_data, "Test")

        train_act_tsv_path = DATA_DIR + '/act/train.tsv'
        # val_act_tsv_path = DATA_DIR + '/act/dev.tsv'
        # test_act_tsv_path = DATA_DIR + '/act/test.tsv'
        train_act_vectors = self._process_act_tsv(train_act_tsv_path)
        # val_act_vectors = self._process_act_tsv(val_act_tsv_path)
        # test_act_vectors = self._process_act_tsv(test_act_tsv_path)

        val_act_pred_path = DATA_DIR + 'act/BERT_dev_prediction.json'
        test_act_pred_path = DATA_DIR + 'act/BERT_test_prediction.json'
        val_act_vec_pred = json.load(open(val_act_pred_path))  # turn id start with 0
        test_act_vec_pred = json.load(open(test_act_pred_path))

        sys_act_seq_path = DATA_DIR + 'act/act_seq.json'
        sys_act_seq = json.load(open(sys_act_seq_path))

        return train_data, valid_data, test_data, train_act_vectors, val_act_vec_pred, test_act_vec_pred, sys_act_seq

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            idx = 0
            for line in reader:
                idx += 1
                # if idx > 100: break
                if sys.version_info[0] == 2:
                    line = list(cell.decode('utf-8') for cell in line)
                lines.append(line)
            return lines

    def _process_act_tsv(self, tsv_path):
        lines = self._read_tsv(tsv_path)
        results = {} # {filename: {turn_id(start with 0): act_vec}}
        for (i, line) in enumerate(lines):
            filename = line[0]
            if filename not in results:
                results[filename] = dict()
            turn_num = line[1]
            label = json.loads(line[5]) # list
            results[filename][turn_num] = label
        return results

    def _process_dialogue(self, data, mode):
        new_dlgs = []
        all_sent_lens = []
        all_dlg_lens = []

        for key, raw_dlg in data.items():
            # raw_dlg.keys: ['sys', 'bs', 'db', 'goal', 'usr']
            # begin of dialog
            norm_dlg = [Pack(speaker=USR, id=-1, utt=[BOS, BOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size)]

            for t_id in range(len(raw_dlg['db'])): # num fo turns
                usr_utt = [BOS] + self.tokenize(raw_dlg['usr'][t_id]) + [EOS]
                sys_utt = [BOS] + self.tokenize(raw_dlg['sys'][t_id]) + [EOS]
                norm_dlg.append(Pack(speaker=USR, id=t_id, utt=usr_utt, db=raw_dlg['db'][t_id], bs=raw_dlg['bs'][t_id]))
                norm_dlg.append(Pack(speaker=SYS, id=t_id, utt=sys_utt, db=raw_dlg['db'][t_id], bs=raw_dlg['bs'][t_id]))
                all_sent_lens.extend([len(usr_utt), len(sys_utt)]) #sent len include special tokens
            # To stop dialog
            norm_dlg.append(Pack(speaker=USR, id=-1, utt=[BOS, EOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size))
            # if self.config.to_learn == 'usr':
            #     norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size))
            all_dlg_lens.append(len(raw_dlg['db']))
            processed_goal = self._process_goal(raw_dlg['goal'])   # {domain: [a|s, a|s|v]}
            new_dlgs.append(Pack(dlg=norm_dlg, goal=processed_goal, key=key))  # key = dialog_filename

        logger.info('(%s) Max utterance len = %d, mean utterance len = %.2f ' % (
            mode, np.max(all_sent_lens), float(np.mean(all_sent_lens))))
        logger.info('(%s) Max dialogue len = %d, mean dialogue len = %.2f ' % (
            mode, np.max(all_dlg_lens), float(np.mean(all_dlg_lens))))
        return new_dlgs

    def _extract_vocab(self):
        all_words = []
        for dlg in self.train_corpus:
            for turn in dlg.dlg:
                all_words.extend(turn.utt)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        keep_vocab_size = min(self.config.max_vocab_size, raw_vocab_size)
        oov_rate = np.sum([c for t, c in vocab_count[0:keep_vocab_size]]) / float(len(all_words))

        logger.info('cut off at word {} with frequency={},\n'.format(vocab_count[keep_vocab_size - 1][0],
                                                                     vocab_count[keep_vocab_size - 1][1]) +
                    'OOV rate = {:.2f}%'.format(100.0 - oov_rate * 100))

        use_vocab_count = vocab_count[0:keep_vocab_size]
        utt_vocab = SPECIAL_TOKENS + [t for t, cnt in use_vocab_count if t not in SPECIAL_TOKENS]
        self.vocab = utt_vocab
        self.vocab_dict = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.vocab_dict[UNK]
        logger.info("Raw vocab size {} in train set and final vocab size {}".format(raw_vocab_size, len(self.vocab)))

    def _process_goal(self, raw_goal):
        res = {}
        for domain in self.domains:
            all_words = []
            d_goal = raw_goal[domain]
            if d_goal:
                for info_type in self.info_types:
                    sv_info = d_goal.get(info_type, dict())  # create default dict for every type of slot
                    if info_type == 'reqt' and isinstance(sv_info, list):
                        all_words.extend([info_type + '|' + item for item in sv_info])  # eg. reqt|phone
                    elif isinstance(sv_info, dict):
                        all_words.extend([info_type + '|' + k + '|' + str(v) for k, v in sv_info.items()])  # eg. info|leaveAt|24:30
                    else:
                        logger.error('Fatal Error!')
                        exit(-1)
            res[domain] = all_words
        return res  # {domain: [a|s, a|s|v]}

    def _extract_goal_vocab(self):
        self.goal_vocab, self.goal_vocab_dict, self.goal_unk_id = {}, {}, {}
        for domain in self.domains:
            all_words = []
            for dlg in self.train_corpus:  # corpus = Pack: dlg=norm_dlg, goal=processed_goal, key=key
                all_words.extend(dlg.goal[domain])  # {domain: [a|s, a|s|v]}
            vocab_count = Counter(all_words).most_common()  # list all element count
            raw_vocab_size = len(vocab_count)
            discard_wc = np.sum([c for t, c in vocab_count])  # actually is sum all, i.e. use all goal vocab

            logger.info('================= domain = {}, \n'.format(domain) +
                  'goal vocab size of train set = %d, \n' % (raw_vocab_size,) +
                  'cut off at word %s with frequency = %d, \n' % (vocab_count[-1][0], vocab_count[-1][1]) +
                  'OOV rate = %.2f' % (1 - float(discard_wc) / len(all_words),))

            self.goal_vocab[domain] = [UNK] + [g for g, cnt in vocab_count]  # each goal domain has an UNK
            self.goal_vocab_dict[domain] = {t: idx for idx, t in enumerate(self.goal_vocab[domain])}
            self.goal_unk_id[domain] = self.goal_vocab_dict[domain][UNK]

    def get_corpus(self):
        # self.xxx_corpus: Pack: dlg=norm_dlg, goal=processed_goal, key=key
        # id_xxx: [Pack: dlg=[Pack: id_utt, speaker, db, bs], goal={domain: one-hot for a-s-v}, key=filename]
        id_train = self._to_id_corpus("Train", self.train_corpus, self.train_act_vectors)
        id_val = self._to_id_corpus("Valid", self.val_corpus, self.val_act_vectors)
        id_test = self._to_id_corpus("Test", self.test_corpus, self.test_act_vectors)
        return id_train, id_val, id_test

    def _to_id_corpus(self, name, data, act_data):
        results = []
        empty_act = [0] * 44
        empty_act_seq = [BOS, EOS]
        for dlg in data:  # data: Pack: dlg=norm_dlg, goal=processed_goal, key=key
            if len(dlg.dlg) < 1:
                continue
            id_dlg = []
            key = dlg.key.split(".")[0]
            if key not in act_data:
                print("No file " + key + " in " + name)
                continue
            for turn in dlg.dlg:  # norm_dlg = Pack: speaker, id, utt, bs, db
                t_id = turn.id # start from 0 also
                if turn.speaker == SYS and t_id >= 0:
                    # if str(t_id + 1) not in self.sys_act_data[key]:
                    #     print("No Sys Act in " + key + ", turn " + str(t_id+1))
                    if str(t_id) not in act_data[key]:
                        print(name + ": No Sys Act in " + key + ", turn " + str(t_id))
                        continue
                    sys_act_vec = list(act_data[key].get(str(t_id), empty_act))
                    sys_act_seq = self.sys_act_seq[key].get(str(t_id + 1), empty_act_seq)
                    id_turn = Pack(utt=self._sent2id(turn.utt),
                                   speaker=turn.speaker,
                                   db=turn.db, bs=turn.bs,
                                   act=sys_act_vec,
                                   act_seq=self._sent2id_act(sys_act_seq)
                                   )
                else:
                    id_turn = Pack(utt=self._sent2id(turn.utt),
                                   speaker=turn.speaker,
                                   db=turn.db, bs=turn.bs
                                   )
                id_dlg.append(id_turn)  # [Pack: id_utt, speaker, db, bs, act_vec]
            id_goal = self._goal2id(dlg.goal) # {domain: one-hot for a-s-v}
            results.append(Pack(dlg=id_dlg, goal=id_goal, key=dlg.key))  # key = dialog filename
        return results

    def _sent2id(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]

    def _sent2id_act(self, sent):
        return [self.act_vocab_dict.get(t, self.unk_id) for t in sent]

    def _goal2id(self, goal):
        res = {}
        # goal: # {domain: [a|s, a|s|v]}
        for domain in self.domains:
            d_bow = [0.0] * len(self.goal_vocab[domain])
            for word in goal[domain]:
                word_id = self.goal_vocab_dict[domain].get(word, self.goal_unk_id[domain])
                d_bow[word_id] += 1.0
            res[domain] = d_bow   # each domain is a one-hot presentation (may>1 for UNK) for all possible a|s/a|s|v
        return res  # {domain: one-hot for a-s-v}

    def id2sent(self, id_list):
        return [self.vocab[i] for i in id_list]

    def id2sent_act(self, id_list):
        return [self.act_vocab_dict[i] for i in id_list]

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens
