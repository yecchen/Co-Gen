import torch.nn as nn
import torch.optim as optim
import numpy as np
from latent_dialog.utils import FLOAT
from latent_dialog.agents import RlAgent


class LatentRlAgent(RlAgent):
    def __init__(self, model, corpus, args, name, tune_pi_only):
        super(LatentRlAgent, self).__init__(model, corpus, args, name, tune_pi_only)
        if self.args.rl_strategy:
            self.opt_startegy = optim.SGD(
                # only update c to z (the rl_strategy level) part
                [p for n, p in self.model.named_parameters() if 'c2z' in n],
                lr=self.args.rl_lr_strategy,
                momentum=self.args.momentum,
                nesterov=(self.args.nesterov and self.args.momentum > 0),
                weight_decay=self.args.weight_decay
            )
        if self.args.rl_realization:
            self.opt_relization = optim.SGD(
                [p for n, p in self.model.named_parameters() if
                 'rsp_generator' in n or 'z_embedding_x2r' in n or 'rsp_projection' in n or 'rsp_decoder_embedding' in n],
                lr=self.args.rl_lr_realization,
                momentum=self.args.momentum,
                nesterov=(self.args.nesterov and self.args.momentum > 0),
                weight_decay=self.args.weight_decay
            )
        self.update_n = 0
        self.all_rewards = {'success': [], 'match': [], 'bleu': []}
        if self.args.rl_realization:
            self.curr_stage = 'rl_realization'
        if self.args.rl_strategy:
            self.curr_stage = 'rl_strategy'
        self.rl_realization_cnt = 0
        self.rl_strategy_cnt = 0
        self.rl_clip = self.args.rl_clip
        self.reward_weight = self.args.reward_weight

    def run(self, batch, evaluator, max_words=None, temp=0.1):
        self.logprobs = dict(rl_strategy=[], rl_realization=[], rl_realization_org=[])
        self.dlg_history = []
        batch_size = len(batch['keys'])
        logprobs, outs, logprob_z, sample_z = self.model.forward_rl(batch, max_words, temp, self.args)

        # for tackling some special case
        if batch_size == 1:
            logprobs = [logprobs]
            outs = [outs]

        key = batch['keys'][0]
        sys_turns = []
        sys_turns_gt = []
        # construct the dialog history for printing
        for turn_id, turn in enumerate(batch['contexts']):
            user_input = self.corpus.id2sent(turn[-1])
            self.dlg_history.append(user_input)
            # collect the groudtruth respnse of system
            sys_output_gt = self.corpus.id2sent(turn[0])
            sys_output = self.corpus.id2sent(outs[turn_id])
            self.dlg_history.append(sys_output)
            sys_turns.append(' '.join(sys_output))
            sys_turns_gt.append(' '.join(sys_output_gt))

        for b_id in range(batch_size):
            self.logprobs['rl_strategy'].append(logprob_z[b_id])

        for log_prob in logprobs:
            self.logprobs['rl_realization'].extend(log_prob)

        self.logprobs['rl_realization_org'] = logprobs

        # compute reward here
        generated_dialog = {key: sys_turns}
        real_dialogues = {key: sys_turns_gt}

        report, success, match, bleu = evaluator.evaluateModel(generated_dialog, real_dialogues=real_dialogues, mode="offline_rl")
        return report, success, match, bleu

    def preprocess_reward(self, reward, all_rewards):
        all_rewards.append(reward)
        # standardize the reward
        r = (reward - np.mean(all_rewards)) / max(self.args.std_threshold, np.std(all_rewards))
        # compute accumulated discounted reward
        g = self.model.np2var(np.array([r]), FLOAT).view(1, 1)
        g_s = self.model.np2var(np.array([r]), FLOAT).view(1, 1) if self.args.diff_discount else None
        rewards_strategy = []
        rewards_realization = []
        assert len(self.logprobs['rl_realization_org']) == len(self.logprobs['rl_strategy'])
        n = len(self.logprobs['rl_realization_org'])
        for b_p in self.logprobs['rl_realization_org']:
            if not self.args.long_term:
                g = self.model.np2var(np.array([r]), FLOAT).view(1, 1) / n
            if self.args.diff_discount:
                if self.args.rl_realization:
                    for w_p in b_p:
                        rewards_realization.insert(0, g)
                        g = g * self.args.gamma
                if self.args.rewards_strategy:
                    rewards_strategy.insert(0, g_s)
                    if self.args.long_term:
                        g_s = g_s * self.args.gamma
            else:
                if self.args.rl_realization:
                    for w_p in b_p:
                        rewards_realization.insert(0, g)
                        g = g * self.args.gamma
                    rewards_strategy.insert(0, g)
                else:
                    rewards_strategy.insert(0, g)
                    if self.args.long_term:
                        g = g * self.args.gamma
        return rewards_strategy, rewards_realization

    def update(self, reward, stats):
        rewards = {'success': {'rl_strategy': [], 'rl_realization': []}, 'match': {'rl_strategy': [], 'rl_realization': []},
                   'bleu': {'rl_strategy': [], 'rl_realization': []}}
        for k in rewards.keys():
            rewards_strategy, rewards_realization = self.preprocess_reward(reward[k], self.all_rewards[k])
            rewards[k]['rl_strategy'] = rewards_strategy.copy()
            rewards[k]['rl_realization'] = rewards_realization.copy()

        self.loss = 0

        # estimate the loss using one MonteCarlo rollout
        if self.args.rl_realization and (self.curr_stage == 'rl_realization' or self.args.synchron):
            for lp, r in zip(self.logprobs['rl_realization'], rewards['success']['rl_realization']):
                self.loss -= lp * r
            if self.reward_weight:
                for lp, r in zip(self.logprobs['rl_realization'], rewards['bleu']['rl_realization']):
                    self.loss -= lp * self.reward_weight * r
            self.rl_realization_cnt += 1

        if self.args.rl_strategy and (self.curr_stage == 'rl_strategy' or self.args.synchron):
            for lp, r in zip(self.logprobs['rl_strategy'], rewards['success']['rl_strategy']):
                self.loss -= lp * r
            self.rl_strategy_cnt += 1

        if self.args.rl_strategy and self.args.rl_realization and self.args.synchron:
            self.opt_startegy.zero_grad()
            self.opt_relization.zero_grad()
            self.loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.rl_clip)
            self.opt_startegy.step()
            self.opt_relization.step()
        else:
            if self.args.rl_strategy and self.curr_stage == 'rl_strategy':
                self.opt_startegy.zero_grad()
                self.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.rl_clip)
                self.opt_startegy.step()
                if self.args.rl_realization:
                    if self.rl_strategy_cnt == self.args.rl_strategy_freq:
                        self.curr_stage = 'rl_realization'
                        self.rl_strategy_cnt = 0
            elif self.args.rl_realization and self.curr_stage == 'rl_realization':
                self.opt_relization.zero_grad()
                self.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.rl_clip)
                self.opt_relization.step()
                if self.args.rl_strategy:
                    if self.rl_realization_cnt == self.args.rl_realization_freq:
                        self.curr_stage = 'rl_strategy'
                        self.rl_realization_cnt = 0

        self.update_n += 1
        if self.args.rl_clip_scheduler and self.update_n % self.args.rl_clip_freq == 0:
            self.rl_clip *= self.args.rl_clip_decay