import os
import sys
import numpy as np
import torch as th
from torch import nn
from datetime import datetime

from latent_dialog.enc2dec.encdec import TEACH_FORCE, GEN
from collections import defaultdict
from latent_dialog.utils import idx2word

import logging
from latent_dialog.utils import TBLogger



logger = logging.getLogger()

PREVIEW_NUM = 0


class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            # print('key = %s\nval = %s' % (key, val))
            if val is not None and type(val) is not bool:
                self.losses[key].append(val.item())  # {key: [val1_scalar, val2_scalar...]} ?

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            aver_loss = np.average(loss) if window is None else np.average(loss[-window:])
            if 'nll' in key:  # TODO: ?
                str_losses.append('{} PPL {:.3f}'.format(key, np.exp(aver_loss)))
            else:
                str_losses.append('{} {:.3f}'.format(key, aver_loss))

        if prefix:
            return '{}: {} {}'.format(prefix, name, ' '.join(str_losses))
        else:
            return '{} {}'.format(name, ' '.join(str_losses))

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def avg_loss(self):
        return np.mean(self.backward_losses)


def train(model, train_data, val_data, config, evaluator):
    # tensorboard
    tb_path = os.path.join(config.saved_path, "tensorboard/")
    tb_logger = TBLogger(tb_path)

    # training parameters
    batch_cnt, best_epoch_score = 0, 0
    best_score = -np.inf
    
    train_loss = LossManager()

    # models
    model.train()  # activate batch normalization and dropout
    optimizer = model.get_optimizer(config, verbose=False)
    saved_models = []
    last_n_model = config.last_n_model 


    logger.info('***** Training Begins at {} *****'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    logger.info('***** Epoch 0/{} *****'.format(config.num_epoch))
    for epoch in range(config.num_epoch):
        # EPOCH
        # for each epoch, reinitialize batches
        train_data.epoch_init(config, shuffle=True, verbose=epoch==0, fix_batch=config.fix_train_batch)
        num_batch = train_data.num_batch

        while True:
            # BATCH
            batch = train_data.next_batch()
            if batch is None:
                break

            optimizer.zero_grad()
            # TEACH_FORCE = decoding directly by groundtruth then adding attn
            loss = model(batch, mode=TEACH_FORCE, epoch=epoch)
            train_loss.add_loss(loss)
            model.backward(loss, batch_cnt)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # learnable parameters
            optimizer.step()
            batch_cnt += 1

            # tensorboard save 
            data_dict = {}
            for key, val in loss.items():
                if val is not None and type(val) is not bool:
                    data_dict["train/%s"%key] = val.item()
            tb_logger.add_scalar_summary(data_dict, batch_cnt)

            # print training loss every print_frequency batch
            if batch_cnt % config.print_frequency == 0:
                logger.info(train_loss.pprint('Train',
                                        window=config.print_frequency,
                                        prefix='{}/{}'.format(batch_cnt%num_batch, num_batch)))
                sys.stdout.flush()

        # Evaluate at the end of every epoch
        logger.info('Checkpoint step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
        logger.info('==== Evaluating Model ====')

        # Generation (bleu/success/match)
        success, match, bleu = generate(model, val_data, config, evaluator)

        # Validation (loss)
        logger.info(train_loss.pprint('Train'))
        valid_loss = validate(model, val_data, config, batch_cnt)

        stats = {'val/success': success, 'val/match': match, 'val/bleu': bleu, "val/loss": valid_loss}
        tb_logger.add_scalar_summary(stats, batch_cnt)

        score = bleu
        if epoch >= config.warmup:
            # save model if score increases
            if score > best_score:
                if config.save_model:
                    cur_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    logger.info('*** Model Saved with overall loss = {}, score = {}, at {}. ***'.format(valid_loss, score, cur_time))
                    model.save(config.saved_path, epoch)
                    best_epoch_score = epoch
                    saved_models.append(epoch)
                    if len(saved_models) > last_n_model:
                        remove_model = saved_models[0]
                        saved_models = saved_models[-last_n_model:]
                        os.remove(os.path.join(config.saved_path, "{}-model".format(remove_model)))
                best_score = score

        # exit val mode
        model.train()
        train_loss.clear()
        logger.info('\n***** Epoch {}/{} *****'.format(epoch+1, config.num_epoch))
        sys.stdout.flush()

    logger.info('Training Ends.')
    logger.info('Best validation score = %f at epoch %d' % (best_score, best_epoch_score))

    return best_epoch_score



def validate(model, data, config, batch_cnt=None):
    model.eval()  # deactivate batch normalization and dropout
    data.epoch_init(config, shuffle=False, verbose=False, fix_batch=True)  # a dialogue as a batch
    losses = LossManager()
    while True:
        batch = data.next_batch()
        if batch is None:
            break
        loss = model(batch, mode=TEACH_FORCE, act_vec_drop=False)
        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))  # a (weighted) sum of loss

    valid_loss = losses.avg_loss()  # mean of self.backward_losses
    logger.info(losses.pprint(data.name))
    logger.info('--- Total loss = {}'.format(valid_loss))
    sys.stdout.flush()
    return valid_loss


def generate(model, data, config, evaluator, verbose=True, dest_f=None, vec_f=None, label_f=None):
    """
        Args:
            - evalutor: this is used to calculate bleu/match/success
    """

    model.eval()
    batch_cnt = 0
    generated_dialogs = defaultdict(list)  # {filename: [pred_responses, ...]}
    real_dialogs = defaultdict(list)  # {filename: [true_responses, ...]}

    data.epoch_init(config, shuffle=False, verbose=False, fix_batch=True)  # all turns in a dialogue as a batch
    logger.debug('Generation: {} batches'.format(data.num_batch))
    while True:
        batch = data.next_batch()
        batch_cnt += 1
        if batch is None:
            break

        act_outputs, act_labels, outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in outputs['sequence']]
        pred_labels = np.array(pred_labels, dtype=int)  # (batch_size, max_dec_len)
        true_labels = labels.data.numpy()  # (batch_size, output_seq_len)

        act_labels = act_labels.cpu()
        pred_acts = [t.cpu().data.numpy() for t in act_outputs['sequence']]
        pred_acts = np.array(pred_acts, dtype=int)  # (batch_size, max_dec_len)
        true_acts = act_labels.data.numpy()

        # get context
        ctx = batch.get('contexts')  # (batch_size, max_ctx_len, max_utt_len)
        ctx_len = batch.get('context_lens')  # (batch_size, ), number of turns, i.e. [#turns_in_context for each pair]
        keys = batch['keys'] # [filenames in this batch]

        sample_z = outputs.get("sample_z", None)
        if sample_z is not None:
            sample_z = sample_z.cpu().data.numpy()

        batch_size = pred_labels.shape[0]
        for b_id in range(batch_size):
            pred_str = idx2word(model.vocab, pred_labels, b_id) # return detokenized result
            true_str = idx2word(model.vocab, true_labels, b_id)
            pred_act = idx2word(model.act_vocab, pred_acts, b_id)
            true_act = idx2word(model.act_vocab, true_acts, b_id)
            prev_ctx = ''
            if ctx is not None:
                ctx_str = []
                for t_id in range(ctx_len[b_id]):
                    temp_str = idx2word(model.vocab, ctx[:, t_id, :], b_id, stop_eos=False)
                    ctx_str.append(temp_str)
                prev_ctx = 'Source context: {}'.format(ctx_str)

            generated_dialogs[keys[b_id]].append(pred_str)  # {filename: [pred_responses, ...]}
            real_dialogs[keys[b_id]].append(true_str)  # {filename: [true_responses, ...]}

            if verbose and batch_cnt <= PREVIEW_NUM:
                logger.debug('%s-prev_ctx = %s' % (keys[b_id], prev_ctx,))
                logger.debug('True: {}'.format(true_str, ))
                logger.debug('Pred: {}'.format(pred_str, ))
                logger.debug('-' * 40)

            if dest_f is not None:
                dest_f.write('%s-prev_ctx = %s\n' % (keys[b_id], prev_ctx,))
                dest_f.write('True Rsp: {}\n'.format(true_str, ))
                dest_f.write('Pred Rsp: {}\n'.format(pred_str, ))
                dest_f.write('True Act: {}\n'.format(true_act, ))
                dest_f.write('Pred Act: {}\n'.format(pred_act, ))
                dest_f.write('-' * 40+"\n")

            if  vec_f is not None and sample_z is not None:
                sample = sample_z[b_id]
                sample_str = "\t".join( str(x) for x in sample )
                vec_f.write(sample_str + "\n")

            if label_f is not None:
                label_f.write("%s\t%s\n"%(true_str, pred_str))


    # data.name = 'Test', generated_dialogs = {filename: [pred_responses, ...]}, real_dialogs = {filename: [true_responses, ...]}
    task_report, success, match, bleu  = evaluator.evaluateModel(generated_dialogs, real_dialogues=real_dialogs, mode=data.name)

    logger.debug('Generation Done')
    logger.info(task_report)
    logger.debug('-' * 40)
    return success, match, bleu


def reinforce(agent, model, train_data, val_data, rl_config, sl_config, evaluator):
    # clone train data for supervised learning
    sl_train_data = train_data.clone()

    # tensorboard
    tb_path = os.path.join(rl_config.saved_path, "tensorboard/")
    tb_logger = TBLogger(tb_path)

    episode_cnt, best_episode = 0, 0
    best_rewards = -1 * np.inf

    # model
    model.train()
    saved_models = []
    last_n_model = rl_config.last_n_model

    logger.info('***** Reinforce Begins at {} *****'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    for epoch_id in range(rl_config.num_epoch):

        train_data.epoch_init(sl_config, shuffle=True, verbose=epoch_id == 0,
                              fix_batch=True)  # fix_batch has to be true for offline reinforce. each batch is an episode
        while True:
            batch = train_data.next_batch() # each batch belong to same dialogue
            if batch is None:
                break

            # reinforcement learning
            assert len(set(batch['keys'])) == 1  # make sure it's the same dialogue
            report, success, match, bleu = agent.run(batch, evaluator, max_words=rl_config.max_words,
                                                     temp=rl_config.temperature)
            # this is the reward function during training
            reward = float(success)
            reward_dict = {'success': float(success), 'match': float(match), 'bleu': float(bleu)}
            stats = {'train/match': match, 'train/success': success, 'train/bleu': bleu}
            agent.update(reward_dict, stats)
            tb_logger.add_scalar_summary(stats, episode_cnt)

            # print loss sometimes
            episode_cnt += 1 # episode = batch * epoch
            if episode_cnt % rl_config.print_frequency == 0:
                logger.info("{}/{} episode: mean_reward {} for last {} episodes".format(episode_cnt,
                                                train_data.num_batch * rl_config.num_epoch,
                                                np.mean(agent.all_rewards['success'][-rl_config.print_frequency:]),
                                                rl_config.print_frequency))

            # record model performance in terms of several evaluation metrics
            if rl_config.record_frequency > 0 and episode_cnt % rl_config.record_frequency == 0:

                logger.info('Checkpoint step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
                logger.info('==== Evaluating Model ====')

                agent.print_dialog(agent.dlg_history, reward, stats)

                logger.info('mean_reward {} for last {} episodes'.format(
                    np.mean(agent.all_rewards['success'][-rl_config.record_frequency:]), rl_config.record_frequency))

                # validation
                valid_loss = validate(model, val_data, sl_config)
                v_success, v_match, v_bleu = generate(model, val_data, sl_config, evaluator)

                # tensorboard
                stats = {'val/success': v_success, 'val/match': v_match, 'val/bleu': v_bleu, "val/loss": valid_loss}
                tb_logger.add_scalar_summary(stats, episode_cnt)

                # save model
                # consider bleu into the evaluation metric
                if (v_success + v_match) / 2 + v_bleu > best_rewards:
                    cur_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    logger.info(
                        '*** Model Saved with match={} success={} bleu={}, at {}. ***\n'.format(v_match, v_success,
                                                                                                v_bleu, cur_time))

                    model.save(rl_config.saved_path, episode_cnt)
                    best_episode = episode_cnt
                    saved_models.append(episode_cnt)
                    if len(saved_models) > last_n_model:
                        remove_model = saved_models[0]
                        saved_models = saved_models[-last_n_model:]
                        os.remove(os.path.join(rl_config.saved_path, "{}-model".format(remove_model)))
                    # new evaluation metric
                    best_rewards = (v_success + v_match) / 2 + v_bleu

            model.train()
            sys.stdout.flush()

    return best_episode









