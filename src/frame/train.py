# -*- coding: utf-8 -*-
from io import open
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from frame.eval_doc import eval_model
from frame.checkpoint_op import *
import data.data_pipe as pipe
import utils.config_loader as config_loader
from utils.config_loader import config_model, path_parser


def check_grads(model):
    paras = [(name, para) for name, para in model.named_parameters() if para.requires_grad]

    for idx, p in enumerate(paras):
        if p[1].grad is None:
            logger.info('{0}: {1}: grad: None'.format(idx, p[0]))
        else:
            grad = p[1].grad.data.abs().sum()
            logger.info('{0}: {1}: grad: {2:3f}'.format(idx, p[0], grad))


def train_model_with_checkpoints(model, restore=False, batch_log=True, batch_eval=True):
    logger.info('Config: {0}'.format(config_loader.model_name))
    max_n_iter = config_model['n_batches']

    checkpoint = join(path_parser.model_save, config_loader.model_name)
    performance_save_fp = join(path_parser.performances, '{0}.txt'.format(config_loader.model_name))

    data_loader = pipe.DomDetDataLoader(dataset_type='train')
    optimizer = make_opt(model)

    if config_loader.placement == 'auto':
        model = nn.DataParallel(model, device_ids=config_loader.device)

    if config_loader.placement in ('auto', 'single'):
        model.cuda()

    if restore:
        global_n_iter = load_checkpoint(checkpoint, model=model, optimizer=optimizer, no_iter_strategy='last')
        global_n_iter -=1 # backward compatible
        logger.info('MODE: restore a pre-trained model and resume training from {}'.format(global_n_iter))
        checkpoint = join(checkpoint, 'resume')
        max_n_iter += global_n_iter
    else:
        logger.info('MODE: create a new model')
        global_n_iter = 0

    train_skip_iter = 50  # 50
    eval_skip_iter = 200

    checkpoint_dict = dict()  # {n_batches: f1}
    batch_loss = 0.0
    for epoch_idx in range(config_model['n_epochs']):
        for batch_idx, batch in enumerate(data_loader):
            model.train(mode=True)
            global_n_iter += 1

            feed_dict = copy.deepcopy(batch)

            for (k, v) in feed_dict.items():
                feed_dict[k] = Variable(v, requires_grad=False)  # fix ids and masks

            loss = model(**feed_dict)[0]

            if config_loader.placement == 'auto':
                loss = loss.mean()  # gather loss from multiple gpus

            batch_loss += loss.data[0]

            optimizer.zero_grad()  # clear history gradients

            loss.backward(retain_graph=True)

            if config_model['grad_clip'] is not None:
                nn.utils.clip_grad_norm(model.parameters(), config_model['grad_clip'])

            optimizer.step()

            if batch_log and global_n_iter % train_skip_iter == 0:
                check_grads(model)
                logger.info('iter: {0}, loss: {1:.6f}'.format(global_n_iter, loss.data[0]))

            if batch_eval and global_n_iter % eval_skip_iter == 0:
                eval_log_info = eval_model(model=model, phase='dev')
                eval_log_info['n_iter'] = global_n_iter

                # res_str = build_res_str(stage='n_iter')
                # logger.info(res_str.format(**eval_log_info))

                mean_batch_loss = batch_loss / eval_skip_iter
                eval_loss = eval_log_info['loss']
                f1 = eval_log_info['avg_f1']

                checkpoint_dict, update, is_best = update_checkpoint_dict(checkpoint_dict, k=global_n_iter, v=f1)
                perf_rec = '{0}\t{1:.6f}\t{2:.6f}\t{3:.6f}\n'.format(global_n_iter, mean_batch_loss, eval_loss, f1)

                state = {'n_iters': global_n_iter + 1,
                         'state_dict': model.state_dict(),
                         'optimizer_dict' : optimizer.state_dict()},
                if update:
                    save_checkpoint(state, checkpoint, global_n_iter, is_best)
                    clean_outdated_checkpoints(checkpoint, checkpoint_dict)

                with open(performance_save_fp, 'a', encoding='utf-8') as f:
                    f.write(perf_rec)
                logger.info(perf_rec.strip('\n'))

                batch_loss = 0.0

            if global_n_iter == max_n_iter:
                break

        if global_n_iter == max_n_iter:
            logger.info('finished expected training: {0} batches!'.format(max_n_iter))
            break


def train_model(model, restore=False, batch_log=True, batch_eval=True):
    train_model_with_checkpoints(model, restore, batch_log, batch_eval)
