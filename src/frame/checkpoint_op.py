# -*- coding: utf-8 -*-
import os
from os import listdir
from os.path import exists, join, dirname, abspath, isfile
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from frame.model import *
import utils.config_loader as config_loader
import shutil

def save_checkpoint(state, checkpoint, n_iter, is_best):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    fp = join(checkpoint, '{}.pth.tar'.format(n_iter))
    if not exists(checkpoint):
        logger.info("Checkpoint directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, fp)
    if is_best:
        shutil.copyfile(fp, os.path.join(checkpoint, 'best.pth.tar'))


def get_min_kv(d, init_v=1.0):
    min_k = None
    min_v = init_v
    for k, v in d.items():
        if v < min_v:
            min_v = v
            min_k = k
    return min_k, min_v


def get_max_v(d, init_v=0.0):
    max_v = init_v
    for k, v in d.items():
        if v > max_v:
            max_v = v
    return max_v


def clean_outdated_checkpoints(checkpoint, checkpoint_dict):
    fns = [fn for fn in listdir(checkpoint) if fn.endswith('.pth.tar') and fn != 'best.pth.tar']
    target_fns = ['{}.pth.tar'.format(n_iter) for n_iter in checkpoint_dict]

    for fn in fns:
        if fn not in target_fns:
            os.remove(join(checkpoint, fn))
            logger.info('Remove checkpoint: {}'.format(fn))

    fns = [fn for fn in listdir(checkpoint) if fn.endswith('.pth.tar') and fn != 'best.pth.tar']
    logger.info('Available #checkpoints: {}'.format(len(fns)))


def update_checkpoint_dict(checkpoint_dict, k, v, max_n_checkpoint=3):
    update = False
    is_best = v > get_max_v(checkpoint_dict)

    if len(checkpoint_dict) < max_n_checkpoint:  # not full
        checkpoint_dict[k] = v
        update = True
        return checkpoint_dict, update, is_best

    min_k, min_v = get_min_kv(checkpoint_dict)
    if v > min_v:  # replace
        checkpoint_dict.pop(min_k)
        checkpoint_dict[k] = v
        update = True
        return checkpoint_dict, update, is_best

    return checkpoint_dict, update, is_best


def load_checkpoint(checkpoint, model, n_iter=None, optimizer=None, filter_keys=None, no_iter_strategy='best'):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    logger.info('checkpoint: {}'.format(checkpoint))
    if n_iter:
        fp = join(checkpoint, '{}.pth.tar'.format(n_iter))
    else:
        if no_iter_strategy == 'best':
            fp = join(checkpoint, 'best.pth.tar')
        elif no_iter_strategy == 'last':
            iters = [int(fn.replace('.pth.tar', '')) for fn in listdir(checkpoint)
                     if fn.endswith('.pth.tar') and fn != 'best.pth.tar']
            fn = '{}.pth.tar'.format(max(iters))
            fp = join(checkpoint, fn)
            # logger.info('Load last checkpoint at: {}'.format(fn))
        else:
            raise ValueError('no_iter_strategy should be either best or last!')

    if exists(fp):
        logger.info('MODE: load a pre-trained model: {0} at iter: {1}'.format(config_loader.model_name, n_iter))
    else:
        raise FileNotFoundError('Checkpoint does not exist {}'.format(fp))

    state = torch.load(fp)[0]
    pretrained_dict = state['state_dict']
    if filter_keys:
        logger.info('State filtering: {}'.format(', '.join(filter_keys)))
        for k in filter_keys:
            pretrained_dict.pop(k, None)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
    else:
        logger.info('Load whole model state...')
        model_dict = pretrained_dict

    model.load_state_dict(model_dict)

    if optimizer:
        optimizer.load_state_dict(state['optimizer_dict'])

    return state['n_iters']