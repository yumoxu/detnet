# -*- coding: utf-8 -*-
from os.path import dirname, abspath
import sys
import torch
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from argparse import ArgumentParser
from utils.res_handler import ResLoader
import utils.config_loader as config
from utils.config_loader import mode, meta_model_name, logger, config_meta
from frame.model import make_detnet_model, make_milnet_model
from frame.train import train_model
import frame.eval_doc as eval_doc
import frame.eval_sent as eval_sent
import frame.eval_word as eval_word
from frame.top_words import top_words_E2E


def init():
    # parse args
    parser = ArgumentParser()
    parser.add_argument('n_devices',
                        nargs='?',
                        default=4,
                        help='num of devices on which model will be running on')

    args = parser.parse_args()
    all_device_ids = [0, 1, 2, 3]
    device = all_device_ids[:int(args.n_devices)]
    config_meta['device'] = device

    if not torch.cuda.is_available():
        placement = 'cpu'
        logger.info('[MAIN INIT] path mode: {0}, placement: {1}'.format(config.path_type, placement))
    else:
        if len(device) == 1:
            placement = 'single'
            torch.cuda.set_device(device[0])
        elif config_meta['auto_parallel']:
            placement = 'auto'
        else:
            placement = 'manual'

        logger.info(
            '[MAIN INIT] path mode: {0}, placement: {1}, n_devices: {2}'.format(config.path_type, placement,
                                                                                args.n_devices))
    config_meta['placement'] = placement


if __name__ == '__main__':
    init()

    # 0 is for padding and index starts from 1. 1 is UNK.
    vocab_size = len(ResLoader().get_vocab()) + 1
    logger.info('Vocab: {}'.format(vocab_size))

    if config.lang == 'en':
        model2iter = {
            'hiernet': 9400,
            'milnet': 4000,
            'detnet1': 6400,
            'detnet2': 10000,
            'detnet': 7400,
        }
    elif config.lang == 'zh':
        model2iter = {
            'hiernet': 9200,
            'milnet': 6200,
            'detnet1': 7000,
            'detnet2': 9200,
            'detnet': 9000,
        }
    else:
        raise ValueError('Invalid language: {}'.format(config.lang))

    model_paras = {
        'vocab_size': vocab_size,
    }

    if meta_model_name.startswith('milnet'):
        logger.info('Baseline making: {}'.format(meta_model_name))
        make_model = make_milnet_model
    else:
        if meta_model_name == 'hiernet':
            logger.info('Baseline making: {}'.format(meta_model_name))
            model_paras['use_sent_embed_attn'] = False
        else:
            logger.info('DetNet making: {}'.format(meta_model_name))
            model_paras['use_sent_embed_attn'] = True

        make_model = make_detnet_model

    if meta_model_name == 'milnet':
        logger.info('Baseline making: {}'.format(meta_model_name))
        make_model = make_milnet_model
    elif meta_model_name == 'hiernet':
        logger.info('Baseline making: {}'.format(meta_model_name))
        model_paras['use_sent_embed_attn'] = False
        make_model = make_detnet_model
    else:
        logger.info('DetNet making: {}'.format(meta_model_name))
        model_paras['use_sent_embed_attn'] = True
        make_model = make_detnet_model

    model = make_model(**model_paras)
    restore = False

    train_paras = {
        'model': model,
        'restore': restore,
        'batch_log': False,
        'batch_eval': True,
    }

    save_pred = False
    save_gold = False
    test_paras = {
        'model': model,
        'save_pred': save_pred,
        'save_gold': save_gold,
        'restore': restore,
    }

    if mode == 'train':
        train_model(**train_paras)
    elif mode == 'test_doc':
        eval_doc.test_model_doc(**test_paras)
    elif mode == 'test_sent_syn':
        test_paras = {
            **test_paras,
            'synthese': True,
        }
        eval_sent.test_model_sent_syn_with_checkpoints(**test_paras)
    elif mode == 'test_sent_mturk':
        test_paras = {
            **test_paras,
            'corpus': 'wiki',  # wiki, nyt
            'n_iter': model2iter[meta_model_name],
        }
        eval_sent.test_model_sent_mturk_with_checkpoints(**test_paras)
    elif mode == 'test_word_mturk':
        test_paras = {
            **test_paras,
            'matching_mode': 'RELAXED',  # [STRICT | RELAXED]
            'corpus': 'wiki',  # wiki, nyt
            'n_iter': model2iter[meta_model_name],
        }
        eval_word.test_model_word_mturk_with_checkpoints(**test_paras)
    elif mode == 'top_words':
        paras = {
            'model': model,
            'n_iter': model2iter[meta_model_name],
            'restore': None,
            'n_top': 100,
        }
        top_words_E2E(**paras)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))
