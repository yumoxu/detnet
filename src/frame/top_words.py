# -*- coding: utf-8 -*-
import io
from os import listdir
import utils.config_loader as config
from utils.config_loader import logger, path_parser
from data.dataset_parser import DatasetParser
from os.path import join, isfile, dirname, abspath
import utils.tools as tools

import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np
from frame.checkpoint_op import *
import data.data_pipe as pipe

data_parser = DatasetParser()


def get_word_scores(model, n_iter, doc_ids, restore=None):
    if config.placement == 'auto':
        model = nn.DataParallel(model, device_ids=config.device)

    if config.placement in ('auto', 'single'):
        model.cuda()

    checkpoint = join(path_parser.model_save, config.model_name)
    if restore:
        checkpoint = join(checkpoint, 'resume')

    filter_keys = None
    if config.reset_size_for_test and not config.set_sep_des_size:
        filter_keys = ['module.word_det.des_ids', 'module.word_det.des_sent_mask', 'module.word_det.des_word_mask']
    load_checkpoint(checkpoint=checkpoint, model=model, n_iter=n_iter, filter_keys=filter_keys)
    model.eval()

    dataset_type = 'dev'
    data_loader = pipe.DomDetDataLoader(dataset_type=dataset_type, collect_doc_ids=True, doc_ids=doc_ids)

    ws_list = []
    for _, batch in enumerate(data_loader):  # only one batch
        logger.info('Batch size: {}'.format(len(batch)))
        c = copy.deepcopy
        feed_dict = c(batch)

        del feed_dict['doc_ids']

        for (k, v) in feed_dict.items():
            feed_dict[k] = Variable(v, requires_grad=False, volatile=True)  # fix ids and masks

        feed_dict['return_sent_attn'] = True
        feed_dict['return_word_attn'] = True
        _, _, _, word_scores, _, word_attn = model(**feed_dict)

        # weight sent scores with their attn
        word_scores = word_scores.data.cpu().numpy()  # n_batch * n_sents * n_words * n_doms
        ws_list.append(word_scores)

    ws = np.concatenate(ws_list, axis=0)
    return ws


def get_word_dom_scores(model, n_iter, restore):
    """
        Get word dom scores for given label ids.
    :return:
    """
    dom2ws = {}  # each value is a dict
    for dom in config.doms_final:
        dom2ws[dom] = {}

    root = path_parser.dataset_type_dp['dev']
    fns = [fn for fn in listdir(root) if isfile(join(root, fn))]
    # fns = ['316858_GOV', '550069_GEN']
    doc_ids = [tools.get_doc_id(fn) for fn in fns]

    # n_batch * n_sents * n_words * n_doms
    word_scores = get_word_scores(model, n_iter=n_iter, doc_ids=doc_ids, restore=restore)

    for doc_idx, fn in enumerate(fns):
        res = data_parser.parse_sents(fp=join(root, fn), clip=True)
        sents = res['sents']
        for sent_idx, ss in enumerate(sents):
            for word_idx, ww in enumerate(ss):
                for dom_idx, dom in enumerate(config.doms_final):
                    dom2ws[dom][ww] = word_scores[doc_idx, sent_idx, word_idx, dom_idx]

    return dom2ws


def dump_dom2ws(dom2ws, n_top):
    for dom, ws in dom2ws.items():
        ws = sorted(ws.items(), key=lambda item: item[1], reverse=True)
        records = ['\t'.join((ww, str(ss))) for ww, ss in ws[:n_top]]
        content = '\n'.join(records)

        out_fp = join(path_parser.top, 'top_{}.txt'.format(dom))
        with io.open(out_fp, encoding='utf-8', mode='a') as f:
            f.write(content)

        logger.info('dom2ws dumped: {}'.format(dom))


def top_words_E2E(model, n_iter, restore, n_top=100):
    dom2ws = get_word_dom_scores(model, n_iter, restore)
    dump_dom2ws(dom2ws, n_top)
