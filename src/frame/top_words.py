# -*- coding: utf-8 -*-
import io
from os import listdir
import utils.config_loader as config
from utils.config_loader import logger, path_parser
import summ.viz as viz
from data.dataset_parser import DatasetParser
from os.path import join, isfile, dirname, abspath
import utils.tools as tools


data_parser = DatasetParser()

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
    word_scores = viz.get_word_scores(model, n_iter=n_iter, doc_ids=doc_ids, restore=restore)

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
