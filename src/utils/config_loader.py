import logging
import logging.config
import yaml
from io import open
import os
import socket
import warnings
import itertools
from os.path import join, dirname, abspath
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))


def deprecated(func):
    """
        This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.
    """

    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


class PathParser:
    def __init__(self, config_path, path_type, lang, mat_combs):
        self.proj_root = config_path['proj_root'][path_type]
        self.proj_paths = config_path['proj_paths']

        # set performances
        self.performances = join(self.proj_root, self.proj_paths['performances'], lang)
        self.sigf = join(self.performances, self.proj_paths['sigf'])  # files for significance test
        self.sigf_doc = join(self.sigf, 'doc')
        self.sigf_sent_mturk = join(self.sigf, 'sent_mturk')
        self.sigf_sent_syn = join(self.sigf, 'sent_syn')

        self.log = join(self.proj_root, self.proj_paths['log'], lang)

        # set data
        self.data = join(self.proj_root, self.proj_paths['data'], lang)

        # set wiki
        self.wiki_url_pattern = config_path['wiki_url_pattern_{0}'.format(lang)]

        if lang == 'en':
            self.wiki_corpus = config_path['wiki_corpus']
            self.wiki_corpus_head = join(self.wiki_corpus, config_path['head'])
        else:
            self.wiki_corpus = join(self.data, '{0}wiki'.format(lang))

        data_r = join(self.data, self.proj_paths['relation'])
        data_r_dom = join(data_r, self.proj_paths['domain_lv'])
        data_r_topic = join(data_r, self.proj_paths['topic_lv'])
        data_r_subtopic = join(data_r, self.proj_paths['subtopic_lv'])
        data_r_subsubtopic = join(data_r, self.proj_paths['subsubtopic_lv'])
        data_r_subsubsubtopic = join(data_r, self.proj_paths['subsubsubtopic_lv'])
        self.lv_dp = {
            'dom': data_r_dom,
            'topic': data_r_topic,
            'subtopic': data_r_subtopic,
            'subsubtopic': data_r_subsubtopic,
            'subsubsubtopic': data_r_subsubsubtopic,
        }

        self.parent_lv_dp = {
            'topic': data_r_dom,
            'subtopic': data_r_topic,
            'subsubtopic': data_r_subtopic,
            'subsubsubtopic': data_r_subsubtopic,
        }

        self.data_doc = join(self.data, self.proj_paths['doc'])
        self.data_doc_dedu = join(self.data, self.proj_paths['doc_dedu'])
        self.data_doc_mixed = join(self.data, self.proj_paths['doc_mixed'])

        self.data_non_ge_ids = join(self.data_doc, self.proj_paths['non_ge_ids'])
        self.data_clean_stats = join(self.data_doc_dedu, self.proj_paths['clean_stats'])

        # set dataset
        self.dataset = join(self.proj_root, self.proj_paths['dataset'], lang)
        self.dataset_bipartite = join(self.dataset, self.proj_paths['bipartite'])
        self.dataset_unproc = join(self.dataset, self.proj_paths['unproc'])
        self.dataset_whole = join(self.dataset, self.proj_paths['whole'])

        self.dataset_train = join(self.dataset, self.proj_paths['train'])
        self.dataset_dev = join(self.dataset, self.proj_paths['dev'])
        self.dataset_test = join(self.dataset, self.proj_paths['test'])

        self.dataset_test_sents = join(self.dataset, self.proj_paths['test_sents'])
        self.dataset_test_filtered = join(self.dataset, self.proj_paths['test_filtered'])
        self.dataset_dom_sent_corpora = join(self.dataset, self.proj_paths['dom_sent_corpora'])
        self.dataset_syn_docs = join(self.dataset, self.proj_paths['syn_docs'])
        self.dataset_test_human_eval = join(self.dataset, self.proj_paths['test_human_eval'])

        self.dataset_lead_sent_asm = join(self.dataset, self.proj_paths['label_consistency_assumption'])

        self.dataset_type_dp = {
            'train': self.dataset_train,
            'dev': self.dataset_dev,
            'test': self.dataset_test,
            'test_filtered': self.dataset_test_filtered,
            'syn_docs': self.dataset_syn_docs,
        }

        self.dataset_des_unproc = join(self.dataset, self.proj_paths['side_des_unproc'])
        self.dataset_des = join(self.dataset, self.proj_paths['side_des'])

        self.mturk = join(self.proj_root, self.proj_paths['mturk'])
        self.zh_wiki = join(self.mturk, self.proj_paths['zh_wiki'])
        self.annot = join(self.mturk, 'annotations')

        # mturk annotations
        self.annot_sent_en_wiki = join(self.annot, 'sent', self.proj_paths['annot_en_wiki'])
        self.annot_sent_zh_wiki = join(self.annot, 'sent', self.proj_paths['annot_zh_wiki'])
        self.annot_sent_en_nyt = join(self.annot, 'sent', self.proj_paths['annot_en_nyt'])
        self.annot_word_en_wiki = join(self.annot, 'word', self.proj_paths['annot_en_wiki'])
        self.annot_word_zh_wiki = join(self.annot, 'word', self.proj_paths['annot_zh_wiki'])
        self.annot_word_en_nyt = join(self.annot, 'word', self.proj_paths['annot_en_nyt'])

        self.label_en_wiki = join(self.annot, 'word', self.proj_paths['label_en_wiki'])
        self.label_en_nyt = join(self.annot, 'word', self.proj_paths['label_en_nyt'])
        self.label_zh_wiki = join(self.annot, 'word', self.proj_paths['label_zh_wiki'])

        self.en_nyt_csv = join(self.mturk, self.proj_paths['en_nyt_csv'])
        self.en_wiki_csv = join(self.mturk, self.proj_paths['en_wiki_csv'])
        self.zh_wiki_csv = join(self.mturk, self.proj_paths['zh_wiki_csv'])

        # set res
        self.res = join(self.proj_root, self.proj_paths['res'], lang)
        self.vocab_full = join(self.res, self.proj_paths['vocab_full'])
        self.vocab_threshold_3 = join(self.res, self.proj_paths['vocab_threshold_3'])
        self.vocab_threshold_5 = join(self.res, self.proj_paths['vocab_threshold_5'])
        self.vocab = self.vocab_threshold_3

        self.model_save = join(self.proj_root, self.proj_paths['model'], lang)
        self.pred = join(self.proj_root, self.proj_paths['pred'])
        self.pred_doc = join(self.pred, self.proj_paths['doc'], lang)
        self.pred_syn = join(self.pred, self.proj_paths['syn'], lang)
        self.pred_mturk_wiki = join(self.pred, self.proj_paths['mturk'], lang)
        self.pred_mturk_nyt = join(self.pred, self.proj_paths['mturk'], 'nyt')
        self.pred_mturk_all = join(self.pred, self.proj_paths['mturk'], 'all')

        self.top = join(self.proj_root,self.proj_paths['top'])
        # set ltp
        LTP_MODEL_DP = 'ltp_data_v3.4.0'
        CWS_FN = 'cws.model'
        LTP_ENV_DP = ''  # specify 
        self.cws = join(LTP_ENV_DP, LTP_MODEL_DP, CWS_FN)


config_root = join(os.path.dirname(os.path.dirname(__file__)), 'config')

# meta
config_meta_fp = os.path.join(config_root, 'config_meta.yml')
config_meta = yaml.load(open(config_meta_fp, 'r', encoding='utf-8'))

hostname = socket.gethostname()

if hostname.endswith('inf.ed.ac.uk'):
    path_type = 'afs'
else:
    path_type = 'local'

print('Hostname: {0}, path mode: {1}'.format(hostname, path_type))

# device
device = config_meta['device']
mode = config_meta['mode']
debug = config_meta['debug']
auto_parallel = config_meta['auto_parallel']

if path_type == 'afs':
    if len(device) == 1:
        placement = 'single'
    elif len(device) >= 3:
        if auto_parallel:
            placement = 'auto'
        else:
            placement = 'manual'
    else:
        placement = None
else:
    placement = 'cpu'

lang = config_meta['lang']
n_ways = config_meta['n_ways']

# doms = config_meta['doms_{0}'.format(lang)]
# doms_final = config_meta['doms_final_{0}'.format(lang)]
if lang == 'en':
    doms = ['Government', 'Politics', 'Lifestyle', 'Business', 'Law', 'Health', 'Military', 'General']
    doms_final = ['GOV', 'LIF', 'BUS', 'LAW', 'HEA', 'MIL', 'GEN']
else:
    doms = ['政府', '政治', '商业', '生活', '法律', '健康', '军事', '普通']
    doms_final = ['政府', '生活', '商业', '法律', '健康', '军事', '普通']

n_doms = len(doms)
target_dom = config_meta['target_dom']
target_lv = config_meta['target_lv']

config_path_fp = os.path.join(config_root, 'config_path.yml')
config_path = yaml.load(open(config_path_fp, 'r'))
mat_combs = list(itertools.product(['1A', '1B'], ['ana1', 'dev1', 'eval1+2']))

path_parser = PathParser(config_path,
                         path_type=path_type,
                         lang=lang,
                         mat_combs=mat_combs)

# model
meta_model_name = config_meta['model_name']
config_model_fn = 'config_model_{0}.yml'.format(meta_model_name)
config_model_fp = os.path.join(config_root, config_model_fn)
config_model = yaml.load(open(config_model_fp, 'r'))

# model name
model_name_str_1 = '{v}-score_f[{score_f}]-ins_attn[{ins_attn}]-activate_f[{act_f}]-dropout[{dp}]-bn[{bn}]-gate[{gate}]'
model_name_str_1 = model_name_str_1.format(v=config_model['variation'],
                                           score_f=config_model['score_func'],
                                           ins_attn=config_model['ins_attn'],
                                           act_f=config_model['activate_func'],
                                           dp=config_model['dropout'],
                                           bn=config_model['bn'],
                                           gate=config_model['gate'])

model_name_str_2 = 'n_epochs[{n_e}]-n_batches[{n_b}]-batch[{batch}]-[{opt}]-lr[{lr}]-decay[{dec}]-clip[{clip}]-ps[{ps}]'
model_name_str_2 = model_name_str_2.format(n_e=config_model['n_epochs'],
                                           n_b=config_model['n_batches'],
                                           batch=config_model['batch_size'],
                                           opt=config_model['opt'],
                                           lr=config_model['lr'],
                                           dec=config_model['weight_decay'],
                                           clip=config_model['grad_clip'],
                                           ps=config_model['ps'])

model_name = '-'.join((model_name_str_1, model_name_str_2))

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_fp = join(path_parser.log, '{0}.log'.format(model_name))
file_handler = logging.FileHandler(log_fp)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

use_checkpoints = config_meta['use_checkpoints']
logger.info('Placement: {0}'.format(placement))
logger.info('Language: {0}'.format(lang))
logger.info('Model: {0}'.format(config_model['variation']))
logger.info('Use checkpoints: {0}'.format(use_checkpoints))

reset_size_for_test = True
if reset_size_for_test:
    if mode in ('test_sent_mturk', 'test_word_mturk'):
        config_model['max_n_sents'] = 15
        config_model['max_n_words'] = 72  # 40, 72
        logger.info('MAX_N_SENTS: {0} and MAX_N_WORDS: {1} have been reset for {2}'.format(config_model['max_n_sents'],
                                                                                           config_model['max_n_words'],
                                                                                           mode))
    elif mode == 'test_sent_syn':
        config_model['max_n_sents'] = 100
        config_model['max_n_words'] = 80  # 40, 70
        # config_model['batch_size'] = 16
        logger.info('MAX_N_SENTS: {0} and MAX_N_WORDS: {1} have been reset for {2}'.format(config_model['max_n_sents'],
                                                                                           config_model['max_n_words'],
                                                                                           mode))

set_sep_des_size = False
if set_sep_des_size:
    config_model['max_n_sents_des'] = 100
    config_model['max_n_words_des'] = 12
    logger.info(
        'Separate MAX_N_SENTS: {0} and MAX_N_WORDS: {1} have been set to DES'.format(config_model['max_n_sents'],
                                                                                     config_model['max_n_words']))
else:
    config_model['max_n_sents_des'] = config_model['max_n_sents']
    config_model['max_n_words_des'] = config_model['max_n_words']
    logger.info(
        'Identical MAX_N_SENTS: {0} and MAX_N_WORDS: {1} have been set to DES'.format(config_model['max_n_sents'],
                                                                                      config_model['max_n_words']))

if debug:
    config_model['batch_size'] = 100
    config_model['n_layers'] = 2
    config_model['n_heads'] = 4
    config_model['d_embed'] = 32
    config_model['d_model'] = 32
    config_model['d_ff'] = 128
    config_model['max_n_sents'] = 10
    config_model['max_n_words'] = 5
