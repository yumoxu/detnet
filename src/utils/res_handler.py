import io
from os import listdir
from multiprocessing.dummy import Pool as ThreadPool
from os.path import isfile, join, dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from utils.config_loader import logger, path_parser, doms
import utils.tools as tools


def mix_two_dicts(wc_dict_1, wc_dict_2):
    """
        Append the 2nd dict to the 1st dict then return the 1st one.

    :param wc_dict_1:
    :param wc_dict_2:
    :return:
    """
    for (w, c) in wc_dict_2.items():
        if w in wc_dict_1:
            wc_dict_1[w] += c
        else:
            wc_dict_1[w] = c
    return wc_dict_1


def slim_vocab(wc_dict, v_threshold):
    """
        Slim the vocab as per the given threshold.

    :param wc_dict: dict, word-count;
    :param v_threshold: int, >= 2.
    """
    return {w: c for (w, c) in wc_dict.items() if c >= v_threshold}


def dump_vocab(vocab, out_fp):
    with io.open(out_fp, 'a+', encoding='utf-8') as f:
        out_str = '\n'.join(vocab)
        # json_out_str = json.dumps(vocab, ensure_ascii=False)  # dump does not work with io.open
        # output_f.write('{}\n'.format(json_out_str).decode('utf-8'))  # strictly write unicode into file
        f.write(out_str)


def mix_multi_dicts(dict_list):
    mixed_dict = {}
    for dict in dict_list:
        mixed_dict = mix_two_dicts(mixed_dict, dict)
    return mixed_dict


def load_vocab(vocab_path):
    with io.open(vocab_path, 'r', encoding='utf-8') as vocab_f:
        return [line.rstrip('\n') for line in vocab_f.readlines() if line]
        # return json.load(vocab_f)


class ResBuilder:

    def __init__(self):
        self.doms = doms
        self.dp_dataset_train = path_parser.dataset_train
        self.dp_dataset_dev = path_parser.dataset_dev
        self.dp_dataset_test = path_parser.dataset_test

        self.train_fns = [fn for fn in listdir(self.dp_dataset_train) if isfile(join(self.dp_dataset_train, fn))]

        # for creation
        self.fp_vocab_full = path_parser.vocab_full
        self.fp_vocab_threshold_3 = path_parser.vocab_threshold_3
        self.fp_vocab_threshold_5 = path_parser.vocab_threshold_5

        self.N_THREADS = 6

    def _proc_doc(self, fn):
        fp = join(self.dp_dataset_train, fn)
        words = tools.get_all_words(fp)
        assert words
        words.remove('')
        wc_dict = {w: words.count(w) for w in set(words)}
        return wc_dict

    def _build_full_vocab(self):
        pool = ThreadPool(self.N_THREADS)
        results = pool.map(self._proc_doc, self.train_fns)  # a list of dicts

        pool.close()
        pool.join()

        return mix_multi_dicts(results)

    def build_vocab_with_thresholds(self):
        """
            Build a full vocab, slim it and save it.
        """
        logger.info('building full_vocab...')
        full_vocab = self._build_full_vocab()
        dump_vocab(full_vocab.keys(), self.fp_vocab_full)

        logger.info('slimming to slimmed_vocab_3...')

        slimmed_vocab_3 = slim_vocab(full_vocab, v_threshold=3)
        dump_vocab(slimmed_vocab_3.keys(), out_fp=self.fp_vocab_threshold_3)

        logger.info('slimming to slimmed_vocab_5...')
        slimmed_vocab_5 = slim_vocab(full_vocab, v_threshold=5)
        dump_vocab(slimmed_vocab_5.keys(), out_fp=self.fp_vocab_threshold_5)

        logger.info('#vocab: {0}, #slimmed-3: {1}, #slimmed-5: {2}'.format(len(full_vocab), len(slimmed_vocab_3), len(slimmed_vocab_5)))


class ResLoader:
    def __init__(self):
        self.vocab = load_vocab(path_parser.vocab)
        self.vocab.insert(0, 'UNK')  # for unknown tokens

    def get_vocab(self):
        return self.vocab

    def get_indexed_vocab(self):
        """
            index vocab starting from 1 (UNK). 0 is for padding.
        :return: vocab_id_dict
        """
        vocab_id_dict = dict()

        for id in range(len(self.vocab)):
            vocab_id_dict[self.vocab[id]] = id + 1

        return vocab_id_dict


if __name__ == '__main__':
    res_builder = ResBuilder()
    res_builder.build_vocab_with_thresholds()
