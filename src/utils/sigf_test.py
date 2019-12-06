import io
from os.path import join, dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from utils.config_loader import path_parser, n_ways, logger
import numpy as np
import utils.metrics as metrics
from multiprocessing.dummy import Pool as ThreadPool


def archived_build_sigf_f(cf_mats, save_fp):
    """
        each line is a confusion matrix for each class in each batch.

    :param cf_mats: cf_mat for each class
    """
    items = list()
    # items = [0, 0, 0]
    for cls_idx in range(n_ways):
        cls_cf_mats = [batch_mats[cls_idx] for batch_mats in cf_mats]
        for cls_cf_mat in cls_cf_mats:
            tn, fp, fn, tp = cls_cf_mat
            # items[0] += tp
            # items[1] += tp + fp
            # items[2] += tp + fn
            item = ' '.join((str(tp), str(tp + fp), str(tp + fn)))
            items.append(item)

    with io.open(save_fp, mode='a', encoding='utf-8') as f:
        f.write('\n'.join(items))
    # with io.open(save_fp, mode='a', encoding='utf-8') as f:
    #     f.write(' '.join([str(item) for item in items]))


def build_sigf_f(cls_f1, save_fp):
    """
        each line is a confusion matrix for each class in each batch.

    :param cf_mats: cf_mat for each class
    """
    cls_f1 = [str(int(10000 * f1)) for f1 in cls_f1]
    with io.open(save_fp, mode='a', encoding='utf-8') as f:
        f.write('\n'.join(cls_f1))
    # with io.open(save_fp, mode='a', encoding='utf-8') as f:
    #     f.write(' '.join([str(item) for item in items]))


def get_obs(fp):
    with io.open(fp, encoding='utf-8') as f:
        obs = [[int(float(item)) for item in line.rstrip('\n').split(' ')] for line in f.readlines()]
    obs = np.array(obs)
    return obs


class SigfTester():

    def __init__(self, model_out_fp_1, model_out_fp_2, gold_fp, iter=10000):
        self.obs_1 = get_obs(fp=model_out_fp_1)
        self.obs_2 = get_obs(fp=model_out_fp_2)
        self.gold = get_obs(fp=gold_fp)
        assert self.obs_1.shape == self.obs_2.shape == self.gold.shape
        self.n_obs = self.obs_1.shape[0]
        logger.info('n_obs: {0}'.format(self.n_obs))

        self.iter = iter

    def _calculate_f1(self, obs):
        eval_res = metrics.metric_eval_with_y_pred(y_true=self.gold, y_pred=obs, cf_mat_only=True)
        cf_mats = eval_res['cf_mat_list']
        _, macro_f1 = metrics.compute_f1_with_confusion_mats([cf_mats])

        return macro_f1

    def _diff(self, obs_1, obs_2, log=False):
        macro_f1_1 = self._calculate_f1(obs=obs_1)
        macro_f1_2 = self._calculate_f1(obs=obs_2)
        diff = abs(macro_f1_1 - macro_f1_2)

        if log:
            logger.info('macro_f1_1: {0}, macro_f1_2: {1}'.format(macro_f1_1, macro_f1_2))
            logger.info('diff: {0}'.format(diff))

        return diff

    def _obs_diff(self):
        return self._diff(self.obs_1, self.obs_2, log=True)

    def approx_rand(self):
        obs_diff = self._obs_diff()

        def _iter(iter_idx):
            logger.info('iter_idx: {0}'.format(iter_idx))
            reserve = np.random.randint(2, size=[self.n_obs, 1])
            reserve_mat = np.tile(reserve, [1, n_ways])
            flip_mat = 1 - reserve_mat

            x_stat = np.multiply(self.obs_1, reserve_mat) + np.multiply(self.obs_2, flip_mat)
            y_stat = np.multiply(self.obs_2, reserve_mat) + np.multiply(self.obs_1, flip_mat)

            diff = self._diff(x_stat, y_stat)
            # logger.info('diff: {0}, x_stat: {1}, y_stat: {2}'.format(diff, x_stat, y_stat))

            if diff >= obs_diff:
                return 1
            else:
                return 0

        pool = ThreadPool(6)
        better = float(sum(pool.map(_iter, range(self.iter))))

        self.p = (better + 1) / (self.iter + 1)

        logger.info('p-value: {0}'.format(self.p))

    def avg_approx_rand(self):
        pass


if __name__ == '__main__':
    root = path_parser.pred_mturk_nyt
    model_name_1 = 'milnet'
    model_name_2 = 'detnet1'
    iter = 1000

    sigf_paras = {
        'model_out_fp_1': join(root, model_name_1),
        'model_out_fp_2': join(root, model_name_2),
        'gold_fp': join(root, 'gold'),
        'iter': iter,
    }

    sigf = SigfTester(**sigf_paras)
    sigf.approx_rand()
