from os import listdir
from os.path import join, dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from frame.model import *
import data.data_pipe as pipe
import utils.metrics as metrics
import utils.config_loader as config_loader
from utils.config_loader import path_parser
from utils.tools import build_res_str
from frame.checkpoint_op import load_checkpoint


def eval_model(model, phase, save_pred=False, save_gold=False):
    assert phase in ('dev', 'test')
    data_loader = pipe.DomDetDataLoader(dataset_type=phase)
    model.eval()

    n_iter, total_loss = 0, 0.0
    n_samples, total_hamming = 0, 0.0
    cf_mats, precision_list, recall_list = list(), list(), list()

    for batch_idx, batch in enumerate(data_loader):
        n_iter += 1

        c = copy.deepcopy
        feed_dict = c(batch)

        for (k, v) in feed_dict.items():
            feed_dict[k] = Variable(v, requires_grad=False, volatile=True)  # fix ids and masks

        loss, doc_scores = model(**feed_dict)[:2]
        total_loss += loss.data[0]

        y_true = batch['labels'].cpu().numpy()  # turn vars to numpy arrays
        hyp_scores = doc_scores.data.cpu().numpy()
        eval_args = {'y_true': y_true,
                     'hyp_scores': hyp_scores,
                     }

        if save_pred:
            eval_args['save_pred_to'] = join(path_parser.pred_doc, config_loader.meta_model_name)

        if save_gold:
            eval_args['save_true_to'] = join(path_parser.pred_doc, 'gold')

        # del model_res
        n_samples += y_true.shape[0]
        # logger.info('batch_size: {0}'.format(y_true.shape[0]))
        eval_res = metrics.metric_eval(**eval_args)

        cf_mats.append(eval_res['cf_mat_list'])
        precision_list.extend(eval_res['precision_list'])
        recall_list.extend(eval_res['recall_list'])
        total_hamming += eval_res['hamming']

    avg_loss = total_loss / n_iter
    cls_f1, avg_f1 = metrics.compute_f1_with_confusion_mats(cf_mats)
    example_based_f1 = metrics.compute_example_based_f1(precision_list=precision_list, recall_list=recall_list)
    hamming = total_hamming / n_samples

    eval_log_info = {
        'ph': phase,
        'loss': avg_loss,
        'example_based_f1': example_based_f1,
        'avg_f1': avg_f1,
        'cls_f1': cls_f1,
        'hamming': hamming,
    }

    return eval_log_info


def test_model_doc_with_checkpoints(model, save_pred=False, save_gold=False, n_iter=None, restore=False):
    if config_loader.placement == 'auto':
        model = nn.DataParallel(model, device_ids=config_loader.device)

    if config_loader.placement in ('auto', 'single'):
        model.cuda()

    logger.info('START: model testing on [DOCS]')
    checkpoint = join(path_parser.model_save, config_loader.model_name)
    if restore:
        checkpoint = join(checkpoint, 'resume')

    filter_keys = None
    if config_loader.reset_size_for_test and not config_loader.set_sep_des_size:
        logger.info('Filter DES pretrained paras...')
        filter_keys = ['module.word_det.des_ids', 'module.word_det.des_sent_mask', 'module.word_det.des_word_mask']
    load_checkpoint(checkpoint=checkpoint, model=model, n_iter=n_iter, filter_keys=filter_keys)

    res_str = build_res_str(stage=None, use_loss=False, use_exam_f1=False, use_hamming=False)
    eval_log_info = eval_model(model, phase='test', save_pred=save_pred, save_gold=save_gold)
    test_eval_log_info = {
        'ph': 'Test',
        'avg_f1': eval_log_info['avg_f1'],
        'cls_f1': eval_log_info['cls_f1'],
    }
    logger.info(res_str.format(**test_eval_log_info))


def test_model_doc(model, save_pred=False, save_gold=False, restore=False):
    checkpoint = join(path_parser.model_save, config_loader.model_name)
    if restore:
        checkpoint = join(checkpoint, 'resume')
    fns = [fn.split('.')[0] for fn in listdir(checkpoint) if fn.endswith('.pth.tar')]
    for n_iter in fns:
        logger.info('===============================')
        test_model_doc_with_checkpoints(model, save_pred, save_gold, n_iter, restore)