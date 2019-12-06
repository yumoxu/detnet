# -*- coding: utf-8 -*-
from os.path import dirname, abspath
import io
import re
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from utils.config_loader import deprecated, logger, path_parser, doms_final, config_model, lang
from utils.langconv import *
from pyltp import Segmentor
from copy import deepcopy
from os import listdir
from tqdm import tqdm
from os.path import isfile, join


class ZHProcessor:
    """
        This class is for processing xml for non-English languages (currently Chinese).

            dataset_unproc => dataset_whole.
    """

    def __init__(self):
        self.dp_dataset_unproc = path_parser.dataset_unproc
        self.dp_dataset_whole = path_parser.dataset_whole

        self.fp_top_unproc = path_parser.dataset_top_unproc
        self.fp_des_unproc = path_parser.dataset_des_unproc
        self.fp_top = path_parser.dataset_top
        self.fp_des = path_parser.dataset_des

        self.segmentor = Segmentor()
        self.segmentor.load(path_parser.cws)
        self.SIDE_PATTERN = '(?<=#s-{0}\n)[\s\S]*?(?=\n#e-{0})'
        # logger.info('CWS model fp: {0}'.format(path_parser.cws))

    @deprecated
    def stanford_stuff(self):
        # from stanfordcorenlp import StanfordCoreNLP
        # from nltk.parse.corenlp import CoreNLPTokenizer
        # import corenlp
        # self.nlp = StanfordCoreNLP('http://localhost',
        #                            port=9000,
        #                            timeout=30000)
        # self.nlp = StanfordCoreNLP(os.environ["CORENLP_HOME"], lang='zh', memory='4g')

        # self.props = {
        #     'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
        #     'pipelineLanguage': 'zh',
        #     'outputFormat': 'json'
        # }
        # seg = StanfordSegmenter()
        # seg.default_config('zh')
        # sent = u'这是斯坦福中文分词器测试'
        # print(seg.segment(sent))
        # with self.nlp as nlp:
            #     for sent in sents:
            #         print(sent)
            #         print(nlp.word_tokenize(sent))

            # with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
            #     ann = client.annotate(text)
            #
            # sentence = ann.sentence[0]
            # assert corenlp.to_text(sentence) == text
            # print(sentence.text)
            # token = sentence.token[0]
            # print(token.lemma)
        pass

    @deprecated
    def sent2words(self, segmentor, sent):
        words = segmentor.segment(sent)
        logger.info('|'.join(words))

    def para2sents(self, paragraph):
        for sent in re.findall('[^!?。\.\!\?]+[!?。\.\!\?]?', paragraph, flags=re.U):
            yield sent

    def proc_content(self, content, is_headline, use_sent_seg=True, convert2simple=False):
        if use_sent_seg:
            content = self.para2sents(content)

        proc_lines = list()
        for sent in content:
            proc_lines.append('#s-sent')
            if convert2simple:
                sent = Converter('zh-hans').convert(sent)
            words = self.segmentor.segment(sent)
            # logger.info('words: {0}'.format(words))
            # logger.info('|'.join(words))
            proc_lines.append('\n'.join(words))
            proc_lines.append('#e-sent')

        if is_headline:
            proc_lines.insert(0, '#s-headline')
            proc_lines.append('#e-headline')
        else:
            proc_lines.insert(0, '#s-para')
            proc_lines.append('#e-para')

        return proc_lines

    def release_seg(self):
        self.segmentor.release()

    def get_xml_elements(self, xml_fp):

        def _get_para_info():
            para_matches = list(re.finditer(re.compile(para_pattern), text))

            if not para_matches:
                logger.error('No para in {0}'.format(xml_fp))
                raise AssertionError

            # logger.info('para_matches {0}'.format(para_matches))

            paras = list()
            para_spans = list()
            for para_m in para_matches:
                # if para_m.group() != '\n':
                paras.append(para_m.group())
                para_spans.append(para_m.span())

            # logger.info('paras: {0}'.format(paras))
            # logger.info('para_spans: {0}'.format(para_spans))

            para_info = list(zip(paras, para_spans))
            # logger.info('para_info {0}'.format(para_info))

            return para_info

        def _get_ind_headline_info():
            ind_headline_info = list()

            headline_matches = list(re.finditer(re.compile(headline_pattern), text))
            if headline_matches:
                headlines = list()
                headline_spans = list()
                for headline_m in headline_matches:
                    # if headline_m.group() != '\n':
                    headlines.append(headline_m.group())
                    headline_spans.append(headline_m.span())

                headline_info = list(zip(headlines, headline_spans))
                # logger.info('headline_info {0}'.format(headline_info))

                for h_info in headline_info:
                    h_start, h_end = h_info[1]
                    in_para = False
                    for p_info in para_info:
                        p_start, p_end = p_info[1]
                        if p_start <= h_start and h_end <= p_end:
                            in_para = True
                            # logger.info('headline in para ...')
                    if not in_para:
                        ind_headline_info.append(h_info)

                # logger.info('ind_headline_info {0}'.format(ind_headline_info))

            return ind_headline_info

        def _sort_paras_and_headlines():
            sorted_items = deepcopy(list(para_info))
            if ind_headline_info:
                for ind_h_info in ind_headline_info:
                    ind_h_start = ind_h_info[1][0]
                    p_span_starts = [p_info[1][0] for p_info in para_info]
                    # logger.info('p_span_starts: {0}'.format(p_span_starts))
                    insert_idx = None
                    for idx, p_span_start in enumerate(p_span_starts):
                        if ind_h_start < p_span_start:
                            insert_idx = idx
                            break
                    item_dict = {
                        'content': ind_h_info[0],
                        'is_headline': True
                    }

                    sorted_items.insert(insert_idx, item_dict)

            # deal with all paras left
            for idx, item in enumerate(sorted_items):
                if type(item) != dict:
                    item_dict = {
                        'content': item[0],
                        'is_headline': False
                    }
                    sorted_items[idx] = item_dict
            return sorted_items

        def _handle_nested_paras():
            for idx, item in enumerate(sorted_items):
                if item['is_headline']:
                    continue

                headline_matches = list(re.finditer(re.compile(headline_pattern), item['content']))
                # headline_match = re.search(re.compile(headline_pattern), item['content'])

                if not headline_matches:
                    continue

                new_items = list()
                for headline_m in headline_matches:
                    inner_headline_item = {
                        'content': headline_m.group(),
                        'is_headline': True,
                    }

                    new_items.insert(0, inner_headline_item)

                rest_pattern = '(?<=\</h>\n)[\s\S]*'
                rest_match = re.search(re.compile(rest_pattern), item['content'])

                if rest_match:
                    # logger.error('No rest in para: {0} of {1}'.format(item['content'], xml_fp))
                    # raise AssertionError
                    rest_para_item = {
                        'content': rest_match.group(),
                        'is_headline': False,
                    }

                    new_items.insert(0, rest_para_item)

                del sorted_items[idx]

                for new_item in new_items:
                    sorted_items.insert(idx, new_item)

        root_pattern = '(?<=\<{0}>\n)[\s\S]*?(?=\n</{0}>)'
        para_pattern = root_pattern.format('p')
        headline_pattern = root_pattern.format('h')

        with io.open(xml_fp, encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # logger.info('text: {0}'.format(text))

        para_info = _get_para_info()
        ind_headline_info = _get_ind_headline_info()
        sorted_items = _sort_paras_and_headlines()
        _handle_nested_paras()

        return sorted_items

    def dump_files(self, fp, text):
        with io.open(fp, mode='a', encoding='utf-8') as f:
             f.write(text)

    def proc_xml(self, xml_fp, out_fp):
        sorted_elements = self.get_xml_elements(xml_fp)
        proc_lines = list()
        # logger.info('sorted_elements: {0}'.format(sorted_elements))
        for element in sorted_elements:
            element_lines = self.proc_content(**element)
            # logger.info('element_lines: {0}'.format(element_lines))
            proc_lines.extend(element_lines)

        proc_lines.insert(0, '#s-doc')
        proc_lines.append('#e-doc')

        out_text = '\n'.join(proc_lines)

        self.dump_files(fp=out_fp, text=out_text)

    def proc_all_docs(self):
        xml_root = self.dp_dataset_unproc
        fns = [fn for fn in listdir(xml_root) if isfile(join(xml_root, fn))]

        for fn in tqdm(fns):
            xml_fp = join(xml_root, fn)
            out_fp = join(self.dp_dataset_whole, fn)
            self.proc_xml(xml_fp=xml_fp, out_fp=out_fp)

    def proc_side_top(self):
        proc_lines = list()
        with io.open(self.fp_top_unproc, encoding='utf-8') as f:
            for dom in doms_final:
                pattern = re.compile(self.SIDE_PATTERN.format(dom))
                topics = re.findall(pattern, f.read())[0].split('\n')
                logger.info('topics: {0}'.format(topics))
                top_proc_lines = self.proc_content(topics, is_headline=False, use_sent_seg=False)

                top_proc_lines.insert(0, '#s-{0}'.format(dom))
                top_proc_lines.append('#e-{0}'.format(dom))

                proc_lines.extend(top_proc_lines)
                f.seek(0, 0)

        with io.open(self.fp_top, mode='a', encoding='utf-8') as f:
            f.write('\n'.join(proc_lines))

    def proc_side_des(self):
        proc_lines = list()
        with io.open(self.fp_des_unproc, encoding='utf-8') as f:
            for dom in doms_final:
                pattern = re.compile(self.SIDE_PATTERN.format(dom))
                des_sents = re.findall(pattern, f.read())[0].split('\n')
                des_proc_lines = self.proc_content(des_sents, is_headline=False, use_sent_seg=False)

                des_proc_lines.insert(0, '#s-{0}'.format(dom))
                des_proc_lines.append('#e-{0}'.format(dom))

                proc_lines.extend(des_proc_lines)
                f.seek(0, 0)

        with io.open(self.fp_des, mode='a', encoding='utf-8') as f:
            f.write('\n'.join(proc_lines))


if __name__ == '__main__':
    proc = ZHProcessor()
    # content = '这是中文分词器测试,这是第一句话!这是第二句话?这是第三句话。'
    # content = proc.proc_content(content=content, is_headline=True)
    # print(content)
    # xml_fp = '/afs/inf.ed.ac.uk/group/project/material/DomainDetection/dataset/zh/whole/黨鞭_政府'
    # proc.proc_xml(xml_fp=xml_fp, out_fp='./test.txt')
    # proc.proc_all_docs()
    proc.proc_side_top()
    proc.proc_side_des()
    proc.release_seg()
