# -*- coding: utf-8 -*-
import io
import os
from tqdm import *
import socket
import time
import urllib.parse
import urllib.request as urlreq
from http.cookiejar import CookieJar
from multiprocessing.dummy import Pool as ThreadPool
from urllib.error import HTTPError, URLError
import lxml.etree as etree
from utils.config_loader import logger, path_parser, doms, target_dom


class Crawler:

    def __init__(self):
        self.wiki_url_pattern = path_parser.wiki_url_pattern
        logger.info('wiki_url_pattern: {0}'.format(self.wiki_url_pattern))

        self.lv_dp = path_parser.lv_dp
        self.parent_lv_dp = path_parser.parent_lv_dp

        # self.dp_dom = [os.path.join(self.dp_data, dom) for dom in doms]
        self.dp_proj_root = path_parser.proj_root

        # crawler settings
        self.n_threads = 5
        user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:35.0) Gecko/20100101 Firefox/35.0'
        connection = 'keep-alive'
        self.headers = {
            'User-Agent': user_agent,
            'Connection': connection,
        }

        cookie_support= urlreq.HTTPCookieProcessor(CookieJar())
        opener = urlreq.build_opener(cookie_support, urlreq.HTTPHandler)
        urlreq.install_opener(opener)

        timeout = 60
        socket.setdefaulttimeout(timeout)
        self.sleep_time = 1
        self.exc_sleep_time = 5  # 30

        # xpath settings
        self.longer_cascade = '//div[@id="mw-{0}"]/div/div[@class="mw-category"]/div[@class="mw-category-group"]/ul/li/'
        self.shorter_cascade = '//div[@id="mw-{0}"]/div/ul/li/'
        self.xpath_subcategories = 'div[@class="CategoryTreeSection"]/div[@class="CategoryTreeItem"]/a/text()'
        self.xpath_pages = 'a/text()'

        # string constants
        self.SUBCATEGORIES = 'subcategories'
        self.PAGES = 'pages'
        self.LONG = 'long'
        self.SHORT = 'short'

    def crawl_with_single_thread(self, topics):
        eligible_topics = list()
        sites = list()
        for topic in topics:
            req = urlreq.Request(url=self.wiki_url_pattern.format(topic), headers=self.headers)
            time.sleep(self.sleep_time)
            try:
                opener = urlreq.urlopen(req)
                wiki_page = opener.read()
                if wiki_page:
                    sites.append(wiki_page)
                    eligible_topics.append(topic)
                urlreq.urlopen(req).close()
            except HTTPError as e:
                print('HTTP error: {0}, code: {1}'.format(e.msg, e.code))
                continue
            except socket.timeout:
                print('Timeout error')
                time.sleep(self.exc_sleep_time)
            except socket.error:
                print('Other socket error')
                time.sleep(self.exc_sleep_time)
            except URLError:
                print('URL socket error')
                time.sleep(self.exc_sleep_time)
        return eligible_topics, sites

    def crawl_with_multi_threads(self, topics):

        def _get_wiki_page(topic):
            parsed_topic = urllib.parse.quote(topic)  # encode non-ascii chars properly
            req = urlreq.Request(url=self.wiki_url_pattern.format(parsed_topic), headers=self.headers)
            try:
                opener = urlreq.urlopen(req)
                wiki_site = opener.read()
                opener.close()
                return (topic, wiki_site) if wiki_site else None  # use original topic string instead of the parsed one
            except HTTPError as e:
                logger.exception('HTTP error: {0}, code: {1}, topic: {2}'.format(e.msg, e.code, topic))
            except socket.timeout:
                logger.exception('Timeout error, topic: {0}'.format(topic))
                logger.info('Sleep for {0}s'.format(self.exc_sleep_time))
                time.sleep(self.exc_sleep_time)
            except socket.error:
                logger.exception('Other socket error, topic: {0}'.format(topic))
                logger.info('Sleep for {0}s'.format(self.exc_sleep_time))
                time.sleep(self.exc_sleep_time)
            except URLError:
                logger.exception('URL socket error, topic: {0}'.format(topic))
                logger.info('Sleep for {0}s'.format(self.exc_sleep_time))
                time.sleep(self.exc_sleep_time)

        # none_ids = [i for i, res in enumerate(results) if not res]
        # topics = [topic for i, topic in enumerate(topics) if i not in none_ids]

        pool = ThreadPool(self.n_threads)
        results = pool.map(_get_wiki_page, topics)
        results = [res for res in results if res]  # remove None site
        if results:
            topics, sites = zip(*results)  # unzip
        else:
            topics, sites = list(), list()

        pool.close()
        pool.join()
        return topics, sites

    def crawl(self, topics, use_multi_threads=True):
        return self.crawl_with_multi_threads(topics) if use_multi_threads else self.crawl_with_single_thread(topics)

    def _get_xpath(self, path_type, cascade_type):
        """
            get xpath for subtopic or doc_head.
            cascade is the xpath shared by the two, and could be longer (default) or shorter.
        """
        assert cascade_type in (self.LONG, self.SHORT)
        assert path_type in (self.SUBCATEGORIES, self.PAGES)

        cascade_pattern = self.longer_cascade if cascade_type == self.LONG else self.shorter_cascade
        cascade = cascade_pattern.format(path_type)
        xpath = cascade + self.xpath_subcategories if path_type == self.SUBCATEGORIES else cascade + self.xpath_pages

        return xpath

    def parse(self, sites):
        all_subcategories = list()  # list of lists
        all_pages = list() # list of lists

        xpath_subcategories = self._get_xpath(path_type=self.SUBCATEGORIES, cascade_type=self.LONG)
        xpath_pages = self._get_xpath(path_type=self.PAGES, cascade_type=self.LONG)

        for site in sites:
            tree = etree.HTML(site.decode('utf-8'))
            subcategories = tree.xpath(xpath_subcategories)
            pages = tree.xpath(xpath_pages)
            if not subcategories:
                shorter_xpath_subcategories = self._get_xpath(path_type=self.SUBCATEGORIES, cascade_type=self.SHORT)
                subcategories = tree.xpath(shorter_xpath_subcategories)
            if not pages:
                shorter_xpath_pages = self._get_xpath(path_type=self.PAGES, cascade_type=self.SHORT)
                pages = tree.xpath(shorter_xpath_pages)

            subcategories = ['_'.join(subcategory.split()) for subcategory in subcategories]  # combine multiple words
            pages = ['_'.join(page.split()) for page in pages]
            all_subcategories.append(subcategories)
            all_pages.append(pages)

        return all_subcategories, all_pages

    def _get_dp(self, lv):
        assert lv in self.lv_dp
        return self.lv_dp[lv]

    def _get_parent_dp(self, lv):
        assert lv in self.parent_lv_dp
        return self.parent_lv_dp[lv]

    def _save(self, all_contents, titles, content_type, lv, is_test):
        """
           save a list of content lists as per their titles.
           e.g.
            all_contents: [['Government by city', 'Government by continent'], ['Military costs']]
            titles: ['Government', 'Military']
           type: 'subcategory' or 'doc_heads'
        """

        assert len(all_contents) == len(titles)
        assert content_type in (self.SUBCATEGORIES, self.PAGES)
        dp = self._get_dp(lv)

        for (contents, title) in zip(all_contents, titles):
            fn = '-'.join((title, content_type))
            if is_test:
                fp = os.path.join(dp, 'test', fn)
            else:
                fp = os.path.join(dp, fn)

            if not os.path.exists(fp):
                with io.open(fp, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(contents))
            else:
                logger.info('file was pre-crawled...')

    def _get_lv(self, parent_topics):
        lv_depth = len(parent_topics)
        assert lv_depth in (1, 2, 3, 4)
        if lv_depth == 1:
            return 'topic'
        elif lv_depth == 2:
            return 'subtopic'
        elif lv_depth == 3:
            return 'subsubtopic'
        else:
            return 'subsubsubtopic'

    def _get_child_topics(self, parent_topics):
        """
            parent_topics: a list of parents. [dom] (topic-lv) or [dom, topic] (subtopic-lv)
        """
        if type(parent_topics) is not list:
            parent_topics = [parent_topics]

        lv = self._get_lv(parent_topics)
        parent_dp = self._get_parent_dp(lv=lv)

        fn = '-'.join(parent_topics + [self.SUBCATEGORIES])
        fn = fn.replace('/', '_')  # todo: recheck this

        with io.open(os.path.join(parent_dp, fn), 'r', encoding='utf-8') as f:
            child_topics = [subcategory.rstrip('\n') for subcategory in f.readlines()]

        return child_topics

    def _dump_links(self, topics, lv, title_prefix, is_test):
        """
            title_prefix: a list. [dom] or [dom, topic]
        """
        # logger.info('Start to crawl...\n')
        eligible_topics, sites = self.crawl(topics=topics)

        # logger.info('Start to parse...\n')
        all_subcategories, all_pages = self.parse(sites=sites)

        titles = self._get_titles(topics=eligible_topics, lv=lv, prefix=title_prefix)

        # logger.info('Start to save...\n')
        self._save(all_contents=all_subcategories, titles=titles, content_type=self.SUBCATEGORIES, lv=lv, is_test=is_test)
        self._save(all_contents=all_pages, titles=titles, content_type=self.PAGES, lv=lv, is_test=is_test)

    @staticmethod
    def _get_titles(topics, lv, prefix=None):
        """
            prefix: a list. [dom] or [dom, topic]
        """
        if lv == 'dom':
            titles = topics
        else:
            assert prefix
            if type(prefix) is not list:
                prefix = [prefix]
            titles = ['-'.join(prefix + [topic]) for topic in topics]

        titles = [title.replace('/', '_') for title in titles]
        return titles

    def _dump_links_from_parent(self, parent_topics, is_test=False):
        lv = self._get_lv(parent_topics)
        child_topics = self._get_child_topics(parent_topics)
        self._dump_links(topics=child_topics, lv=lv, title_prefix=parent_topics, is_test=is_test)

    def dump_dom_links(self, is_test=False):
        self._dump_links(topics=doms, lv='dom', title_prefix=None, is_test=is_test)

    def dump_topic_links(self, is_test=False):
        for dom in tqdm(doms):
            self._dump_links_from_parent(parent_topics=[dom], is_test=is_test)

    def dump_subtopic_links(self, is_test=False):
        for dom in tqdm(doms):
            topics = self._get_child_topics(parent_topics=[dom])
            for topic in topics:
                self._dump_links_from_parent(parent_topics=[dom, topic], is_test=is_test)

    def dump_subsubtopic_links(self, is_test=False):
        for dom in tqdm(doms):
            topics = self._get_child_topics(parent_topics=[dom])
            for topic in topics:
                subtopics = self._get_child_topics(parent_topics=[dom, topic])
                for subtopic in subtopics:
                    self._dump_links_from_parent(parent_topics=[dom, topic, subtopic], is_test=is_test)

    def dump_topic_links_for_a_domain(self, dom, is_test=False):
        self._dump_links_from_parent(parent_topics=[dom], is_test=is_test)

    def dump_subtopic_links_for_a_domain(self, dom, is_test=False):
        topics = self._get_child_topics(parent_topics=[dom])
        for topic in topics:
            self._dump_links_from_parent(parent_topics=[dom, topic], is_test=is_test)

    def dump_subsubtopic_links_for_a_domain(self, dom, is_test=False):
        topics = self._get_child_topics(parent_topics=[dom])
        for topic in topics:
            subtopics = self._get_child_topics(parent_topics=[dom, topic])
            for subtopic in subtopics:
                self._dump_links_from_parent(parent_topics=[dom, topic, subtopic], is_test=is_test)

    def dump_subsubsubtopic_links_for_a_domain(self, dom, is_test=False):
        topics = self._get_child_topics(parent_topics=[dom])
        for topic in topics:
            subtopics = self._get_child_topics(parent_topics=[dom, topic])
            for subtopic in subtopics:
                subsubtopics = self._get_child_topics(parent_topics=[dom, topic, subtopic])
                for subsubtopic in subsubtopics:
                    self._dump_links_from_parent(parent_topics=[dom, topic, subtopic, subsubtopic], is_test=is_test)


if __name__ == '__main__':
    crawler = Crawler()
    # test
    # doms = ['Government']
    # pages = crawler.crawl(topics)
    # print(pages)
    # all_subtopics, all_docs = crawler.parse(pages)
    # print(all_subtopics, all_docs)
    # crawler.manage(topics=doms)
    # domain => topics => subtopics
    # crawler.doms = ['Government']
    # crawler.dump_dom_links()
    # crawler.dump_topic_links()
    # crawler.dump_subtopic_links()
    # crawler.dump_subsubtopic_links()

    # for lifestyle, General, Sports, 体育 only
    # crawler.dump_topic_links_for_a_domain(dom='体育')
    # crawler.dump_subtopic_links_for_a_domain(dom='体育')

    # for Business, Military, Sports, 体育 only
    crawler.dump_subsubtopic_links_for_a_domain(dom='体育')

    # for 健康 & 法律 & 军事 & 商业 only
    # crawler.dump_subsubsubtopic_links_for_a_domain(dom=target_dom)
