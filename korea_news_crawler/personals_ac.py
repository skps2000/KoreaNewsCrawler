#!/usr/bin/env python
# -*- coding: utf-8, euc-kr -*-

import os
import platform
import calendar
import sys
import requests
import re
from datetime import datetime as dt
import pytz

from time import sleep
from bs4 import BeautifulSoup
from multiprocessing import Process
from korea_news_crawler.exceptions import *
from korea_news_crawler.articleparser import ArticleParser
from korea_news_crawler.writer import Writer



class ArticleCrawler(object):
    def __init__(self):
        self.categories = {'정치': 100, '경제': 101, '사회': 102, '생활문화': 103, '세계': 104, 'IT과학': 105, '오피니언': 110,
                           'politics': 100, 'economy': 101, 'society': 102, 'living_culture': 103, 'world': 104, 'IT_science': 105, 'opinion': 110}
        self.selected_categories = []
        self.date = {'start_year': 0, 'start_month': 0, 'start_day' : 0, 'end_year': 0, 'end_month': 0, 'end_day':0}
        self.user_operating_system = str(platform.system())

    def set_category(self, *args):
        for key in args:
            if self.categories.get(key) is None:
                raise InvalidCategory(key)
        self.selected_categories = args

    def set_date_range(self, start_date:str, end_date:str):
        start = list(map(int, start_date.split("-")))
        end = list(map(int, end_date.split("-")))
        
        # Setting Start Date
        if len(start) == 1: # Input Only Year
            start_year = start[0]
            start_month = 1
            start_day = 1
        elif len(start) == 2: # Input Year and month
            start_year, start_month = start
            start_day = 1
        elif len(start) == 3: # Input Year, month and day
            start_year, start_month, start_day = start
        
        # Setting End Date
        if len(end) == 1: # Input Only Year
            end_year = end[0]
            end_month = 12
            end_day = 31
        elif len(end) == 2: # Input Year and month
            end_year, end_month = end
            end_day = calendar.monthrange(end_year, end_month)[1]
        elif len(end) == 3: # Input Year, month and day
            end_year, end_month, end_day = end

        args = [start_year, start_month, start_day, end_year, end_month, end_day]

        if start_year > end_year:
            raise InvalidYear(start_year, end_year)
        if start_month < 1 or start_month > 12:
            raise InvalidMonth(start_month)
        if end_month < 1 or end_month > 12:
            raise InvalidMonth(end_month)
        if start_day < 1 or calendar.monthrange(start_year, start_month)[1] < start_day:
            raise InvalidDay(start_day)
        if end_day < 1 or calendar.monthrange(end_year, end_month)[1] < end_day:
            raise InvalidDay(end_day)
        if start_year == end_year and start_month > end_month:
            raise OverbalanceMonth(start_month, end_month)
        if start_year == end_year and start_month == end_month and start_day > end_day:
            raise OverbalanceDay(start_day, end_day)

        for key, date in zip(self.date, args):
            self.date[key] = date
        # print(self.date)

    @staticmethod
    def make_news_page_url(category_url, date):
        
        made_urls = []

        for year in range(date['start_year'], date['end_year'] + 1):
            if date['start_year'] == date['end_year']:
                target_start_month = date['start_month']
                target_end_month = date['end_month']
            else:
                if year == date['start_year']:
                    target_start_month = date['start_month']
                    target_end_month = 12
                elif year == date['end_year']:
                    target_start_month = 1
                    target_end_month = date['end_month']
                else:
                    target_start_month = 1
                    target_end_month = 12

            for month in range(target_start_month, target_end_month + 1):
                if date['start_month'] == date['end_month']:
                    target_start_day = date['start_day']
                    target_end_day = date['end_day']
                else:
                    if year == date['start_year'] and month == date['start_month']:
                        target_start_day = date['start_day']
                        target_end_day = calendar.monthrange(year, month)[1]
                    elif year == date['end_year'] and month == date['end_month']:
                        target_start_day = 1
                        target_end_day = date['end_day']
                    else:
                        target_start_day = 1
                        target_end_day = calendar.monthrange(year, month)[1]

                for day in range(target_start_day, target_end_day + 1):
                    if len(str(month)) == 1:
                        month = "0" + str(month)
                    if len(str(day)) == 1:
                        day = "0" + str(day)
                        
                    # 날짜별로 Page Url 생성
                    url = category_url + str(year) + str(month) + str(day)

                    # totalpage는 네이버 페이지 구조를 이용해서 page=10000으로 지정해 totalpage를 알아냄
                    # page=10000을 입력할 경우 페이지가 존재하지 않기 때문에 page=totalpage로 이동 됨 (Redirect)
                    totalpage = ArticleParser.find_news_totalpage(url + "&page=10000")
                    for page in range(1, totalpage + 1):
                        made_urls.append(url + "&page=" + str(page))
                    
        return made_urls

    @staticmethod
    def get_url_data(url, max_tries=5):
        remaining_tries = int(max_tries)
        while remaining_tries > 0:
            try:
                return requests.get(url, headers={'User-Agent':'Mozilla/5.0'})
            except requests.exceptions:
                sleep(1)
            remaining_tries = remaining_tries - 1
        raise ResponseTimeout()

    def crawling(self, category_name):

        print('crawling START')

        # Multi Process PID
        # print(category_name + " PID: " + str(os.getpid()))    
        writer = Writer(category='Article', article_category=category_name, date=self.date)

        # 기사 url 형식
        url_format = f'http://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1={self.categories.get(category_name)}&date='

        # start_year년 start_month월 start_day일 부터 ~ end_year년 end_month월 end_day일까지 기사를 수집합니다.
        target_urls = self.make_news_page_url(url_format, self.date)

        print(f'{category_name} is collecting ...')

        for url in target_urls:
            
            request = self.get_url_data(url)
            document = BeautifulSoup(request.content, 'html.parser')

            # html - newsflash_body - type06_headline, type06
            # 각 페이지에 있는 기사들 가져오기
            temp_post = document.select('.newsflash_body .type06_headline li dl')
            temp_post.extend(document.select('.newsflash_body .type06 li dl'))
            
            # 각 페이지에 있는 기사들의 url 저장
            post_urls = []
            for line in temp_post:
                # 해당되는 page에서 모든 기사들의 URL을 post_urls 리스트에 넣음
                post_urls.append(line.a.get('href'))
            del temp_post

            for content_url in post_urls:  # 기사 url
                # 크롤링 대기 시간
                sleep(0.01)
                
                # 기사 HTML 가져옴
                request_content = self.get_url_data(content_url)

                try:
                    document_content = BeautifulSoup(request_content.content, 'html.parser')
                except:
                    # print('ArticleCrawler', 'error : line 180')
                    continue

                try:
                    
                    # print([time, category_name, text_company, text_headline, text_sentence, content_url])

                    # 기사 제목 가져옴
                    # tag_headline = document_content.find_all('h3', {'id': 'articleTitle'}, {'class': 'tts_head'})
                    tag_headline = document_content.find_all('h2', {'class': 'media_end_head_headline'})
                    
                    # 뉴스 기사 제목 초기화
                    text_headline = ''
                    text_headline = text_headline + ArticleParser.clear_headline(str(tag_headline[0].find_all(text=True)))
                    
                    # 공백일 경우 기사 제외 처리
                    # if not text_headline:
                        # continue

                    # 기사 본문 가져옴
                    # tag_content = document_content.find_all('div', {'id': 'articleBodyContents'})
                    tag_content = document_content.find_all('div', {'id':'dic_area'})

                    # 뉴스 기사 본문 초기화
                    text_sentence = ''
                    text_sentence = text_sentence + ArticleParser.clear_content(str(tag_content[0].find_all(text=True)))

                    # 공백일 경우 기사 제외 처리
                    # if not text_sentence:
                        # continue

                    # 기사 언론사 가져옴
                    # $('meta[name="twitter:creator"]').attr('content')
                    # tag_company = document_content.find_all('meta', {'property': 'me2:category1'})
                    tag_company = document_content.find_all('meta', {'name': 'twitter:creator'})

                    # 언론사 초기화
                    text_company = ''
                    text_company = text_company + str(tag_company[0].get('content'))

                    # 공백일 경우 기사 제외 처리
                    # if not text_company:
                    #     continue

                    # 기사 시간대 가져옴

                    # time = re.findall('<span class="t11">(.*)</span>',request_content.text)[0]
                    # time = re.findall('<span class="media_end_head_info_datestamp_time _ARTICLE_DATE_TIME">(.*)</span>',request_content.text)[0]
                    time = document_content.find('span', {'class': 'media_end_head_info_datestamp_time _ARTICLE_DATE_TIME'}).text

                    # CSV 작성
                    # print([time, category_name, text_company, text_headline, text_sentence, content_url])
                    writer.write_row([time, category_name, text_company, text_headline, text_sentence, content_url])
                    
                    del time
                    del text_company, text_sentence, text_headline
                    del tag_company 
                    del tag_content, tag_headline
                    del request_content, document_content

                # UnicodeEncodeError
                except Exception as ex:
                    print(ex)
                    del request_content, document_content
                    pass

        writer.close()

    def start(self):
        # MultiProcess 크롤링 시작
        for category_name in self.selected_categories:
            proc = Process(target=self.crawling, args=(category_name,))
            proc.start()

if __name__ == "__main__":
    
    KST = pytz.timezone('Asia/Seoul')
    currDt = dt.now(KST).date().isoformat()

    print(currDt)

    Crawler = ArticleCrawler()
    Crawler.set_category('politics','economy','society','living_culture','world','IT_science','opinion')
    Crawler.set_date_range(currDt, currDt)
    # Crawler.set_date_range('2022-05-25', '2022-05-25')
    Crawler.start()
    
    # Crawler.set_category('IT_science')
    # Crawler.set_category(sys.argv[3])
    # Crawler.set_date_range(sys.argv[1], sys.argv[2])
    # Crawler.start()