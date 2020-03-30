"""
Collect Twitter sample (English, non-deleted tweets) for NER labelling.
Random sample: take one day of tweets per month (first day).
"""
import gzip
import json
from argparse import ArgumentParser
import datetime
import os, re
import codecs
from langid import classify

def main():
    parser = ArgumentParser()
    parser.add_argument('--corpus_dir', default='/hg190/corpora/twitter-crawl/new-archive/')
    parser.add_argument('--tweet_year', type=int, default=2016) # default year = 2016
    parser.add_argument('--out_dir', default='../../data/mined_tweets/sample_ner_tweets/')
    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    tweet_year = args.tweet_year
    out_dir = args.out_dir
    
    ## load data
    month_range = range(1,13)
    tweet_day = 1
    all_dates = map(lambda x: datetime.datetime.strftime(datetime.date(tweet_year, x, tweet_day), '%b-%d-%y'), month_range)
    date_matcher = re.compile('|'.join(all_dates))
    tweet_files = filter(lambda x: date_matcher.search(x), os.listdir(corpus_dir))
    
    ## extract data
    LANG = 'en'
    DELETED = '[deleted]'
    EMPTY = ''
    MIN_CHARS = 20
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    for tweet_file in tweet_files:
        print('processing file %s'%(tweet_file))
        out_file_name = os.path.join(out_dir, tweet_file.replace('.gz', '.txt'))
        with codecs.open(out_file_name, 'w', encoding='utf-8') as out_file:
            in_file_name = os.path.join(corpus_dir, tweet_file)
            for i, l in enumerate(gzip.open(in_file_name, 'r')):
                try:
                    j = json.loads(l.strip())
                    j_text = j['text'].replace('\n', '').strip()
                    j_lang = j['lang']
                    if(j_text != DELETED and j_text != EMPTY and len(j_text) > MIN_CHARS and j_lang == LANG):
                        out_file.write('%s\n'%(j_text))
                except Exception, e:
                    pass
                if(i % 1000000 == 0):
                    print('processed %d lines'%(i))
        
if __name__ == '__main__':
    main()