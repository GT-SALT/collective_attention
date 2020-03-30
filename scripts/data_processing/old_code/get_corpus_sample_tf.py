"""
Count tokens in Spanish corpus
and output to file.
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.casual import TweetTokenizer
from argparse import ArgumentParser
import re
import pandas as pd
import numpy as np
import gzip
import json
import os

HANDLE_MATCHER = re.compile('@\w+')
HASHTAG_MATCHER = re.compile('#\w+')
URL_MATCHER = re.compile('https?://\S+|pic.twitter.com/\S+')
NUM_MATCHER = re.compile('[\d\.]+')
MATCHERS = [(HANDLE_MATCHER, ''), (HASHTAG_MATCHER, ''), 
            (URL_MATCHER, '<URL>'), (NUM_MATCHER, '<NUM>')]
def tweet_generator(f, lang='es'):
    # cutoff = 1000000
    for i, l in enumerate(gzip.open(f, 'r')):
        try:
            l_json = json.loads(l.strip())
            l_txt = l_json.get('text')
            l_lang = l_json.get('lang')
            # TODO: deal with English words too!!
            if(l_txt is not None and l_lang == lang):
                l_txt = l_txt.lower()
                for m, sub in MATCHERS:
                    l_txt = m.sub(sub, l_txt)
                yield l_txt
            else:
                yield ''
        except Exception, e:
            yield ''
        if(i % 1000000 == 0):
            print('processed %d lines'%(i))
        #if(i >= cutoff):
        #    break

def get_tf(corpus_file, ngram_range=(1,3), lang='es'):
    """
    Compute frequency of all n-grams in corpus.
    Assume .gz format with JSON objects.

    Parameters:
    -----------
    corpus_file : str
    ngram_range : (int, int)
    lang : str
    Tagged tweet language.
    
    Returns:
    --------
    tf : pandas.Series
    """
    tokenizer = TweetTokenizer()
    cv = CountVectorizer(lowercase=True, ngram_range=ngram_range, tokenizer=tokenizer.tokenize)
    dtm = cv.fit_transform(tweet_generator(corpus_file, lang=lang))
    ivoc = {v : k for k,v in cv.vocabulary_.iteritems()}
    tf = np.array(dtm.sum(axis=0))[0]
    tf = pd.Series(dict([(ivoc[i], x) for i, x in enumerate(tf)]))
    return tf

def main():
    parser = ArgumentParser()
    parser.add_argument('--corpus_file', default='/hg190/corpora/twitter-crawl/new-archive/tweets-Sep-20-17-03-57.gz')
    parser.add_argument('--out_dir', default='../../data/')
    args = parser.parse_args()
    corpus_file = args.corpus_file
    out_dir = args.out_dir
    
    ## count
    tf = get_tf(corpus_file, ngram_range=(1,3), lang='es')
    tf.sort_values(inplace=True, ascending=False)
    
    ## write to file
    date_str_matcher = re.compile('[A-Z][a-z]{2}-[0-3][0-9]-1[0-9]')
    date_str = date_str_matcher.findall(corpus_file)[0]
    out_file = os.path.join(out_dir, '%s_tf.tsv'%(date_str))
    tf.to_csv(out_file, sep='\t', encoding='utf-8')

if __name__ == '__main__':
    main()
