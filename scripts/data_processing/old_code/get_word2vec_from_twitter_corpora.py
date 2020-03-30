"""
Generate Word2Vec embeddings from sample Twitter corpora.
"""
from gensim.models import Word2Vec
import re, os
import json
from argparse import ArgumentParser
from nltk.tokenize.casual import TweetTokenizer
import gzip

URL_MATCHER = re.compile('https?://[^\s]+')
REPLY_MATCHER = re.compile('@\w+')
HASHTAG_MATCHER = re.compile('#[^\s]+')
NUMBER_MATCHER = re.compile('[0-9\.]+')
REPLACE_PAIRS = [(URL_MATCHER, '<URL>'),
                 (REPLY_MATCHER, '@USER'),
                 (HASHTAG_MATCHER, '#HASHTAG'),
                 (NUMBER_MATCHER, '<NUM>')]
DELETED = '[deleted]'
LANG = 'en'
TOKENIZER = TweetTokenizer()
def clean_txt(txt):
    for matcher, sub in REPLACE_PAIRS:
        txt = matcher.sub(sub, txt)
    return txt

def extract_tweet_txt(l, lower=False):
    tweet_txt = ''
    try:
        tweet_data = json.loads(l)
        tweet_txt = tweet_data.get('text')
        tweet_lang = tweet_data.get('lang')
        if(tweet_txt is None or tweet_txt == DELETED or tweet_lang != LANG):
            tweet_txt = ''
        else:
            # clean
            if(lower):
                tweet_txt = tweet_txt.lower()
            tweet_txt = clean_txt(tweet_txt)
            tweet_txt = TOKENIZER.tokenize(tweet_txt)
    except Exception, e:
        pass
    return tweet_txt
    
class MakeIter:
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
        
    def __iter__(self):
        return self.generator_func(**self.kwargs)
        
def tweet_iter(corpora_files, verbose=True, lower=False):
    ctr = 0
    for f in corpora_files:
        for l in gzip.open(f, 'r'):
            ctr += 1
            if(verbose and ctr % 100000 == 0):
                print('processed %d sentences'%(ctr))
            yield extract_tweet_txt(l, lower=lower)

def main():
    parser = ArgumentParser()
    parser.add_argument('--corpora_dir', default='/hg190/corpora/twitter-crawl/new-archive/')
    parser.add_argument('--corpora_dates', default=['Jan-01-16', 'Feb-01-16', 'Mar-01-16', 
                                                    'Apr-01-16', 'May-01-16', 'Jun-01-16',
                                                    'Jul-01-16', 'Aug-01-16', 'Sep-01-16', 
                                                    'Oct-01-16', 'Nov-01-16', 'Dec-01-16'])
    parser.add_argument('--out_dir', default='../../data/embeddings/')
    parser.add_argument('--dim', type=int, default=300)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    corpora_dir = args.corpora_dir
    corpora_dates = args.corpora_dates
    out_dir = args.out_dir
    dim = args.dim
    window = args.window
    min_count = args.min_count
    epochs = args.epochs
    
    ## get corpora files
    corpora_file_matcher = re.compile('|'.join(corpora_dates))
    corpora_files = map(lambda f: os.path.join(corpora_dir, f), filter(lambda x: len(corpora_file_matcher.findall(x)) > 0, os.listdir(corpora_dir)))
    # testing: only do one file
    corpora_files = corpora_files[:1]
    
    ## get iterator
    lower = True
    corpora_sent_iterator = MakeIter(tweet_iter, corpora_files=corpora_files, lower=lower)
    
    ## get total tweet count
#     tweet_count = 0
#     for s in corpora_sent_iterator:
#         tweet_count += 1
#     print('%d tweets total'%(tweet_count))
    
    ## get model
    WORKERS = 4
    MAX_VOCAB = 1000000
    # TODO: try upper and lower
    model = Word2Vec(corpora_sent_iterator, size=dim, window=window, min_count=min_count, 
                     workers=WORKERS, max_vocab_size=MAX_VOCAB, iter=epochs)

    ## write to file
    if(lower):
        file_base = 'word2vec_lower_%d_%d_%d'%(dim, window, min_count)
    else:
        file_base = 'word2vec_%d_%d_%d'%(dim, window, min_count)
    out_file = os.path.join(out_dir, file_base)
#     model.save(out_file)
    model.wv.save_word2vec_format(out_file, binary=False)
    
if __name__ == '__main__':
    main()