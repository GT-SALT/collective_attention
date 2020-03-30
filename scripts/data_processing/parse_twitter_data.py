"""
Parse all tweets with Spacy because it's easier than Google and can handle short dependencies well (fine for Twitter, less fine for FB with long sentences).
"""
import spacy
from argparse import ArgumentParser
from langid import langid
import pandas as pd
import numpy as np
from data_helpers import build_parse
import os
from nltk.tokenize import sent_tokenize
import math
import re
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
from spacy.tokens import Doc
import logging

class NLTKTokenizerSpacy(object):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __call__(self, text):
        words = self.tokenizer.tokenize(text)
        # All tokens 'own' a subsequent space character in this tokenizer
#         spaces = [True] * len(words)
        return Doc(self.vocab, words=words)

def clean_parse_txt(x):
    """
    Clean text for TweeboParser.
    """
    x = x.replace(',', ' ')
    return x

def parse_data_tweebo(data, tmp_dir='../../data/mined_tweets/'):
    """
    Parse all unique sentences in data,
    using TweeboParser for English and 
    spacy for Spanish. Assumes that TweeboParser
    has already been installed.
    TODO: extract from ugly CoNLL output
    
    :param data: pandas.DataFrame with text data
    :returns parsed_data:: pandas.DataFrame with text data
    """
    ## need to write English data to file to get TweeboParser to run...FML 
    parser_es = spacy.load('es_core_news_sm', disable=['tagger', 'ner', 'textcat'])
    data.loc[:, 'lang'] = data.loc[:, 'txt'].apply(lambda x: langid.classify(x)[0])
    
    data_es = data[data.loc[:, 'lang'] == 'es']
    data_en = data[data.loc[:, 'lang'] != 'es']
    
    ## EN parses
    # write to temp file
    en_txt_file = os.path.join(tmp_dir, 'en_parse_tmp.txt')
    with open(en_txt_file, 'w') as en_txt_out:
        # clean text
        txt_clean = data_en.loc[:, 'txt'].apply(lambda x: clean_parse_txt(x)).values
        en_txt_out.write('\n'.join(txt_clean))
    # parse
    en_txt_parse_file = '%s.predict'%(en_txt_file)
    parse_command = 'cd TweeboParser && ./run.sh %s'%(en_txt_file)
    process = subprocess.Popen(parse_command.split(), stdout=subprocess.PIPE)
    output, err = process.communicate()
    # read parses
    parsed_data_en = read_tweeboparse_output(en_txt_parse_file)
    # remove files
    os.remove(en_txt_file)
    os.remove(en_txt_parse_file)
    
    ## ES parses
    parsed_data_es = data_es.loc[:, 'txt'].apply(lambda x: [build_parse(parser_es(y), parse_type='spacy') for y in sent_tokenize(x)])
    
PUNCT_MATCHER = re.compile('[-,]')
TWEET_MATCHER = re.compile('<URL>|@USER|#HASH')

def clean_data_for_spacy(x):
    """
    Clean data before parsing with spacy.
    """
    x = PUNCT_MATCHER.sub(' ', x)
    x = TWEET_MATCHER.sub('', x)
    return x
    
def parse_data(data):
    """
    Parse all unique sentences in data.
    
    :param data: pandas.DataFrame with text data
    :returns parsed_data:: pandas.DataFrame with text data
    """
    parser_en = spacy.load('en_core_web_md', disable=['ner', 'textcat'])
    parser_es = spacy.load('es_core_news_sm', disable=['ner', 'textcat'])
    # custom tokenizers because duh
    parser_en.tokenizer = NLTKTokenizerSpacy(parser_en.vocab, TweetTokenizer())
    parser_es.tokenizer = NLTKTokenizerSpacy(parser_es.vocab, ToktokTokenizer())
    data.loc[:, 'lang'] = data.loc[:, 'txt'].apply(lambda x: langid.classify(x)[0])
    parsed_data = []
    for i, data_i in data.iterrows():
        txt = data_i.loc['txt']
        txt = clean_data_for_spacy(txt)
        sents = sent_tokenize(txt)
        parsed_data_i = []
        for sent in sents:
            if(data_i.loc['lang'] == 'es'):
                parse_i = parser_es(sent)
            else:
                parse_i = parser_en(sent)
            # extract tree
            tree_i = build_parse(parse_i, parse_type='spacy')
            parsed_data_i.append(tree_i)
        parsed_data_i = pd.DataFrame(pd.Series(parsed_data_i), columns=['parse'])
#         logging.debug('processing id %s/%s'%(data_i.loc['id'], int(data_i.loc['id'])))
        parsed_data_i = parsed_data_i.assign(**{'id' : int(data_i.loc['id'])})
        parsed_data.append(parsed_data_i)
    parsed_data = pd.concat(parsed_data, axis=0)
#     parsed_data.loc[:, 'id'] = parsed_data.loc[:, 'id'].astype(np.int64)
    return parsed_data

def main():
    parser = ArgumentParser()
    parser.add_argument('--tweet_file', default='../../data/mined_tweets/combined_tweet_NE_flat_data.gz')
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--chunk_count', type=int, default=0)
    parser.add_argument('--total_tweets', type=int, default=1000)
#     parser.add_argument('--start_idx', type=int, default=0) # inclusive
#     parser.add_argument('--end_idx', type=int, default=1e8) # the last possible row
    args = vars(parser.parse_args())
    log_file_name = '../../output/parse_twitter_data.txt'
    if(os.path.exists(log_file_name)):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)

    ## load data
#     valid_idx = range(args['start_idx'], args['end_idx']+1)
    # tmp debugging
    
    if(args['chunk_count'] > 0):
        chunk_size = int(math.ceil(args['total_tweets'] / args['chunk_count']))
        start_idx = chunk_size*(args['chunk_idx']-1)
        end_idx = chunk_size*(args['chunk_idx'])
        end_idx = min(end_idx, args['total_tweets']-1)
        valid_idx = set([0] + list(range(start_idx, end_idx))) # include 0 for header
        logging.debug('chunk idx %d size %d; start/end = %d/%d'%(args['chunk_idx'], chunk_size, start_idx, end_idx))
        data = pd.read_csv(args['tweet_file'], sep='\t', compression='gzip', index_col=False, skiprows=lambda x: x not in valid_idx, usecols=['id', 'txt'])
    else:
        data = pd.read_csv(args['tweet_file'], sep='\t', compression='gzip', index_col=False, usecols=['id', 'txt'])
#         data = data.iloc[start_idx:end_idx, :]
    logging.debug('%d rows loaded'%(data.shape[0]))
    # deduplicate
    data_dedup = data.drop_duplicates('id', inplace=False)
    logging.debug('%d/%d rows after dropping duplicates'%(data_dedup.shape[0], data.shape[0]))
    
    ## parse all the things
    parsed_data = parse_data(data_dedup)
#     logging.debug(parsed_data.head())
    
    ## write to file
    data_dir = os.path.dirname(args['tweet_file'])
    parse_out_file_name = os.path.join(data_dir, os.path.basename(args['tweet_file']).replace('.gz', '_parsed_%d.gz'%(args['chunk_idx'])))
    parsed_data.to_csv(parse_out_file_name, sep='\t', index=False, compression='gzip')
    
if __name__ == '__main__':
    main()