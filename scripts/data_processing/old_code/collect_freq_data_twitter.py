"""
Collect frequency statistics over time 
for all tagged tweets.
This is gonna get messy!
"""
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os
import re
import dateutil
import gzip
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from data_helpers import clean_tweet_txt
from functools import reduce
from stop_words import get_stop_words
from unidecode import unidecode
import logging

def compute_days(data, time_var='date'):
    """
    Round down date value to nearest day.
    
    :param data: pandas.DataFrame with combined txt/time data
    :returns data_time:: updated txt/time data with rounded date
    """
    data_time = data.assign(**{'%s_day'%(time_var) : data.loc[:, time_var].apply(lambda x: datetime(day=x.day, month=x.month, year=x.year))})
    return data_time

from data_helpers import extract_NEs_from_tags
def convert_tags_to_txt(tags):
    tag_list = extract_NEs_from_tags(tags, keep_all_tokens=True)
    return tag_list

def load_combined_data_from_tag_files(tag_files):
    """
    Load combined text/status/tag data from 
    tagged data files.
    
    :param tag_files: list of tag files
    :returns combined_txt_data:: combined text/status/tag data
    """
    combined_txt_data = []
    data_name_matcher = re.compile('[a-z]+(?=_txt_tags\.gz)')
    for f in tag_files:
        logging.debug('processing file %s'%(os.path.basename(f)))
        f_status = f.replace('_txt_tags.gz', '_status.gz')
        f_txt = f.replace('_txt_tags.gz', '_txt.txt')
        f_data = pd.read_csv(f_status, sep='\t', index_col=False, compression='gzip', header=None, converters={2:dateutil.parser.parse})
        f_data.columns = ['id', 'username', 'date', 'retweets', 'likes', 'lang']
        f_data = f_data.assign(**{'id' : f_data.loc[:, 'id'].astype(int)})
        f_txt = [l.strip() for l in open(f_txt, 'r')]
        f_data = f_data.assign(**{'txt':f_txt})
        # label with data name
        f_name = data_name_matcher.search(os.path.basename(f)).group(0)
        f_data = f_data.assign(**{'data_name':f_name})
        f_tag_data = pd.DataFrame([l.decode('utf-8').strip().split('\t') for l in gzip.open(f, 'rb')], columns=['id', 'tags'])
        # convert tags to list
        f_tag_data = f_tag_data.assign(**{'tags' : f_tag_data.loc[:, 'tags'].apply(lambda x: [y.split('/') for y in x.split(' ')])})
        ## fix tag triples??
        f_tag_data = f_tag_data.assign(**{'tags' : f_tag_data.loc[:, 'tags'].apply(lambda x: [[' '.join(y[:-1]), y[-1]] for y in x])})
        f_tag_data = f_tag_data.assign(**{'id':f_tag_data.loc[:, 'id'].astype(int)})
        f_data = pd.merge(f_data, f_tag_data, on='id')
    #     logging.debug('%d/%d data'%(len(f_txt), f_data.shape[0]))
    #     display(f_data.head())
        combined_txt_data.append(f_data)
    combined_txt_data = pd.concat(combined_txt_data, axis=0)
    logging.debug('%d combined statuses'%(combined_txt_data.shape[0]))
    # deduplicate
    combined_txt_data.drop_duplicates(['id', 'data_name'], inplace=True)
    # add time
    combined_txt_data_time = compute_days(combined_txt_data)
    # extract tags/non-tags
    combined_txt_data_time = combined_txt_data_time.assign(**{'tagged_txt' : combined_txt_data_time.loc[:, 'tags'].apply(lambda x: convert_tags_to_txt(x))})
    tag_type_counts = pd.Series([y[1] for x in combined_txt_data_time.loc[:, 'tagged_txt'].values for y in x]).value_counts()
    logging.debug(tag_type_counts)
    return combined_txt_data_time

def norm_entity_type(txt, entity_type_matchers):
    """
    Normalize entity type if text matches.
    
    :param txt: raw text to normalize
    :param entity_type_matchers: (regex, substitute marker) tuples ex. ('_geo', '_LOC')
    :return txt:: normalized entity text
    """
    for entity_type_matcher, entity_marker in entity_type_matchers:
        if(entity_type_matcher.search(txt) is not None):
            txt = entity_type_matcher.sub(entity_marker, txt)
    return txt

QUOTE_MATCHER = re.compile('\'|"|”|“')
HANGING_UNDERSCORE_MATCHER = re.compile('^_|_$')
PUNCT_MATCHER = re.compile('_*[!,\.;:]_*')
GENERIC_MATCHER = re.compile('.*(#+|#\s*hash|\s*<\s*url\s*>|\s*<\s*num\s*>).*')
WORD_NE_MATCHER = re.compile('[a-zA-Z_]+')
def clean_strip_txt(x):
    """
    Clean and strip tweet text.
    
    :param x: raw text
    :returns x:: cleaned text
    """
    x = x.strip()
    x = x.replace(' ', '_')
    x = clean_tweet_txt(x)
    x = unidecode(x)
    # strip quotes, punct, underscores
    x = QUOTE_MATCHER.sub('', x)
    x = PUNCT_MATCHER.sub('', x)
    x = GENERIC_MATCHER.sub('', x)
    x = HANGING_UNDERSCORE_MATCHER.sub('', x)
    # remove non-words
    x_valid = WORD_NE_MATCHER.search(x)
    if(x_valid is None):
        x = ''
    else:
        x = x_valid.group(0)
    return x

def compute_time_frequency(data, time_var='date_day', stopword_langs=['en', 'es']):
    """
    Compute frequency over time of text tokens.
    
    :param data: pandas.DataFrame with raw text, tags, time
    :returns freq:: pandas.DataFrame with normalized type frequency (row) per date (col)
    """
    # group tag data by date
    # compute raw NE frequencies at each t
    # normalize for token count at t
    freq = []
    LOC_TYPES = set(['COUNTRY', 'LOCATION', 'CITY', 'geo'])
    LOC_TYPE_MATCHER = re.compile('|'.join(['_%s$'%(x) for x in LOC_TYPES]))
    LOC_MARKER = '_LOC'
    ENTITY_TYPE_MARKERS = [[LOC_TYPE_MATCHER, LOC_MARKER]]
    min_df = 0.00001
    max_df = 0.75
    min_df_per_time = 10
    stopwords = set(reduce(lambda x,y: x|y, [set(get_stop_words(x)) for x in stopword_langs]))
    TKNZR = lambda x: x.split(' ')
    
    ## convert all text to DTM, then group by time
    txt = data.loc[:, 'tagged_txt']
    txt_tokens_tags = txt.apply(lambda x: [[clean_strip_txt(y[0].replace(' ', '_')), y[1]] for y in x])
    # remove blanks
    txt_tokens_tags = txt_tokens_tags.apply(lambda x: [y for y in x if y[0] != ''])
    txt_tokens = txt_tokens_tags.apply(lambda x: ' '.join(['%s_%s'%(y[0], y[1]) if y[1] != 'O' else y[0] for y in x]))
    # normalize entity types to same marker
    # ex. COUNTRY/LOCATION/CITY/GEO => LOC
    txt_tokens = txt_tokens.apply(lambda x: ' '.join([norm_entity_type(y, ENTITY_TYPE_MARKERS) for y in x.split(' ')]))
    cv = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1,1), stop_words=stopwords, tokenizer=TKNZR)
    dtm = cv.fit_transform(txt_tokens)
    cv_vocab = sorted(cv.vocabulary_.keys(), key=lambda x: cv.vocabulary_[x])
    for time_i in data.loc[:, time_var].unique():
        time_i_idx = np.where(data.loc[:, time_var] == time_i)[0]
        dtm_i = dtm[time_i_idx, :]
        dtm_i = pd.Series(np.squeeze(np.asarray(dtm_i.sum(axis=0))), index=cv_vocab)
        dtm_i = dtm_i / dtm_i.sum()
        freq.append([time_i, dtm_i])
    freq_time, freq_counts = list(zip(*freq))
    freq_data = pd.concat(freq_counts, axis=1).transpose()
    freq_data = freq_data.assign(**{'TIME' : freq_time})
    freq_data.fillna(0., inplace=True)
    return freq_data

def compute_frequency_by_time(data, group_var='data_name', time_var='date_day'):
    """
    Compute frequency within each time period.
    
    :param data: frequency dataframe
    :param group_var: data group var
    :param time_var: data time var
    :return freq_data:: time + frequency dataframe
    """
    ## group by day, compute token frequency
    freq_data = []
    for data_name_i, data_i in data.groupby(group_var):
        logging.debug('processing data %s'%(data_name_i))
        freq_data_i = compute_time_frequency(data_i, time_var=time_var)
    #     display(freq_data_i.head())
    #     freq_data_i_sum = freq_data_i.drop('TIME', axis=1, inplace=False).sum(axis=0).sort_values(inplace=False, ascending=False)
    #     display(freq_data_i_sum.loc[[x for x in freq_data_i_sum.index if x.endswith('_geo')]])
        freq_data_i = freq_data_i.assign(**{'DATA_NAME' : data_name_i})
        freq_data.append(freq_data_i)
    freq_data = pd.concat(freq_data, axis=0)
    freq_data.fillna(0, inplace=True)
    # test top-k words
    top_k = 20
    logging.debug('%d/%d LOC cols'%(len([x for x in freq_data.columns if x.endswith('_loc')]), freq_data.shape[1]))
    for data_name_i, data_i in freq_data.groupby('DATA_NAME'):
        logging.debug('testing data=%s'%(data_name_i))
        data_i_freq = data_i.drop(['TIME', 'DATA_NAME'], axis=1, inplace=False)
        # overall freq
        logging.debug(data_i_freq.mean(axis=0).sort_values(inplace=False, ascending=False).iloc[:top_k])
        # LOC word freq
        data_i_freq_loc = data_i_freq.loc[:, [x for x in data_i_freq.columns if x.endswith('_loc')]]
        logging.debug(data_i_freq_loc.mean(axis=0).sort_values(inplace=False, ascending=False).iloc[:top_k])
    return freq_data

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/mined_tweets/')
    parser.add_argument('--out_file_name', default='combined_tweet_tag_data_NE')
    args = vars(parser.parse_args())
    log_file_name = '../../output/regression_results/collect_freq_data_twitter.txt'
    if(os.path.exists(log_file_name)):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)

    ## collect tag files and load data
    data_dir = args['data_dir']
    txt_file_matcher = re.compile('_txt\.txt$')
    status_file_matcher = re.compile('_status\.gz$')
    tag_file_matcher = re.compile('(archive|stream|historical)_[a-z]+_txt_tags\.gz$')
    tag_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if tag_file_matcher.search(x) is not None]
    combined_txt_data = load_combined_data_from_tag_files(tag_files)
    
    ## compute frequency
    combined_txt_data_time = compute_frequency_by_time(combined_txt_data, group_var='data_name')
    
    ## save to file
    out_file_name = os.path.join(args['data_dir'], '%s_freq.gz'%(args['out_file_name']))
    combined_txt_data_time.to_csv(out_file_name, sep='\t', index=False, compression='gzip')

if __name__ == '__main__':
    main()