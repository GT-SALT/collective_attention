"""
Extract clean status text from raw, pre-tag data.
We need this to extract image/video data from URLs,
which were erased in the tagging step (because of weird
tokenization).

We need this to generate the clean data for regression.
"""
from argparse import ArgumentParser
import logging
import os
import gzip
from tag_NE_tweets import process_line, load_json
import pandas as pd
import re

def main():
    parser = ArgumentParser()
    # original data
#     parser.add_argument('--tweet_files', default=[
#         '../../data/mined_tweets/stream_maria.gz',
#         '../../data/mined_tweets/archive_maria.gz',
#         '../../data/mined_tweets/historical_maria.gz',
#         '../../data/mined_tweets/stream_harvey.gz',
#         '../../data/mined_tweets/archive_harvey.gz',
#         '../../data/mined_tweets/historical_harvey.gz',
#         '../../data/mined_tweets/stream_irma.gz',
#         '../../data/mined_tweets/archive_irma.gz',
#         '../../data/mined_tweets/historical_irma.gz',
#         '../../data/mined_tweets/east_coast_geo_twitter_2018/geo_stream_florence.gz',
#         '../../data/mined_tweets/archive_florence.gz',
#         '../../data/mined_tweets/historical_florence.gz',
#         '../../data/mined_tweets/east_coast_geo_twitter_2018/geo_stream_michael.gz',
#         '../../data/mined_tweets/archive_michael.gz',
#         '../../data/mined_tweets/historical_michael.gz',
#     ])
    # power user data
    parser.add_argument('--tweet_files', default=[
        '../../data/mined_tweets/combined_data_power_user_tweets.gz',
    ])
    # original data
#     parser.add_argument('--clean_txt_file_name', default='../../data/mined_tweets/combined_tweet_clean_status.gz')
    # power user data
    parser.add_argument('--clean_txt_file_name', default='../../data/mined_tweets/combined_data_power_user_tweet_clean_status.gz')
    args = vars(parser.parse_args())
    logging_file = '../../output/extract_clean_status_txt_from_raw_data.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)

    ## stream tweets to new file
    RETURN_MATCHER = re.compile('\n|\r')
    URL_MATCHER = re.compile('(https?:|pic.twitter.com|www\.)[^#]+') # anything except a hashtag can be part of URL...yikes
#     line_ctr = 0
    if(not os.path.exists(args['clean_txt_file_name'])):
        with gzip.open(args['clean_txt_file_name'], 'wb') as out_file:
            for tweet_file in args['tweet_files']:
                logging.debug('processing file %s'%(tweet_file))
                with gzip.open(tweet_file, 'rb') as tweet_file_input:
                    for i, x in enumerate(tweet_file_input):
                        # check first line for JSON
                        if(i == 0):
                            x_str = x.decode('utf-8').strip()
                            try:
                                x_test = load_json(x_str)
                                input_file_json = type(x_test) is dict
                            except Exception as e:
                                logging.debug('json error %s with tweet %s'%(e, str(x)))
                                input_file_json = False
                        # check for dummy .tsv first line
                        if(not input_file_json and x.decode('utf-8').split('\t')[0]=='username'):
                            continue
                        else:
                            x_data = process_line(x, input_file_json, verbose=False)
                            x_txt = RETURN_MATCHER.sub(' ', x_data.loc['text'])
                            # fix URLs: some are broken by spaces
                            # TODO: find URL start index, remove all following spaces
                            url_match = URL_MATCHER.search(x_txt)
                            while(url_match is not None):
                                url_start_idx, url_end_idx = url_match.span()
                                x_txt = x_txt[0:url_start_idx] + x_txt[url_start_idx:url_end_idx].replace(' ','') + ' ' + x_txt[url_end_idx:]
                                url_match = URL_MATCHER.search(x_txt, pos=url_end_idx)
                            x_id = x_data.loc['id']
                            out_file.write(('%s\n'%('\t'.join([x_txt, str(x_id)]))).encode('utf-8'))
                        if(i % 1000 == 0):
                            logging.debug('file=%s line=%d'%(tweet_file, i))
#                         line_ctr += 1
#                         if(line_ctr >= 1000):
#                             break
#                 if(line_ctr >= 1000):
#                     break
    
    ## add to combined data file
#     combined_data = pd.read_csv(args['combined_tweet_file_name'], sep='\t', index_col=False, compression='gzip', dtype={'id' : int})
#     clean_txt_data = pd.read_csv(args['clean_txt_file_name'], sep='\t', index_col=False, compression='gzip', header=None, names=['txt_clean', 'id'], dtype={1 : int})
#     # remove duplicates
#     combined_data.drop_duplicates(['id', 'data_name_fixed'], inplace=True)
#     clean_txt_data.drop_duplicates('id', inplace=True)
    
#     clean_txt_data_lines = [x.decode('utf-8').strip().split('\t') for x in gzip.open(args['clean_txt_file_name'])]
#     clean_txt_data = pd.DataFrame(clean_txt_data_lines, columns=['txt_clean', 'id'])
#     clean_txt_data = clean_txt_data.assign(**{'id' : clean_txt_data.loc[:, 'id'].astype(int)})
#     shared_IDs = set(clean_txt_data.loc[:, 'id'].unique()) & set(combined_data.loc[:, 'id'].unique())
#     logging.debug('%d shared IDs'%(len(shared_IDs)))
#     logging.debug('%d unshared IDs'%(len(shared_IDs)))
#     # restrict to shared IDs
#     clean_txt_data = clean_txt_data[clean_txt_data.loc[:, 'id'].isin(shared_IDs)]
#     combined_data_clean = pd.merge(combined_data, clean_txt_data, on='id', how='left')
#     logging.debug('%d/%d lines merged'%(combined_data_clean.shape[0], combined_data.shape[0]))
    # original data
#     combined_file_name = args['combined_tweet_file_name'].replace('.gz', '_clean_txt.gz')
    # power user data
#     combined_file_name = args['combined_tweet_file_name'].replace('.gz', '_clean_txt.gz')    
#     combined_data_clean.to_csv(combined_file_name, sep='\t', index=False, compression='gzip')
    
if __name__ == '__main__':
    main()