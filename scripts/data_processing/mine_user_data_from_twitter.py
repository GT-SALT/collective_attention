"""
Mine user data from Twitter using query:
https://developer.twitter.com/en/docs/accounts-and-users/follow-search-get-users/api-reference/get-users-lookup
"""
import pandas as pd
from tweepy import API, OAuthHandler
from time import sleep
from datetime import datetime
from argparse import ArgumentParser
import logging
import os
from math import ceil
import gzip
import re

def mine_write_author_data(authors, out_file_name, api):
    author_attributes = ['name', 'id', 'location', 'description', 'followers_count', 'friends_count', 'statuses_count', 'created_at', 'verified', 'listed_count', 'screen_name']
    author_attribute_cols = ['name', 'id', 'location', 'description', 'followers_count', 'friends_count', 'statuses_count', 'created_at', 'verified', 'listed_count', 'username']
    txt_attributes = ['location', 'description']
    mine_date_fmt = '%d-%m-%Y'
    # remove return chars from text
    return_char_matcher = re.compile('[\n\r]')
    # sleep 15 mins in case of timeout
    sleep_time = 15 * 60
    author_chunk_size = 100
    author_chunks = int(ceil(len(authors) / author_chunk_size))
    with gzip.open(out_file_name, 'wb') as out_file:
        out_file.write(('%s\n'%('\t'.join(author_attribute_cols + ['date']))).encode('utf-8'))
        for i in range(author_chunks):
            authors_i = authors[(i*author_chunk_size):((i+1)*author_chunk_size)]
            author_lookup_success = False
            while(not author_lookup_success):
                try:
#                     logging.debug('authors = %s'%(str(authors_i)))
                    author_info_i = api.lookup_users(screen_names=authors_i)
                    author_lookup_success = True
                except Exception as e:
                    logging.debug('exception %s'%(e))
                    logging.debug('timeout after %d authors; sleeping for %d sec'%(i*author_chunk_size, sleep_time))
                    sleep(sleep_time)
            author_data_i = pd.DataFrame([[author_info.__getattribute__(author_attribute) for author_attribute in author_attributes] for author_info in author_info_i], columns=author_attributes)
            # add mine date
            mine_date_i = datetime.strftime(datetime.now(), mine_date_fmt)
            author_data_i = author_data_i.assign(**{'date' : mine_date_i})
            # fix txt data
            author_data_i = author_data_i.assign(**{
                txt_attribute : author_data_i.loc[:, txt_attribute].apply(lambda y: return_char_matcher.sub('', y)) 
                for txt_attribute in txt_attributes
            })
            # write to file
            out_file.write(author_data_i.to_csv(sep='\t', index=False, header=False).encode('utf-8'))
            if(i % 10 == 0):
                logging.debug('mined %d/%d authors'%(i*author_chunk_size, len(authors)))

def main():
    parser = ArgumentParser()
    parser.add_argument('--full_data_file', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor.gz')
    parser.add_argument('--auth_file', default='../../data/auth.csv')
    parser.add_argument('--out_dir', default='../../data/mined_tweets/tweet_user_data/')
    args = vars(parser.parse_args())
    logging_file = '../../output/mine_user_data_from_twitter.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)

    ## load original data
    use_cols = ['username']
    full_data = pd.read_csv(args['full_data_file'], sep='\t', index_col=False, compression='gzip', usecols=use_cols)
    # fix author names
    author_var = 'username'
    full_data = full_data.assign(**{
        author_var : full_data.loc[:, author_var].apply(lambda x: x.split(':')[-1])
    })
    
    ## set up API connection
    auth_data = pd.read_csv(args['auth_file'], header=None, index_col=0).iloc[:, 0]
    auth = OAuthHandler(auth_data.loc['consumer_key'], auth_data.loc['consumer_secret'])
    api = API(auth)
    auth.set_access_token(auth_data.loc['access_token'], auth_data.loc['access_secret'])

    ## search for author data
    unique_authors = list(full_data.loc[:, author_var].unique())
    # tmp debugging
#     unique_authors = unique_authors[:200]
    out_file_name = os.path.join(args['out_dir'], 'user_data_twitter_mine.gz')
    mine_write_author_data(unique_authors, out_file_name, api)
    
if __name__ == '__main__':
    main()