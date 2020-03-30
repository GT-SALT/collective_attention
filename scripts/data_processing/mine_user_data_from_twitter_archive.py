"""
Mine user data from Twitter archive, from the time
period during which the users were active.
"""
import pandas as pd
from argparse import ArgumentParser
import logging
import os
from datetime import datetime, timedelta
from data_helpers import fix_timezone
import dateutil
import re
import gzip
import json

def extract_write_user_data(files, authors, data_name, out_dir):
    data_name_var = 'data_name_fixed'
    author_var = 'username'
    date_var = 'date'
    tweet_file_date_matcher = re.compile('(?<=tweets-)[A-Za-z]{3,}-[0-9]{2}-[0-9]{2}')
    return_char_matcher = re.compile('[\n\r]')
    author_tweet_attributes = ['name', 'id', 'location', 'description', 'followers_count', 'friends_count', 'statuses_count', 'created_at', 'verified']
    author_data_attributes = author_tweet_attributes + [date_var, data_name_var, author_var]
    txt_attributes = ['location', 'description']
    for file_j in files:
        # extract date
        file_date_str_j = tweet_file_date_matcher.search(file_j).group(0)
        file_date_j = dateutil.parser.parse(file_date_str_j)
        logging.debug('processing file %s with date %s'%(file_j, file_date_j))
        user_data = []
        # tmp debugging
#             line_ctr = 0
        for l in gzip.open(file_j):
            try:
                j = json.loads(l.strip())
#                     j_date = j['created_at']
                if('delete' not in j):
                    j_user = j['user']
                    j_user_name = j_user['screen_name'].lower()
                    if(j_user_name in authors):
                        user_data_j = [j_user[x] for x in author_tweet_attributes] + [file_date_j, data_name, j_user_name]
                        user_data.append(user_data_j)
#                             logging.debug('found valid tweet %s'%(json.dumps(j)))
#                             break
                        # tmp debugging
#                             line_ctr += 1
#                             if(line_ctr >= 10):
#                                 logging.debug('stop mining')
#                                 break
            except Exception as e:
                logging.debug('error %s'%(e))
                logging.debug('skipping bad line %s'%(l.strip()))
                pass
        # combine data
        user_data = pd.DataFrame(user_data)
        user_data.columns = author_data_attributes
        user_data.fillna('', inplace=True)
        # fix text data
        user_data = user_data.assign(**{
            x : user_data.loc[:, x].apply(lambda y: return_char_matcher.sub('', y)) 
            for x in txt_attributes
        })
        # drop duplicate author/date pairs
        user_data.drop_duplicates([author_var, date_var], inplace=True)
        # write to file
        user_data_file = os.path.join(out_dir, '%s_%s_user_data.gz'%(data_name, file_date_str_j))
        user_data.to_csv(user_data_file, sep='\t', index=False, compression='gzip')

def main():
    parser = ArgumentParser()
    parser.add_argument('--tweet_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor.gz')
    parser.add_argument('--archive_dir', default='/hg190/corpora/twitter-crawl/new-archive/')
    args = vars(parser.parse_args())
    logging_file = '../../output/mine_user_data_from_twitter_archive.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    ## load data
    date_var = 'date'
    author_var = 'username'
    data_name_var = 'data_name_fixed'
    tweet_data = pd.read_csv(args['tweet_data'], sep='\t', index_col=False, compression='gzip', usecols=[date_var, author_var, data_name_var], converters={date_var : dateutil.parser.parse})
    # fix username
    tweet_data = tweet_data.assign(**{author_var : tweet_data.loc[:, author_var].apply(lambda x: x.split(':')[-1])})
    # fix timezones
    tweet_data = tweet_data.assign(**{date_var : tweet_data.loc[:, date_var].apply(lambda x: fix_timezone(x))})
    # need author | data_name | date range
    data_date_ranges = tweet_data.groupby(data_name_var).apply(lambda x: [x.loc[:, date_var].min() + timedelta(days=i) for i in range((x.loc[:, date_var].max() - x.loc[:, date_var].min()).days)])
    # round to nearest day
    data_date_ranges = data_date_ranges.apply(lambda x: [datetime(*y.timetuple()[:3]) for y in x])
    # clean
    date_range_var = 'date_range'
    data_date_ranges = data_date_ranges.reset_index().rename(columns={0 : date_range_var})
    # combine with author names
    author_data_names = tweet_data.loc[:, [author_var, data_name_var]].drop_duplicates(inplace=False)
    author_date_ranges = pd.merge(data_date_ranges, author_data_names, on=data_name_var)
    
    ## mine!
    ## author | tweet date | location | description | followers | friends | status count | created date
    tweet_file_pattern = os.path.join(args['archive_dir'], 'tweets-%s-*.gz')
    tweet_file_date_fmt = '%b-%d-%y'
    out_dir = os.path.join(os.path.dirname(args['tweet_data']), 'tweet_user_data')
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    for name_i, data_i in author_date_ranges.groupby(data_name_var):
        logging.debug('mining authors from data %s'%(name_i))
        authors_i = set(data_i.loc[:, author_var].apply(lambda x: x.lower()).unique())
        dates_i = data_i.loc[:, date_range_var].iloc[0]
        files_i = []
        for date_j in dates_i:
            tweet_file_pattern_j = datetime.strftime(date_j, tweet_file_date_fmt)
            tweet_file_matcher_j = re.compile(tweet_file_pattern_j)
            files_j = [os.path.join(args['archive_dir'], f) for f in os.listdir(args['archive_dir']) if tweet_file_matcher_j.search(f) is not None]
            files_i += files_j
        user_data_i = extract_write_user_data(files_i, authors_i, name_i, out_dir)
    
    ## combine all user data
    user_data_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if 'user_data.gz' in f]
    user_data_combined = []
    for f in user_data_files:
        data_f = pd.read_csv(f, sep='\t', index_col=False, compression='gzip')
        user_data_combined.append(data_f)
    user_data_combined = pd.concat(user_data_combined, axis=0)
    combined_out_file = os.path.join(out_dir, 'user_data_archive.gz')
    if(not os.path.exists(combined_out_file)):
        user_data_combined.to_csv(combined_out_file, sep='\t', index=False, compression='gzip')
    
if __name__ == '__main__':
    main()