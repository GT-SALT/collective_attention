"""
Collect info (followers, bio, etc.) for all available
users from the archive for the time periods specified.
"""
from argparse import ArgumentParser
from datetime import datetime, timedelta
import re
import os
import pandas as pd
import json
import gzip
import logging
from functools import reduce

def collect_user_data_from_files(data_files, users_existing=set(), out_dir='../../data/mined_tweets/user_data/', langs=set(['en', 'es'])):
    """
    Collect all user data from tweet archive files.
    
    :param data_files: data files
    :param users_existing: existing users to target
    :param out_dir: output directory
    :returns user_data:: pandas.DataFrame with user data (one user/tweet-time combo per row)
    """
    user_cols = ['id', 'name', 'screen_name', 'location', 'url', 'description', 'verified', 'followers_count', 'friends_count', 'created_at', 'lang']
    data_cols = user_cols + ['tweet_date']
    user_data = []
    tweet_ctr = 0
    ## TODO: restrict to certain users based on lang?
    
    user_out_files = []
    get_all_users = len(users_existing) == 0
    # clean text data
    txt_cols = ['description', 'location']
    txt_cleaner = re.compile('[\n\r\t]+')
    for files_i in data_files:
        logging.debug('processing files %s'%(','.join(files_i)))
        for f in files_i:
            logging.debug('processing file %s'%(f))
            user_out_file = os.path.join(out_dir, os.path.basename(f).replace('.gz', '_users.gz'))
            user_out_files.append(user_out_file)
            logging.debug('output to file %s'%(user_out_file))
            if(not os.path.exists(user_out_file)):
                with gzip.open(user_out_file, 'wt') as user_output:
                    for l in gzip.open(f, 'r'):
                        try:
                            j = json.loads(l.strip())
                            if('delete' not in j.keys()):
                                if(j['user']['lang'] in langs):
                                    j_user_name = j['user']['screen_name']
                                    if(get_all_users or j_user_name in users_existing):
                                        # get main user data
                                        for x in txt_cols:
                                            j['user'][x] = txt_cleaner.sub('', str(j['user'][x]))
                                        j_user_data = [j['user'][x] for x in user_cols] + [j['created_at']]
        #                                 print('\t'.join([str(x).encode('utf-8') for x in j_user_data]) + '\n')
        #                                 logging.debug('\t'.join([str(x) for x in j_user_data]) + '\n')
                                        user_output.write('\t'.join([str(x) for x in j_user_data]) + '\n')
        #                             user_data.append(j_user_data)
                                    # if available, get retweeted user data
                                    if('retweeted_status' in j.keys()):
                                        j_user_name_rt = j['retweeted_status']['user']['screen_name']
                                        if(get_all_users or j_user_name_rt in users_existing):
                                            for x in txt_cols:
                                                j['retweeted_status']['user'][x] = txt_cleaner.sub('', str(j['retweeted_status']['user'][x]))
                                            j_user_data_rt = [j['retweeted_status']['user'][x] for x in user_cols] + [j['retweeted_status']['created_at']]
        #                                     logging.debug('\t'.join([str(x) for x in j_user_data_rt]) + '\n')
                                            user_output.write('\t'.join([str(x) for x in j_user_data_rt]) + '\n')
        #                                 user_data.append(j_user_data_rt)
                            tweet_ctr += 1
                            if(tweet_ctr % 1000000 == 0):
                                logging.debug('processed %d tweets'%(tweet_ctr))
#                             if(tweet_ctr >= 1000):
#                                 break
                        except Exception as e:
                            # badly formed tweet
                            logging.debug('tweet error %s'%(e))
#             break
#         break
        
    ## combine user data!
#     user_data = [pd.read_csv(f, sep='\t', index_col=False, compression='gzip') for f in user_out_files]
    # have to build it manually because of write errors??? ugh!!
    user_col_count = 12
    user_data = [pd.DataFrame([l.strip().split('\t') for l in gzip.open(f, 'rt') if len(l.strip().split('\t')) == user_col_count]) for f in user_out_files]
    
    user_data = pd.concat(user_data, axis=0)
    user_data.columns = data_cols
    
#     user_data = pd.DataFrame(user_data, columns=data_cols)
    logging.debug('collected %d user records'%(user_data.shape[0]))
    logging.debug(user_data.head())
    user_data.fillna('', inplace=True)
    for c in txt_cols:
        user_data = user_data.assign(**{c : user_data.loc[:, c].apply(lambda x: txt_cleaner.sub('', x))})
    return user_data
        
def main():
    parser = ArgumentParser()
    parser.add_argument('--archive_dir', default='/hg190/corpora/twitter-crawl/new-archive/')
    parser.add_argument('--out_dir', default='../../data/mined_tweets/user_data/')
    parser.add_argument('--date_ranges_str', default=[['30-08-18', '26-09-18'], ['15-08-17', '09-09-17'], ['30-08-17', '20-09-17'], ['18-09-17', '31-10-17'], ['07-10-18', '23-10-18']])
    parser.add_argument('--user_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat.gz')
#     parser.add_argument('--user_data', default=None)
    ## TODO: limit to relevant users 
    args = vars(parser.parse_args())
    log_file_name = '../../output/collect_user_info_from_archive.txt'
    if(os.path.exists(log_file_name)):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    
    ## convert date ranges
    date_ranges = [[datetime.strptime(y[0], '%d-%m-%y') + timedelta(days=x) for x in range((datetime.strptime(y[1], '%d-%m-%y') - datetime.strptime(y[0], '%d-%m-%y')).days+1)] for y in args['date_ranges_str']]
    
    ## get relevant files
    file_date_format = '%b-%d-%y'
    data_files = []
    for date_range_i in date_ranges:
        date_range_str_i = '|'.join([datetime.strftime(x, file_date_format) for x in date_range_i])
        data_matcher_i = re.compile('tweets-(%s)-.*\.gz'%(date_range_str_i))
        files_i = [os.path.join(args['archive_dir'], x) for x in os.listdir(args['archive_dir']) if data_matcher_i.search(x) is not None]
#         logging.debug('found matching files\n%s'%('\n'.join(files_i)))
        data_files.append(files_i)
    
    ## collect existing user info e.g. from processed tweet file
    if(args['user_data'] is not None):
        user_data_existing = pd.read_csv(args['user_data'], sep='\t', index_col=False, compression='gzip')
        users_existing = set(user_data_existing.loc[:, 'username'].unique())
        # also add RT @ users
        rt_matcher = re.compile('^RT @([a-zA-Z0-9_]+)')
        user_data_rt = user_data_existing.loc[:, 'txt'].apply(lambda x: rt_matcher.search(x))
        user_data_rt = [x.group(1) for x in user_data_rt if x is not None]
        users_existing_rt = set(user_data_rt)
        users_existing.update(users_existing_rt)
        logging.debug('%d existing users'%(len(users_existing)))
    else:
        users_existing = set()
    user_data = collect_user_data_from_files(data_files, users_existing=users_existing)
    
    ## save to file
    # combine with existing data
    user_file_name = os.path.join(args['out_dir'], 'user_data.gz')
    if(os.path.exists(user_file_name)):
        user_data_old = pd.read_csv(user_file_name, sep='\t', index_col=False, compression='gzip')
        user_data = pd.concat([user_data_old, user_data], axis=0)
        # deduplicate
        user_data.drop_duplicates(['id', 'tweet_date'], inplace=True)
    user_data.to_csv(user_file_name, sep='\t', index=False, compression='gzip')
    
    ## remove tmp files
    tmp_user_file_matcher = re.compile('tweets-.*_users\.gz')
    user_files_tmp = [f for f in os.path.listdir(args['out_dir']) if tmp_user_file_matcher.search(f) is not None]
    for x in user_files_tmp:
        os.remove(x)
    
if __name__ == '__main__':
    main()