"""
Mine tweets for power users in data.

We'll also have to tag NEs, parse, detect anchors...ugh.
"""
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import os
import logging
import sys
if('GetOldTweets/' not in sys.path):
    sys.path.append('GetOldTweets/')
import got3
from got3.manager import TweetCriteria, TweetManager
from datetime import datetime
from mine_old_tweets_GOT import convert_tweet
from time import sleep
import gzip
from unidecode import unidecode

def mine_power_users(power_user_data, start_dates, end_dates):
    """
    Mine tweets for all power users within
    specified time frame.
    
    :param power_user_data: pandas.DataFrame with usernames, data names
    :param start_dates: pandas.Series with start dates
    :param end_dates: pandas.Series with end dates
    """
    ## within specified time period
    MAX_MINE_ATTEMPTS = 5
    MINE_TIMEOUT_WAIT = 30
    # iteratively write to file
    tweet_cols = ['username','date','retweets','favorites','text','geo','mentions','hashtags','id','permalink']
#     power_user_tweet_out_file_name = os.path.join(out_dir, 'combined_data_power_user_tweets.gz')
    ## TODO: restrict to relevant hashtags? no we might need the full activity
#     data_names = power_user_data.loc[:, 'data_name'].unique()
#     data_name_hashtags = {
#         k : ['#%s'%(k), '#hurricane%s'%(k)] for k in data_names
#     }
#     # add Spanish one
#     data_name_hashtags['maria'].append('#huracanmaria')
    # TODO: remove users that we've already mined?
    # need separate file for each data set to make the
    # rest of processing work
    for name_i, data_i in power_user_data.groupby('data_name'):
        out_file_name = os.path.join(out_dir, 'combined_data_power_user_%s.gz'%(name_i))
        with gzip.open(out_file_name, 'wt') as power_user_tweet_out_file:
            # write header
            power_user_tweet_out_file.write('%s\n'%('\t'.join(tweet_cols)))
            NE_data_i = NE_data[NE_data.loc[:, 'data_name_fixed']==name_i]
            power_user_i = data_i.loc[:, 'username'].values
            start_date_i = start_dates.loc[name_i]
            end_date_i = end_dates.loc[name_i]
            # format to string
            start_date_i = datetime.strftime(start_date_i, '%Y-%m-%d')
            end_date_i = datetime.strftime(end_date_i, '%Y-%m-%d')
            power_user_tweets_i = []
            for u in power_user_i:
                logging.debug('mining user %s for %s-%s'%(u, start_date_i, end_date_i))
                mine_success = False
                attempt_ctr = 0
                tweets = []
                while(not mine_success and attempt_ctr < MAX_MINE_ATTEMPTS):
                    try:
                        ## TODO: fix GetOldTweets to return full JSON object??
                        ## need to extract from HTML which could be messy
                        tweet_criteria = TweetCriteria().setUsername(u).setSince(start_date_i).setUntil(end_date_i)
                        tweets = TweetManager.getTweets(tweet_criteria)
                        mine_success = True
                    except Exception as e:
                        print('mine failed for %s because error: %s'%(u, e))
                        print('sleeping for %d sec'%(MINE_TIMEOUT_WAIT))
                        sleep(MINE_TIMEOUT_WAIT)
                    attempt_ctr += 1
                # remove null tweets
                tweets = [tweet for tweet in tweets if not (tweet.text is None or np.isnan(tweet.text))]
                # convert to raw data
                if(len(tweets) > 0):
                    tweets = list(map(convert_tweet, tweets))
                    if(len(tweets) > 0):
                    tweets = pd.concat(tweets, axis=1).transpose()
                    tweets = tweets.loc[:, tweet_cols]
                    # write to file 
                    power_user_tweet_out_file.write(tweets.to_csv(sep='\t', header=None, index=False))
                logging.debug('collected %d tweets'%(len(tweets)))

def get_power_users(data, power_pct_lower, power_pct_upper, top_k=200, data_name_var='data_name_fixed'):
    """
    Collect power users from data based on top-k
    users within percentile bounds.
    """
    user_var = 'username'
    power_user_data = []
    # get user counts for each data name
    for name_i, data_i in data.groupby(data_name_var):
        logging.debug('processing data=%s'%(name_i))
        user_counts_i = np.log(data_i.loc[:, user_var].value_counts())
        user_counts_i_upper = np.percentile(user_counts_i, power_pct_upper)
        user_counts_i_lower = np.percentile(user_counts_i, power_pct_lower)
        user_counts_i_power = user_counts_i[(user_counts_i >= user_counts_i_lower) &
                                            (user_counts_i <= user_counts_i_upper)]
        logging.debug('filtered %d/%d power users with %d <= posts <= %d'%
                      (len(user_counts_i_power), len(user_counts_i), np.exp(user_counts_i_lower), np.exp(user_counts_i_upper)))
        # filter for top K
        user_counts_i_power = user_counts_i_power.sort_values(inplace=False, ascending=False).iloc[:top_k]
        user_counts_i_power = user_counts_i_power.index.tolist()
        power_user_data += [(name_i,x) for x in user_counts_i_power]
    power_user_data = pd.DataFrame(power_user_data, columns=['data_name', 'username'])
    return power_user_data

def main():
    parser = ArgumentParser()
    parser.add_argument('--NE_data_file', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat.gz')
    parser.add_argument('--tweet_creds', default='../../data/auth.csv')
    parser.add_argument('--event_date_file', default='../../data/hurricane_data/hurricane_dates.tsv')
    parser.add_argument('--power_pct_lower', type=int, default=95)
    parser.add_argument('--power_pct_upper', type=int, default=100)
    args = vars(parser.parse_args())
    logging_file = '../../output/mine_tweets_for_power_users.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    out_dir = os.path.dirname(args['NE_data_file'])
    
    ## load data
    NE_data = pd.read_csv(args['NE_data_file'], sep='\t', index_col=False, compression='gzip')
    # fix usernames!!
    NE_data = NE_data.assign(**{'username' : NE_data.loc[:, 'username'].apply(lambda x: x.split(':')[-1])})
    
    ## collect power users
    # can only mine K per data set without breaking Twitter ;_;
    top_k = 200
    power_user_file = os.path.join(out_dir, os.path.basename(args['NE_data_file']).replace('.gz', '_power_users.tsv'))
    if(not os.path.exists(power_user_file)):
        power_user_data = get_power_users(NE_data, args['power_pct_lower'], args['power_pct_upper'], top_k=top_k)
        logging.debug('%d power users'%(power_user_data.shape[0]))
        power_user_data.to_csv(power_user_file, sep='\t', index=False)
    else:
        power_user_data = pd.read_csv(power_user_file, sep='\t', index_col=False)

    ## define date periods
    date_periods = pd.read_csv(args['event_date_file'], sep='\t', index_col=False)
    start_dates = date_periods.groupby('hurricane_name').apply(lambda x: datetime.strptime(x[x.loc[:, 'hurricane_type']=='form'].loc[:, 'hurricane_date'].iloc[0], '%d-%m-%y'))
    end_dates = date_periods.groupby('hurricane_name').apply(lambda x: datetime.strptime(x[x.loc[:, 'hurricane_type']=='dissipation'].loc[:, 'hurricane_date'].iloc[0], '%d-%m-%y'))
    
    ## mine all power users 
    mine_power_users(power_user_data, start_dates, end_dates)
                
    ## restrict to relevant hashtags
    # clean hashtags for matching
    def clean_txt(x):
        return unidecode(x.lower())
    for name_i, data_i in power_user_data.groupby('data_name'):
        out_file_name = os.path.join(out_dir, 'combined_data_power_user_%s.gz'%(name_i))
        hashtags_i = set(['#%s'%(name_i), '#hurricane%s'%(name_i)])
        if(name_i == 'maria'):
            hashtags_i.add('#huracanmaria')
        power_user_data_i = pd.read_csv(out_file_name, sep='\t', index_col=False, compression='gzip')
        power_user_data_i = power_user_data_i.assign(**{'hashtags_fixed' : power_user_data_i.loc[:, 'hashtags'].apply(lambda x: set([clean_txt(y) for y in x]))})
        power_user_data_i = power_user_data_i.assign(**{'contains_relevant_hashtag' : power_user_data_i.loc[:, 'hashtags_fixed'].apply(lambda x: len(hashtags_i & x) > 0)})
        power_user_data_i_relevant = power_user_data_i[power_user_data_i.loc[:, 'contains_relevant_hashtag']]
        logging.debug('%d/%d relevant posts'%(power_user_data_i_relevant.shape[0], power_user_data_i.shape[0]))
        out_relevant_file_name = os.path.join(out_dir, 'combined_data_power_user_%s_relevant.gz'%(name_i))
        power_user_data_i_relevant.drop('contains_relevant_hashtag', axis=1, inplace=True)
        power_user_data_i_relevant.to_csv(out_relevant_file_name, sep='\t', index=False, compression='gzip')
        
if __name__ == '__main__':
    main()