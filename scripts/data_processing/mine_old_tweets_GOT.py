"""
Mine old tweets with GOT library:
https://github.com/Jefferson-Henrique/GetOldTweets-python
"""
from argparse import ArgumentParser
import sys
if('GetOldTweets/' not in sys.path):
    sys.path.append('GetOldTweets/')
import got3
import pandas as pd
import os

TWEET_IDX = ['username', 'date', 'retweets', 'favorites', 'text', 'geo', 'mentions', 'hashtags', 'id', 'permalink']
def convert_tweet(t):
    t_data = pd.Series([t.username, t.date, t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink], index=TWEET_IDX)
    return t_data

def mine_write_tweets(keyword_query, date_range, out_dir, data_name):
    """
    Mine tweets and write to tsv file.
    """
    tweet_criteria = got.manager.TweetCriteria().setQuerySearch(keyword_query).setSince(date_range[0]).setUntil(date_range[1])
    ## TODO: add getTweets option for full JSON data
    tweets = got.manager.TweetManager.getTweets(tweet_criteria)
    print('returned %d tweets'%(len(tweets)))
    tweet_data = list(map(convert_tweet, tweets))
    tweet_data = pd.DataFrame(tweet_data)

    ## write output
    # too complicated for single filename!!
#     keyword_str = ','.join(args['keywords'])
#     out_file = os.path.join(args['out_dir'], '%s_%s-%s.gz'%(keyword_str, args['date_range'][0], args['date_range'][1]))
    # simpler
    out_file = os.path.join(out_dir, 'historical_%s.gz'%(data_name))
    tweet_data.to_csv(out_file, compression='gzip', sep='\t', encoding='utf-8', index=False)

def main():
    parser = ArgumentParser()
    # keywords
    # Maria
#     parser.add_argument('--keywords', default=['#hurricanemaria', '#maria', '#huracanmaria'])
    # Harvey
#     parser.add_argument('--keywords', default=['#hurricaneharvey', '#harvey'])
    # Irma
    parser.add_argument('--keywords', default=['#hurricaneirma', '#irma'])
    # Florence
#     parser.add_argument('--keywords', default=['#hurricaneflorence', '#florence'])
    # Michael
#     parser.add_argument('--keywords', default=['#hurricanemichael', '#michael'])
    # time
    # [hurricane_start - 1, hurricane_end + 7]
    # Maria
#     parser.add_argument('--date_range', default=['2017-09-18', '2017-10-09'])
    # Harvey
#     parser.add_argument('--date_range', default=['2017-08-15', '2017-09-09'])
    # Irma
    parser.add_argument('--date_range', default=['2017-08-30', '2017-09-20'])
    # Florence
#     parser.add_argument('--date_range', default=['2018-08-30', '2018-09-26'])
    # Michael
#     parser.add_argument('--date_range', default=['2018-10-07', '2018-10-23'])
    parser.add_argument('--out_dir', default='../../data/mined_tweets/')
    # Maria
#     parser.add_argument('--data_name', default='maria')
    # Harvey
#     parser.add_argument('--data_name', default='harvey')
    # Irma
    parser.add_argument('--data_name', default='irma')
    # Florence
#     parser.add_argument('--data_name', default='florence')
    # Michael
#     parser.add_argument('--data_name', default='michael')
    # geo??
    args = vars(parser.parse_args())
    
    ## query
    keyword_query = ' OR '.join(args['keywords'])
    print('querying %s'%(keyword_query))
    mine_write_tweets(keyword_query, args['date_range'], args['out_dir'], args['data_name'])
    
if __name__ == '__main__':
    main()