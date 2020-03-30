"""
Mine Twitter for specific IDs.
"""
from __future__ import division
import pandas as pd
import tweepy
from argparse import ArgumentParser
from math import ceil
from itertools import izip

def extract_status_data(status, attributes=['text', 'id']):
    """
    Extract relevant data from status.
    
    Parameters:
    -----------
    status : tweepy.Status
    attributes : list
    
    Returns:
    --------
    status_data : list
    """
    basic_attributes = filter(lambda x: 'user.' not in x, attributes)
    user_attributes = filter(lambda x: 'user.' in x, attributes)
    user_attributes_clean = map(lambda x: x.replace('user.', ''), user_attributes)
    status_data = {}
    status_data.update({a : status.__getattribute__(a) for a in basic_attributes})
    status_data.update({a : status.__getattribute__('user').__getattribute__(a_c) for a, a_c in izip(user_attributes, user_attributes_clean)})
    # fix text, geo, place
    status_data['text'] = status_data['text'].replace('\n', '').replace('\t', '')
    if(status_data['geo'] is not None):
        status_data['geo'] = status_data['geo']['coordinates']
    if(status_data['place'] is not None):
        status_data['place.coords'] = status_data['place'].__getattribute__('bounding_box').__getattribute__('coordinates')
        status_data['place.name'] = status_data['place'].__getattribute__('full_name')
    del(status_data['place'])
    return status_data

def main():
    parser = ArgumentParser()
    parser.add_argument('--twitter_id_file', default='../../data/mined_tweets/GeoCorpora/geocorpora_1506879947339.tsv')
    parser.add_argument('--output_file', default='../../data/mined_tweets/GeoCorpora/geocorpora_rehydrated.tsv')
    parser.add_argument('--auth_file', default='../../data/auth.csv')
    args = parser.parse_args()
    twitter_id_file = args.twitter_id_file
    output_file = args.output_file
    auth_file = args.auth_file
    
    ## load data
    twitter_id_df = pd.read_csv(twitter_id_file, sep='\t', index_col=False)
    twitter_ids = twitter_id_df.loc[:, 'tweet_id_str'].values.tolist()
    # turn into batches of 100
    N = len(twitter_ids)
    batch_size = 100
    batch_count = int(ceil(N / batch_size))

    ## set up API
    auth_data = {l.split(',')[0] : l.split(',')[1].strip() for l in open(auth_file)}
    consumer_key = auth_data['consumer_key']
    consumer_secret = auth_data['consumer_secret']
    access_token = auth_data['access_token']
    access_secret = auth_data['access_secret']
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    
    ## mine!
    tweet_attributes = ['created_at', 'favorite_count', 'geo', 'id', 'lang', 'place', 'retweet_count', 'text', 'user.screen_name', 'user.id', 'user.location']
    tweet_data = []
    for i in range(batch_count):
        id_batch = twitter_ids[i*batch_size:(i+1)*batch_size]
        statuses = api.statuses_lookup(id_batch)
        status_data = map(lambda x: extract_status_data(x, attributes=tweet_attributes), statuses)
        tweet_data += status_data
        #print(statuses[0])
        #print(dir(statuses[0]))
        #print(statuses[0].__getattribute__('user').__getattribute__('screen_name'))
        #print(statuses[0].user.screen_name)
        #print(json.dumps(statuses[0], indent=2, sort_keys=True))
    
    ## convert, write to file
    tweet_data = [pd.Series(t) for t in tweet_data]
    tweet_data = pd.concat(tweet_data, axis=1).transpose()
    tweet_data.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    

if __name__ == '__main__':
    main()
