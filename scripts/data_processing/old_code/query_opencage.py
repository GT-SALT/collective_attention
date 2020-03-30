"""
Query OpenCage (https://geocoder.open_cagedata.com) for OSM and
GeoNames estimates for the gold-labelled data.
"""
from opencage.geocoder import OpenCageGeocode, RateLimitExceededError
from argparse import ArgumentParser
from time import sleep
import pandas as pd
import os

OPEN_CAGE_QUERY_LIMIT=100
OPEN_CAGE_WAIT_TIME = 1
def query_open_cage(query, open_cage_api):
    result = open_cage_api.geocode(query, limit=OPEN_CAGE_QUERY_LIMIT)
    # mandatory sleep after query
    sleep(OPEN_CAGE_WAIT_TIME)
    if(len(result) == 0):
        result_df = pd.DataFrame()
    else:
        result_df = pd.DataFrame(result)
        result_df.loc[:, 'query'] = query
    return result_df

def main():
    parser = ArgumentParser()
    parser.add_argument('--geocorpora_file', default='../../data/mined_tweets/GeoCorpora/geocorpora_1506879947339.tsv')
    parser.add_argument('--open_cage_key', default='557e2cc5cc164f59a1243b83747b4aba')
    parser.add_argument('--out_file', default='../../data/mined_tweets/GeoCorpora/geocorpora_opencage_query_results_temp.tsv')
    args = parser.parse_args()
    geocorpora_file = args.geocorpora_file
    open_cage_key = args.open_cage_key
    out_file = args.out_file
    
    ## load data
    geocorpora = pd.read_csv(geocorpora_file, sep='\t', index_col=False, encoding='utf-8')
    geocorpora_data = geocorpora.loc[:, ['text', 'tweet_id_str', 'latitude', 'longitude']]
    geocorpora_data.rename(columns={'latitude' : 'gold_latitude', 
                                    'longitude' : 'gold_longitude'},
                           inplace=True)
    open_cage_api = OpenCageGeocode(open_cage_key)
    
    # mine!
    
    if(not os.path.exists(out_file)):
        mentions_mined = {}    
        open_cage_query_results = pd.DataFrame()
    else:
        open_cage_query_results = pd.read_csv(out_file, sep='\t', index_col=False, encoding='utf-8')
        mentions_mined = {}
        for g_mention, g_mention_group in open_cage_query_results.groupby('query'):
            mentions_mined[g_mention] = g_mention_group
    for i, g_data in geocorpora_data.iterrows():
        g_mention = g_data.loc['text']
        if(g_mention not in mentions_mined.keys()):
            try:
                g_mention = g_mention.decode('utf-8')
            except UnicodeEncodeError, e:
                print('decoding error at mention %s'%(g_mention))
            try:
                query_results = query_open_cage(g_mention, open_cage_api)
                mentions_mined[g_mention] = query_results
            except RateLimitExceededError, e:
                print('rate limit exceeded at %d mentions: mention %s'%(g_mention))
                break
            if(len(mentions_mined) % 10 == 0):
                print('processed %d mention strings'%(len(mentions_mined)))
        else:
            query_results = mentions_mined[g_mention]
        # if no valid results, just make an empty data frame
        if(query_results.shape[0] == 0):
            print('empty query result from mention %s'%(g_mention))
            query_results.loc[:, 'query'] = [g_mention]
        # also include tweet ID, lat, lon
        # convert ID to string to prevent clipping
        query_results.loc[:, 'tweet_id_str'] = g_data.loc['tweet_id_str'].apply(lambda x: str(x))
        query_results.loc[:, 'gold_latitude'] = g_data.loc['gold_latitude']
        query_results.loc[:, 'gold_longitude'] = g_data.loc['gold_longitude']
        open_cage_query_results = open_cage_query_results.append(query_results)
        # repeatedly write to file because paranoia
        open_cage_query_results.to_csv(out_file, sep='\t', index=False, encoding='utf-8')
    print('finished mining %d queries'%(len(mentions_mined)))
    
if __name__ == '__main__':
    main()