"""
In which we rehydrate tweet IDs and store
in separate file.
"""
from __future__ import division
import gzip
import tweepy
from argparse import ArgumentParser
from time import time, sleep
import json

# seconds to wait before querying again after
# getting a rate limit
RATE_LIMIT_DELAY = 15
def rehydrate_statuses(api, id_file, out_file):
    max_status_count = 100
    status_list = []
    start_time = time()
#     relevant_keys = ['text', 'id', 'favorite_count', 'place', 'geo', 'lang', 'created_at', 'retweet_count', 'screen_name', '']
    with gzip.open(out_file, 'w') as output_file:
        for i, l in enumerate(gzip.open(id_file, 'r')):
            l_id = int(l.strip())
            status_list.append(l_id)
            if(len(status_list) == max_status_count):
                lookup_success = False
                while(not lookup_success):
                    try:
                        statuses_rehydrated = api.statuses_lookup(status_list)
                        lookup_success = True
                    except tweepy.error.RateLimitError, e:
                        # wait for rate limit to go away
                        print('rate limited at %d tweets'%(i))
                        sleep(RATE_LIMIT_DELAY)
                    except tweepy.error.TweepError, e:
                        # unknown internal error...maybe if we wait it will go away
                        print('got internal error at %d tweets'%(i))
                        sleep(RATE_LIMIT_DELAY)
                # TODO: limit status to keys of interest to save space??
                for s in statuses_rehydrated:
                    status_str = json.dumps(s._json)
                    output_file.write('%s\n'%(status_str))
                # reset status list
                status_list = []
            if(i % 100000 == 0):
                print("processed %d IDs"%(i))
        if(len(status_list) > 0):
            for s in status_list:
                output_file.write('%s\n'%(s))
    end_time = time()
    elapsed_time = end_time - start_time
    print("finished writing to %s"%(out_file))
    print("finished mining in %.3f"%(elapsed_time))
    sec_per_tweet = elapsed_time / i
    print('average %.3E seconds per tweet'%(sec_per_tweet))

def main():
    parser = ArgumentParser()
    parser.add_argument('id_file')
    parser.add_argument('auth_file')
    parser.add_argument('out_file')
    args = parser.parse_args()
    id_file = args.id_file
    auth_file = args.auth_file
    out_file = args.out_file
    # load auth data
    auth_lines = [x.strip().split(',') for x in open(auth_file, 'r')]
    auth_info = {k : v for k,v in auth_lines}
    # set up API
    auth_handler = tweepy.OAuthHandler(auth_info['consumer_key'], auth_info['consumer_secret'])
    auth_handler.set_access_token(auth_info['access_token'], auth_info['access_secret'])
    api = tweepy.API(auth_handler)
    # start rehydrating
    rehydrate_statuses(api, id_file, out_file)
    
if __name__ == '__main__':
    main()