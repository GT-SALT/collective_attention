"""
Mine Twitter in streaming fashion for specified hashtags.
"""
from tweepy import API, OAuthHandler, Stream
from tweepy.streaming import StreamListener
from argparse import ArgumentParser
import pandas as pd
import gzip

class TweetListener(StreamListener):
    def __init__(self, file_name):
        self.file_name = file_name

    def on_data(self, data):
        try:
            with gzip.open(self.file_name, 'a') as f:
                data_str = '%s\n'%(data)
                f.write(data)
                return True
        except BaseException, e:
            print("error on data: %s"%(e))
        return True

    def on_error(self, status):
        print(status)
        return True

MAX_HASHTAGS=400
def main():
    parser = ArgumentParser()
    parser.add_argument('hashtags', nargs='+')
    parser.add_argument('--auth_file')
    parser.add_argument('--out_file')
    args = parser.parse_args()
    hashtags = args.hashtags
    auth_file = args.auth_file
    out_file = args.out_file
    if(len(hashtags) > MAX_HASHTAGS):
        hashtags = hashtags[:MAX_HASHTAGS]
    # extract authentication from file and load
    auth_info = pd.read_csv(auth_file, header=None, index_col=0)
    consumer_key = auth_info.loc['consumer_key', 1]
    consumer_secret = auth_info.loc['consumer_secret', 1]
    access_token = auth_info.loc['access_token', 1]
    access_secret = auth_info.loc['access_secret', 1]
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = API(auth)
    # start stream
    listener = TweetListener(out_file)
    twitter_stream = Stream(auth, listener)
    hashtags_to_track = map(lambda h: '#%s'%(h), hashtags)
    twitter_stream.filter(track=hashtags_to_track)
    
if __name__ == '__main__':
    main()
