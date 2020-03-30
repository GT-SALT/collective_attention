"""
Clean user data for classification via demographer.
Mainly convert from TSV to JSON.
"""
from argparse import ArgumentParser
import logging
import os
import dateutil
from datetime import datetime
import pandas as pd
from data_helpers import fix_timezone

def main():
    parser = ArgumentParser()
    # Twitter mine
#     parser.add_argument('--user_data_file', default='../../data/mined_tweets/tweet_user_data/user_data_twitter_mine.gz')
    # archive mine
    parser.add_argument('--user_data_file', default='../../data/mined_tweets/tweet_user_data/user_data_archive.gz')
    args = vars(parser.parse_args())
    logging_file = '../../output/clean_user_data_for_classification.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)

    ## load data
    user_data = pd.read_csv(args['user_data_file'], sep='\t', index_col=False, converters={'created_at' : dateutil.parser.parse})
    # fix timezones
    user_data = user_data.assign(**{
        'created_at' : user_data.loc[:, 'created_at'].apply(lambda x: fix_timezone(x))
    })
    user_data.fillna('', inplace=True)
    author_var = 'username'
    date_var = 'date'
    user_meta_vars = ['location', 'description', 'friends_count', 'followers_count', 'created_at', 'statuses_count', 'name', 'verified']
    user_data.sort_values([author_var, date_var], inplace=True, ascending=True)
    user_dedup = user_data.drop_duplicates(author_var).loc[:, user_meta_vars + [author_var]]
    logging.debug('loaded %d users'%(user_data.shape[0]))
    
    ## convert to JSON
    classify_cols = ['location', 'description', 'username', 'friends_count', 'followers_count', 'created_at', 'statuses_count', 'verified', 'name', 'listed_count']
    author_var = 'username'
    classify_data = user_dedup.drop_duplicates(author_var, inplace=False).loc[:, classify_cols].rename(columns={author_var : 'screen_name'})
    # fill in missing data with 0
    classify_data.fillna(0, inplace=True)
#     classify_data = classify_data.assign(**{k : 0. for k in classify_null_cols})
    # remove users with invalid creation date
    max_create_date_str = '10-31-2018'
    max_create_date = datetime.strptime(max_create_date_str, '%m-%d-%Y')
    max_create_date = fix_timezone(max_create_date)
    classify_data = classify_data[classify_data.loc[:, 'created_at'] <= max_create_date]
    # fix date format
    classify_data = classify_data.assign(**{'created_at' : classify_data.loc[:, 'created_at'].apply(lambda x: datetime.strftime(x, '%a %b %d %H:%M:%S +0000 %Y'))})
    # classify_data = classify_data.assign(**{'name' : classify_data.loc[:, 'screen_name'].apply(lambda x: x.replace('_', ' '))})
    classify_data_json = classify_data.apply(lambda x: x.to_json(), axis=1).values
    classify_input_file = args['user_data_file'].replace('.gz', '_json.txt')
    with open(classify_input_file, 'w') as classify_input:
        classify_input.write('\n'.join(classify_data_json))
    
if __name__ == '__main__':
    main()