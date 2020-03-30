"""
Clean user data after ORG/non-ORG classification.
"""
from argparse import ArgumentParser
import logging
import os
import json
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('--user_label_file', default='../../data/mined_tweets/tweet_user_data/user_data_twitter_mine_json.txt.labelled')
    args = vars(parser.parse_args())
    logging_file = '../../output/output.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    ## load data
    org_detect_output = [json.loads(x) for x in open(args['user_label_file'], 'r')]
    # convert to dataframe
    label_cols = ['screen_name']
    org_data_label = pd.DataFrame([[x[c] for c in label_cols] + [x['demographics']['indorg']['value'], x['demographics']['indorg']['scores'][x['demographics']['indorg']['value']]] for x in org_detect_output])
    org_data_label.columns = label_cols + ['label', 'label_score']
    org_data_label = org_data_label.assign(**{'organization' : (org_data_label.loc[:, 'label']=='org').astype(int)})
    org_data_label = org_data_label.rename(columns={'screen_name': 'username'})
    
    ## write to file
    out_data_file_name = args['user_label_file'].replace('_json.txt.labelled', '_labelled.tsv')
    org_data_label.to_csv(out_data_file_name, sep='\t', index=False)

if __name__ == '__main__':
    main()