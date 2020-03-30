"""
Combine author metadata for easier data 
cleaning during regression.
"""
import pandas as pd
from argparse import ArgumentParser
import logging
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--org_meta_data_files', type=list, default=['../../data/mined_tweets/tweet_user_data/user_data_archive_labelled.tsv', '../../data/mined_tweets/tweet_user_data/user_data_twitter_mine_labelled.tsv'])
    parser.add_argument('--loc_meta_data_files', type=list, default=['../../data/mined_tweets/tweet_user_data/user_data_local.tsv'])
    parser.add_argument('--out_dir', default='../../data/mined_tweets/tweet_user_data/')
    args = vars(parser.parse_args())
    logging_file = '../../output/combine_author_meta_data.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)

    ## load data
    org_meta_data = pd.concat([pd.read_csv(meta_file, sep='\t', index_col=False) for meta_file in args['org_meta_data_files']], axis=0)
    local_meta_data = pd.concat([pd.read_csv(meta_file, sep='\t', index_col=False) for meta_file in args['loc_meta_data_files']], axis=0)
    # normalize local var => convert to int
    local_var = 'is_local'
    local_meta_data = local_meta_data.assign(**{
        local_var : local_meta_data.loc[:, local_var].apply(lambda x: 1 if x=='True' else 0 if x=='False' else x)
    })
    
    # drop duplicate authors, but only the non-archived authors (less accurate ORG info)
    author_var = 'username'
    org_meta_data.drop_duplicates(author_var, keep='first', inplace=True)
    
    ## combine
    org_var = 'organization'
    org_cols = [org_var] + [author_var]
    local_var = 'is_local'
    data_name_var = 'data_name_fixed'
    local_cols = [local_var] + [author_var, data_name_var]
    meta_data = pd.merge(org_meta_data.loc[:, org_cols], local_meta_data.loc[:, local_cols], on=author_var, how='outer')
    # check for NAs
    logging.debug('%d NA rows'%(meta_data[meta_data.isnull().max(axis=1)].shape[0]))
    # fill rows with NA location
    meta_null_val = -1
    meta_data = meta_data.assign(**{
        local_var : meta_data.loc[:, local_var].fillna(meta_null_val, inplace=False)
    })
    
    ## save
    meta_data_file = os.path.join(args['out_dir'], 'user_meta_data_clean.tsv')
    meta_data.to_csv(meta_data_file, sep='\t', index=False)
    
if __name__ == '__main__':
    main()