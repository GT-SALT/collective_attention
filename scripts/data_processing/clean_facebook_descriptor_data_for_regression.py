"""
Clean Facebook descriptor data for regression.
"""
from argparse import ArgumentParser
import logging
import os
import pandas as pd
import numpy as np
from data_helpers import get_text_with_no_context

def main():
    parser = ArgumentParser()
#     parser.add_argument('--group_data', default='../../data/facebook-maria/combined_group_data_es_tagged_valid_anchor_group_contain.tsv')
    parser.add_argument('--group_data', default='../../data/facebook-maria/combined_group_data_es_tagged_parsed_spacy_anchor_group_contain.tsv')
    parser.add_argument('--out_dir', default='../../data/facebook-maria/')
    args = vars(parser.parse_args())
    logging_file = '../../output/clean_facebook_descriptor_data_for_regression.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    ## load data
    group_data = pd.read_csv(args['group_data'], sep='\t', index_col=False)
    # importance: NE frequency
    NE_var = 'NE_fixed'
    NE_counts = group_data.groupby(NE_var).apply(lambda x: x.shape[0]).reset_index().rename(columns={0 : 'NE_count'})
    group_data = pd.merge(group_data, NE_counts, on=NE_var)
    # information: post length
    group_data = get_text_with_no_context(group_data)
    txt_var = 'status_message'
    no_context_txt_var = '%s_no_context'%(txt_var)
    group_data = group_data.assign(**{
        'txt_len_norm' : group_data.loc[:, no_context_txt_var].apply(lambda x: np.log(len(x)+1))
    })
    # audience: number of posts per author per group
    author_var = 'status_author_id'
    group_var = 'group_name'
    author_group_counts = group_data.groupby([author_var, group_var]).apply(lambda x: x.shape[0]).reset_index().rename(columns={0 : 'author_group_count'})
    group_data = pd.merge(group_data, author_group_counts, on=[author_var, group_var])
    # audience: group size
    group_counts = group_data.groupby(group_var).apply(lambda x: x.loc[:, author_var].nunique()).reset_index().rename(columns={0 : 'group_size'})
    group_data = pd.merge(group_data, group_counts, on=group_var)
    
    ## restrict to consistently-posting authors, NEs, groups
    min_author_count = 20
    min_NE_count = 20
    min_group_count = 20
    author_var = 'status_author_id'
    NE_var = 'NE_fixed'
    group_var = 'group_name'
    min_counts = [min_author_count, min_NE_count, min_group_count]
    cat_vars = ['status_author_id', 'NE_fixed', 'group_name']
    for cat_var, min_count in zip(cat_vars, min_counts):
        cat_counts = group_data.loc[:, cat_var].value_counts()
        group_data = group_data.assign(**{
            '%s_cap'%(cat_var) : group_data.loc[:, cat_var].apply(lambda x: 'RARE' if cat_counts.loc[x] < min_count else x)
        })
    
    ## name cleanup
    group_data.rename(columns={'status_id' : 'id', 'status_author_id' : 'username'}, inplace=True)
    
    ## write to file
    out_file = os.path.join(args['out_dir'], 'combined_group_data_clean_regression.tsv')
#     out_file = os.path.join(args['out_dir'], 'combined_group_data_spacy_clean_regression.tsv')
    if(not os.path.exists(out_file)):
        group_data.to_csv(out_file, sep='\t', index=False)

if __name__ == '__main__':
    main()