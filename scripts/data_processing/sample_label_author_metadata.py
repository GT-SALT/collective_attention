"""
Sample and label author metadata for:
(1) local/non-local
(2) organization/non-organization
"""
from argparse import ArgumentParser
import logging
import os
import numpy as np
import pandas as pd
from data_helpers import DATA_NAME_STATES_SHORT_LOOKUP
import dateutil

np.random.seed(123)

def score_labels(data, label_var='organization', gold_var='organization_gold'):
    # score precision/recall on meta labels
    tp_data = (data.loc[:, label_var] - data.loc[:, gold_var]==0) & (data.loc[:, label_var]==1)
    fp_data = (data.loc[:, label_var] - data.loc[:, gold_var] == 1)
    fn_data = (data.loc[:, label_var] - data.loc[:, gold_var] == -1)
    tp = tp_data.sum()
    fp = fp_data.sum()
    fn = fn_data.sum()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return tp_data, fp_data, fn_data, prec, rec

def main():
    parser = ArgumentParser()
    parser.add_argument('--archive_data_file', default='../../data/mined_tweets/tweet_user_data/user_data_archive.gz')
    parser.add_argument('--author_data_prelabelled', default='../../data/mined_tweets/tweet_user_data/user_data_media_label_clean.tsv') # generated from notebook scripts/data_processing/classify_authors_by_metadata.ipynb
    parser.add_argument('--sample_size', type=int, default=100)
    # metadata classify results
    # local: classify_locals.py
    # organization: classify_org.sh, clean_user_data_from_classification.py
    # combine: combine_author_meta_data.py
    parser.add_argument('--meta_classify_data_file', default='../../data/mined_tweets/tweet_user_data/user_meta_data_clean.tsv')
    args = vars(parser.parse_args())
    logging_file = '../../output/sample_label_author_metadata.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)

    ## load data
    author_data = pd.read_csv(args['archive_data_file'], sep='\t', index_col=False, compression='gzip', converters={'date' : dateutil.parser.parse, 'created_at' : dateutil.parser.parse})
    author_data.fillna('', inplace=True)
    author_var = 'username'
    data_name_var = 'data_name_fixed'
    date_var = 'date'
    author_meta_vars = ['location', 'description', 'followers_count', 'friends_count', 'created_at']
    author_data.sort_values([author_var, date_var], inplace=True, ascending=True)
    author_data_dedup = author_data.drop_duplicates([author_var, data_name_var]).loc[:, author_meta_vars + [author_var, data_name_var]]
    logging.debug('sampling %d/%d authors'%(author_data_dedup.loc[:, data_name_var].nunique() * args['sample_size'], author_data_dedup.shape[0]))
    
    ## if we haven't already sampled and labelled,
    ## sample S authors from each data set to label
    ## otherwise load the labelled data and restrict
    ## to the relevant authors
    if(args['author_data_prelabelled'] is None):
        author_label_data = author_data_dedup.groupby(data_name_var).apply(lambda x: x.loc[np.random.choice(x.index, args['sample_size'], replace=False), :]).reset_index(drop=True)
    else:
        author_prelabeled_data = pd.read_csv(args['author_data_prelabelled'], sep='\t', index_col=False)
        label_vars = ['organization', 'local']
        label_vars = list(set(label_vars) & set(author_prelabeled_data.columns))
        author_label_data = pd.merge(author_data_dedup, author_prelabeled_data.loc[:, [author_var, data_name_var] + label_vars], on=[author_var, data_name_var], how='inner')
        print('loaded %d/%d authors to label'%(author_label_data.shape[0], author_data_dedup.shape[0]))
    
    ## for local classification: need list of affected states for each event
    data_names = author_label_data.loc[:, data_name_var].unique()
    affected_states = pd.DataFrame([[data_name, DATA_NAME_STATES_SHORT_LOOKUP[data_name]] for data_name in data_names], columns=[data_name_var, 'affected_states'])
    author_label_data = pd.merge(author_label_data, affected_states, on=data_name_var)
    
    ## write to file
    out_file = args['archive_data_file'].replace('.gz', '_sample.tsv')
    # add dummy values for label vars
    label_vars = ['local', 'organization']
    label_vars_to_label = list(set(label_vars) - set(author_label_data.columns))
    author_label_data = author_label_data.assign(**{
        label_var : -1
        for label_var in label_vars_to_label
    })
    if(not os.path.exists(out_file)):
        author_label_data.to_csv(out_file, sep='\t', index=False)
        
    ## after labelling, detect local, organization
    ## compute precision, recall
    full_label_author_out_file = args['archive_data_file'].replace('.gz', '_sample_labelled.tsv')
    full_label_author_data = pd.read_csv(full_label_author_out_file, sep='\t', index_col=False)
    full_label_author_data.rename(columns={label_var : '%s_gold'%(label_var) for label_var in label_vars}, inplace=True)
    # load metadata classify labels
    meta_classify_data = pd.read_csv(args['meta_classify_data_file'], sep='\t', index_col=False).rename(columns={'is_local':'local'}, inplace=False)
    full_label_author_data = pd.merge(full_label_author_data, meta_classify_data, on=[author_var, data_name_var], how='inner')  
    ## precision/recall
    for label_var in label_vars:
        tp_data, fp_data, fn_data, prec, rec = score_labels(full_label_author_data, label_var=label_var, gold_var='%s_gold'%(label_var))
        logging.debug('var=%s, prec=%.3f, rec=%.3f'%(label_var, prec, rec))
    
if __name__ == '__main__':
    main()