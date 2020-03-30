"""
Clean raw anchor data for regression:

1. add peak times for all NEs
2. restrict to consistent NEs (at least D unique dates)
3. restrict to consistent authors (at least P unique posts, otherwise RARE_AUTHOR)
4. optional: add metadata
"""
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import logging
import os
import dateutil
from datetime import datetime, timedelta
import re
from data_helpers import assign_relative_peak_time_vars, compute_post_length, round_to_day, assign_peak_date, shift_dates, fix_timezone, add_post_length

def compute_shift_counts(data, date_var='date_day', prior_shift=1, count_var='post_count_prior', aggregate_var=None, null_val=0.):
    """
    Compute and shift date counts (ex. number of posts at t-1).
    
    :param data: post DataFrame
    :param date_var: date variable
    :param prior_shift: N days to shift
    :param count_var: count variable
    :param aggregate_var: optional variable to aggregate at each date (ex. mean likes at t-1)
    :return data: shifted data
    """
    data.sort_values(date_var, ascending=True, inplace=True)
    if(aggregate_var is None):
        date_counts = data.loc[:, date_var].value_counts().sort_index()
    else:
        date_counts = data.groupby(date_var).apply(lambda x: x.loc[:, aggregate_var].mean())
    date_range = (data.loc[:, date_var].max() - data.loc[:, date_var].min()).days
    date_range = [data.loc[:, date_var].min() + timedelta(days=x) for x in range(date_range)]
    date_counts = date_counts.loc[date_range].fillna(0, inplace=False)
    # shift dates forward
    date_counts_idx = date_counts.index[prior_shift:]
    date_counts_shift = date_counts.iloc[:-prior_shift]
    date_counts_shift.index = date_counts_idx
    # add values for null dates (the first N dates)
    date_counts_shift = date_counts_shift.append(pd.Series([null_val]*prior_shift, index=date_counts.index[:prior_shift]))
    date_counts_shift.sort_index(inplace=True)
    # reorganize for merge
    date_counts_shift = pd.DataFrame(date_counts_shift).reset_index().rename(columns={'index' : date_var, 0 : count_var})
    # add back to original data
    data = pd.merge(data, date_counts_shift, on=date_var)
    return data

def main():
    parser = ArgumentParser()
#     parser.add_argument('--full_data_file', default='../../data/mined_tweets/combined_tweet_tag_data.gz')
    parser.add_argument('--full_data_file', default='../../data/mined_tweets/combined_tweet_tag_data.gz')
    parser.add_argument('--anchor_data_file', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor.gz')
    # clean text for image/video feature
    parser.add_argument('--clean_txt_file', default='../../data/mined_tweets/combined_tweet_clean_status.gz') # from extract_clean_status_txt_from_raw_data.py
    # user metadata for ORG/non-ORG, local/non-local
    parser.add_argument('--author_meta_file', default='../../data/mined_tweets/tweet_user_data/user_meta_data_clean.tsv')
    parser.add_argument('--NE_min_count', type=int, default=5)
    parser.add_argument('--peak_date_buffer', type=float, default=1) # number of days before/after peak to consider pre/post
    # author metadata about posting exclusively in a time period (or across all time periods)
    parser.add_argument('--include_author_meta_time_period', default=False)
#     parser.add_argument('--include_author_meta_time_period', default=True)
    args = vars(parser.parse_args())
    logging_file = '../../output/clean_anchor_data_for_regression.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    ## collect data
    # load full data
    full_data = pd.read_csv(args['full_data_file'], sep='\t', index_col=False, compression='gzip', converters={'date' : dateutil.parser.parse})
    # load anchor data
    anchor_data = pd.read_csv(args['anchor_data_file'], sep='\t', index_col=False, compression='gzip', converters={'date' : dateutil.parser.parse})
    # fix bad usernames
    anchor_data = anchor_data.assign(**{'username' : anchor_data.loc[:, 'username'].apply(lambda x: x.split(':')[-1])})
    # fix date timezone problem
    date_var = 'date'
    full_data = full_data.assign(**{date_var : full_data.loc[:, date_var].apply(lambda x: fix_timezone(x))})
    anchor_data = anchor_data.assign(**{date_var : anchor_data.loc[:, date_var].apply(lambda x: fix_timezone(x))})
    if(args.get('clean_txt_file') is not None):
        clean_txt_data = pd.read_csv(args['clean_txt_file'], sep='\t', index_col=False, compression='gzip', header=None, names=['txt', 'id'], dtype={'id' : int})
    ## add rounded time var
    round_date_var = '%s_day'%(date_var)
    full_data = full_data.assign(**{round_date_var : full_data.loc[:, date_var].apply(lambda x: round_to_day(x))})
    anchor_data = anchor_data.assign(**{round_date_var : anchor_data.loc[:, date_var].apply(lambda x: round_to_day(x))})
    # compute peak times per-NE
    NE_var = 'NE_fixed'
    round_date_var = 'date_day'
    data_name_var = 'data_name_fixed'
    NE_counts = anchor_data.groupby([NE_var, data_name_var, round_date_var]).apply(lambda x: x.shape[0]).reset_index().rename(columns={0 : 'NE_count'})
    id_var = 'id'
    doc_counts = full_data.groupby([data_name_var, round_date_var]).apply(lambda x: x.loc[:, id_var].nunique()).reset_index().rename(columns={0 : 'doc_count'})
    NE_doc_counts = pd.merge(NE_counts, doc_counts, on=[data_name_var, round_date_var])
    NE_doc_counts = NE_doc_counts.assign(**{'NE_freq' : NE_doc_counts.loc[:, 'NE_count'] / NE_doc_counts.loc[:, 'doc_count']})
    
    ## limit to data with consistent NEs
    # restrict to NEs that occur on at least k dates
    NE_unique_date_counts = NE_doc_counts.groupby([data_name_var, NE_var]).apply(lambda x: x.loc[:, round_date_var].nunique())
    NE_consistent = NE_unique_date_counts[NE_unique_date_counts >= args['NE_min_count']].reset_index().loc[:, [data_name_var, NE_var]]
    NE_doc_counts_consistent = pd.merge(NE_doc_counts, NE_consistent, on=[data_name_var, NE_var], how='inner')
    # compute peaks
    count_var = 'NE_count'
#     count_var = 'NE_freq'
    NE_count_peaks = NE_doc_counts_consistent.groupby([data_name_var, NE_var]).apply(lambda x: assign_peak_date(x, count_var, date_var=round_date_var)).reset_index().rename(columns={0 : 'peak_date'})
    anchor_data_consistent = pd.merge(anchor_data, NE_count_peaks, on=[NE_var, data_name_var], how='inner')
    logging.debug('%d/%d consistent data'%(anchor_data_consistent.shape[0], anchor_data.shape[0]))

    ## assign peak times
    anchor_data_consistent = assign_relative_peak_time_vars(anchor_data_consistent, args['peak_date_buffer'])
    logging.debug('%d data with peak time'%(anchor_data_consistent.shape[0]))
    
    ## compute prior post frequency 
    prior_post_shift = 1
    post_null = 0.
    full_data_prior_posts = []
    count_var = 'post_count_prior'
    for name_i, data_i in full_data.groupby(data_name_var):
        data_i = compute_shift_counts(data_i, date_var=round_date_var, prior_shift=prior_post_shift, count_var=count_var, null_val=post_null)
        full_data_prior_posts.append(data_i)
    full_data_prior_posts = pd.concat(full_data_prior_posts, axis=0)
    # deduplicate
    full_data_prior_posts.drop_duplicates([data_name_var, round_date_var, count_var], inplace=True)
    logging.debug('%d/%d data with prior post counts'%(full_data_prior_posts.shape[0], full_data.shape[0]))
    anchor_data_consistent = pd.merge(anchor_data_consistent, full_data_prior_posts.loc[:, [data_name_var, round_date_var, count_var]], on=[data_name_var, round_date_var], how='left')
    logging.debug('%d data with prior post counts'%(anchor_data_consistent.shape[0]))
    
    ## compute prior NE frequency (instead of peak frequency, use f_{t-1} to predict context)
    prior_freq_shift = 1
    freq_null = 0.
    anchor_data_prior_freq = []
    count_var = 'NE_count_prior'
    for (name_i, NE_i), data_i in anchor_data_consistent.groupby([data_name_var, NE_var]):
        data_i = compute_shift_counts(data_i, date_var=round_date_var, prior_shift=prior_freq_shift, count_var=count_var, null_val=freq_null)
        anchor_data_prior_freq.append(data_i)
    ## TODO: does this get rid of data without prior frequency counts?
    anchor_data_prior_freq = pd.concat(anchor_data_prior_freq, axis=0)
    logging.debug('%d/%d data with prior freq'%(anchor_data_prior_freq.shape[0], anchor_data_consistent.shape[0]))
    anchor_data_consistent = anchor_data_prior_freq.copy()
    logging.debug('%d data with prior freq'%(anchor_data_consistent.shape[0]))
    
    ## "regular" author status: determine if author is <= 50% of activity for each event
    # TODO: why does this drop ~7K rows?? we have missing authors??
    full_data_regular_authors = []
    data_name_var = 'data_name_fixed'
    author_var = 'username'
    id_var = 'id'
    regular_author_var = 'regular_author'
    regular_cutoff_pct = 95
    for name_i, data_i in full_data.groupby(data_name_var):
#         author_counts_i = data_i.drop_duplicates([author_var, id_var], inplace=False).loc[:, author_var].value_counts()
        # log-transform to reduce skew
        author_counts_i = np.log(data_i.drop_duplicates([author_var, id_var], inplace=False).loc[:, author_var].value_counts())
        # remove authors with min count? (extreme skew)
#         filter_author_counts_i = author_counts_i[author_counts_i]
        regular_cutoff_count_i = np.percentile(author_counts_i, regular_cutoff_pct)
        regular_authors_i = (author_counts_i <= regular_cutoff_count_i).astype(int).reset_index(name=regular_author_var).rename(columns={'index' : author_var})
        data_i = pd.merge(data_i, regular_authors_i, on=author_var)
        full_data_regular_authors.append(data_i)
    full_data_regular_authors = pd.concat(full_data_regular_authors, axis=0)
    full_data_regular_authors_dedup = full_data_regular_authors.drop_duplicates([data_name_var, author_var], inplace=False)
    logging.debug('%d/%d regular authors'%(full_data_regular_authors_dedup.loc[:, regular_author_var].sum(), full_data_regular_authors.shape[0]))
    # add to anchor data
    anchor_data_consistent = pd.merge(anchor_data_consistent, full_data_regular_authors_dedup.loc[:, [data_name_var, author_var, regular_author_var]], on=[data_name_var, author_var], how='left')
    
    ## fix rare authors
    rare_author_var_val = 'RARE_AUTHOR'
    author_var = 'username'
    data_name_var = 'data_name_fixed'
    min_author_count = 5
    anchor_data_author_counts = anchor_data_consistent.groupby([data_name_var, author_var]).apply(lambda x: x.shape[0]).reset_index().rename(columns={0 : 'author_count'})
    anchor_data_consistent_authors_fixed = pd.merge(anchor_data_consistent, anchor_data_author_counts, on=[data_name_var, author_var])
    anchor_data_consistent_authors_fixed = anchor_data_consistent_authors_fixed.assign(**{
        author_var : anchor_data_consistent_authors_fixed.apply(lambda x: x.loc[author_var] if x.loc['author_count'] >= min_author_count else rare_author_var_val, axis=1)
    })
    logging.debug('%d data with consistent authors'%(anchor_data_consistent_authors_fixed.shape[0]))
    
    ## add URL feature
    url_matcher = re.compile('<URL>')
    txt_var = 'txt'
    anchor_data_consistent_authors_fixed = anchor_data_consistent_authors_fixed.assign(**{'has_URL' : anchor_data_consistent_authors_fixed.loc[:, txt_var].apply(lambda x: int(url_matcher.search(x) is not None))})
    
    ## add URL content features: contains image/video?
    if(args.get('clean_txt_file') is not None):
        image_video_url_matcher = re.compile('pic.twitter.com/|instagram.com/')
        image_video_url_var = 'image_video_URL'
        clean_txt_data = clean_txt_data.assign(**{image_video_url_var : clean_txt_data.loc[:, 'txt'].apply(lambda x: int(image_video_url_matcher.search(x) is not None))})
        clean_txt_data = clean_txt_data[clean_txt_data.loc[:, id_var].isin(anchor_data_consistent_authors_fixed.loc[:, id_var].unique())]
        anchor_data_consistent_authors_fixed = pd.merge(anchor_data_consistent_authors_fixed, clean_txt_data.loc[:, [id_var, image_video_url_var]], on=id_var, how='inner')
        logging.debug('%d clean text data'%(anchor_data_consistent_authors_fixed.shape[0]))
        
    ## add character length of post without context
    post_len_bins = 11
    anchor_data_consistent_authors_fixed = compute_post_length(anchor_data_consistent_authors_fixed, bins=post_len_bins)
    logging.debug('%d post length data'%(anchor_data_consistent_authors_fixed.shape[0]))
    
    ## add author metadata
    if(args['author_meta_file'] is not None):
        author_meta_data = pd.read_csv(args['author_meta_file'], sep='\t', index_col=False)
        author_var = 'username'
        data_name_var = 'data_name_fixed'
        # clean author names for merging
        author_meta_data = author_meta_data.assign(**{author_var : author_meta_data.loc[:, author_var].apply(lambda x: x.lower())})
        anchor_data_consistent_authors_fixed = anchor_data_consistent_authors_fixed.assign(**{author_var : anchor_data_consistent_authors_fixed.loc[:, author_var].apply(lambda x: x.lower())})
        # merge
        logging.debug('metadata for %d authors'%(author_meta_data.shape[0]))
        meta_data_vars = ['organization', 'is_local']
        anchor_data_meta = pd.merge(anchor_data_consistent_authors_fixed, author_meta_data.loc[:, [author_var, data_name_var] + meta_data_vars], on=[author_var, data_name_var], how='inner')
        logging.debug('%d/%d authors in combined meta_data+context_data'%(anchor_data_meta.loc[:, author_var].nunique(), anchor_data_consistent_authors_fixed.loc[:, author_var].nunique()))
        anchor_data_consistent_authors_fixed = anchor_data_meta.copy()
    
    ## TODO: mark authors by time period:
    ## do they post in all time periods or only during one?
    ## this is kind of cheating so don't do this for the main regression
#     if(args['include_author_meta_time_period']):
#         author_var = 'username'
#         data_name_var = 'data_name_fixed'
#         author_data_var = '%s_%s'%(author_var, data_name_var)
#         anchor_data_consistent_authors_fixed = anchor_data_consistent_authors_fixed.assign(**{
#             author_data_var : anchor_data_consistent_authors_fixed.loc[:, author_var] + '_' + anchor_data_consistent_authors_fixed.loc[:, data_name_var]
#         })
#         anchor_data_author_time_period = []
#         author_time_period_var = 'author_time_period_group'
#         def assign_shared_time_period(data, time_periods=['pre_peak', 'during_peak', 'post_peak']):
#             for time_period in time_periods:
#                 if(data.loc[time_period]==1):
#                     return '%s_shared'%(time_period)
#             return ''
#         for author_i, data_i in anchor_data_consistent_authors_fixed.groupby(author_data_var):
#             author_time_period_i = 'shared'
#             if(data_i.loc[:, 'pre_peak'].sum() == data_i.shape[0]):
#                 author_time_period_i = 'pre_peak_only'
#             elif(data_i.loc[:, 'during_peak'].sum() == data_i.shape[0]):
#                 author_time_period_i = 'during_peak_only'
#             elif(data_i.loc[:, 'post_peak'].sum() == data_i.shape[0]):
#                 author_time_period_i = 'post_peak_only'
#             # if not one of the above categories, assign "shared" within time period
#             else:
#                 author_time_period_i = data_i.apply(lambda x: assign_shared_time_period(x), axis=1)
#             data_i = data_i.assign(**{
#                 author_time_period_var : author_time_period_i
#             })
#             anchor_data_author_time_period.append(data_i)
#         anchor_data_author_time_period = pd.concat(anchor_data_author_time_period, axis=0)
#         logging.debug('author time period group counts:\n%s'%(anchor_data_author_time_period.loc[:, author_time_period_var].value_counts()))
    
    ## save to file
    ## TODO: remove irrelevant columns?? saves on memory and is cleaner
    out_file = args['anchor_data_file'].replace('.gz', '_NE_peak_times_consistent_authors.gz')
    if(not os.path.exists(out_file)):
        anchor_data_consistent_authors_fixed.to_csv(out_file, sep='\t', index=False, compression='gzip')
    
if __name__ == '__main__':
    main()