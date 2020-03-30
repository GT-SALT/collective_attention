"""
Clean regular author anchor data for regression:

1. add peak times for all NEs
2. restrict to consistent NEs (at least D unique dates)
3. add prior posts, entity counts, engagement
"""
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import logging
import os
import dateutil
from datetime import datetime, timedelta
import re
from data_helpers import assign_relative_peak_time_vars, compute_post_length, round_to_day, assign_peak_date, shift_dates, fix_timezone
from sklearn.preprocessing import StandardScaler

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

def compute_data_change(data, change_var, date_var='date_day', period_shift=1, data_cols=['username', 'data_name_fixed']):
    """
    Compute change in variable between dates, for a given set of data (ex. per-author per-dataset).
    """
    data_dates = data.loc[:, date_var].sort_values(inplace=False).drop_duplicates(inplace=False)
    # compute per-date mean of change var
    data_sum = data.groupby(date_var).apply(lambda x: x.loc[:, change_var].mean()).reset_index().rename(columns={0 : change_var}).sort_values(date_var, inplace=False, ascending=True)
    # compute difference
    data_change = data_sum.loc[:, change_var].diff(periods=period_shift).dropna(inplace=False)
    # shift date index
    data_change.index = data_dates[period_shift:]
    shift_change_var = '%s_change'%(change_var)
    data_change = data_change.reset_index().rename(columns={'index' : date_var, change_var : shift_change_var})
    # add extra data values (ex. author)
    if(len(data_cols) > 0):
        data_change = data_change.assign(**{
            data_col : data.loc[:, data_col].iloc[0]
            for data_col in data_cols
        })
    return data_change

def main():
    parser = ArgumentParser()
    parser.add_argument('--full_data_file', default='../../data/mined_tweets/combined_tweet_tag_data.gz')
    parser.add_argument('--anchor_data_file', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor.gz')
    parser.add_argument('--clean_txt_file', default='../../data/mined_tweets/combined_tweet_clean_status.gz') # from extract_clean_status_txt_from_raw_data.py
    # user metadata for ORG/non-ORG, local/non-local
    parser.add_argument('--author_meta_file', default='../../data/mined_tweets/tweet_user_data/user_meta_data_clean.tsv')
    parser.add_argument('--NE_min_count', type=int, default=5) # number of unique dates on which an NE must occur to be counted
    parser.add_argument('--peak_date_buffer', type=float, default=1) # number of days before/after peak to consider pre/post
    args = vars(parser.parse_args())
    logging_file = '../../output/clean_regular_author_anchor_data_for_regression.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    ## load/clean data
    # full data (for peaks)
    full_data = pd.read_csv(args['full_data_file'], sep='\t', index_col=False, compression='gzip', converters={'date' : dateutil.parser.parse})
    # anchor data (for peaks)
    anchor_data = pd.read_csv(args['anchor_data_file'], sep='\t', index_col=False, compression='gzip', converters={'date' : dateutil.parser.parse})
    # clean text for URL info
    if(args.get('clean_txt_file') is not None):
        clean_txt_data = pd.read_csv(args['clean_txt_file'], sep='\t', index_col=False, compression='gzip', header=None, names=['txt', 'id'], dtype={'id' : int})
        logging.debug('%d NA rows'%(clean_txt_data[clean_txt_data.loc[:, 'txt'].apply(lambda x: np.isnan(x) if type(x) is not str else False)].shape[0]))
        clean_txt_data.fillna('', inplace=True)
    id_var = 'id'
    
    ## clean up user, time data
    # fix bad usernames
    anchor_data = anchor_data.assign(**{'username' : anchor_data.loc[:, 'username'].apply(lambda x: x.split(':')[-1])})
    # fix timezones
    date_var = 'date'
    full_data = full_data.assign(**{date_var : full_data.loc[:, date_var].apply(lambda x: fix_timezone(x))})
    anchor_data = anchor_data.assign(**{date_var : anchor_data.loc[:, date_var].apply(lambda x: fix_timezone(x))})
    # add time var
    round_date_var = '%s_day'%(date_var)
    full_data = full_data.assign(**{round_date_var : full_data.loc[:, date_var].apply(lambda x: round_to_day(x))})
    anchor_data = anchor_data.assign(**{round_date_var : anchor_data.loc[:, date_var].apply(lambda x: round_to_day(x))}) 

    ## peak times
    # compute peak times per-NE
    NE_var = 'NE_fixed'
    round_date_var = 'date_day'
    data_name_var = 'data_name_fixed'
    NE_counts = anchor_data.groupby([NE_var, data_name_var, round_date_var]).apply(lambda x: x.shape[0]).reset_index().rename(columns={0 : 'NE_count'})
    id_var = 'id'
    doc_counts = full_data.groupby([data_name_var, round_date_var]).apply(lambda x: x.loc[:, id_var].nunique()).reset_index().rename(columns={0 : 'doc_count'})
    NE_doc_counts = pd.merge(NE_counts, doc_counts, on=[data_name_var, round_date_var])
    NE_doc_counts = NE_doc_counts.assign(**{'NE_freq' : NE_doc_counts.loc[:, 'NE_count'] / NE_doc_counts.loc[:, 'doc_count']})
    
    ## compute author counts per dataset
    ## limit to authors <= 95th percentile
    regular_author_data_clean = []
    data_name_var = 'data_name_fixed'
    author_var = 'username'
    author_cutoff_pct = 95
    for name_i, data_i in anchor_data.groupby(data_name_var):
        author_counts_i = np.log(data_i.loc[:, author_var].value_counts())
        author_cutoff_i = np.percentile(author_counts_i, author_cutoff_pct)
        data_i = data_i[data_i.loc[:, author_var].apply(lambda x: author_counts_i.loc[x] <= author_cutoff_i)]
        regular_author_data_clean.append(data_i)
    regular_author_data_clean = pd.concat(regular_author_data_clean, axis=0)
    logging.debug('%d/%d regular authors'%(regular_author_data_clean.loc[:, author_var].nunique(), anchor_data.loc[:, author_var].nunique()))
    regular_author_data = regular_author_data_clean.copy()
    
    ## limit to data with consistent NEs
    ## add resulting peak data
    # restrict to NEs that occur on at least k dates
    NE_unique_date_counts = NE_doc_counts.groupby([data_name_var, NE_var]).apply(lambda x: x.loc[:, round_date_var].nunique())
    NE_consistent = NE_unique_date_counts[NE_unique_date_counts >= args['NE_min_count']].reset_index().loc[:, [data_name_var, NE_var]]
    NE_doc_counts_consistent = pd.merge(NE_doc_counts, NE_consistent, on=[data_name_var, NE_var], how='inner')
    count_var = 'NE_count'
#     count_var = 'NE_freq'
    NE_count_peaks = NE_doc_counts_consistent.groupby([data_name_var, NE_var]).apply(lambda x: assign_peak_date(x, count_var, date_var=round_date_var)).reset_index().rename(columns={0 : 'peak_date'})
    anchor_data_consistent = pd.merge(anchor_data, NE_count_peaks, on=[NE_var, data_name_var], how='inner')
    # merge with regular author data
    regular_author_data_consistent = pd.merge(regular_author_data, NE_count_peaks, on=[NE_var, data_name_var], how='inner')
    logging.debug('%d/%d consistent data'%(regular_author_data_consistent.shape[0], regular_author_data.shape[0]))
    
    ## assign peak times
    peak_relative_time_data = assign_relative_peak_time_vars(anchor_data_consistent, args['peak_date_buffer'])
    time_period_vars = ['pre_peak', 'during_peak', 'post_peak']
    logging.debug('about to merge peak data: N=%d'%(peak_relative_time_data.shape[0]))
    peak_relative_time_data = peak_relative_time_data.drop_duplicates([NE_var, data_name_var], inplace=False).loc[:, [NE_var, data_name_var] + time_period_vars]
    regular_author_data_consistent = pd.merge(regular_author_data_consistent, peak_relative_time_data, on=[NE_var, data_name_var], how='inner')
    logging.debug('%d regular author peak data'%(regular_author_data_consistent.shape[0]))
    ## remove authors who don't post during all time periods
    ## because we need to analyze per-individual change
#     author_var = 'username'
#     regular_author_data_consistent_time_periods = []
#     for author_i, data_i in regular_author_data_consistent.groupby(author_var):
#         time_periods_i = data_i.loc[:, time_period_vars].max(axis=0).sum()
#         if(time_periods_i == len(time_period_vars)):
#             regular_author_data_consistent_time_periods.append(data_i)
#         else:
#             logging.debug('removed %d data from author %s with %d time periods'%(data_i.shape[0], author_i, time_periods_i))
#     regular_author_data_consistent_time_periods = pd.concat(regular_author_data_consistent_time_periods, axis=0)
#     logging.debug('after removing authors with inconsistent time periods: %d/%d authors, %d/%d data retained'%
#                   (regular_author_data_consistent_time_periods.loc[:, author_var].nunique(), regular_author_data_consistent.loc[:, author_var].nunique(), regular_author_data_consistent_time_periods.shape[0], regular_author_data_consistent.shape[0]))
#     # per-dataset
#     data_name_var = 'data_name_fixed'
#     logging.debug('after removing authors with inconsistent time periods: per-data author counts\n%s'%
#                   (regular_author_data_consistent_time_periods.groupby(data_name_var).apply(lambda x: x.loc[:, author_var].nunique())))
#     regular_author_data_consistent = regular_author_data_consistent_time_periods.copy()
#     # sanity check: did we get rid of sub-optimal authors?
#     for author_i, data_i in regular_author_data_consistent.groupby(author_var):
#         time_periods_i = data_i.loc[:, time_period_vars].max(axis=0).sum()
#         if(time_periods_i < len(time_period_vars)):
#             print('after cleaning: author %s has %d time periods'%(author_i, time_periods_i))
    
    ## compute prior NE frequency (instead of peak frequency, use f_{t-1} to predict context)
    prior_freq_shift = 1
    count_null = 0.
    anchor_data_prior_freq = []
    data_name_var = 'data_name_fixed'
    NE_var = 'NE_fixed'
    prior_NE_count_var = 'NE_count_prior'
    for (name_i, NE_i), data_i in anchor_data_consistent.groupby([data_name_var, NE_var]):
        data_i = compute_shift_counts(data_i, date_var=round_date_var, prior_shift=prior_freq_shift, count_var=prior_NE_count_var, aggregate_var=None, null_val=count_null)
        anchor_data_prior_freq.append(data_i)
    anchor_data_prior_freq = pd.concat(anchor_data_prior_freq, axis=0)
    logging.debug('%d/%d data with prior freq'%(anchor_data_prior_freq.shape[0], anchor_data_consistent.shape[0]))
    # merge with power user data
    anchor_data_prior_freq.drop_duplicates([NE_var, data_name_var, round_date_var], inplace=True)
    regular_author_data_consistent = pd.merge(regular_author_data_consistent, anchor_data_prior_freq.loc[:, [NE_var, data_name_var, round_date_var, prior_NE_count_var]], on=[NE_var, data_name_var, round_date_var], how='inner')
    logging.debug('%d data with prior freq'%(regular_author_data_consistent.shape[0]))
    
    ## add prior author stats
    ## post count, NE count, engagement count
    ## TODO: get prior stats from full data (even non-NE tweets)!!
    # full data
    def get_roll_sum(data, sum_var):
        data_sum = data.loc[:, sum_var].value_counts().sort_index().cumsum()
        return data_sum
    # post counts
    author_var = 'username'
    post_count_var = 'post_count'
    author_post_counts = regular_author_data_consistent.groupby([data_name_var, author_var]).apply(lambda x: get_roll_sum(x, round_date_var)).reset_index(name=post_count_var).rename(columns={'level_2':round_date_var})
    # NE counts
    NE_count_var = 'NE_count'
    author_NE_counts = regular_author_data_consistent.groupby([data_name_var, author_var, NE_var]).apply(lambda x: get_roll_sum(x, round_date_var)).reset_index(name=NE_count_var).rename(columns={'level_3':round_date_var})
    logging.debug('author NE counts:\n%s'%(author_NE_counts.head()))
    ## TODO: engagement from full data
    # engagement
#     engagement_vars = ['favorites', 'retweets']
#     engagement_data_change = None
#     engagement_data = None
#     engagement_change_shift = 1
#     for engagement_var in engagement_vars:
#         prior_engagement_var = 'prior_%s'%(engagement_var)
#         engagement_counts = full_data.groupby([data_name_var, author_var, round_date_var]).apply(lambda x: x.loc[:, engagement_var].mean()).reset_index(name=prior_engagement_var)
#         engagement_counts_change = full_data.groupby([data_name_var, author_var]).apply(lambda x: compute_data_change(x, engagement_var, date_var=round_date_var, period_shift=engagement_change_shift, data_cols=[data_name_var, author_var]))
#         logging.debug('engagement counts change\n%s'%(engagement_counts_change.head()))
#         engagement_counts_change = engagement_counts_change.reset_index(drop=True)#.drop('level_2', axis=1, inplace=False)
#         engagement_counts_change.drop_duplicates([author_var, data_name_var, round_date_var], inplace=True)
#         logging.debug('cleaned engagement counts change\n%s'%(engagement_counts_change.head()))
#         if(engagement_data is None):
#             engagement_data = engagement_counts
#         else:
#             engagement_data = pd.merge(engagement_data, engagement_counts, on=[data_name_var, author_var, round_date_var], how='inner')
#         if(engagement_data_change is None):
#             engagement_data_change = engagement_counts_change
#         else:
#             engagement_data_change = pd.merge(engagement_data_change, engagement_counts_change, on=[data_name_var, author_var, round_date_var], how='inner')
#     logging.debug('engagement data:\n%s'%(engagement_data.head()))
#     logging.debug('engagement change data:\n%s'%(engagement_data_change.head()))
    # log-scale, Z-norm vars
#     scaler = StandardScaler()
#     engagement_data = engagement_data.assign(**{
#         'prior_%s'%(engagement_var) : scaler.fit_transform(np.log(engagement_data.loc[:, 'prior_%s'%(engagement_var)]+1.).values.reshape(-1,1))
#         for engagement_var in engagement_vars
#     })
#     engagement_data_change = engagement_data_change.assign(**{
#         '%s_change'%(engagement_var) : scaler.fit_transform(engagement_data_change.loc[:, '%s_change'%(engagement_var)].values.reshape(-1,1))
#         for engagement_var in engagement_vars
#     })
    # get sum over all engagement vars
#     engagement_data = engagement_data.assign(**{
#         'prior_engagement' : engagement_data.loc[:, ['prior_%s'%(engagement_var) for engagement_var in engagement_vars]].sum(axis=1)
#     })
#     engagement_data_change = engagement_data_change.assign(**{
#         'engagement_change' : engagement_data_change.loc[:, ['%s_change'%(engagement_var) for engagement_var in engagement_vars]].sum(axis=1)
#     })
    # fix NA vals
#     engagement_data.fillna(0, inplace=True)
#     engagement_data_change.fillna(0, inplace=True)
    
    ## add back to original data
    ## we need to shift all dates to the previous date
    date_shift = 1
    author_post_counts = author_post_counts.groupby([data_name_var, author_var]).apply(lambda x: shift_dates(x, date_var=round_date_var, date_shift=date_shift)).reset_index(drop=True)
    author_NE_counts = author_NE_counts.groupby([data_name_var, author_var, NE_var]).apply(lambda x: shift_dates(x, date_var=round_date_var, date_shift=date_shift)).reset_index(drop=True)
#     engagement_data = engagement_data.groupby([data_name_var, author_var]).apply(lambda x: shift_dates(x, date_var=round_date_var, date_shift=date_shift)).reset_index(drop=True)
    # fix shifted date column
    shift_date_var = '%s_shift'%(round_date_var)
    author_post_counts = author_post_counts.drop(round_date_var, axis=1, inplace=False).rename(columns={shift_date_var : round_date_var})
    author_NE_counts = author_NE_counts.drop(round_date_var, axis=1, inplace=False).rename(columns={shift_date_var : round_date_var})
#     engagement_data = engagement_data.drop(round_date_var, axis=1, inplace=False).rename(columns={shift_date_var : round_date_var})
    # merge with prior data
    regular_author_data_with_prior_stats = pd.merge(regular_author_data_consistent, author_post_counts, on=[data_name_var, author_var, round_date_var], how='inner')
    regular_author_data_with_prior_stats = pd.merge(regular_author_data_with_prior_stats, author_NE_counts, on=[data_name_var, author_var, round_date_var, NE_var], how='inner')
#     regular_author_data_with_prior_stats = pd.merge(regular_author_data_with_prior_stats, engagement_data, on=[data_name_var, author_var, round_date_var], how='inner')
    # add engagement change without shifting
#     regular_author_data_with_prior_stats = pd.merge(regular_author_data_with_prior_stats, engagement_data_change, on=[data_name_var, author_var, round_date_var], how='left')
    # fix NA values for engagement change vars
#     regular_author_data_with_prior_stats.fillna(value={'%s_change'%(engagement_var) : 0. for engagement_var in engagement_vars}, inplace=True)
    # deduplicate
    regular_author_data_with_prior_stats.drop_duplicates([id_var, NE_var], inplace=True)
    logging.debug('regular user data has cols %s'%(','.join(sorted(regular_author_data_with_prior_stats.columns))))
    logging.debug('regular user data with prior stats\n%s'%(regular_author_data_with_prior_stats.head()))
    # fix NA vals from engagement data merge
#     combined_engagement_vars = ['prior_engagement', 'engagement_change']
#     regular_author_data_with_prior_stats.fillna(value={v : 0. for v in combined_engagement_vars}, inplace=True)
    
    ## add URL feature
    url_matcher = re.compile('<URL>')
    regular_author_data_with_prior_stats = regular_author_data_with_prior_stats.assign(**{'has_URL' : regular_author_data_with_prior_stats.loc[:, 'txt'].apply(lambda x: int(url_matcher.search(x) is not None))})
    ## add URL content features: contains image/video?
    if(args.get('clean_txt_file') is not None):
        image_video_url_matcher = re.compile('pic.twitter.com/|instagram.com/')
        image_video_url_var = 'image_video_URL'
        clean_txt_data = clean_txt_data.assign(**{image_video_url_var : clean_txt_data.loc[:, 'txt'].apply(lambda x: int(image_video_url_matcher.search(x) is not None))})
        clean_txt_data = clean_txt_data[clean_txt_data.loc[:, id_var].isin(regular_author_data_with_prior_stats.loc[:, id_var].unique())]
        regular_author_data_with_prior_stats = pd.merge(regular_author_data_with_prior_stats, clean_txt_data.loc[:, [id_var, image_video_url_var]], on=id_var, how='inner')
    
    ## add character length of post without context
    post_len_bins = 11
    regular_author_data_with_prior_stats = compute_post_length(regular_author_data_with_prior_stats, bins=post_len_bins)
    
    ## add author metadata
    if(args['author_meta_file'] is not None):
        author_meta_data = pd.read_csv(args['author_meta_file'], sep='\t', index_col=False)
        author_var = 'username'
        data_name_var = 'data_name_fixed'
        # clean author names for merging
        author_meta_data = author_meta_data.assign(**{author_var : author_meta_data.loc[:, author_var].apply(lambda x: x.lower())})
        regular_author_data_with_prior_stats = regular_author_data_with_prior_stats.assign(**{author_var : regular_author_data_with_prior_stats.loc[:, author_var].apply(lambda x: x.lower())})
        # merge
        logging.debug('metadata for %d authors'%(author_meta_data.shape[0]))
        meta_data_vars = ['organization', 'is_local']
        anchor_data_meta = pd.merge(regular_author_data_with_prior_stats, author_meta_data.loc[:, [author_var, data_name_var] + meta_data_vars], on=[author_var, data_name_var], how='inner')
        logging.debug('%d/%d authors in combined meta_data+context_data'%(anchor_data_meta.loc[:, author_var].nunique(), regular_author_data_with_prior_stats.loc[:, author_var].nunique()))
        regular_author_data_with_prior_stats = anchor_data_meta.copy()
    
    ## cleaning spurious correlations: author/NE with 1 time period
#     regular_author_data_consistent_time_periods = []
#     for author_i, data_i in regular_author_data_with_prior_stats.groupby(author_var):
#         time_periods_i = data_i.loc[:, time_period_vars].max(axis=0).sum()
# #         if(time_periods_i == len(time_period_vars)):
#         if(time_periods_i == 1):
#             logging.debug('final cleaning: author %s has %d time periods'%(author_i, time_periods_i))
#         else:
#             regular_author_data_consistent_time_periods.append(data_i)
#     regular_author_data_with_prior_stats = pd.concat(regular_author_data_consistent_time_periods, axis=0)
#     regular_author_data_consistent_time_periods = []
#     for NE_i, data_i in regular_author_data_with_prior_stats.groupby(NE_var):
#         time_periods_i = data_i.loc[:, time_period_vars].max(axis=0).sum()
# #         if(time_periods_i == len(time_period_vars)):
#         if(time_periods_i == 1):
#             logging.debug('final cleaning: NE %s has %d time periods'%(NE_i, time_periods_i))
#         else:
#             regular_author_data_consistent_time_periods.append(data_i)
#     regular_author_data_with_prior_stats = pd.concat(regular_author_data_consistent_time_periods, axis=0)
    
    ## cleaning spurious correlations: author/NE combos
#     regular_author_data_consistent_author_NEs = []
#     for NE_i, data_i in regular_author_data_with_prior_stats.groupby(NE_var):
#         if(data_i.loc[:, author_var].nunique() == 1):
#             logging.debug('final cleaning: NE %s has %d unique authors'%(NE_i, data_i.loc[:, author_var].nunique()))
#         else:
#             regular_author_data_consistent_author_NEs.append(data_i)
#     regular_author_data_consistent_author_NEs = pd.concat(regular_author_data_consistent_author_NEs, axis=0)
#     regular_author_data_consistent_authors = []
#     for author_i, data_i in regular_author_data_consistent_author_NEs.groupby(author_var):
#         if(data_i.loc[:, NE_var].nunique() == 1):
#             logging.debug('final cleaning: author %s has %d unique NEs'%(author_i, data_i.loc[:, NE_var].nunique()))
#         else:
#             regular_author_data_consistent_authors.append(data_i)
#     regular_author_data_consistent_authors = pd.concat(regular_author_data_consistent_authors, axis=0)
#     regular_author_data_with_prior_stats = regular_author_data_consistent_authors.copy()
    
    ## cleaning spurious correlations: authors with constant scalar vars
#     scalar_vars = ['post_count', 'NE_count', 'prior_engagement', 'txt_len_norm', 'engagement_change']
#     regular_author_data_scalar_clean = []
#     err_min = 1e-3
#     for author_i, data_i in regular_author_data_with_prior_stats.groupby(author_var):
#         scalar_var_err = (data_i.loc[:, scalar_vars].std(axis=0) / data_i.loc[:, scalar_vars].mean(axis=0)).min()
#         if(scalar_var_err <= err_min):
#             logging.debug('final cleaning: author %s has ~0 err for scalar var'%(author_i))
#         else:
#             regular_author_data_scalar_clean.append(data_i)
#     regular_author_data_scalar_clean = pd.concat(regular_author_data_scalar_clean, axis=0)
#     regular_author_data_with_prior_stats = regular_author_data_scalar_clean.copy()
    
    ## save to file
    out_file = args['anchor_data_file'].replace('.gz', '_regular_authors.gz')
#     if(args['author_meta_file'] is not None):
#         out_file = args['anchor_data_file'].replace('.gz', '_prior_author_stats_with_meta.gz')
    if(not os.path.exists(out_file)):
        regular_author_data_with_prior_stats.to_csv(out_file, sep='\t', index=False, compression='gzip')
    
if __name__ == '__main__':
    main()