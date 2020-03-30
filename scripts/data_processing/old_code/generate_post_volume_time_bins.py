"""
Assign a bin to each post based on:
(1) relative post volume of the group in which the post was made ("high", "medium" or "low" volume group)
(2) relative time of the post (initial post-storm, short-term, long-term)

These bins are based on cumulative percentages.
"""
from __future__ import division
from argparse import ArgumentParser
import os
from datetime import datetime
from dateutil import parser as date_parser
import pandas as pd
import numpy as np

def round_to_min(x):
    return x.round('min')

START = datetime(1970,1,1)
def convert_time(x):
    return (x.to_datetime() - START).total_seconds() * 1e9

def main():
    parser = ArgumentParser()
    parser.add_argument('--post_file', default='../../data/facebook-maria/combined_group_data.tsv')
    parser.add_argument('--group_location_file', default='../../data/facebook-maria/location_group_data.tsv')
    parser.add_argument('--out_dir', default='../../data/facebook-maria/')
    args = parser.parse_args()
    post_file = args.post_file
    location_file = args.group_location_file
    out_dir = args.out_dir
    
    ## load data
    post_data = pd.read_csv(post_file, sep='\t', index_col=False, encoding='utf-8')
    # convert dates 
    post_data.loc[:, 'status_published'] = post_data.loc[:, 'status_published'].apply(lambda x: date_parser.parse(x))
    location_data = pd.read_csv(location_file, sep='\t', index_col=False, encoding='utf-8')
    post_data = pd.merge(post_data, location_data.loc[:, ['location_name', 'group_id']], on='group_id', how='outer')
    # restrict to Spanish because that's ultimately what we're going to sample from
    post_data = post_data[post_data.loc[:, 'status_lang'] == 'es']
    
    ## compute split points
    group_count_splits = 3
    group_counts = post_data.loc[:, 'location_name'].value_counts().sort_values(inplace=False, ascending=True)
    split_pcts = np.linspace(0., 1., group_count_splits+1)
    # cumulative percent
    group_cumulative_pct = group_counts.cumsum() / group_counts.sum()
    cutoff_labels = ['Low', 'Medium', 'High']
    # map group counts to cutoff group
    group_cutoff_data = pd.DataFrame()
    for i in range(group_count_splits):
        split_lower = split_pcts[i]
        split_upper = split_pcts[i+1]
        cutoff_label = cutoff_labels[i]
        if(i == 0):
            relevant_data = group_counts[(group_cumulative_pct >= split_lower) &
                                         (group_cumulative_pct <= split_upper)]
        else:
            relevant_data = group_counts[(group_cumulative_pct > split_lower) &
                                         (group_cumulative_pct <= split_upper)]
        relevant_df = pd.DataFrame(relevant_data)
        relevant_df.columns = ['post_count']
        relevant_df.loc[:, 'location_name'] = relevant_df.index
        relevant_df.loc[:, 'post_relative_volume'] = cutoff_label
        group_cutoff_data = group_cutoff_data.append(relevant_df)

    ## same thing but with post times
    post_data.loc[:, 'status_published_min'] = post_data.loc[:, 'status_published'].apply(round_to_min)
    start_time = post_data.loc[:, 'status_published_min'].min()
    end_time = post_data.loc[:, 'status_published_min'].max()
    time_full = pd.date_range(start_time, end_time, freq='min')
    time_counts = post_data.loc[:, 'status_published_min'].value_counts()
    time_counts = time_counts.loc[time_counts.index.sort_values()]
    time_cumulative_pcts = time_counts.cumsum() / time_counts.sum()
    time_full_pcts = pd.Series(pd.np.zeros(len(time_full)), index=time_full)
    time_full_pcts.loc[time_cumulative_pcts.index] = time_cumulative_pcts.values
    # fill up the zeroes
    time_full_pcts = time_full_pcts.replace(to_replace=0, method='ffill')
    # split by time
    time_splits = 3
    time_labels = ['Early', 'Mid', 'Late']
    split_pcts = np.linspace(0., 1., time_splits+1)
    time_cutoff_data = pd.DataFrame()
    for i in range(time_splits):
        split_lower = split_pcts[i]
        split_upper = split_pcts[i+1]
        time_label = time_labels[i]
        if(i == 0):
            cutoff_lower = time_full_pcts.index[np.where(time_full_pcts >= split_lower)[0][0]]
        else:
            cutoff_lower = time_full_pcts.index[np.where(time_full_pcts > split_lower)[0][0]]
        cutoff_upper = time_full_pcts.index[np.where(time_full_pcts <= split_upper)[0][-1]]
        cutoff_df = pd.DataFrame(pd.Series([cutoff_lower, cutoff_upper, time_label])).transpose()
        cutoff_df.columns = ['start', 'end', 'time_period']
        time_cutoff_data = time_cutoff_data.append(cutoff_df)

    ## write to file
    group_out_file = os.path.join(out_dir, 'group_post_volume_bins.tsv')
    group_cutoff_data.to_csv(group_out_file, sep='\t', index=False, encoding='utf-8')
    time_out_file = os.path.join(out_dir, 'post_time_bins.tsv')
    time_cutoff_data.to_csv(time_out_file, sep='\t', index=False)

if __name__ == '__main__':
    main()