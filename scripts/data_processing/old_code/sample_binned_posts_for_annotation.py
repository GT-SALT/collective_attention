"""
First, bin all Facebook posts by 
(1) group (based on relative post volume high/medium/low), and
(2) time (based on relative time early/mid/late). (bins generated in scripts/data_processing/generate_post_volume_time_bins.py)
Next, sample posts from each bin with equal frequency (if possible)
to achieve even sample.
"""
from __future__ import division
import pandas as pd
from argparse import ArgumentParser
from data_helpers import load_combined_group_data, load_group_location_data
from dateutil import parser as date_parser
from math import floor
import os

pd.np.random.seed(123)
def main():
    parser = ArgumentParser()
    parser.add_argument('--group_bin_file', default='../../data/facebook-maria/group_post_volume_bins.tsv')
    parser.add_argument('--time_bin_file', default='../../data/facebook-maria/post_time_bins.tsv')
    parser.add_argument('--out_dir', default='../../data/facebook-maria/')
    parser.add_argument('--sample_size', type=int, default=1000)
    args = parser.parse_args()
    group_bin_file = args.group_bin_file
    time_bin_file = args.time_bin_file
    out_dir = args.out_dir
    sample_size = args.sample_size
    
    ## load data
    post_data = load_combined_group_data()
    location_data = load_group_location_data()
    group_bins = pd.read_csv(group_bin_file, sep='\t', index_col=False, encoding='utf-8')
    time_bins = pd.read_csv(time_bin_file, sep='\t', index_col=False, encoding='utf-8')
    # combine with locations
    post_data = pd.merge(post_data, location_data.loc[:, ['location_name', 'group_id']], on='group_id', how='outer')
    # restrict to Spanish data
    post_data = post_data[post_data.loc[:, 'status_lang'] == 'es']
    time_bins.loc[:, 'start'] = time_bins.loc[:, 'start'].apply(lambda x: date_parser.parse(x))
    time_bins.loc[:, 'end'] = time_bins.loc[:, 'end'].apply(lambda x: date_parser.parse(x))
    
    ## add group and time bins
    post_data = pd.merge(post_data, group_bins.loc[:, ['location_name', 'post_relative_volume']], on='location_name', how='outer')
    post_data.rename(columns={'post_relative_volume' : 'group_bin'}, inplace=True)
    post_data.sort_values('status_published', inplace=False, ascending=True)
    post_data_fixed = pd.DataFrame()
    for idx, time_row in time_bins.iterrows():
        start, end, time_period = time_row.values.tolist()
        if(idx == 0):
            post_data_relevant = post_data[(post_data.loc[:, 'status_published'] >= start) & 
                                           (post_data.loc[:, 'status_published'] <= end)]
        else:
            post_data_relevant = post_data[(post_data.loc[:, 'status_published'] > start) & 
                                           (post_data.loc[:, 'status_published'] <= end)]
        post_data_relevant.loc[:, 'time_bin'] = time_period
        post_data_fixed = post_data_fixed.append(post_data_relevant)
    post_data = post_data_fixed

    ## sample!! 
    # organize bin values
    group_bin_vals = group_bins.loc[:, 'post_relative_volume'].unique().tolist()
    time_bin_vals = time_bins.loc[:, 'time_period'].unique().tolist()
    bins = len(group_bin_vals) * len(time_bin_vals)
    posts_per_bin = sample_size / bins
    bin_splits = [floor(posts_per_bin*i) for i in range(bins+1)]
    sampled_posts = pd.DataFrame()
    ctr = 0 
    for group_bin in group_bin_vals:
        for time_bin in time_bin_vals:
            bin_size = int(bin_splits[ctr+1] - bin_splits[ctr])
            binned_data = post_data[(post_data.loc[:, 'group_bin'] == group_bin) &
                                    (post_data.loc[:, 'time_bin'] == time_bin)]
            print('%d values in bin size=%d for group=%s, time=%s'%(binned_data.shape[0], bin_size, group_bin, time_bin))
            sample_binned_data = binned_data.loc[pd.np.random.choice(binned_data.index, replace=False, size=bin_size), :]
            sampled_posts = sampled_posts.append(sample_binned_data)
            ctr += 1
            
    ## write to file!
    out_file = os.path.join(out_dir, 'volume_time_binned_post_sample.tsv')
    sampled_posts.to_csv(out_file, sep='\t', index=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()