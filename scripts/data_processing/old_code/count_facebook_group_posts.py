"""
Count number of posts in specified FB groups.
"""
from argparse import ArgumentParser
import pandas as pd
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--group_file', default='../../data/facebook-maria/location_group_data.tsv')
    parser.add_argument('--start_end_dates', nargs='+', default=['2017-09-20', '2017-10-20'])
    args = parser.parse_args()
    group_file = args.group_file
    start_date, end_date = args.start_end_dates
    
    ## load data
    group_data = pd.read_csv(group_file, sep='\t', index_col=False)
    data_dir = os.path.dirname(group_file)
    
    ## load each file and count number of statuses collected
    group_id_list = group_data.loc[:, 'group_id'].unique().tolist()
    group_counts = []
    for g in group_id_list:
        group_post_file_name = os.path.join(data_dir, '%s_%s_%s_facebook_posts.tsv'%(g, start_date, end_date))
        g_df = pd.read_csv(group_post_file_name, sep='\t', index_col=False)
        g_count = g_df.shape[0]
        group_counts.append([g, g_count])
    group_counts = pd.DataFrame(group_counts, columns=['group_id', 'post_count'])

    ## write to file
    out_file = os.path.join(data_dir, 'group_counts.tsv')
    group_counts.to_csv(out_file, sep='\t', index=False)

if __name__ == '__main__':
    main()
