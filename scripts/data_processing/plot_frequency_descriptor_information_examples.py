"""
Plot example trajectories for frequency and 
descriptor probability.
"""
from argparse import ArgumentParser
import logging
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import dateutil
from datetime import datetime, timedelta
from data_helpers import fix_timezone, round_to_day, assign_peak_date
from dateutil import parser as date_parser
from math import ceil, floor

def generate_date_range(data, inter_date_days=1):
    date_buffer_bins = int((data.max() - data.min()).days / inter_date_days)
    date_range = [data.min() + timedelta(days=(i*inter_date_days)) for i in range(date_buffer_bins+1)]
    return date_range

def add_zero_counts(counts_data, date_var='date_day'):
    NE_var = 'NE_fixed'
    data_name_var = 'data_name_fixed'
    date_range_var = 'data_range'
    counts_zeros = []
    for NE_i, data_i in counts_data.groupby([NE_var, data_name_var]):
        date_range_N = len(data_i.loc[:, date_range_var].values[0])
        data_i_zeros = pd.concat([pd.Series(np.repeat(data_i.loc[:, NE_var].iloc[0], date_range_N)),
                                  pd.Series(np.repeat(data_i.loc[:, data_name_var].iloc[0], date_range_N)),
                                  pd.Series(data_i.loc[:, date_range_var].values[0])], axis=1)
        data_i_zeros.columns = [NE_var, data_name_var, date_var]
        data_i = pd.merge(data_i, data_i_zeros, on=[NE_var, data_name_var, date_var], how='outer')
        data_i = data_i.drop(date_range_var, axis=1, inplace=False).fillna(0, inplace=False)
        counts_zeros.append(data_i)
    counts_zeros = pd.concat(counts_zeros, axis=0)
    return counts_zeros

def get_ylim_lower(ylim, buffer_pct):
    ylim_lower =  ylim[0]-(ylim[1]-ylim[0])*buffer_pct
    return ylim_lower

def plot_data(data_names_to_plot, NEs_to_plot, pre_peak_dates_all, post_peak_dates_all, count_data, context_data, data_date_ranges, date_var='date_day', days_per_timestep=1, out_dir='../../output/'):
    NE_var = 'NE_fixed'
    count_var = 'NE_count'
    context_pct_var = 'descriptor_pct'
    data_name_var = 'data_name_fixed'
    # plot params
    x_range_buffer_pct = 0.1
    title_font_size = 18
    label_font_size = 18
    tick_font_size = 14
    fig_height = 5
    fig_width = 10
    tick_date_fmt = '%d-%m-%Y'
    dates_per_plot = 6
    X_AXIS_BUFFER_PCT = 0.025

    for idx, (name_i, NE_j) in enumerate(zip(data_names_to_plot, NEs_to_plot)):
    ## get count/context data
        date_range_i = data_date_ranges.loc[name_i]
        x_range = [min(date_range_i), max(date_range_i)]
        x_lim_buffer = timedelta(days=((x_range[1]-x_range[0]).days * x_range_buffer_pct))
        x_lim = [min(date_range_i)-x_lim_buffer, max(date_range_i)+x_lim_buffer]
        counts_i = count_data[count_data.loc[:, data_name_var]==name_i].sort_values(date_var, inplace=False, ascending=True)
        context_pcts_i = context_data[context_data.loc[:, data_name_var]==name_i].sort_values(date_var, inplace=False, ascending=True)

        ## organize time, counts
        raw_counts_j = counts_i[counts_i.loc[:, NE_var]==NE_j].loc[:, count_var]
        # mark date with max frequency (for final plot)
        max_count_date_j = date_range_i[np.where(raw_counts_j==raw_counts_j.max())[0][0]]
        # limit x range to nonzero count dates
        min_date_j = date_range_i[min(np.where(raw_counts_j > 0.)[0])]
        max_date_j = date_range_i[max(np.where(raw_counts_j > 0.)[0])]
        # generate custom date range
        time_step_count = int(floor((max_date_j - min_date_j).days / days_per_timestep))
        date_range_j = [min_date_j + timedelta(days=k*days_per_timestep) for k in range(time_step_count + 1)]
        # restrict counts to valid dates
        # raw counts
        counts_j = counts_i[counts_i.loc[:, NE_var]==NE_j].loc[:, count_var]
        # log counts
        counts_i_valid_dates = counts_i[(counts_i.loc[:, date_var] >= min_date_j) & 
                                        (counts_i.loc[:, date_var] <= max_date_j)]
        context_pcts_i_valid_dates = context_pcts_i[(context_pcts_i.loc[:, date_var] >= min_date_j) & 
                                                    (context_pcts_i.loc[:, date_var] <= max_date_j)]
        raw_counts_j_valid = counts_i_valid_dates[counts_i_valid_dates.loc[:, NE_var]==NE_j].loc[:, count_var]
        counts_j = np.log(raw_counts_j_valid+1)
        context_pcts_j = context_pcts_i_valid_dates[context_pcts_i_valid_dates.loc[:, NE_var]==NE_j].loc[:, context_pct_var]

        ## plot
        fig_height = 5
        fig_width = 12
        f, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        ax2 = ax1.twinx()
        # set axis tick sizes
        ax1.tick_params(axis='both', which='major', labelsize=tick_font_size)
        ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)
        # log counts
        ax1_line = ax1.plot(date_range_j, counts_j, color='k', label='Log frequency')
        ax2_line = ax2.plot(date_range_j, context_pcts_j, color='r', linestyle='-.', label='P(descriptor)')
        # vertical line for peak date
        plt.axvline(x=max_count_date_j, linestyle='--', color='k') # line from bottom to top of plot
        ax1.set_ylabel('Log frequency', fontsize=label_font_size)
        ax2.set_ylabel('P(descriptor)', rotation=270, labelpad=18, fontsize=label_font_size)
        plt.title('"%s" time series'%(NE_j), fontsize=title_font_size)

        ## set legend, ticks for multiple axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1+lines2, labels1+labels2, fontsize=tick_font_size)
        # add timeline
        plt.axhline(y=min(context_pcts_j), color='k')
        # set x-ticks so they actually fit ;_;
        date_chunks = int(len(date_range_j) / dates_per_plot)
        x_ticks = date_range_j[::(date_chunks+1)]
        ax1.set_xticks(x_ticks)
#         ax1.set_xticklabels([date_k.strftime(tick_date_fmt) for date_k in x_ticks], rotation=45) # rotated ticks
        ax1.set_xticklabels([date_k.strftime(tick_date_fmt) for date_k in x_ticks], size=12) # small ticks

        ## align y axes
        ylim_1 = ax1.get_ylim()
        ylim_2 = ax2.get_ylim()
        ax1.set_ylim([get_ylim_lower(ylim_1, X_AXIS_BUFFER_PCT), ylim_1[1]])
        ax2.set_ylim([get_ylim_lower(ylim_2, X_AXIS_BUFFER_PCT), ylim_2[1]])
        pre_peak_dates_i = list(map(date_parser.parse, pre_peak_dates_all[idx]))
        post_peak_dates_i = list(map(date_parser.parse, post_peak_dates_all[idx]))

        ax2.scatter(pre_peak_dates_i, [0,]*len(pre_peak_dates_i), color='k')
        ax2.scatter(post_peak_dates_i, [0,]*len(post_peak_dates_i), color='k')
        
        # save to file
        out_file = os.path.join(out_dir, f'example_frequency_descriptor_plot_{NE_j}_days={days_per_timestep}.pdf')
        plt.savefig(out_file)

def main():
    parser = ArgumentParser()
    parser.add_argument('context_data') # '../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor.gz'
    parser.add_argument('--data_names_to_plot', nargs='+')
    parser.add_argument('--NEs_to_plot', nargs='+')
    parser.add_argument('--out_dir', default='../../output/')
    args = vars(parser.parse_args())
    logging_file = '../../output/plot_frequency_descriptor_information_examples.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    context_data_file = args['context_data']
    descriptor_data = pd.read_csv(context_data_file, sep='\t', index_col=False, compression='gzip', converters={'date' : dateutil.parser.parse})
    # fix names
    descriptor_data.rename(columns={'anchor' : 'descriptor'}, inplace=True)
    
    ## clean data
    # fix bad usernames
    descriptor_data = descriptor_data.assign(**{'username' : descriptor_data.loc[:, 'username'].apply(lambda x: x.split(':')[-1])})
    # fix date timezone problem
    date_var = 'date'
    descriptor_data = descriptor_data.assign(**{date_var : descriptor_data.loc[:, date_var].apply(lambda x: fix_timezone(x))})
    ## add rounded time var
    round_date_var = '%s_day'%(date_var)
    descriptor_data = descriptor_data.assign(**{round_date_var : descriptor_data.loc[:, date_var].apply(lambda x: round_to_day(x, round_day=1))})
    # also do weekly because daily data is too spiky
    week_date_var = '%s_week'%(date_var)
    descriptor_data = descriptor_data.assign(**{week_date_var : descriptor_data.loc[:, date_var].apply(lambda x: round_to_day(x, round_day=7))})
    # compute peak times per-NE
    NE_var = 'NE_fixed'
    round_date_var = 'date_day'
    data_name_var = 'data_name_fixed'
    NE_counts = descriptor_data.groupby([NE_var, data_name_var, round_date_var]).apply(lambda x: x.shape[0]).reset_index().rename(columns={0 : 'NE_count'})
    NE_week_counts = descriptor_data.groupby([NE_var, data_name_var, week_date_var]).apply(lambda x: x.shape[0]).reset_index().rename(columns={0 : 'NE_count'})

    ## limit to data with consistent NEs
    # restrict to NEs that occur on at least k dates
    # compute peaks
    count_var = 'NE_count'
    NE_count_peaks = NE_counts.groupby([data_name_var, NE_var]).apply(lambda x: assign_peak_date(x, count_var, date_var=round_date_var)).reset_index().rename(columns={0 : 'peak_date'})
    descriptor_data = pd.merge(descriptor_data, NE_count_peaks, on=[NE_var, data_name_var], how='inner')
    NE_week_count_peaks = NE_week_counts.groupby([data_name_var, NE_var]).apply(lambda x: assign_peak_date(x, count_var, date_var=week_date_var)).reset_index().rename(columns={0 : 'peak_date'})
    week_descriptor_data = pd.merge(descriptor_data, NE_week_count_peaks, on=[NE_var, data_name_var], how='inner')
    
    data_name_var = 'data_name_fixed'
    round_date_var = 'date_day'
    inter_date_days = 1
    week_inter_date_days = 7
    data_date_ranges = descriptor_data.groupby(data_name_var).apply(lambda x: generate_date_range(x.loc[:, round_date_var], inter_date_days=inter_date_days))
    week_data_date_ranges = week_descriptor_data.groupby(data_name_var).apply(lambda x: generate_date_range(x.loc[:, week_date_var], inter_date_days=week_inter_date_days))
    
    NE_counts_date_ranges = pd.merge(NE_counts, data_date_ranges.reset_index().rename(columns={0 : 'data_range'}), on=data_name_var)
    NE_week_counts_date_ranges = pd.merge(NE_week_counts, week_data_date_ranges.reset_index().rename(columns={0 : 'data_range'}), on=data_name_var)
    
    date_range_var = 'data_range'
    NE_counts_zeros = add_zero_counts(NE_counts_date_ranges, date_var='date_day')
    NE_week_counts_zeros = add_zero_counts(NE_week_counts_date_ranges, date_var='date_week')
    
    context_var = 'descriptor'
    context_pct_var = '%s_pct'%(context_var)
    NE_context_pcts = descriptor_data.groupby([NE_var, data_name_var, round_date_var]).apply(lambda x: x.loc[:, context_var].mean()).reset_index().rename(columns={0 : context_pct_var})
    NE_week_context_pcts = week_descriptor_data.groupby([NE_var, data_name_var, week_date_var]).apply(lambda x: x.loc[:, context_var].mean()).reset_index().rename(columns={0 : context_pct_var})
    NE_context_pcts_date_ranges = pd.merge(NE_context_pcts, data_date_ranges.reset_index().rename(columns={0 : date_range_var}), on=data_name_var)
    NE_week_context_pcts_date_ranges = pd.merge(NE_week_context_pcts, week_data_date_ranges.reset_index().rename(columns={0 : date_range_var}), on=data_name_var)
    
    NE_context_zeros = add_zero_counts(NE_context_pcts_date_ranges, date_var=round_date_var)
    NE_week_context_zeros = add_zero_counts(NE_week_context_pcts_date_ranges, date_var=week_date_var)
    
    out_dir = args['out_dir']
    data_names_to_plot = args['data_names_to_plot']
    NEs_to_plot = args['NEs_to_plot']
    # dates to plot small dots for example tweets
    pre_peak_dates_all = [
    [ '2017-09-18', '2017-09-17', ],
    [ '2018-09-12', '2018-09-13', ]
    ]

    post_peak_dates_all = [
        [ '2017-10-01', '2017-10-04', ],
        [ '2018-09-15', '2018-09-20', ]
    ]
    
    ## plot
    # per-day
    plot_data(data_names_to_plot, NEs_to_plot, pre_peak_dates_all, post_peak_dates_all, NE_counts_zeros, NE_context_zeros, data_date_ranges, date_var=round_date_var, days_per_timestep=1)
    # per-week
    plot_data(data_names_to_plot, NEs_to_plot, pre_peak_dates_all, post_peak_dates_all, NE_week_counts_zeros, NE_week_context_zeros, week_data_date_ranges, date_var=week_date_var, days_per_timestep=7)
    
if __name__ == '__main__':
    main()