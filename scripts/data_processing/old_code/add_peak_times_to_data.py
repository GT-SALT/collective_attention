"""
Add peak times to anchor data:
(1) one peak per dataset
(2) one peak per NE per dataset
"""
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import logging
import os
import dateutil
from datetime import datetime, timedelta
def fix_timezone(x, date_fmt='%Y-%m-%d %H:%M:%S', timezone_str='+0000'):
    # add timezone offset for "naive" dates
    if(x.utcoffset() is None):
        x = datetime.strptime('%s%s'%(x.strftime(date_fmt), timezone_str), '%s%%z'%(date_fmt))
    return x

def round_to_day(x):
    # TODO: make this more flexible? ex. 8 hour bins
    x_day = datetime(day=x.day, month=x.month, year=x.year)
    return x_day

def compute_peak_time(data, date_var='date_day', data_name_var='data_name_fixed', verbose=True):
    """
    Compute peak frequency time for data. 
    We use the max frequency time as the peak.
    TODO: find peak based on frequency that falls outside standard deviation
    """
    data_freq = data.groupby(date_var).apply(lambda x: x.shape[0])
    if(verbose):
        logging.debug('data %s'%(data.loc[:, data_name_var].iloc[0]))
        logging.debug(data_freq)
    peak_time = data_freq.sort_values(inplace=False, ascending=False).index[0]
    return peak_time

def main():
    parser = ArgumentParser()
    parser.add_argument('--full_data', default='../../data/mined_tweets/combined_tweet_tag_data.gz')
    parser.add_argument('--anchor_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor.gz')
    args = vars(parser.parse_args())

    logging_file = '../../output/add_peak_times_to_data.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    ## load data
    full_data = pd.read_csv(args['full_data'], sep='\t', index_col=False, compression='gzip', converters={'date' : dateutil.parser.parse})
    anchor_data = pd.read_csv(args['anchor_data'], sep='\t', index_col=False, compression='gzip', converters={'date' : dateutil.parser.parse})
    # fix bad usernames
    anchor_data = anchor_data.assign(**{'username' : anchor_data.loc[:, 'username'].apply(lambda x: x.split(':')[-1])})
    # fix timezones
    date_var = 'date'
    full_data = full_data.assign(**{date_var : full_data.loc[:, date_var].apply(lambda x: fix_timezone(x))})
    anchor_data = anchor_data.assign(**{date_var : anchor_data.loc[:, date_var].apply(lambda x: fix_timezone(x))})
    # round time to nearest day
    round_date_var = '%s_day'%(date_var)
    full_data = full_data.assign(**{round_date_var : full_data.loc[:, date_var].apply(lambda x: round_to_day(x))})
    anchor_data = anchor_data.assign(**{round_date_var : anchor_data.loc[:, date_var].apply(lambda x: round_to_day(x))})
    
    ## add peak times
    data_name_var = 'data_name_fixed'
    data_peak_times = full_data.groupby(data_name_var).apply(lambda x: compute_peak_time(x, date_var=round_date_var, data_name_var=data_name_var))
    logging.debug('data peak times\n%s'%(data_peak_times))
    anchor_data = anchor_data.assign(**{'data_peak_date' : anchor_data.loc[:, 'data_name_fixed'].apply(lambda x: data_peak_times.loc[x])})
    NE_var = 'NE_fixed'
    min_NE_count = 20
    anchor_data_NE_peak = []
    for name_i, data_i in anchor_data.groupby(data_name_var):
        NE_counts = data_i.loc[:, NE_var].value_counts()
        NE_consistent = NE_counts[NE_counts >= min_NE_count].index.tolist()
        data_i_NE = data_i[data_i.loc[:, NE_var].isin(NE_consistent)]
        data_i_NE_peak_times = data_i_NE.groupby(NE_var).apply(lambda x: compute_peak_time(x, date_var=round_date_var, verbose=False))
        logging.debug('data=%s, peak times per-NE=\n%s'%((name_i, data_i_NE_peak_times)))
        data_i_NE_peak_times = pd.DataFrame(data_i_NE_peak_times).rename(columns={0:'NE_peak_date'})
        data_i_NE_peak_times = data_i_NE_peak_times.assign(**{'NE_fixed' : data_i_NE_peak_times.index})
        data_i_NE_peak_times.index = np.arange(data_i_NE_peak_times.shape[0])
        logging.debug(data_i_NE_peak_times.head())
        data_i = pd.merge(data_i, data_i_NE_peak_times, on='NE_fixed', how='inner')
        anchor_data_NE_peak.append(data_i)
    anchor_data_NE_peak = pd.concat(anchor_data_NE_peak, axis=0)
    logging.debug('%d/%d data retained'%(anchor_data_NE_peak.shape[0], anchor_data.shape[0]))
    ## assign binary before/after peak values
    data_name_var = 'data_name_fixed'
    peak_var = 'data_peak_date'
    NE_peak_var = 'NE_peak_date'
    logging.debug('date var has type %s'%(type(anchor_data_NE_peak.loc[:, date_var].iloc[0])))
    logging.debug('round date var has type %s'%(type(anchor_data_NE_peak.loc[:, round_date_var].iloc[0])))
    logging.debug('peak var has type %s'%(type(anchor_data_NE_peak.loc[:, peak_var].iloc[0])))
    logging.debug('NE peak var has type %s'%(type(anchor_data_NE_peak.loc[:, NE_peak_var].iloc[0])))
    # TODO: try different date buffers for NE peaks
    date_buffer = timedelta(days=1)
    # per-data peaks 
    anchor_data_NE_peak = anchor_data_NE_peak.assign(**{'pre_peak' : (anchor_data_NE_peak.loc[:, round_date_var] <= anchor_data_NE_peak.loc[:, peak_var] - date_buffer).astype(int)})
    anchor_data_NE_peak = anchor_data_NE_peak.assign(**{'post_peak' : (anchor_data_NE_peak.loc[:, round_date_var] >= anchor_data_NE_peak.loc[:, peak_var] + date_buffer).astype(int)})
    # per-NE peaks
    anchor_data_NE_peak = anchor_data_NE_peak.assign(**{'pre_peak_NE' : (anchor_data_NE_peak.loc[:, round_date_var] <= anchor_data_NE_peak.loc[:, NE_peak_var] - date_buffer).astype(int)})
    anchor_data_NE_peak = anchor_data_NE_peak.assign(**{'post_peak_NE' : (anchor_data_NE_peak.loc[:, round_date_var] >= anchor_data_NE_peak.loc[:, NE_peak_var] + date_buffer).astype(int)})
    
    ## write to file
    out_file_name = args['anchor_data'].replace('.gz', '_peak_times.gz')
    if(not os.path.exists(out_file_name)):
        anchor_data_NE_peak.to_csv(out_file_name, sep='\t', index=False, compression='gzip')
    
if __name__ == '__main__':
    main()