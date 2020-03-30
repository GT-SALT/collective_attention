"""
Mine Twitter archive file that is 
in .tsv format.
"""
from argparse import ArgumentParser
from datetime import datetime
import os
import re
from zipfile import ZipFile
import gzip

def process_status_line(x):
    x_lat, x_lon, x_id, x_user, x_status, x_time = x.strip().split('\t')
    x_lat = float(x_lat)
    x_lon = float(x_lon)
    x_id = int(x_id)
    x_time = datetime.strptime(x_time, '%a %b %d %H:%M:%S %Z %Y')
    return x_lat, x_lon, x_id, x_user, x_status, x_time

def contains_geo(lat, lon, geo_bounds):
    return lat >= geo_bounds[0][0] and lat <= geo_bounds[0][1] and lon >= geo_bounds[1][0] and lon <= geo_bounds[1][1]

def contains_time(time, date_range):
    return time >= date_range[0] and time <= date_range[1]

def contains_phrase(txt, matcher):
    return matcher.search(txt) is not None

def main():
    parser = ArgumentParser()
    parser.add_argument('--archive_file', default='../../data/mined_tweets/east_coast_geo_twitter_2018/east_coast_geo_twitter_2018_clean.zip')
    parser.add_argument('--phrases', default='.*')
#     parser.add_argument('--phrases', default='#HurricaneMichael,#Michael')
    # Michael bounds
#     parser.add_argument('--geo_bounds', default=[[24.7, 34.9], [-88.3, -75.3]]) # [[lat1, lat2], [lon1, lon2]]
    # Florence bounds
    parser.add_argument('--geo_bounds', default=[[31.0, 36.6], [-85.0, -76.6]])
    # Michael time
#     parser.add_argument('--date_range', default=['07-10-18 00:00:00', '23-10-18 23:59:59'])
    # Florence time
    parser.add_argument('--date_range', default=['30-08-18 00:00:00', '26-09-18 23:59:59'])
    args = vars(parser.parse_args())
    date_range_str = '-'.join(args['date_range']).replace(' ', '_')
    args['date_range'] = [datetime.strptime(x, '%d-%m-%y %H:%M:%S') for x in args['date_range']]
    args['phrases'] = args['phrases'].split(',')
    geo_bounds_str = '_'.join([','.join(['%.1f'%(y) for y in x]) for x in args['geo_bounds']])
    phrase_str = ','.join(args['phrases'])
    out_file_name = args['archive_file'].replace('.zip', '_%s.gz'%('_'.join([phrase_str, geo_bounds_str, date_range_str])))
    phrase_matcher = re.compile('|'.join([' %s[# ]|%s$|^%s[# ]'%(x,x,x) for x in args['phrases']]))
    print(phrase_matcher.pattern)
    
    ## assume same order of data
    archive_file_cols = ['lat', 'lon', 'status_id', 'username', 'status', 'timestamp']
    archive_file_data = []
    with gzip.open(out_file_name, 'w') as out_file:
        with ZipFile(args['archive_file']) as archive_dir:
            for archive_file in archive_dir.filelist:
    #             print(dir(archive_file))
                for x in archive_dir.open(archive_file.filename, 'r'):
                    x = x.decode('utf-8')
                    x_lat, x_lon, x_id, x_user, x_status, x_time = process_status_line(x)
    #                 print(x_lat)
    #                 print(x_lon)
    #                 print(x_time)
    #                 print(x_status)
#                     if(contains_geo(x_lat, x_lon, args['geo_bounds'])):
#                     if(contains_time(x_time, args['date_range'])):
#                     if(contains_phrase(x_status, phrase_matcher)):
#                         print(x_status)
                    if(contains_geo(x_lat, x_lon, args['geo_bounds']) and contains_time(x_time, args['date_range']) and contains_phrase(x_status, phrase_matcher)):
                        out_file.write(x.encode('utf-8'))
    #                     print(x)
    #                     break
    
if __name__ == '__main__':
    main()