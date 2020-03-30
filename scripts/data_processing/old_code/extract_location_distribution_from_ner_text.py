"""
From NER text, extract distribution of locations
that match to the tagged LOCATION strings that
we can use for later training.
"""
from data_helpers import collect_entities_from_txt, load_name_data, query_norm, load_simplified_geonames_data, load_flat_name_data
from argparse import ArgumentParser
import re, os
import codecs
from collections import Counter
import pandas as pd
from bz2 import BZ2File
from multiprocessing import Pool
from itertools import izip, repeat
from functools import partial
from time import time

def get_matches(loc_pair, name_data):
    """
    Find matches for location in name data
    and return with the location+matches+count.
    
    Parameters:
    -----------
    loc_pair : location, count
    name_data : pandas.DataFrame
    
    Returns:
    --------
    loc_matches : pandas.DataFrame
    Row = location name, count, geonames ID.
    """
    l, l_count = loc_pair
    l_norm = query_norm(l)
    loc_matches = name_data[name_data.loc[:, 'name'] == l_norm]
    if(loc_matches.shape[0] > 0):
        loc_matches = loc_matches.loc[:, ['geonames_ID']] # only care about the geonames ID
        loc_matches.loc[:, 'query'] = l_norm
        loc_matches.loc[:, 'query_count'] = l_count
    else:
        loc_matches = pd.DataFrame([-1, l_norm, l_count], index=['geonames_ID', 'query', 'query_count']).transpose()
    return loc_matches

def get_matches_star(args):
    return get_matches(**args)

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/mined_tweets/sample_ner_tweets/')
    parser.add_argument('--file_pattern', default='.*ner.txt')
    args = parser.parse_args()
    data_dir = args.data_dir
    file_pattern = args.file_pattern
    
    ## collect files
    file_matcher = re.compile(file_pattern)
    ner_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if file_matcher.match(f)]
    
    ## collect entities
    # cleaning up all tags
    tag_matcher = re.compile('/O/\S+|/[BI]-[a-zA-Z\-]+/\S+')
    loc_tag_matcher = re.compile('/(geo-loc|facility)')
    include_type = True
    delim = '/'
    use_bio = True
    LANG = 'en'
    use_POS_tag = True
    location_counts = Counter()
    location_out_file = os.path.join(data_dir, 'sample_ner_location_counts.tsv')
    if(not os.path.exists(location_out_file)):
        for f in ner_files:
            print('processing %s'%(f))
            out_file_name = os.path.join(data_dir, f.replace('.txt', '_loc.txt'))
            for i, l in enumerate(codecs.open(f, 'r', encoding='utf-8')):
                if(l.strip() != ''):
                    # check lang?
                    l_clean = tag_matcher.sub('', l)
                    # English already determined by Twitter
                    try:
                        f_entities = collect_entities_from_txt(l, include_type=include_type,
                                                               delim=delim, use_bio=use_bio, 
                                                               use_POS_tag=use_POS_tag)
                    except Exception, e:
                        print('error with line\n%s'%(l))
                    # restrict to locations; get rid of location tags
                    loc_entities = map(lambda y: loc_tag_matcher.sub('', y), filter(lambda x: loc_tag_matcher.search(x), f_entities))
                    if(len(loc_entities) > 0):
                        location_counts.update(loc_entities)
#                     print(','.join(loc_entities))
#                     if(i > 100):
#                         break
                if(i % 1000000 == 0):
                    print('processed %d lines'%(i))
        # write to file!!
        location_counts_series = pd.Series(location_counts).sort_values(inplace=False, ascending=False)
        location_counts_series.to_csv(location_out_file, sep='\t', index=True, encoding='utf-8')
    else:
        location_counts = pd.read_csv(location_out_file, sep='\t', index_col=0, encoding='utf-8').iloc[:, 0].to_dict()
        # restrict to strings
        location_counts = {(str(k) if (type(k) is not str and type(k) is not unicode) else k) : v
                           for k,v in location_counts.iteritems()}

    # sample
#     location_counts = {k : v for i, (k, v) in enumerate(location_counts.iteritems()) if i < 100}
    print('%d unique locations to query'%(len(location_counts)))
    
    ## match location names in database
    norm_data_file = os.path.join(data_dir, 'flat_names_norm.tsv')
    if(not os.path.exists(norm_data_file)):
        flat_name_data = load_flat_name_data()
        # normalize names
        # write to file because this takes freaking forever
        flat_name_data.loc[:, 'name'] = flat_name_data.loc[:, 'name'].apply(lambda x: query_norm(x))
        flat_name_data.drop_duplicates(['name', 'geonames_ID'], inplace=True)
        flat_name_data.to_csv(norm_data_file, sep='\t', index=False, encoding='utf-8')
    else:
        flat_name_data = pd.read_csv(norm_data_file, sep='\t', index_col=False)

    ## match over all location names
    geonames_data = load_simplified_geonames_data()
    location_names_norm = map(query_norm, location_counts.keys())
    location_name_data = pd.DataFrame([location_counts.keys(), location_names_norm, location_counts.values()]).transpose()
    location_name_data.columns = ['name', 'name_norm', 'count']
    # combine counts by name_norm
    location_name_data = location_name_data.groupby('name_norm').apply(lambda x: pd.Series([x.loc[:, 'count'].sum(), x.loc[:, 'name_norm'].iloc[0]], index=['count', 'name_norm']))
    location_names_norm_set = set(location_names_norm)
    print('%d unique normalized locations'%(len(location_names_norm_set)))
    
    ## join with flat name data
    out_file = os.path.join(data_dir, 'name_location_matches.bz2')
    flat_name_data_match = flat_name_data[flat_name_data.loc[:, 'name'].isin(location_names_norm_set)]
    
    # only keep ID, lat/lon, count
    relevant_data = pd.merge(flat_name_data_match.loc[:, ['geonames_ID', 'name']],
                             location_name_data.loc[:, ['name_norm', 'count']],
                             left_on='name', right_on='name_norm', how='inner').loc[:, ['geonames_ID', 'count', 'name', 'name_norm']]
    print('%d relevant data matches'%(relevant_data.shape[0]))
    # add lat/lon
    relevant_data = pd.merge(relevant_data, geonames_data.loc[:, ['geonames_ID', 'latitude', 'longitude']], 
                             how='inner', on='geonames_ID')
    print('%d relevant data matches after joining with geonames'%(relevant_data.shape[0]))
    relevant_data.to_csv(out_file, sep='\t', index=False, compression='bz2')

if __name__ == '__main__':
    main()
