"""
Classify authors as local, non-local or UNK
based on their location field in metadata
relative to the event during which they were active.
Ex. author posting during Maria with location="San Juan"
counts as local.
"""
import pandas as pd
from argparse import ArgumentParser
import logging
import os
import pickle
from collect_validate_NEs_in_tweets import DATA_LOC_BOXES
from unidecode import unidecode
import re
from data_helpers import DATA_NAME_STATES_LOOKUP

def contains(coord, box):
    return coord[0] >= box[0][0] and coord[0] <= box[0][1] and coord[1] >= box[1][0] and coord[1] <= box[1][1]

def is_local(location, data_name, event_gazetteers):
    gazetteer = event_gazetteers[data_name]
#     location_local = (location in gazetteer) and (gazetteer[location].shape[0] <= 1)
    location_local = location in gazetteer
    return location_local

## deduplicate copy locations
def dedup_locations(data, importance_var='population', coord_cols=['latitude', 'longitude'], round_place=1):
    data = data.assign(**{
        '%s_norm'%(coord_col) : (data.loc[:, coord_col] / round_place).astype(int)
        for coord_col in coord_cols
    })
    data.sort_values(importance_var, inplace=True, ascending=False)
    norm_coord_cols = ['%s_norm'%(coord_col) for coord_col in coord_cols]
    data.drop_duplicates(norm_coord_cols, inplace=True, keep='first')
    return data

def main():
    parser = ArgumentParser()
    # can only use archive meta data because the locations may have changed since the event
    parser.add_argument('--author_meta_data_files', type=list, default=['../../data/mined_tweets/tweet_user_data/user_data_archive.gz'])
    parser.add_argument('--out_dir', default='../../data/mined_tweets/tweet_user_data/')
    parser.add_argument('--gazetteer_file', default='/hg190/corpora/GeoNames/allCountriesSimplified_lookup_US.pickle')
    args = vars(parser.parse_args())
    logging_file = '../../output/classify_locals.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)

    ## load data
    author_meta_data = pd.concat([pd.read_csv(x, sep='\t', index_col=False, compression='gzip') for x in args['author_meta_data_files']], axis=0)
    author_meta_data.fillna('', inplace=True)
    author_var = 'username'
    data_name_var = 'data_name_fixed'
    author_meta_data.drop_duplicates([author_var, data_name_var], inplace=True)
    # fix location strings
    location_var = 'location'
    author_meta_data = author_meta_data.assign(**{
        location_var : author_meta_data.loc[:, location_var].apply(lambda x: unidecode(x.lower()))
    })
    
    logging.debug('loaded %d authors'%(author_meta_data.shape[0]))
    logging.debug('top locations in author metadata:\n%s'%(author_meta_data.loc[:, 'location'].value_counts().head(20)))
    gazetteer = pickle.load(open(args['gazetteer_file'], 'rb'))
    logging.debug('loaded %d locations in gazetteer'%(len(gazetteer)))
    
    ## subset gazetteer for each event
    lat_var = 'latitude'
    lon_var = 'longitude'
    event_gazetteers = {
        data_name : {k : v[v.loc[:, [lat_var, lon_var]].apply(lambda x: contains(x, data_loc_box), axis=1)] for k,v in gazetteer.items()}
        for data_name, data_loc_box in DATA_LOC_BOXES.items()
    }
    # get rid of empty entries
    event_gazetteers = {
        data_name : {k : v for k,v in gazetteer.items() if v.shape[0] > 0}
        for data_name, gazetteer in event_gazetteers.items()
    }
    # deduplicate copy locations
    event_gazetteers_dedup = {
        data_name : {k : dedup_locations(v) for k,v in gazetteer.items()}
        for data_name, gazetteer in event_gazetteers.items()
    }
    assert is_local('san juan', 'maria', event_gazetteers_dedup)
    assert is_local('houston', 'harvey', event_gazetteers_dedup)
    assert not is_local('san juan', 'irma', event_gazetteers_dedup)
    for name_i, gazetteer_i in event_gazetteers_dedup.items():
        logging.debug('event=%s, gazetteer has %d LOC'%(name_i, len(gazetteer_i)))
    
    ## TODO: re-configure using parallel processing ex. https://medium.com/@jmcarpenter2/swiftapply-automatically-efficient-pandas-apply-operations-50e1058909f9
    
    ## label locals
    data_name_var = 'data_name_fixed'
    # state
    DATA_NAME_STATES_MATCHER = {
        k : re.compile('|'.join(['(?<=[, ])%s$|^%s,\s+'%((loc.lower(),)*2) for loc in v]))
        for k,v in DATA_NAME_STATES_LOOKUP.items()
    }
    state_local_var = 'is_state_local'
    author_meta_data = author_meta_data.assign(**{
        state_local_var : author_meta_data.apply(lambda x: DATA_NAME_STATES_MATCHER[x.loc[data_name_var]].search(x.loc[location_var]) is not None, axis=1)
    })
    logging.debug('%d/%d state locals'%(author_meta_data.loc[:, state_local_var].sum(), author_meta_data.shape[0]))
    # city
    city_local_var = 'is_city_local'
    author_meta_data = author_meta_data.assign(**{
        city_local_var : author_meta_data.apply(lambda x: is_local(x.loc[location_var], x.loc[data_name_var], event_gazetteers_dedup), axis=1)
    })
    logging.debug('%d/%d city locals'%(author_meta_data.loc[:, city_local_var].sum(), author_meta_data.shape[0]))
    # combine state/city
    local_var = 'is_local'
    author_meta_data = author_meta_data.assign(**{
        local_var : author_meta_data.loc[:, [state_local_var, city_local_var]].max(axis=1)
    })
    # assign -1 to all authors without location data
    author_meta_data = author_meta_data.assign(**{
        local_var : author_meta_data.apply(lambda x: -1 if x.loc[location_var]=='' else x.loc[local_var], axis=1)
    })
    logging.debug('%d/%d/%d locals/non-locals/UNK'%
                  (author_meta_data[author_meta_data.loc[:, local_var]==1].shape[0],
                   author_meta_data[author_meta_data.loc[:, local_var]==0].shape[0],
                   author_meta_data[author_meta_data.loc[:, local_var]==-1].shape[0]))
    
    ## write to file
    local_meta_file = os.path.join(args['out_dir'], 'user_data_local.tsv')
    author_meta_data_loc = author_meta_data.loc[:, [author_var, data_name_var, local_var]]
    author_meta_data.to_csv(local_meta_file, sep='\t', index=False)
    
if __name__ == '__main__':
    main()