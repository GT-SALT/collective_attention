"""
Combine all tweet files, extract
NEs, determine valid NEs based on
proximity to event, and approximate
the NEs geographic information based on
matches in Gazetteer.
"""
import pandas as pd
from argparse import ArgumentParser
from data_helpers import process_status_txt_tag_files
import re
from unidecode import unidecode
import pickle
import os
import logging
from dateutil.parser import parse as parse_date
from ast import literal_eval
from functools import reduce
from collections import defaultdict
import numpy as np

def build_name_regex(x):
    try:
        x_regex = re.compile('|'.join(['^%s$'%(y.lower()) for y in set([x.loc['name']]+str(x.loc['alternate_names']).split(',')) - set([''])]))
    except Exception as e:
        x_regex = ''
    return x_regex

DATA_LOC_BOXES = {
    'florence' : [[31.0,36.6], [-85.0,-76.6]],
    'harvey' : [[26.7,36.3], [-105.1,-89.9]],
    'irma' : [[25.5,34.4], [-88.1,-79.0]],
    'maria' : [[17.8,18.5], [-67.4,-65.1]],
    'michael' : [[24.7,34.9], [-88.3,-75.3]],
}
def allow_loc_bound(x, geo_dict, geonames_country_state, verbose=False):
    """
    Determine whether NE has entry in gazetteer, fits in the 
    bounding box, and is not a country/state.
    
    :param x: Series containing NE info
    :param geo_dict: dict to map name to gazetteer info
    :param geonames_country_state: set of countries and U.S. states in gazetteer
    :returns loc_allowed:: loc allowed
    """
    x_txt = x.loc['NE'].replace('_', ' ')
    x_data_name = x.loc['data_name_fixed']
    (lat1, lat2), (lon1, lon2) = DATA_LOC_BOXES[x_data_name]
    x_txt_clean = unidecode(x_txt.lower())
    has_ref = geo_dict.get(x_txt_clean) is not None and geo_dict[x_txt_clean].shape[0] > 0
    # if has_ref, check for bounding box
    if(has_ref):
        geo_data = geo_dict[x_txt_clean]
        geo_data_in_box = geo_data[(geo_data.loc[:, 'latitude'] >= lat1) &
                                   (geo_data.loc[:, 'latitude'] <= lat2) &
                                   (geo_data.loc[:, 'longitude'] >= lon1) &
                                   (geo_data.loc[:, 'longitude'] <= lon2)]
        has_bound_ref = geo_data_in_box.shape[0] > 0
    else:
        has_bound_ref = False
#     not_country_state = geonames_country_state[geonames_country_state.loc[:, 'name_regex'].apply(lambda y: y.search(x_txt_clean) is not None)].shape[0] == 0
    not_country_state = x_txt_clean not in geonames_country_state
    if(verbose):
        logging.debug('%s has ref %s bound_ref %s not_country_state %s'%(x_txt_clean, has_ref, has_bound_ref, not_country_state))
    loc_allowed = has_ref and has_bound_ref and not_country_state
    return loc_allowed
    
STATES_SHORT_FULL_LOOKUP = {
    'FL' : 'Florida', 'NC' : 'North Carolina', 'SC' : 'South Carolina', 
    'VA' : 'Virginia', 'GA' : 'Georgia', 'PR' : 'Puerto Rico',
    'LA' : 'Louisiana', 'TX' : 'Texas',
    
}
DATA_NAME_STATES_SHORT_LOOKUP = {
    'florence' : ['FL', 'NC', 'SC', 'VA'],
    'irma' : ['FL', 'GA'],
    'harvey' : ['TX', 'LA'],
    'maria' : ['PR', 'NC'],
    'michael' : ['FL', 'GA', 'NC', 'SC']
}
DATA_NAME_STATES_LONG_LOOKUP = {k : [STATES_SHORT_FULL_LOOKUP[v] for v in vs] for k, vs in DATA_NAME_STATES_SHORT_LOOKUP.items()}
DATA_NAME_STATES_LOOKUP = {k : DATA_NAME_STATES_SHORT_LOOKUP[k]+DATA_NAME_STATES_LONG_LOOKUP[k] for k in DATA_NAME_STATES_SHORT_LOOKUP.keys()}
def find_descriptor(data, verbose=False):
    """
    Extract descriptor from text when it follows
    NE.
    
    :param data: NE data row
    :param verbose: bool, logging.debug more information
    :returns data_descriptor: bool, text contains descriptor
    """
    data_NE = data.loc['NE'].replace('_', ' ').lower()
    if(verbose):
        logging.debug('testing NE %s'%(data_NE))
    data_name = data.loc['data_name_fixed']
    data_name_states = DATA_NAME_STATES_LOOKUP[data_name]
    try:
        data_NE_txt_matcher = re.compile('|'.join(['%s[, ]\s?%s|%s\s?\(%s\)'%(data_NE, x.lower(), data_NE, x.lower()) for x in data_name_states]))
    except Exception as e:
        logging.debug('bad pattern %s'%('|'.join(['%s[, ]\s?%s|%s\s?\(%s\)'%(data_NE, x.lower(), data_NE, x.lower()) for x in data_name_states])))
        data_NE_txt_matcher = re.compile('<NULL>')
    try:
        data_NE_matcher = re.compile('|'.join(['[, ]\s?%s|\(%s\)'%((x.lower(),)*2) for x in data_name_states]))
    except Exception as e:
        logging.debug('bad pattern %s'%('|'.join(['[, ]\s?%s|\(%s\)'%((x.lower(),)*2) for x in data_name_states])))
        data_NE_matcher = re.compile('<NULL>')
    data_txt = data.loc['txt'].lower()
    
    data_NE_txt_descriptor = data_NE_txt_matcher.search(data_txt) is not None
    data_NE_descriptor = data_NE_matcher.search(data_NE) is not None
    if(verbose):
        logging.debug('data NE txt pattern %s'%(data_NE_txt_matcher.pattern))
        logging.debug('data NE pattern %s'%(data_NE_matcher.pattern))
        logging.debug('data NE txt %s'%(data_NE_txt_descriptor))
        logging.debug('data NE %s'%(data_NE_descriptor))
    data_descriptor = data_NE_descriptor or data_NE_txt_descriptor
    return data_descriptor

def load_geonames_data(geonames_dict_data_file, geonames_data_file, loc_feat_codes=['ADM1', 'ADM2', 'ADM3', 'ADM4', 'ADM5', 'ADMD', 'BLDG', 'CH', 'CMTY', 'ISL', 'ISLS', 'LCTY', 'PCLD', 'PCLH', 'PCLI', 'PPL', 'PPLA', 'PPLA2', 'PPLA3', 'PPLA4', 'PPLC', 'PPLG', 'PPLL', 'PPLS', 'PPLX', 'SCH'], geonames_countries=['US', 'PR']):
    """
    Load relevant GeoNames data from file.

    :param geonames_data_file: full dict data file
    :param geonames_data_file: flat data file
    :param loc_feat_codes: location feature codes
    :param geonames_countries: two-letter country codes
    :returns geonames_data_loc:: relevant location data (based on feature codes, countries)
    :return geonames_country_state:: countries/states to leave out of location matching
    """
    ## new GeoNames dict (bigger and faster)
    # warning: takes a long time to load ~1G
    geonames_dict = pickle.load(open(geonames_dict_data_file, 'rb'))
    # restrict to specified features, countries
    # assume that dict is already restricted!!
#     geonames_loc = {k : v[(v.loc[:, 'feature_code'].isin(loc_feat_codes)) & (v.loc[:, 'country'].isin(geonames_countries))] for k,v in geonames_dict.items()}
#     geonames_loc = {k : v for k,v in geonames_loc.items() if v.shape[0] > 0}
    logging.debug('loaded GeoNames dict with %d items'%(len(geonames_dict)))
    # get country/states for filtering
    geonames_data = pd.read_csv(geonames_data_file, sep='\t', index_col=False)
    ## restrict GeoNames to relevant data
    relevant_cols = list(geonames_dict.items())[0][1].columns
    geonames_data = geonames_data.loc[:, relevant_cols]
    state_codes = set(['ADM1'])
    territory_codes = set(['PCLD'])
    country_code = 'PCLI'
    us_countries = set(['US'])
    us_territory_countries = set(['US', 'PR'])
    geonames_country_state = geonames_data[((geonames_data.loc[:, 'feature_code'].isin(state_codes)) & (geonames_data.loc[:, 'country'].isin(us_countries))) | 
                                                ((geonames_data.loc[:, 'feature_code'].isin(territory_codes)) & (geonames_data.loc[:, 'country'].isin(us_territory_countries))) | 
                                                (geonames_data.loc[:, 'feature_code'] == country_code)]
    geonames_country_state_dict = defaultdict(list)
    for idx, data_i in geonames_country_state.iterrows():
        names_i = set([data_i.loc['name'].lower()]) | set([y.lower() for y in data_i.loc['alternate_names'].split(',')]) - set([''])
        for name_i in names_i:
            geonames_country_state_dict[name_i].append(data_i)
    geonames_country_state_dict = {k : pd.concat(v, axis=1).transpose() for k,v in geonames_country_state_dict.items()}
    geonames_country_state_names = geonames_country_state.loc[:, 'name'].apply(lambda x: x.lower()).unique()
    geonames_country_state_names_alt = list(reduce(lambda a,b: a|b, geonames_country_state.loc[:, 'alternate_names'].apply(lambda x: set([y.lower() for y in x.split(',')]))))
    geonames_country_state = set(geonames_country_state_names) | set(geonames_country_state_names_alt)
    logging.debug('loaded %d country/state names'%(len(geonames_country_state)))
    logging.debug('example country/state names %s'%(list(geonames_country_state)[:10]))
    ## add country, states to geonames data loc for anchors
    geonames_combined_dict = {k : v for k,v in geonames_dict.items()}
    for k,v in geonames_country_state_dict.items():
        if(k not in geonames_combined_dict.keys()):
            geonames_combined_dict[k] = v
        else:
            geonames_combined_dict[k] = pd.concat([geonames_combined_dict[k], v], axis=0)
            geonames_combined_dict[k].index = np.arange(geonames_combined_dict[k].shape[0])
    logging.debug('%d total entries in combined lookup'%(len(geonames_combined_dict)))
    
    return geonames_dict, geonames_combined_dict, geonames_country_state
    
    ## old flat GeoNames data
#     geonames_data = pd.read_csv(geonames_data_file, sep='\t', index_col=False, usecols=['geonames_ID', 'name', 'alternate_names', 'feature_code', 'country', 'latitude', 'longitude', 'population', 'alternate_name_count'])
#     geonames_data.rename(columns={'geonames_ID':'id'}, inplace=True)
#     geonames_data.fillna('', inplace=True)
    # restrict to features/countries
#     geonames_data_loc = geonames_data[geonames_data.loc[:, 'feature_code'].isin(loc_feat_codes)]
#     geonames_data_loc = geonames_data_loc[geonames_data_loc.loc[:, 'country'].isin(geonames_countries)]
    # match each location name to GeoNames entry
    # using regex of alternate names
#     geonames_data_loc = geonames_data_loc.assign(**{'name_regex' : geonames_data_loc.apply(lambda x: build_name_regex(x), axis=1)})
#     geonames_data_loc = geonames_data_loc[geonames_data_loc.loc[:, 'name_regex'] != '']
#     logging.debug('%d valid GeoNames locations'%(geonames_data_loc.shape[0]))
    # get country/state names for filtering
#     state_codes = set(['ADM1'])
#     territory_codes = set(['PCLD'])
#     country_code = 'PCLI'
#     us_countries = set(['US'])
#     us_territory_countries = set(['US', 'PR'])
#     geonames_country_state = geonames_data[((geonames_data.loc[:, 'feature_code'].isin(state_codes)) & (geonames_data.loc[:, 'country'].isin(us_countries))) | 
#                                                 ((geonames_data.loc[:, 'feature_code'].isin(territory_codes)) & (geonames_data.loc[:, 'country'].isin(us_territory_countries))) |
#                                                 (geonames_data.loc[:, 'feature_code'] == country_code)]
#     geonames_country_state = geonames_country_state.assign(**{'name_regex' : geonames_country_state.apply(build_name_regex, axis=1)})
#     logging.debug('%d country/state names'%(geonames_country_state.shape[0]))
#     return geonames_data_loc, geonames_country_state

def flatten_NE_data(combined_tag_data):
    """
    Flatten tagged data to include one NE per row.
   
    :param combined_tag_data: pandas.DataFrame containing tag data
    :returns combined_tag_data_NE_flat: pandas.DataFrame containing flat NE data
    """
    combined_tag_data_NE_flat = []
    flat_rows = ['id', 'txt', 'data_name_fixed', 'username', 'date', 'lang']
    NE_ctr = 0
    tweet_ctr = 0
    for idx_i, data_i in combined_tag_data.iterrows():
        for NE, NE_type in data_i.loc['NE_list']:
            data_j = data_i.loc[flat_rows]
            data_j.loc['NE'] = NE
            data_j.loc['NE_type'] = NE_type
            combined_tag_data_NE_flat.append(data_j)
            NE_ctr += 1
        # add null data for posterity
        if(len(data_i.loc['NE_list']) == 0):
            data_j = data_i.loc[flat_rows]
            data_j.loc['NE'] = ''
            data_j.loc['NE_type'] = ''
#             data_j = data_j.assign(**{'NE':'', 'NE_type':''})
            combined_tag_data_NE_flat.append(data_j)
        tweet_ctr += 1
        if(tweet_ctr % 1000 == 0):
            logging.debug('processed %d/%d tweets; %d NEs total'%(tweet_ctr, combined_tag_data.shape[0], NE_ctr))
    combined_tag_data_NE_flat = pd.concat(combined_tag_data_NE_flat, axis=1).transpose()
    logging.debug('%d NEs total'%(combined_tag_data_NE_flat.shape[0]))
    # fix text
    combined_tag_data_NE_flat = combined_tag_data_NE_flat.assign(**{'txt':combined_tag_data_NE_flat.loc[:, 'txt'].apply(lambda x: x.strip())})
    # fix nans
    combined_tag_data_NE_flat.fillna('', inplace=True)
    # fix username: 
    # in some files the username has file name attached
    combined_tag_data_NE_flat = combined_tag_data_NE_flat.assign(**{'username' : combined_tag_data_NE_flat.loc[:, 'username'].apply(lambda x: x.split(':')[-1])})
    return combined_tag_data_NE_flat
    
def find_valid_loc_NE(combined_tag_data_NE_flat, geo_dict, geonames_country_state, data_dir, data_name='combined_data_NE_tweets'):
    """
    Find NEs that are LOCATION and exist within GeoNames as city/county/sub-country region.

    :param combined_tag_data_NE_flat: flat NE data
    :param geo_dict: dictionary of name : GeoNames data
    :param geonames_data_loc: GeoNames data with valid locations
    :param geonames_country_state: set of country/state names (i.e. invalid)
    :param data_dir: data directory (to save/load geo dict for lookup)
    :param data_name: data name 
    :returns combined_tag_data_NE_flat: updated flat NE data
    """
    LOC_TYPES = set(['CITY', 'LOCATION'])
    combined_tag_data_NE_flat = combined_tag_data_NE_flat.assign(**{'NE_LOC':combined_tag_data_NE_flat.loc[:, 'NE_type'].apply(lambda x: x in LOC_TYPES)})
    combined_tag_data_NE_flat_names = combined_tag_data_NE_flat[combined_tag_data_NE_flat.loc[:, 'NE_LOC']].loc[:, 'NE'].apply(lambda x: unidecode(x.lower()).replace('_', ' ')).unique()
    logging.debug('%d unique names'%(len(combined_tag_data_NE_flat_names)))
    ## old code to collect geo_dict
    # get dict of name : geoname data
    # load existing data if possible
#     geo_dict_file_name = os.path.join(data_dir, '%s_geo_dict.pickle'%(data_name))
#     if(os.path.exists(geo_dict_file_name)):
#         geo_dict = pickle.load(open(geo_dict_file_name, 'rb'))
#     else:
#         geo_dict = {}
#     combined_tag_data_NE_flat_names_to_add = list(set(combined_tag_data_NE_flat_names) - set(geo_dict.keys()))
#     # optional: replace the entire dict
# #     replace_geo_dict = True
#     replace_geo_dict = False
#     if(replace_geo_dict):
#         combined_tag_data_NE_flat_names_to_add = list(set(combined_tag_data_NE_flat_names))
#     # optional: update with gazetteer entries
#     # that were not present before
#     update_geo_dict = True
# #     update_geo_dict = False
#     if(update_geo_dict):
#         combined_tag_data_NE_flat_names_to_add = list(set(combined_tag_data_NE_flat_names))
#         geonames_prev_ids = set(reduce(lambda x,y: x|y, [set(v.loc[:, 'id'].unique()) for k,v in geo_dict.items()]))
#         logging.debug('%d previously collected IDs'%(len(geonames_prev_ids)))
#         geonames_data_loc = geonames_data_loc[~geonames_data_loc.loc[:, 'id'].isin(geonames_prev_ids)]
#         logging.debug('%d possible gazetteer entries to add'%(geonames_data_loc.shape[0]))
#     logging.debug('%d names to add'%(len(combined_tag_data_NE_flat_names_to_add)))
#     for x in combined_tag_data_NE_flat_names_to_add:
#         if(update_geo_dict):
#             geo_dict[x] = pd.concat([geo_dict[x], geonames_data_loc[geonames_data_loc.loc[:, 'name_regex'].apply(lambda y: y.search(x) is not None)]], axis=0).drop_duplicates('id', inplace=False)
#         else:
#             geo_dict[x] = geonames_data_loc[geonames_data_loc.loc[:, 'name_regex'].apply(lambda y: y.search(x) is not None)]
#     # save for later
#         pickle.dump(geo_dict, open(geo_dict_file_name, 'wb'))
    # determine valid locations
    # based on match in geo_dict
    combined_tag_data_NE_flat_valid = []
    for i, (idx_i, data_i) in enumerate(combined_tag_data_NE_flat.iterrows()):
        if(data_i.loc['NE_LOC']):
            valid_loc_i = allow_loc_bound(data_i, geo_dict, geonames_country_state, verbose=False)
        else:
            valid_loc_i = False
        combined_tag_data_NE_flat_valid.append(valid_loc_i)
        if(i % 10000 == 0):
            logging.debug('processed %d/%d NEs as valid'%(i, combined_tag_data_NE_flat.shape[0]))
    combined_tag_data_NE_flat = combined_tag_data_NE_flat.assign(**{'valid_loc' : combined_tag_data_NE_flat_valid})
    logging.debug('%d/%d valid locations'%(combined_tag_data_NE_flat.loc[:, 'valid_loc'].sum(), combined_tag_data_NE_flat.shape[0]))
    return combined_tag_data_NE_flat
    
def find_NE_descriptors(combined_tag_data_NE_flat):
    """
    Find descriptors for LOCATION NEs.

    :param combined_tag_data_NE_flat: flat NE data
    :returns combined_tag_data_NE_flat: flat NE data with descriptors
    """
    combined_tag_data_NE_flat_has_descriptor = []
    descriptor_ctr = 0
    for idx_i, data_i in combined_tag_data_NE_flat.iterrows():
        if(data_i.loc['NE_LOC']):
            data_i_has_descriptor = find_descriptor(data_i, verbose=False)
        else:
            data_i_has_descriptor = False
        combined_tag_data_NE_flat_has_descriptor.append(data_i_has_descriptor)
        if(descriptor_ctr % 1000 == 0):
            logging.debug('processed %d NEs with descriptor'%(descriptor_ctr))
        descriptor_ctr += 1
    combined_tag_data_NE_flat = combined_tag_data_NE_flat.assign(**{'has_descriptor' : combined_tag_data_NE_flat_has_descriptor})
    logging.debug('%d/%d NEs have standard descriptor'%(combined_tag_data_NE_flat.loc[:, 'has_descriptor'].astype(int).sum(), combined_tag_data_NE_flat.shape[0]))
    NE_valid_loc = combined_tag_data_NE_flat[combined_tag_data_NE_flat.loc[:, 'valid_loc']]
    logging.debug('%d/%d valid NE locations'%(NE_valid_loc.shape[0], combined_tag_data_NE_flat.shape[0]))
    NE_valid_loc_descriptor = combined_tag_data_NE_flat[(combined_tag_data_NE_flat.loc[:, 'has_descriptor']) & (combined_tag_data_NE_flat.loc[:, 'valid_loc'])]
    logging.debug('%d/%d NEs are valid and have standard descriptor'%(NE_valid_loc_descriptor.shape[0], combined_tag_data_NE_flat.shape[0]))
    # unique NEs with/without descriptor
    NE_valid_loc_counts_ne = NE_valid_loc.loc[:, 'NE'].value_counts()
    logging.debug('top-k valid NEs')
    logging.debug(NE_valid_loc_counts_ne.head(20))
    NE_valid_loc_descriptor = combined_tag_data_NE_flat[(combined_tag_data_NE_flat.loc[:, 'has_descriptor']) & (combined_tag_data_NE_flat.loc[:, 'valid_loc'])]
    logging.debug('%d/%d NEs are valid and have standard descriptor'%(NE_valid_loc_descriptor.shape[0], combined_tag_data_NE_flat.shape[0]))
    return combined_tag_data_NE_flat
    
def clean_tag_data(data):
    """
    Clean redundant and irrelevant tag data.

    :param data: NE tagged data
    :returns data:: clean NE tagged data
    """
    # remove nan data
    data = data[~data.loc[:, 'id'].apply(lambda x: type(x) is float and np.isnan(x))]
    # convert date stamp to float for binning
    data = data.assign(**{'date_stamp' : data.loc[:, 'date'].apply(lambda x: x.timestamp())})
    # get rid of duplicate statuses
    data_N = data.shape[0]
    data.drop_duplicates(['id', 'data_name_fixed'], inplace=True)
    data.drop_duplicates(['txt', 'data_name_fixed'], inplace=True)
    data_dedup_N = data.shape[0]
    # get rid of hanging underscores!!
    SPACE_UNDERSCORE_MATCHER = re.compile('[ _]+$')
    data = data.assign(**{'NE_list' : data.loc[:, 'NE_list'].apply(lambda x: [[SPACE_UNDERSCORE_MATCHER.sub('', y[0]), y[1]] for y in x] if type(x) is list else [])})
    logging.debug('%d/%d deduplicated statuses'%(data_dedup_N, data_N))
    # clean tag types
    data = data.assign(**{'NE_list' : data.loc[:, 'NE_list'].apply(lambda x: [(y[0], y[1].replace('"', '')) for y in x])})
    # unify tag types: "geo-loc" => "LOCATION", etc.
    NE_type_lookup = {'geo' : 'LOCATION', 'person':'PERSON'}
    data = data.assign(**{'NE_list' : data.loc[:, 'NE_list'].apply(lambda x: [(y[0], NE_type_lookup[y[1]]) if y[1] in NE_type_lookup else y for y in x])})
    # filter for locations
    LOC_TYPES = set(['CITY', 'LOCATION'])
    data = data.assign(**{'NE_list_LOC' : data.loc[:, 'NE_list'].apply(lambda x: [y for y in x if y[1] in LOC_TYPES])})
    return data
    
def find_NE_anchors(data, geo_dict, importance_stats=['population', 'alternate_name_count'], verbose=False):
    """
    Find anchors for LOCATION NEs.
    Anchor = LOCATION NE with more importance.
    Ex. "San Juan is located in Puerto Rico"
    "San Juan" has the anchor "Puerto Rico".

    :param data: flat NE data
    :param geo_dict: gazetteer with importance data
    :param importance_stats: importance stats to treat as anchor
    :param verbose: print debug statements
    :returns anchor_data:: flat NE data with anchor annotations (true/false)
    """
    ## TODO: load anchor data separately? all nonzero population referents
    ## assign anchor data
    data_loc = data[data.loc[:, 'NE_type']=='LOCATION']
    data_loc = data_loc.assign(**{'NE_fixed_clean':data_loc.loc[:, 'NE_fixed'].apply(lambda x: x.replace('_', ' '))})
    NULL_VAL = 0.
    for importance_stat in importance_stats:
        importance_stat_max = 'max_%s'%(importance_stat)
        max_importance = defaultdict(lambda: NULL_VAL)
        max_importance.update({k : v.loc[:, importance_stat].max() for k,v in geo_dict.items()})
#         for x in data_loc.loc[:, 'NE_fixed_clean']:
#             try:
#                 max_importance[x]
#             except Exception as e:
#                 logging.debug('error %s:\n access max importance of NE %s'%(e, x))
#                 break
        data_loc = data_loc.assign(**{importance_stat_max : data_loc.loc[:, 'NE_fixed_clean'].apply(lambda x: max_importance[x])})
    if(verbose):
        print('generated data_loc:\n%s'%(data_loc))
    anchor_data = []
    anchor_cols = ['id', 'data_name_fixed']+['%s_anchor'%(x) for x in importance_stats]
    for (id_i, name_i), data_i in data_loc.groupby(['id', 'data_name_fixed']):
        if(verbose):
            print('processing data:\n%s'%(data_i))
        for importance_stat in importance_stats:
            importance_stat_max = 'max_%s'%(importance_stat)
            data_i_max_anchor = data_i.loc[:, importance_stat_max].max()
            data_i = data_i.assign(**{'%s_anchor'%(importance_stat_max) : data_i.apply(lambda x: x.loc[importance_stat_max] < data_i_max_anchor, axis=1)})
            # also include diff for housekeeping
            data_i = data_i.assign(**{'%s_diff'%(importance_stat_max) : data_i.apply(lambda x: data_i_max_anchor - x.loc[importance_stat_max], axis=1)})
            if(verbose):
                print('updated data:\n%s'%(data_i))
#         data_i = data_i.loc[:, anchor_cols]
        anchor_data.append(data_i)
    anchor_data = pd.concat(anchor_data, axis=0)

    ## rejoin to original data
#     data = pd.merge(data, anchor_data, on=['id', 'data_name_fixed'])
    logging.debug('%d/%d anchors'%(anchor_data.loc[:, 'max_population_anchor'].sum(), anchor_data.shape[0]))
    return anchor_data
    
def main():
    parser = ArgumentParser()
    ## assume that each tag file has corresponding tweet/status file for complete info
    parser.add_argument('--tag_files', default=[
        # hashtag tweets
    # Maria
#     '../../data/mined_tweets/archive_maria_txt_tags.gz', '../../data/mined_tweets/stream_maria_txt_tags.gz', '../../data/mined_tweets/historical_maria_txt_tags.gz',
    # Harvey
#     '../../data/mined_tweets/archive_harvey_txt_tags.gz', '../../data/mined_tweets/stream_harvey_txt_tags.gz', '../../data/mined_tweets/historical_harvey_txt_tags.gz',
    # Irma
#     '../../data/mined_tweets/archive_irma_txt_tags.gz', '../../data/mined_tweets/stream_irma_txt_tags.gz', '../../data/mined_tweets/historical_irma_txt_tags.gz',
    # Florence
#     '../../data/mined_tweets/archive_florence_txt_tags.gz', '../../data/mined_tweets/east_coast_geo_twitter_2018/geo_stream_florence_txt_tags.gz', '../../data/mined_tweets/historical_florence_txt_tags.gz',
    # Michael
#     '../../data/mined_tweets/archive_michael_txt_tags.gz', '../../data/mined_tweets/east_coast_geo_twitter_2018/geo_stream_michael_txt_tags.gz', '../../data/mined_tweets/historical_michael_txt_tags.gz',
        # location-phrase tweets
        # Florence
#         '../../data/mined_tweets/archive_location_phrases_florence_txt_tags.gz',
# #         Harvey
#         '../../data/mined_tweets/archive_location_phrases_harvey_txt_tags.gz',
# #         Irma
#         '../../data/mined_tweets/archive_location_phrases_irma_txt_tags.gz',
# #         Maria
#         '../../data/mined_tweets/archive_location_phrases_maria_txt_tags.gz',
# #         Michael
#         '../../data/mined_tweets/archive_location_phrases_michael_txt_tags.gz',
        # power user tweets
            # Florence
        '../../data/mined_tweets/combined_data_power_user_florence_txt_tags.gz',
# #         Harvey
        '../../data/mined_tweets/combined_data_power_user_harvey_txt_tags.gz',
# #         Irma
        '../../data/mined_tweets/combined_data_power_user_irma_txt_tags.gz',
# #         Maria
        '../../data/mined_tweets/combined_data_power_user_maria_txt_tags.gz',
        # Michael
        '../../data/mined_tweets/combined_data_power_user_michael_txt_tags.gz',
])
    
    ## gazetteer data
    # conair
    parser.add_argument('--geonames_data', default='/hg190/corpora/GeoNames/allCountriesSimplified.tsv')
    parser.add_argument('--geonames_data_dict', default='/hg190/corpora/GeoNames/allCountriesSimplified_lookup_US.pickle')
    # arizona
#     parser.add_argument('--geonames_data', default='../../data/GeoNames/allCountriesSimplified.tsv')
#     parser.add_argument('--geonames_data_dict', default='../../data/GeoNames/allCountriesSimplified_lookup_US.pickle')
    ## output file name
    # original
#     parser.add_argument('--out_file_name', default='combined_tweet_tag_data')
#     parser.add_argument('--out_file_name', default='combined_location_phrase_tweet_tag_data')
    # power user 
    parser.add_argument('--out_file_name', default='combined_data_power_user')
    args = vars(parser.parse_args())
    log_file_name = '../../output/collect_validate_NEs_in_tweets_output.txt'
    if(os.path.exists(log_file_name)):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    
    ## get data directory to save later
    data_dir = os.path.dirname(args['tag_files'][0])
    
    ## combine tag/status data
    ## TODO: filter spammers by percentile, bad tags
    combined_tag_data_out_file = os.path.join(data_dir, '%s.gz'%(args['out_file_name']))
    replace_combined_tag_data_file = False
#     replace_combined_tag_data_file = True
    if(not os.path.exists(combined_tag_data_out_file) or replace_combined_tag_data_file):
        combined_tag_data = []
        for f in args['tag_files']:
            f_combined = process_status_txt_tag_files(f)
            combined_tag_data.append(f_combined)
        combined_tag_data = pd.concat(combined_tag_data, axis=0)
        combined_tag_data.to_csv(combined_tag_data_out_file, sep='\t', index=False, compression='gzip')
    else:
        combined_tag_data = pd.read_csv(combined_tag_data_out_file, sep='\t', index_col=False, compression='gzip', parse_dates=['date'], converters={'NE_list':literal_eval})
        # read date
#         combined_tag_data = combined_tag_data.assign(**{'date' : combined_tag_data.loc[:, 'date'].apply(parse_date)})
        # read list
#         combined_tag_data = combined_tag_data.assign(**{'NE_list' : combined_tag_data.loc[:, 'NE_list'].apply(lambda x: literal_eval(x))})
    logging.debug('%d lines total'%(combined_tag_data.shape[0]))
    # tmp debugging
#     combined_tag_data = combined_tag_data.head(1000)
#     print(combined_tag_data.head())
    
    ## clean tag data 
    combined_tag_data = clean_tag_data(combined_tag_data)
    logging.debug('combined tag data data names\n%s'%(combined_tag_data.loc[:, 'data_name_fixed'].value_counts()))
    
    ## flatten data to one LOCATION NE per row
    combined_tag_data_NE_flat_full_file = os.path.join(data_dir, '%s_NE_flat_all.gz'%(args['out_file_name']))
    if(not os.path.exists(combined_tag_data_NE_flat_full_file)):
        combined_tag_data_NE_flat = flatten_NE_data(combined_tag_data)
        combined_tag_data_NE_flat.to_csv(combined_tag_data_NE_flat_full_file, sep='\t', index=False, compression='gzip')
    else:
        combined_tag_data_NE_flat = pd.read_csv(combined_tag_data_NE_flat_full_file, sep='\t', index_col=False, compression='gzip')
#     print(combined_tag_data_NE_flat.head())
    ## debug: top-k NEs per dataset
    for data_name_i, data_i in combined_tag_data_NE_flat.groupby('data_name_fixed'):
        logging.debug('data %s top-k NEs'%(data_name_i))
        data_i_NE_counts = data_i.loc[:, 'NE'].apply(lambda x: str(x).lower()).value_counts()
        logging.debug(data_i_NE_counts.head(20))
    ## tmp debugging
#     combined_tag_data_NE_flat = combined_tag_data_NE_flat.head(1000)
    
    ## remove NEs with data names
#     data_names = ['#%s'%(x) for x in combined_tag_data_NE_flat.loc[:, 'data_name_fixed'].unique()]
#     combined_tag_data_NE_flat = combined_tag_data_NE_flat
    
    ## validate NEs
    geonames_data_loc, geonames_data_loc_combined, geonames_country_state_names = load_geonames_data(args['geonames_data_dict'], args['geonames_data'])
    combined_tag_data_NE_flat_valid_file = os.path.join(data_dir, '%s_NE_flat_valid.gz'%(args['out_file_name']))
    if(not os.path.exists(combined_tag_data_NE_flat_valid_file)):
        ## find geo matches for all valid NEs
        ## restrict to locations!
        combined_tag_data_NE_flat = find_valid_loc_NE(combined_tag_data_NE_flat, geonames_data_loc, geonames_country_state_names, data_dir)
        # fix NE name for easier counting later
        combined_tag_data_NE_flat = combined_tag_data_NE_flat.assign(**{'NE_fixed' : combined_tag_data_NE_flat.loc[:, 'NE'].apply(lambda x: unidecode(str(x).lower()))})
        combined_tag_data_NE_flat.to_csv(combined_tag_data_NE_flat_valid_file, sep='\t', index=False, compression='gzip')
    else:
        combined_tag_data_NE_flat = pd.read_csv(combined_tag_data_NE_flat_valid_file, sep='\t', index_col=False, compression='gzip')
    
    ## find descriptors! in the format "CITY, <STATE>"
    ## TODO: add this as separate class "extract_anchors" etc.
#     combined_tag_data_NE_flat = find_NE_descriptors(combined_tag_data_NE_flat)
    ## also: find anchoring coccurrence! e.g. "San Juan was the latest city in Puerto Rico to be hit"
    ## TODO: add dependency condition
    ## only counts as anchoring if larger NE
    ## occurs in acl, prep, appos subclause
    ## need geo data
#     data_name = 'combined_data_NE_tweets'
#     geo_dict_file_name = os.path.join(data_dir, '%s_geo_dict.pickle'%(data_name))
#     geo_dict = pickle.load(open(geo_dict_file_name, 'rb'))
#     logging.debug('computing anchors')
#     combined_tag_data_NE_flat = find_NE_anchors(combined_tag_data_NE_flat, geonames_data_loc_combined)
    
    ## remove bad NEs
    bad_loc_NEs = set(['hurricane'] + list(combined_tag_data_NE_flat.loc[:, 'data_name_fixed'].unique()))
    combined_tag_data_NE_flat = combined_tag_data_NE_flat[combined_tag_data_NE_flat.apply(lambda x: x.loc['NE_fixed'] not in bad_loc_NEs or not x.loc['valid_loc'], axis=1)]
    
    ## save final data!
#     combined_tag_data_NE_flat_file_name = os.path.join(data_dir, 'combined_tweet_NE_flat_data.gz')
    combined_tag_data_NE_flat_file_name = os.path.join(data_dir, '%s_NE_flat.gz'%(args['out_file_name']))
    combined_tag_data_NE_flat.to_csv(combined_tag_data_NE_flat_file_name, sep='\t', compression='gzip', index=False)
    
if __name__ == '__main__':
    main()
