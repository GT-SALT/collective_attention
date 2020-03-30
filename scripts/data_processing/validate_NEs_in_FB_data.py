"""
Validate LOC NEs from tagged FB data based on 
occurrence in OSM gazetteer or GN gazetteer.
"""
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from ast import literal_eval
import os
import logging
import pickle
import re
from unidecode import unidecode
from functools import reduce

def clean_geonames_name(x):
    x = unidecode(x.lower())
    return x

STREET_MATCHER = re.compile('^Calle |^Camino ')
NEIGHBORHOOD_MATCHER = re.compile('^Barrio |^Bo\.? ')
def clean_osm_name(x):
    if(STREET_MATCHER.search(x) is not None):
        x = '|'.join([x, STREET_MATCHER.sub('', x)])
    if(NEIGHBORHOOD_MATCHER.search(x) is not None):
        x = '|'.join([x, NEIGHBORHOOD_MATCHER.sub('', x)])
    x = unidecode(x.lower())
    return x

def expand_alt_names(data):
    """
    Expand gazetteer data by removing suffixes
    from certain names (ex. "Rincon Barrio" => "Rincon")
    and adding the new name as separate entry.
    
    :param data: gazetteer data
    :returns data:: updated gazeteer data
    """
    ALT_NAME_MATCHER = re.compile(' barrio$| barrio-pueblo$| municipio$| subbarrio$')
    for k in list(data.keys()):
        v = data[k]
        if(ALT_NAME_MATCHER.search(k) is not None):
            k_fix = ALT_NAME_MATCHER.sub('', k)
            if(k_fix in data.keys()):
                data[k_fix] = pd.concat([data[k_fix], v], axis=0)
            else:
                data[k_fix] = v
    return data

def main():
    parser = ArgumentParser()
    parser.add_argument('--tag_data_file', default='../../data/facebook-maria/combined_group_data_es_tagged.tsv')
    parser.add_argument('--osm_data', default='../../data/geo_files/PR_OSM/combined_shp_xml_data.tsv')
    parser.add_argument('--geonames_data', default='/hg190/corpora/GeoNames/allCountriesSimplified_lookup_US.pickle')
    args = vars(parser.parse_args())
    log_file_name = '../../output/validate_NEs_in_FB_data.txt'
    if(os.path.exists(log_file_name)):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    
    ## load tag data
    tag_data_cols = ['group_id', 'status_author_id', 'status_message', 'status_lang', 'status_id', 'status_published', 'status_message_clean', 'status_message_tags_ne', 'status_message_tags']
    tag_var = 'status_message_tags_ne'
    tag_data = pd.read_csv(args['tag_data_file'], sep='\t', index_col=False, usecols=tag_data_cols, converters={tag_var : literal_eval})
    ## filter spam messages
    tag_len_var = 'status_len'
    tag_data = tag_data.assign(**{tag_len_var : tag_data.loc[:, 'status_message'].apply(len)})
    message_len_cutoff_pct = 95
    message_len_cutoff = np.percentile(tag_data.loc[:, tag_len_var].values, message_len_cutoff_pct)
    logging.debug('filtering all statuses with length > %d'%(int(message_len_cutoff)))
    tag_data_filtered = tag_data[tag_data.loc[:, tag_len_var] < message_len_cutoff]
    logging.debug('%d/%d statuses remaining'%(tag_data_filtered.shape[0], tag_data.shape[0]))
    tag_data = tag_data_filtered.copy()
    
    ## load OSM, GN gazetteers
    ## restrict to PR obviously
    data_countries = set(['PR'])
    geonames_data = pickle.load(open(args['geonames_data'], 'rb'))
    geonames_data = {k : v[v.loc[:, 'country'].isin(data_countries)] for k,v in geonames_data.items()}
    geonames_data = {k : v for k,v in geonames_data.items() if v.shape[0] > 0}
    ## restrict to allowed types: neighborhoods (ADM2, ADM3), cities (PPL*), municipalities (ADM1)
    GEONAMES_ALLOWED_TYPES = set(['ADM1', 'ADM2', 'ADM3', 'ADM5', 'PPL', 'PPLA'])
    geonames_data = {k : v[v.loc[:, 'feature_code'].isin(GEONAMES_ALLOWED_TYPES)] for k,v in geonames_data.items()}
    geonames_data = {k : v for k,v in geonames_data.items() if v.shape[0] > 0}
    
    ## expand gazetteer alternate names => more chances to catch candidates!
    logging.debug('%d gazetteer entries before expanding'%(len(geonames_data)))
    geonames_data = expand_alt_names(geonames_data)
    logging.debug('%d gazetteer entries after expanding'%(len(geonames_data)))
    
    ## load OSM data
    logging.debug('%d geonames candidates'%(len(geonames_data)))
    osm_cols = ['lat', 'lon', 'name']
    osm_data = pd.read_csv(args['osm_data'], sep='\t', usecols=osm_cols)
    name_col = 'name'
    osm_data = osm_data[osm_data.loc[:, name_col].apply(lambda x: type(x) is str)]
    # relabel names
    osm_data = osm_data.assign(**{name_col : osm_data.loc[:, name_col].apply(clean_osm_name)})
    # organize by name
    osm_data_dict = {k : v.drop(name_col, axis=1, inplace=False) for k, v in osm_data.groupby(name_col)}
    logging.debug('%d OSM candidates'%(len(osm_data)))
    
    ## filter for LOC, ORG NEs
    LOC_TAGS = ['LOCATION', 'ORGANIZATION', 'CITY']
    tag_var_LOC = '%s_LOC'%(tag_var)
    tag_data = tag_data.assign(**{tag_var_LOC : tag_data.loc[:, tag_var].apply(lambda x: [y[0] for y in x if y[1] in LOC_TAGS])})
#     tag_data = tag_data.assign(**{'%s_LOC_GN'%(tag_var) : tag_data.loc[:, '%s_LOC'%(tag_var)].apply(lambda x: [y for y in x if clean_geonames_name(y) in geonames_data.keys()])})
#     tag_data = tag_data.assign(**{'%s_LOC_OSM'%(tag_var) : tag_data.loc[:, '%s_LOC'%(tag_var)].apply(lambda x: [y for y in x if clean_osm_name(y) in osm_data_dict.keys()])})
#     tag_data_GN_loc = pd.Series(list(reduce(lambda x,y: x+y, tag_data.loc[:, '%s_LOC_GN'%(tag_var)].values)))
#     tag_data_OSM_loc = pd.Series(list(reduce(lambda x,y: x+y, tag_data.loc[:, '%s_LOC_OSM'%(tag_var)].values)))
#     logging.debug('top GeoNames LOC =\n%s'%(tag_data_GN_loc.value_counts().iloc[:100]))
#     logging.debug('top OSM LOC =\n%s'%(tag_data_OSM_loc.value_counts().iloc[:100]))
    
    ## flatten
    tag_data_flat = []
    for i, data_i in tag_data.iterrows():
        data_i_flat = [pd.Series(list(data_i) + [int(clean_geonames_name(x) in geonames_data.keys())]+[int(clean_osm_name(x) in osm_data_dict.keys())] + [x], index=list(data_i.index)+['LOC_GN', 'LOC_OSM', 'NE']) for x in data_i.loc[tag_var_LOC]]
        tag_data_flat += data_i_flat
    tag_data_flat = pd.concat(tag_data_flat, axis=1).transpose()
    logging.debug('tag data flat:\n%s'%(tag_data_flat.head()))
    logging.debug('GN=%d, OSM=%d, total=%d'%(tag_data_flat.loc[:, 'LOC_GN'].sum(), tag_data_flat.loc[:, 'LOC_OSM'].sum(), tag_data_flat.shape[0]))
    
    ## TODO: add max importance vars where possible
    
    ## write to file
    tag_data_flat_out_file = args['tag_data_file'].replace('.tsv', '_valid.tsv')
    tag_data_flat.to_csv(tag_data_flat_out_file, sep='\t', index=False)
    
if __name__ == '__main__':
    main()