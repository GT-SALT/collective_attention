"""
Check validated NEs for containment within FB groups' 
associated municipalities.
"""
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import logging
import os
import fiona
from shapely.geometry import Polygon, MultiPolygon, Point
from unidecode import unidecode
import pickle
import re

def invert_coords(coords):
    return [(y,x) for x,y in coords]

def convert_coords_to_shp(coords, shp_type='Polygon'):
    if(shp_type=='MultiPolygon'):
        return MultiPolygon([Polygon(invert_coords(y[0])) for y in coords])
    return Polygon(invert_coords(coords[0]))

def convert_data_to_shp(data):
    geo = data['geometry']
    coords = geo['coordinates']
    shp = convert_coords_to_shp(coords, shp_type=geo['type'])
    return shp

def add_abbreviated_locations_to_gazetteer(data, abbreviation_words=['barrio']):
    """
    Add abbreviated locations to gazetteer by collecting all entries containing
    abbreviation and updating the entry without the abbreviation.
    Ex. "quebradillas barrio" => "quebradillas" => update the "quebradillas" entry

    :param data: gazetteer dict
    :param abbreviation_words: words to remove from gazetteer entries
    :returns data: updated gazetteer dict
    """
    for w in abbreviation_words:
        w_matcher = re.compile('%s$|^%s'%((w,)*2))
        k_match = [(k,v) for k,v in data.items() if w_matcher.search(k) is not None]
        for (k,v) in k_match:
            k_abbreviation = w_matcher.sub('', k).strip()
            if(k_abbreviation not in data):
                data[k_abbreviation] = v
            else:
                data[k_abbreviation] = pd.concat([v, data[k_abbreviation]], axis=0)
    return data

def group_contains_NE(NE, group_loc_names, GN_data, OSM_data, PR_shp_lookup, verbose=False):
    """
    Check if group municipalities contain the NE mentioned.
    
    :param NE: NE to check
    :param group_loc_names: group municipality names
    :param GN_data: GeoNames data
    :param OSM_data: OSM data
    :param PR_shp_lookup: shape files for all municipalities
    :returns contains_NE:: if group contains NE
    """
    # get combined shape for group locations
    group_shp = MultiPolygon([PR_shp_lookup[x] for x in group_loc_names])
    # disambiguate NE to valid location using GN/OSM data
    NE_coords = [[-300, -300]] # by default, shape cannot contain coordinate
    if(NE in GN_data.keys()):
        NE_coords = GN_data[NE]
    elif(NE in OSM_data.keys()):
        NE_coords = OSM_data[NE]
    # containment: if any of the coordinates are contained in group
    if(verbose):
        logging.debug('NE=%s, group shape %s'%(NE, group_shp))
        logging.debug('NE=%s, NE_coords=%s'%(NE, str(NE_coords)))
    contains_NE = any([group_shp.contains(Point(NE_coord)) for NE_coord in NE_coords])
    return contains_NE
    
def get_most_likely_locs(loc_data, importance_var='population'):
    """
    Get most likely locations for set of locations,
    based on max importance.
    
    :param loc_data: location data
    :param importance_var: inferred importance (ex. population)
    :returns loc_coords:: coordinates of maximum importance location
    """
    max_importance = loc_data.loc[:, importance_var].min()
    loc_matches = loc_data[loc_data.loc[:, importance_var]==max_importance]
    loc_coords = loc_matches.loc[:, ['latitude', 'longitude']].values
    return loc_coords
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--county_shp_file', default='../../data/geo_files/county_shape_files/tl_2018_us_county.shp')
#     parser.add_argument('--NE_file', default='../../data/facebook-maria/combined_group_data_es_tagged_valid_anchor.tsv')
    parser.add_argument('--NE_file', default='../../data/facebook-maria/combined_group_data_es_tagged_parsed_spacy_anchor.tsv')
    parser.add_argument('--geonames_file', default='/hg190/corpora/GeoNames/allCountriesSimplified_lookup_US.pickle')
    parser.add_argument('--OSM_file', default='../../data/geo_files/PR_OSM/combined_shp_xml_data.tsv')
    parser.add_argument('--group_metadata_file', default='../../data/facebook-maria/location_group_data.tsv')
    args = vars(parser.parse_args())
    
    logging_file = '../../output/check_NE_group_containment.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    ## process shape data
    county_shp_data = fiona.open(args['county_shp_file'])
    PR_code = 72
    PR_shp = [x[1] for x in county_shp_data.items() if x[1]['properties']['STATEFP']==str(PR_code)]
    logging.debug('%d/%d PR counties'%(len(PR_shp), len(county_shp_data)))
    PR_shp_names = [x['properties']['NAME'] for x in PR_shp]
    PR_shp_names = [unidecode(x) for x in PR_shp_names]
    PR_shp_data = [convert_data_to_shp(x) for x in PR_shp]
    PR_shp_lookup = {k : v for k,v in zip(PR_shp_names, PR_shp_data)}
    logging.debug('shp names = %s'%(', '.join(sorted(PR_shp_names))))
    ## test containment
    test_name = 'Quebradillas'
    test_coord = Point([18.471211, -66.937679])
    logging.debug('test shp = %s'%(PR_shp_lookup[test_name]))
    assert PR_shp_lookup[test_name].contains(test_coord)
    
    ## load data
    NE_data = pd.read_csv(args['NE_file'], sep='\t', index_col=False)
    geonames_data = pickle.load(open(args['geonames_file'], 'rb'))
    # restrict to PR locations
    PR_country = 'PR'
    geonames_data = {k : v[v.loc[:, 'country']==PR_country] for k,v in geonames_data.items()}
    geonames_data = {k : v for k,v in geonames_data.items() if v.shape[0] > 0}
    # add entries to gazetteer according to abbreviations
    # ex. "Quebradillas Barrio" => "Quebradillas" if entry does not exist 
#     geonames_data = add_abbreviations_to_gazetteer(geonames_data)
    abbreviation_words = ['barrio', '-pueblo']
    geonames_data = add_abbreviated_locations_to_gazetteer(geonames_data, abbreviation_words=abbreviation_words)
    OSM_data = pd.read_csv(args['OSM_file'], sep='\t', index_col=False)
    group_metadata = pd.read_csv(args['group_metadata_file'], sep='\t', index_col=False)
    group_var = 'group_id'
    loc_name_var = 'location_name'
    group_name_var = 'group_name'
    NE_var = 'NE'
    group_metadata = pd.DataFrame([[i, [unidecode(y) for y in x.loc[:, loc_name_var].values], x.loc[:, group_name_var].iloc[0]] for i,x in group_metadata.groupby(group_var)])
    group_metadata.columns = [group_var, loc_name_var, group_name_var]
    NE_data = pd.merge(NE_data, group_metadata, on=group_var)
    NE_data = NE_data.assign(**{NE_var : NE_data.loc[:, NE_var].apply(lambda x: unidecode(x.lower()))})
    # limit to valid locations
    logging.debug('%d/%d valid NEs'%(NE_data.loc[:, ['LOC_OSM', 'LOC_GN']].max(axis=1).sum(), NE_data.shape[0]))
    NE_data = NE_data[NE_data.loc[:, ['LOC_OSM', 'LOC_GN']].max(axis=1)==1]
    
    ## match each NE to its most likely location
    ## and check containment
    importance_var = 'population'
    # multiple loc candidates
    geonames_data_matches = {k : get_most_likely_locs(v, importance_var=importance_var) for k,v in geonames_data.items()}
    # one loc candidate
#     geonames_data_matches = {k : v.sort_values(importance_var, inplace=False, ascending=False).iloc[0].loc[['latitude', 'longitude']].values for k,v in geonames_data.items()}
    logging.debug('%d GeoNames data'%(len(geonames_data_matches)))
    coord_vars = ['lat', 'lon']
    # multiple NE candidates
    OSM_data_matches = {unidecode(k).lower() : v.loc[:, coord_vars].values for k,v in OSM_data.groupby('name')}
    # one loc candidate
#     OSM_data_matches = {unidecode(k).lower() : v.loc[:, coord_vars].values for k,v in OSM_data.groupby('name')}
    logging.debug('%d OSM data'%(len(OSM_data_matches)))
    NE_data = NE_data.assign(**{'group_contains_NE' : NE_data.apply(lambda x: group_contains_NE(x.loc[NE_var], x.loc[loc_name_var], geonames_data_matches, OSM_data_matches, PR_shp_lookup), axis=1)})
    ## test containment
    group_contains_NE('quebradillas barrio', ['Quebradillas'], geonames_data_matches, OSM_data_matches, PR_shp_lookup, verbose=True)
    assert group_contains_NE('quebradillas barrio', ['Quebradillas'], geonames_data_matches, OSM_data_matches, PR_shp_lookup)

    ## write to file
    out_file_name = args['NE_file'].replace('.tsv', '_group_contain.tsv')
    if(not os.path.exists(out_file_name)):
        NE_data.to_csv(out_file_name, sep='\t', index=False)
    
if __name__ == '__main__':
    main()