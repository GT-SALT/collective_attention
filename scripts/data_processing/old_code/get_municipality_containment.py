"""
For each location in the combined dataset, extract the municipality most likely to contain
that location's centroid (precomputed).
"""
from argparse import ArgumentParser
import os, re
import json
import pandas as pd
import fiona
from shapely.geometry import shape, Point

def containment_test(lon_lat, municipality_shapes):
    """
    Find municipality that contains the specified lat/lon point.
    
    Parameters:
    -----------
    lon_lat : [float]
    municipality_shapes : list
    Name, Polygon pairs.
    
    Returns:
    --------
    municipality_name : str
    Name of best guess for municipality containing point;
    returns None if no municipality contains point.
    """
    point = Point(lon_lat)
    for municipality_name, municipality_shape in municipality_shapes:
        if(municipality_shape.contains(point)):
            return municipality_name
    return None

def main():
    parser = ArgumentParser()
#     parser.add_argument('--shp_dir', default='../../data/geo_files/PR_OSM/shp_files/')
    parser.add_argument('--muni_shape_file', default='../../data/geo_files/county_shape_files/cb_2016_us_county_500k.shp')
    parser.add_argument('--group_location_file', default='../../data/facebook-maria/facebook_group_location_data.tsv')
    parser.add_argument('--geo_data_file', default='../../data/geo_files/combined_geo_data.tsv')
    args = parser.parse_args()
#     shp_dir = args.shp_dir
    muni_shape_file = args.muni_shape_file
    group_location_file = args.group_location_file
    geo_data_file = args.geo_data_file
    out_dir = os.path.dirname(geo_data_file)
    
    ## load data
#     file_matcher = re.compile('*.places.*.shp')
#     place_files = [os.path.join(shp_dir, f) for f in os.listdir(shp_dir) if file_matcher.match(f)]
#     place_list = reduce(lambda x,y: x+y, [list(fiona.open(f)) for f in place_files])
#     place_names = filter(lambda x: x!='', sorted([p['properties']['name'] for p in place_list]))
    county_shapes = fiona.open(muni_shape_file)
    county_shape_list = list(county_shapes)
    PR_state_FP = "72"
    PR_county_shape_list = filter(lambda x: x['properties']['STATEFP'] == PR_state_FP, county_shape_list)
    municipality_geoms = [(c['properties']['NAME'], c['geometry']) for c in PR_county_shape_list]
    municipality_shapes = [(n, shape(g)) for n,g in municipality_geoms]
    municipality_shape_dict = dict(municipality_shapes)
    group_data = pd.read_csv(group_location_file, sep='\t', index_col=0)
    geo_data = pd.read_csv(geo_data_file, sep='\t', index_col=False, encoding='utf-8', low_memory=False)
    
    ## containment test
    # apply containment test to all locations in combined data
    test_municipality_name = 'Guayama'
    test_location = geo_data[geo_data.loc[:, 'name'] == test_municipality_name].loc[:, ['lon', 'lat']].values[0]
    test_municipality = containment_test(test_location, municipality_shapes)
    assert test_municipality == test_municipality_name
    geo_municipalities = geo_data.loc[:, ['lon', 'lat']].apply(lambda p: containment_test(p.values.tolist(), municipality_shapes), axis=1)
    
    ## write to file
    geo_municipalities = pd.concat([geo_data.loc[:, 'id'], geo_municipalities], axis=1)
    geo_municipalities.columns = ['id', 'municipality']
    print(geo_municipalities.head())
    print(geo_municipalities.loc[:, 'municipality'].value_counts())
    out_file = os.path.join(out_dir, 'geo_data_municipalities.tsv')
    geo_municipalities.to_csv(out_file, sep='\t', index=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()