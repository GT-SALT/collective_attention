"""
Combine all OSM data from .xml and .shp files,
giving precedence to .shp files in case of ID collision
because the .shp files have cleaner data.
"""
from argparse import ArgumentParser
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('--shp_data_file', default='../../data/geo_files/PR_OSM/shp_files/combined_shp_data.tsv')
    parser.add_argument('--xml_data_files', nargs='+', 
                        default=['../../data/geo_files/PR_OSM/xml_files/xml_node_data.tsv',
                                 '../../data/geo_files/PR_OSM/xml_files/xml_way_data.tsv',
                                 '../../data/geo_files/PR_OSM/xml_files/xml_relation_data.tsv'])
    parser.add_argument('--out_file', default='../../data/geo_files/PR_OSM/combined_shp_xml_data.tsv')
    args = parser.parse_args()
    shp_data_file = args.shp_data_file
    xml_data_files = args.xml_data_files
    out_file = args.out_file
    
    ## load data
    shp_data = pd.read_csv(shp_data_file, sep='\t', index_col=False, encoding='utf-8')
    xml_data = pd.concat([pd.read_csv(f, sep='\t', index_col=False, encoding='utf-8') for f in xml_data_files], axis=0)
    
    ## remove duplicate IDs
    duplicate_IDs = set(shp_data.loc[:, 'osm_id'].values.tolist()) & set(xml_data.loc[:, 'id'].values.tolist())
    xml_data = xml_data[~ xml_data.loc[:, 'id'].isin(duplicate_IDs)]
    xml_data.rename(columns={'id' : 'osm_id'}, inplace=True)
    xml_data.rename(columns={'lat_lons' : 'all_points'}, inplace=True)
    combined_data = pd.concat([shp_data, xml_data], axis=0).fillna('', inplace=False)
    
    ## write
    combined_data.to_csv(out_file, sep='\t', index=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()