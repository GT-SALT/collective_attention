"""
Combine GeoNames and OSM data into one giant database with 
normalized names.
"""
from data_helpers import load_combined_geo_data
from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--out_file', default='../../data/geo_files/combined_geo_data.tsv')
    args = parser.parse_args()
    out_file = args.out_file
    
    ## load data
    if(os.path.exists(out_file)):
        os.remove(out_file)
    geo_data = load_combined_geo_data()
    
    ## write to file
    geo_data.to_csv(out_file, sep='\t', index=False)

if __name__ == '__main__':
    main()
