"""
Extact alternative names from GeoNames data
into an easier-to-read format: 
columns = geonames_ID, name, alternative name.
"""
from data_helpers import load_simplified_geonames_data
from argparse import ArgumentParser
import os
import pandas as pd
from itertools import izip  

def main():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='/hg190/corpora/GeoNames/')
    args = parser.parse_args()
    out_dir = args.out_dir
    
    ## load data
    geonames_data = load_simplified_geonames_data()
    geonames_data.fillna('', inplace=True)    
    
    ## separate alternative names
    name_data = []
#     cutoff = 1000
    for i, (idx, g_row) in enumerate(geonames_data.iterrows()):
        g_id = g_row.loc['geonames_ID']
        real_name = g_row.loc['name']
        alt_names = g_row.loc['alternate_names']
        alt_names = alt_names.split(',')
        alt_names = filter(lambda x: x != '' and x != real_name, alt_names)
        name_data.append([[g_id, real_name, alt_name] for alt_name in alt_names])
        if(i % 100000 == 0):
            print('processed %d names'%(i))
#         if(i >= cutoff):
#             break
    name_data = pd.DataFrame(name_data)
    name_data.columns = ['geonames_ID', 'name', 'alternate_name']
    
    ## write to file
    out_file = os.path.join(out_dir, 'alternate_names.tsv')
    name_data.to_csv(out_file, sep='\t', index=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()