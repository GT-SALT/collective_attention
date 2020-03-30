"""
Get acronyms for all GeoNames toponyms:
"United States of America" => "USA".
"""
from data_helpers import load_simplified_geonames_data
import os
import pandas as pd
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='/hg190/corpora/GeoNames/')
    args = parser.parse_args()
    out_dir = args.out_dir
    
    ## load data
    geonames_data = load_simplified_geonames_data()
    
    ## extract acronyms
    acronym_data = []
#     cutoff = 1000
    for i, (g, g_row) in enumerate(geonames_data.iterrows()):
        g_id = g_row.loc['geonames_ID']
        g_name = g_row.loc['name']
        g_name_split = filter(lambda x: x != '', g_name.split(' '))
        if(len(g_name_split) > 1):
            try:
                g_acronym = ''.join(map(lambda x: x[0].upper(), g_name_split))
                acronym_data.append([g_id, g_name, g_acronym])
            except Exception, e:
                print(g_name_split)
#         if(i > cutoff):
#             break
    acronym_data = pd.DataFrame(acronym_data)
    acronym_data.columns = ['geonames_ID', 'name', 'acronym']
    
    ## write to file
    out_file = os.path.join(out_dir, 'acronyms.tsv')
    acronym_data.to_csv(out_file, sep='\t', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()