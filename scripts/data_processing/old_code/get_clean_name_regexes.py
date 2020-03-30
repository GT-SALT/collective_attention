"""
Get clean strings from all names and alternate names
in GeoNames, combine into a single regex for 
faster matching, save all ID/regex pairs to file.
"""
import pandas as pd
import os, re
from unidecode import unidecode
from argparse import ArgumentParser
from data_helpers import load_simplified_geonames_data, query_norm

## string parsing helpers
# PUNCT_MATCHER = re.compile('[\.,\*\?\+"\']+')
# def normalize_str(s):
#     """
#     Normalize string
#     """
#     s = s.lower()
#     s = unidecode(s)
#     s = PUNCT_MATCHER.sub('', s)
#     s = re.escape(s)
#     # fix space escapes
#     s = s.replace('\ ', ' ')
#     return s
def get_all_names(g_row):
    """
    Extract and normalize all names
    from GeoNames row.
    
    Parameters:
    -----------
    g_row : pandas.Series
    
    Returns:
    --------
    all_names : list
    """
    base_name = g_row.loc['name']
    all_names = []
    if(g_row.loc['alternate_names'] != ''):
        alt_names = filter(lambda x: x != base_name, g_row.loc['alternate_names'].split(','))
        all_names = alt_names
        # normalize names first
        all_names = list(set(map(lambda x: query_norm(x, regex=True), all_names)))
    # stick original name onto front of list
    all_names = [query_norm(base_name, regex=True)] + all_names
    return all_names

def get_id_regex_pairs(geonames_data):
    alt_names_regex = geonames_data.apply(lambda x: re.compile('|'.join(get_all_names(x))), axis=1)
    id_regex_pairs = zip(geonames_data.loc[:, 'geonames_ID'], alt_names_regex)
    return id_regex_pairs

def main():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='/hg190/corpora/GeoNames')
    args = parser.parse_args()
    out_dir = args.out_dir
    
    ## load data
    geonames_data = load_simplified_geonames_data()
    geonames_data.fillna('', inplace=True)
    geonames_data.loc[:, 'alternate_names'] = geonames_data.loc[:, 'alternate_names'].apply(lambda s: str(s) if type(s) is not str and type(s) is not unicode else s)
    
    ## get names
    id_regex_pairs = get_id_regex_pairs(geonames_data)
    # convert to data frame
    geonames_id_regex_df = pd.DataFrame([[x, y.pattern] for x,y in id_regex_pairs])
    geonames_id_regex_df.columns = ['geonames_ID', 'name_regex']
    
    ## write to file
    out_file = os.path.join(out_dir, 'geonames_clean_combined_names.tsv')
    geonames_id_regex_df.to_csv(out_file, sep='\t', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()