"""
Simplify the complicated GeoNames data that
we downloaded from here: http://download.geonames.org/export/dump/allCountries.zip
"""
import pandas as pd
from argparse import ArgumentParser
from zipfile import ZipFile
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--geonames_file', default='/hg190/corpora/GeoNames/allCountries.zip')
    parser.add_argument('--out_file', default='/hg190/corpora/GeoNames/allCountriesSimplified.tsv')
    args = parser.parse_args()
    geonames_file = args.geonames_file
    geonames_unarchived_file = os.path.basename(geonames_file).replace('.zip', '.txt')
    out_file = args.out_file
    
    ## load data, but only keeping the necessary rows:
    ## 0=ID, 1=name, 3=alternate names, 4=latitude, 5=longitude, 6=feature class, 7=feature_code, 14=population
    use_cols = [0, 1, 3, 4, 5, 6, 7, 8, 14]
    col_names = ['geonames_ID', 'name', 'alternate_names', 'latitude', 'longitude', 'feature_class', 'feature_code', 'country', 'population']
    # can't read csv like this; it drops rows without warning possibly due to extra spacing
    #     geonames_data = pd.read_csv(geonames_file, sep='\t', compression='zip', usecols=use_cols, header=None)
    # read one row at a time ;_;
    geonames_data = []
    
    with ZipFile(geonames_file, 'r') as geo_archive:
        geo_input = geo_archive.open('allCountries.txt')
        for i, l in enumerate(geo_input):
            l_vals = l.split('\t')
            use_vals = [l_vals[u] for u in use_cols]
            geonames_data.append(use_vals)
            if(i % 1000000 == 0):
                print('processed %d lines'%(i))
    geonames_data = pd.DataFrame(geonames_data, columns=col_names)
    print('loaded %d GeoNames entries'%(geonames_data.shape[0]))
    geonames_data.columns = col_names
    geonames_data.loc[:, 'alternate_names'] = geonames_data.loc[:, 'alternate_names'].apply(lambda x: str(x))
    # add country names to alternate names
    # for toponyms with feature class == country = {PCLI, PCLD, PCLS, PCLF, PCL, PCLIX}
    country_feature_codes = set(['PCL', 'PCLD', 'PCLF', 'PCLI', 'PCLIX', 'PCLS'])
    updated_alternate_names = geonames_data.loc[:, ['alternate_names', 'feature_code', 'country']].apply(lambda x: ','.join(x.loc['alternate_names'].split(',') + [x.loc['country']]) if x.loc['feature_code'] in country_feature_codes else x.loc['alternate_names'], axis=1)
    geonames_data.loc[:, 'alternate_names'] = updated_alternate_names
    # generate extra column for alternate name count
    geonames_data.loc[:, 'alternate_name_count'] = geonames_data.loc[:, 'alternate_names'].apply(lambda x: len(str(x).split(',')))
    
    ## write to file
    geonames_data.to_csv(out_file, sep='\t', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()