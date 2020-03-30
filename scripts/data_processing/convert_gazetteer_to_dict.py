"""
Convert flat gazetteer file to dict for faster lookup.
"""
from data_helpers import load_gazetteer
from argparse import ArgumentParser
import pickle
from collections import defaultdict
from unidecode import unidecode
import pandas as pd
import logging
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--gazetteer_file', default='/hg190/corpora/GeoNames/allCountriesSimplified.tsv')
    args = vars(parser.parse_args())
    if(os.path.exists('../../output/convert_gazetteer_to_dict.txt')):
        os.remove('../../output/convert_gazetteer_to_dict.txt')
    logging.basicConfig(filename='../../output/convert_gazetteer_to_dict.txt', level=logging.DEBUG)
    
    ## load data
    gazetteer_data = load_gazetteer(args['gazetteer_file'])
    gazetteer_data.fillna('', inplace=True)
    # tmp debugging
#     gazetteer_data = gazetteer_data.head(1000)
    gazetteer_dict_file = args['gazetteer_file'].replace('.tsv', '_lookup.pickle')
    
    ## collect data for each unique name 
    if(not os.path.exists(gazetteer_dict_file)):
        gazetteer_dict = defaultdict(list)
        for i, (idx_i, gazetteer_entry_i) in enumerate(gazetteer_data.iterrows()):
            names_i = set([gazetteer_entry_i.loc['name']]) | set([x for x in gazetteer_entry_i.loc['alternate_names'].split(',') if x != ''])
            names_i = set([unidecode(x.lower()) for x in names_i])
            for name_j in names_i:
                gazetteer_dict[name_j].append(gazetteer_entry_i)
            if(i % 100000 == 0):
                logging.debug('processed %d/%d gazetteer entries'%(i, gazetteer_data.shape[0]))
        # list => Dataframe
        gazetteer_dict = {k : pd.concat(v, axis=1).transpose() for k,v in gazetteer_dict.items()}
        ## save
        pickle.dump(gazetteer_dict, open(gazetteer_dict_file, 'wb'))
    else:
        gazetteer_dict = pickle.load(open(gazetteer_dict_file, 'rb'))
    logging.debug('%d entries total'%(len(gazetteer_dict)))
    
    ## save US only version with specified locations for easier access
    loc_feat_codes=set(['ADM1', 'ADM2', 'ADM3', 'ADM4', 'ADM5', 'ADMD', 'BLDG', 'CH', 'CMTY', 'ISL', 'ISLS', 'LCTY', 'PCLD', 'PCLH', 'PCLI', 'PPL', 'PPLA', 'PPLA2', 'PPLA3', 'PPLA4', 'PPLC', 'PPLG', 'PPLL', 'PPLS', 'PPLX', 'SCH'])
    gazetteer_dict_US_file = args['gazetteer_file'].replace('.tsv', '_lookup_US.pickle')
    if(not os.path.exists(gazetteer_dict_US_file) or True):
        US_countries = set(['US', 'PR'])
        gazetteer_dict_US = {k : v[(v.loc[:, 'country'].isin(US_countries)) & (v.loc[:, 'feature_code'].isin(loc_feat_codes))] for k,v in gazetteer_dict.items()}
        gazetteer_dict_US = {k : v for k,v in gazetteer_dict_US.items() if v.shape[0] > 0}
        logging.debug('%d/%d entries in US dict'%(len(gazetteer_dict_US), len(gazetteer_dict)))
        pickle.dump(gazetteer_dict_US, open(gazetteer_dict_US_file, 'wb'))
    
if __name__ == '__main__':
    main()