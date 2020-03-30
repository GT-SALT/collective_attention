"""
Extract anchor descriptor phrases from parses as follows:
[LOC_{OSM, GN}], [LOC_{GN}]_{acl,appos,prep,nmod}
Following the format from tests on annotated data:
scripts/data_processing/test_anchor_detection_on_annotated_twitter_data.ipynb#
"""
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval
from argparse import ArgumentParser
import os
import logging
from data_helpers import detect_anchor_by_type

def extract_anchors(data, id_var='id', parse_var='parse', anchor_var='max_population', data_var='data_name_fixed', txt_var='txt'):
    """
    Extract all anchor instances for NEs.
    """
    all_anchor_types = ['state', 'descriptor', 'compound', 'list']
    anchor_data = []
    ctr = 0
    id_ctr = data.loc[:, id_var].nunique()
    for id_i, data_i in data.groupby(id_var):
        anchor_state, anchor_descriptor, anchor_compound, anchor_list, anchor_phrases = detect_anchor_by_type(data_i, parse_var=parse_var, anchor_var=anchor_var, data_name_var=data_var, txt_var=txt_var, verbose=False)
        anchor_pred_i = pd.concat([anchor_state, anchor_descriptor, anchor_compound, anchor_list], axis=1)
        anchor_pred_i.columns = all_anchor_types
        anchor_pred_i = anchor_pred_i.loc[:, all_anchor_types].max(axis=1)
        # add NEs and phrases to data
        pred_i = pd.DataFrame(anchor_pred_i).assign(**{'NE' : anchor_pred_i.index, 'anchor_phrase' : anchor_phrases}).rename(columns={0 : 'anchor'}, inplace=False)
        pred_i.index = np.arange(pred_i.shape[0])
        data_i = pd.merge(data_i, pred_i, on='NE')
#         data_i.loc[:, '%s_anchor'%(anchor_var)] = pred_i.values
        anchor_data.append(data_i)
        ctr += 1
        if(ctr % 1000 == 0):
            print('processed %d/%d ID groups'%(ctr, id_ctr))
    anchor_data = pd.concat(anchor_data, axis=0)
    return anchor_data

def main():
    parser = ArgumentParser()
    # tag data
    parser.add_argument('--tag_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat.gz')
    # power user data
#     parser.add_argument('--tag_data', default='../../data/mined_tweets/combined_data_power_user_NE_flat.gz')
    # parse data
    parser.add_argument('--parse_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed.gz')
    # power user data
#     parser.add_argument('--parse_data', default='../../data/mined_tweets/combined_data_power_user_NE_flat_parsed.gz')
    # gazetteer data
    parser.add_argument('--geonames_data', default='/hg190/corpora/GeoNames/allCountriesSimplified_lookup_US.pickle')
    args = vars(parser.parse_args())
    log_file_name = '../../output/extract_anchors_in_twitter_data.txt'
    if(os.path.exists(log_file_name)):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    
    ## load importance data
    ## TODO: customize importance per data set
    geonames_data = pickle.load(open(args['geonames_data'], 'rb'))
    geonames_max_pop = {k : v.loc[:, 'population'].max() for k,v in geonames_data.items()}
    geonames_max_alt_names = {k : v.loc[:, 'alternate_name_count'].max() for k,v in geonames_data.items()}
    
    ## load data
    tag_data = pd.read_csv(args['tag_data'], sep='\t', index_col=False, compression='gzip')
    # remove NAN NEs
    tag_data = tag_data[~tag_data.loc[:, 'NE'].apply(lambda x: np.isnan(x) if type(x) is float else False)]
    tag_data = tag_data[~tag_data.loc[:, 'NE_fixed'].apply(lambda x: np.isnan(x) if type(x) is float else False)]
    # fix NE names
    tag_data = tag_data.assign(**{'NE' : tag_data.loc[:, 'NE'].apply(lambda x: x.replace('_', ' '))})
    tag_data = tag_data.assign(**{'NE_fixed' : tag_data.loc[:, 'NE_fixed'].apply(lambda x: x.replace('_', ' '))})
    # fix parse data
    parse_data = pd.read_csv(args['parse_data'], sep='\t', index_col=False, compression='gzip', converters={'parse' : lambda x: literal_eval(x), 'id' : np.int64})
    # flatten to one ID per row
    parse_data = pd.concat([pd.Series([i, list(x.loc[:, 'parse'].values)], index=['id', 'parse']) for i,x in parse_data.groupby('id')], axis=1).transpose()
    # merge
    tag_parse_data = pd.merge(tag_data, parse_data, on='id', how='inner')
    logging.debug('merged %d NEs'%(tag_parse_data.shape[0]))
    logging.debug('top NEs after merge %s'%(tag_parse_data.loc[:, 'NE'].value_counts().head(100)))
    
    ## add importance stats
    tag_parse_data = tag_parse_data.assign(**{
        'max_population' : tag_parse_data.loc[:, 'NE_fixed'].apply(lambda x: geonames_max_pop[x] if x in geonames_max_pop else 0.),
        'max_alternate_names' : tag_parse_data.loc[:, 'NE_fixed'].apply(lambda x: geonames_max_alt_names[x] if x in geonames_max_alt_names else 0.),
    })
    
    ## extract anchors
    anchor_var = 'max_population'
    tag_parse_data = extract_anchors(tag_parse_data, anchor_var=anchor_var)
    
    ## write to file
    out_file = args['parse_data'].replace('.gz', '_anchor.gz')
    if(not os.path.exists(out_file)):
        tag_parse_data.to_csv(out_file, sep='\t', index=False, compression='gzip')
    
if __name__ == '__main__':
    main()