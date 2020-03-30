"""
Extract anchor descriptor phrases from parses as follows:
[LOC_{OSM, GN}], [LOC_{GN}]_{acl,appos,prep,nmod}
"""
from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
import re
from data_helpers import extract_full_tree, extract_NE_subtree
from extract_descriptors_in_twitter_data import extract_anchors
from nltk.tokenize.toktok import ToktokTokenizer
import os
from unidecode import unidecode
import pickle

def extract_NE_subtrees(data):
    """
    Extract subtrees and trees for all NEs in data.
    
    :param data: pandas.DataFrame with NE, parses
    :returns data_trees:: pandas.DataFrame with subtrees
    """
    data_trees = []
    for id_i, data_i in data.groupby('status_id'):
        NE_i = data_i.loc[:, 'NE'].values
        NE_i = [x.replace('_', ' ') for x in NE_i]
        trees = data_i.loc[:, 'parse'].iloc[0]
        tree_graphs = [extract_full_tree(x) for x in trees]
        tree_ctr = 0
        tree_token_ctr = 0
        data_trees_i = []
        data_subtrees_i = []
#         print('trees %s'%(str(trees)))
#         print('tree graphs %s'%(str([list(x.edges) for x in tree_graphs])))
        for NE_j in NE_i:
#             print('testing NE=%s, tree=%d, tree_token=%d'%(NE_j, tree_ctr, tree_token_ctr))
            data_NE_tree, data_NE_subtree, NE_tree_ctr, tree_ctr, tree_token_ctr = extract_NE_subtree(NE_j, trees, tree_graphs, tree_ctr, tree_token_ctr)
            data_trees_i.append(data_NE_tree)
            data_subtrees_i.append(data_NE_subtree)
        data_i = data_i.assign(**{'subtree' : data_subtrees_i})
        data_i = data_i.assign(**{'tree' : data_trees_i})
        data_trees.append(data_i)
    data_trees = pd.concat(data_trees, axis=0)
    return data_trees

PARSE_MATCHER = re.compile('(.+)/(.+)/(.+)/(.+)/(.+)')
def main():
    parser = ArgumentParser()
    parser.add_argument('--tag_data_file', default='../../data/facebook-maria/combined_group_data_es_tagged_valid.tsv')
#     parser.add_argument('--parse_data_file', default='../../data/facebook-maria/combined_group_data_es_tagged_parsed.txt')
    parser.add_argument('--parse_data_file', default='../../data/facebook-maria/combined_group_data_es_tagged_parsed_spacy.txt')
    parser.add_argument('--geonames_data', default='/hg190/corpora/GeoNames/allCountriesSimplified_lookup_US.pickle')
    args = vars(parser.parse_args())
    log_file_name = '../../output/extract_anchors_in_FB_data.txt'
    if(os.path.exists(log_file_name)):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    
    ## load tagged/marked flat NE data
    tag_cols = ['group_id', 'status_author_id', 'status_message', 'status_id', 'NE', 'status_published', 'LOC_GN', 'LOC_OSM']
    tag_data = pd.read_csv(args['tag_data_file'], sep='\t', index_col=False, usecols=tag_cols)
    # restrict to valid NEs
    tag_data = tag_data[tag_data.loc[:, ['LOC_GN', 'LOC_OSM']].sum(axis=1) > 0.]
    logging.debug(tag_data.head())
    logging.debug('%d NEs'%(tag_data.shape[0]))
    parse_data = pd.read_csv(args['parse_data_file'], sep='\t', index_col=False, header=None)
    parse_data.columns = ['status_id', 'parse']
    # TODO: remove NAN?
    # extract parse data
#     parse_data = parse_data.assign(**{'parse' : parse_data.loc[:, 'parse'].apply(lambda x: [list(PARSE_MATCHER.findall(y)[0]) + [i] for i, y in enumerate(x.split(' ')) if PARSE_MATCHER.search(y) is not None])})
    parse_data = parse_data.assign(**{'parse' : parse_data.loc[:, 'parse'].apply(lambda x: [list(PARSE_MATCHER.findall(y)[0]) for y in x.split(' ') if PARSE_MATCHER.search(y) is not None])})
    # add numbers
#     parse_data = parse_data.assign(**{'parse' : parse_data.loc[:, 'parse'].apply(lambda x: [[y[0], y[1], int(y[2]), y[3], y[4]] for y in x])})
    parse_data = parse_data.assign(**{'parse' : parse_data.loc[:, 'parse'].apply(lambda x: [[y[0], y[1], int(y[2]), y[3], int(y[4])] for y in x])})
#     print('parse = %s'%(parse_data.loc[:, 'parse'].head(5)))
    # make it one line per parse
    parse_data = pd.DataFrame([[i, list(x.loc[:, 'parse'].values)] for i, x in parse_data.groupby('status_id')], columns=['status_id', 'parse'])
    logging.debug(parse_data.head())
    logging.debug('%d parses'%(parse_data.shape[0]))
    tag_parse_ids = set(tag_data.loc[:, 'status_id'].unique()) & set(parse_data.loc[:, 'status_id'].unique())
    logging.debug('%d shared IDs'%(len(tag_parse_ids)))
    
    # merge tag/parse data
    tag_data_1 = tag_data[tag_data.loc[:, 'status_id'].isin(tag_parse_ids)]
    parse_data_1 = parse_data[parse_data.loc[:, 'status_id'].isin(tag_parse_ids)]
    tag_parse_data = pd.merge(tag_data_1, parse_data_1, on='status_id', how='inner')
    logging.debug('%d/%d merged data'%(tag_parse_data.shape[0], tag_data.shape[0]))
    logging.debug(tag_parse_data.loc[:, ['status_id', 'status_message', 'NE', 'parse']].head(20))
    logging.debug(tag_parse_data.loc[:, ['status_id', 'status_message', 'NE', 'parse']].tail(20))

    ## extract data for context annotation
    parse_data_file = args['parse_data_file']
    annotate_out_file = parse_data_file.replace('.txt', '_annotation.tsv')
    if(not os.path.exists(annotate_out_file)):
        np.random.seed(123)
        sample_size = 100
        tag_parse_data_sample = tag_parse_data.loc[np.random.choice(tag_parse_data.index, sample_size, replace=False), :]
        tag_parse_data_sample = tag_parse_data_sample.assign(**{'state_gold':0, 'descriptor_gold':0})
        tag_parse_data_sample.to_csv(annotate_out_file, sep='\t', index=False)
    
    ## extract descriptor phrases
    ## states: look for LOC, STATE patterns
#     states = ['PR', 'Puerto Rico']
#     state_pattern = [',?\s?%s'%(x) for x in states]
#     # need to tokenize status message for proper matching
#     TKNZR = ToktokTokenizer()
#     tag_parse_data = tag_parse_data.assign(**{'status_message_tokenized' : tag_parse_data.loc[:, 'status_message'].apply(lambda x: ' '.join(TKNZR.tokenize(x)))})
#     tag_parse_data = tag_parse_data.assign(**{'descriptor_state' : tag_parse_data.apply(lambda x: int(re.compile('|'.join(['%s,?\s?%s'%(x.loc['NE'], y) for y in states]).lower()).search(x.loc['status_message_tokenized'].lower()) is not None), axis=1)})
#     logging.debug('%d/%d data with descriptor_state'%(tag_parse_data.loc[:, 'descriptor_state'].sum(), tag_parse_data.shape[0]))
#     sample_size = 10
#     tag_parse_data_with_descriptor = tag_parse_data[tag_parse_data.loc[:, 'descriptor_state'] == 1]
#     for i in range(sample_size):
#         data_i = tag_parse_data_with_descriptor.iloc[i, :]
#         logging.debug('NE=%s, message=%s'%(data_i.loc['NE'], data_i.loc['status_message']))
    
    ## descriptors: find children phrases for all NEs
    tag_parse_data = extract_NE_subtrees(tag_parse_data)
    # debug: determine how many NEs may have valid anchors
    tag_parse_data_with_tree = tag_parse_data[tag_parse_data.loc[:, 'tree'].apply(lambda x: len(x) > 0)]
    tag_parse_data_with_subtree = tag_parse_data[tag_parse_data.loc[:, 'subtree'].apply(lambda x: len(x) > 0)]
    logging.debug('%d/%d data with tree'%(tag_parse_data_with_tree.shape[0], tag_parse_data.shape[0]))
    logging.debug('%d/%d data with subtree'%(tag_parse_data_with_subtree.shape[0], tag_parse_data.shape[0]))
    sample_size = 10
    for i in range(sample_size):
        data_i = tag_parse_data_with_tree.iloc[i, :]
        logging.debug('NE=%s, tree=%s'%(data_i.loc['NE'], data_i.loc['tree']))
    for i in range(sample_size):
        data_i = tag_parse_data_with_subtree.iloc[i, :]
        logging.debug('NE=%s, subtree=%s'%(data_i.loc['NE'], data_i.loc['subtree']))
#     ## filter for allowed dependencies
#     ## highest head in subtree should match dependency (ex. prep phrase)
#     descriptor_dep_types = set(['acl', 'prep', 'appos', 'nmod'])
#     tag_parse_data_with_subtree_dep = tag_parse_data_with_subtree[tag_parse_data_with_subtree.loc[:, 'subtree'].apply(lambda x: len(set([y[3] for y in x if y[2]==min(x, key=lambda z: z[2])]) & descriptor_dep_types) > 0)]
#     logging.debug('%d/%d data with deps'%(tag_parse_data_with_subtree_dep.shape[0], tag_parse_data_with_subtree.shape[0]))
    
    ## process anchor data
    geonames_data = pickle.load(open(args['geonames_data'], 'rb'))
    # restrict to PR
    valid_countries = ['PR']
    geonames_data = {k : v[v.loc[:, 'country'].isin(valid_countries)] for k,v in geonames_data.items()}
    geonames_data = {k : v for k,v in geonames_data.items() if v.shape[0] > 0}
    geonames_max_pop = {k : v.loc[:, 'population'].max() for k,v in geonames_data.items()}
    geonames_max_alt_names = {k : v.loc[:, 'alternate_name_count'].max() for k,v in geonames_data.items()}
    tag_parse_data = tag_parse_data.assign(**{'NE_fixed' : tag_parse_data.loc[:, 'NE'].apply(lambda x: unidecode(x.lower()))})
    tag_parse_data = tag_parse_data.assign(**{
        'max_population' : tag_parse_data.loc[:, 'NE_fixed'].apply(lambda x: geonames_max_pop[x] if x in geonames_max_pop else 0.),
        'max_alternate_names' : tag_parse_data.loc[:, 'NE_fixed'].apply(lambda x: geonames_max_alt_names[x] if x in geonames_max_alt_names else 0.),
    })
    
    ## extract anchors
    anchor_var = 'max_population'
    id_var = 'status_author_id'
    # add valid var as column
    valid_var = 'valid_loc'
    tag_parse_data = tag_parse_data.assign(**{valid_var : tag_parse_data.loc[:, ['LOC_GN', 'LOC_OSM']].max(axis=1)})
    # rename text var
    txt_var = 'status_message'
    data_var = 'data_name'
    tag_parse_data = tag_parse_data.assign(**{data_var : 'maria_fb'})
    tag_parse_data = extract_anchors(tag_parse_data, anchor_var=anchor_var, id_var=id_var, txt_var=txt_var, data_var=data_var)
    
    ## write to file
    out_file = parse_data_file.replace('.txt', '_anchor.tsv')
    if(not os.path.exists(out_file)):
        tag_parse_data.to_csv(out_file, sep='\t', index=False)
    
if __name__ == '__main__':
    main()