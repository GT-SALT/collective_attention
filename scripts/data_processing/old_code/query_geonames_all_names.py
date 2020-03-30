"""
Query the downloaded GeoNames database
for all queries in the test data
based on exact/approximate matching.
"""
import pandas as pd
from argparse import ArgumentParser
from data_helpers import load_simplified_geonames_data, load_name_data, get_logger, process_query, query_norm
import re, os
from itertools import izip

def main():
    parser = ArgumentParser()
    parser.add_argument('--query_file', default='../../data/mined_tweets/GeoCorpora/geocorpora.tsv')
    parser.add_argument('--query_col', default='text')
    parser.add_argument('--out_dir', default='../../data/mined_tweets/GeoCorpora/')
    args = parser.parse_args()
    query_file = args.query_file
    query_col = args.query_col
    out_dir = args.out_dir
    logger = get_logger('../../output/query_geonames_all_names.txt')
    
    ## load data
    query_data = pd.read_csv(query_file, sep='\t', index_col=False, encoding='utf-8')
    all_queries = query_data.loc[:, query_col].unique()#[:50] #testing
    name_data = load_name_data()
    name_ids = name_data.loc[:, 'geonames_ID'].values.tolist()
    name_regexes = name_data.loc[:, 'name_regex']
    name_regexes.fillna('', inplace=True)
    # fix regexes with anchors
    name_regexes = name_regexes.apply(lambda x: re.compile('^(%s)$'%(x)))
#     name_regexes = name_regexes.apply(lambda x: re.compile('|'.join(map(lambda y: '^%s$'%(y), x.split('|'))))).values.tolist()
    geonames = load_simplified_geonames_data()
    geonames.loc[:, 'name_norm'] = geonames.loc[:, 'name'].apply(query_norm)
    flat_name_data = load_flat_name_data()
    # normalize ALL the names
    flat_name_data.loc[:, 'name'] = flat_name_data.loc[:, 'name'].apply(query_norm)
    
    ## query 
    ## separate exact and approx results
    query_exact_results = []
    query_approx_results = []
    for i, q in enumerate(all_queries):
        logger.debug('processing candidate %s'%(q))
        q_exact_results, q_approx_results = process_query(q, flat_name_data)
#         q_exact_results, q_approx_results = process_query(q, geonames, name_ids, name_regexes)
        # TODO: containment match "Ohio State" => "Ohio State University"
        # TODO: acronym matching => "USA"
        if(len(q_exact_results) + len(q_approx_results) == 0):
            logger.debug('got 0 candidates for query %s'%(q))
        query_exact_results += q_exact_results
        query_approx_results += q_approx_results
        if(i % 100 == 0):
            logger.debug('processed %d candidates'%(i))
    query_exact_results = pd.DataFrame(query_exact_results)
    query_approx_results = pd.DataFrame(query_approx_results)
    query_exact_results.loc[:, 'match'] = 'exact'
    query_approx_results.loc[:, 'match'] = 'approx'
    # combine!!
    query_results = pd.concat([query_exact_results, query_approx_results], axis=0)
    query_results.columns = ['query', 'geonames_ID', 'match']
    query_results.sort_values('query', inplace=True, ascending=True)
    
    ## write to file
    out_file = os.path.join(out_dir, 'geocorpora_names_query_results_full.tsv')
    query_results.to_csv(out_file, sep='\t', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()