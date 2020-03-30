"""
Query GeoNames for GeoCorpora strings,
because their under-the-hood matching
is probably better than our current candidate
generation strategy.
"""
from data_helpers import query_norm
from geonames_python import geonames
from argparse import ArgumentParser
import pandas as pd
import os
from time import sleep
import codecs

# sleep between queries to avoid rate-limiting
SLEEP_TIME=1.0

def mine_queries(all_queries, username, output_file, relevant_fields):
    queries_mined = set()
    for i, q in enumerate(all_queries):
        q_normed = query_norm(q)
        if(q_normed not in queries_mined):
            print('querying %s'%(q_normed))
            success = False
            while(not success):
                try:
                    q_results = geonames.search(username=username, q=q_normed)
                    if(len(q_results) > 0):
                        q_results = map(lambda x: [x[k] for k in relevant_fields], q_results)
                        # add query as column
                        if(type(q) is unicode):
                            q = q.encode('utf-8')
                        q_results = map(lambda x: [q] + x, q_results)
                    else:
                        print('0 results for query "%s"'%(q))
                        q_results = [[q_normed, 0]]
                    # write
                    for q_row in q_results:
                        output_file.write('%s\n'%('\t'.join(map(str, q_row))))
                    queries_mined.add(q)
                    success = True
                except Exception, e:
                    print('error reason %s'%(e.message))
                    print('skipping query "%s" because of error %s'%(q, e))
                    success = True
            sleep(SLEEP_TIME)
        if(i % 100 == 0):
            print('%d/%d queries mined'%(i, len(all_queries)))

def main():
    parser = ArgumentParser()
    parser.add_argument('--username', default='istewart')
    parser.add_argument('--query_file', default='../../data/mined_tweets/GeoCorpora/geocorpora.tsv')
    parser.add_argument('--out_dir', default='../../data/mined_tweets/GeoCorpora/')
    args = parser.parse_args()
    username = args.username
    query_file = args.query_file
    out_dir = args.out_dir
    
    ## load data
    query_col = 'text'
    query_data = pd.read_csv(query_file, sep='\t', index_col=False, encoding='utf-8')
    all_queries = query_data.loc[:, query_col].unique()
    
    ## make queries
    # write to file iteratively
    base_name = os.path.basename(query_file).replace('.tsv', '')
    out_file = os.path.join(out_dir, '%s_geonames_query_results.tsv'%(base_name))
    # only need geo ID for later use
    relevant_fields = ['geonameId']
    output_cols = ['query'] + relevant_fields
    if(os.path.exists(out_file)):
        query_results = pd.read_csv(out_file, sep='\t', index_col=False, encoding='utf-8')
        valid_queries = query_results[query_results.loc[:, 'geonameId'] != 0]
        all_queries = list(set(all_queries) - set(valid_queries.loc[:, 'query'].unique()))
        new_file = os.path.join(out_dir, '%s_geonames_query_results.tsv'%(base_name))
        with codecs.open(new_file, 'w') as output_file:
            # rewrite valid queries
            output_file.write('%s\n'%('\t'.join(output_cols)))
            for q_idx, q_row in valid_queries.iterrows():
                output_file.write('%s\n'%('\t'.join(q_row.values.astype(str))))
            mine_queries(all_queries, username, output_file, relevant_fields)
    else:
        with codecs.open(out_file, 'w') as output_file:
            output_file.write('%s\n'%('\t'.join(output_cols)))
            mine_queries(all_queries, username, output_file, relevant_fields)
    
if __name__ == '__main__':
    main()