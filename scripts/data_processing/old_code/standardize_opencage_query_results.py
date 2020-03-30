"""
Get standardized version of OpenCage query results.
"""
import pandas as pd
from argparse import ArgumentParser
from data_helpers import decode_dict

def main():
    parser = ArgumentParser()
    parser.add_argument('--query_result_file', default='../../data/mined_tweets/GeoCorpora/geocorpora_opencage_query_results_temp.tsv')
    args = parser.parse_args()
    query_result_file = args.query_result_file
    
    ## load data
    query_results = pd.read_csv(query_result_file, sep='\t', index_col=False)
    
    ## add relevance counter
    query_results_clean = pd.DataFrame()
    for i, (t, group) in enumerate(query_results.groupby('tweet_id_str')):
        relevance_ctr = pd.np.arange(group.shape[0])+1
        group.loc[:, 'relevance'] = relevance_ctr
        query_results_clean = query_results_clean.append(group)
        if(i % 100 == 0):
            print('processed %d unique tweets'%(i))
    
    ## normalize the dictionaries and extract relevant data
    query_result_norm = pd.DataFrame()
    keep_cols = ['confidence', 'relevance', 'gold_latitude', 'gold_longitude', 'tweet_id_str', 'query']
    dict_cols = ['annotations', 'components', 'geometry']
    dict_attrs = { 'annotations' : ['wikidata'], 
                   'components' : ['city', 'state', 'country', 'country_code'],
                   'geometry' : ['lat', 'lng']}
    for i, (r, q_row) in enumerate(query_results_clean.iterrows()):
        q_data = q_row.loc[keep_cols]
        # convert dicts
        for d_c in dict_cols:
            q_dict = decode_dict(q_row.loc[d_c])
            for c_a in dict_attrs[d_c]:
                q_data.loc[c_a] = q_dict.get(c_a)
        q_bounds = decode_dict(q_row.loc['bounds'])
        if(len(q_bounds) > 0 
           and 'northeast' in q_bounds
           and 'southwest' in q_bounds):
            q_data.loc['NE_lat'] = q_bounds['northeast']['lat']
            q_data.loc['NE_lng'] = q_bounds['northeast']['lng']
            q_data.loc['SW_lat'] = q_bounds['southwest']['lat']
            q_data.loc['SW_lng'] = q_bounds['southwest']['lng']
        query_result_norm = query_result_norm.append(pd.DataFrame(q_data).transpose())
        if(i % 1000 == 0):
            print('processed %d rows'%(i))
    query_result_norm.fillna('', inplace=True)
    print(query_result_norm.head())
    
    # write to file!!
    out_file = query_result_file.replace('.tsv', '_norm.tsv')
    print(out_file)
    query_result_norm.to_csv(out_file, sep='\t', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()