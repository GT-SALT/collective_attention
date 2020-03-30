"""
Convert GeoNames data into tree of enclosing locations,
such that the parent of each location is a bounding region
such as a country, state, etc.
               LOC1
              /    \
           LOC2    LOC3
          /
        LOC4
                etc.
"""
import pandas as pd
from data_helpers import load_full_geonames_data, load_country_info, load_admin1_data, load_admin2_data, load_geonames_hierarchy, get_logger
from argparse import ArgumentParser
import networkx as nx

def main():
    parser = ArgumentParser()
#     parser.add_argument('--hierarchy_out_file', default='/hg190/corpora/GeoNames/combined_hierarchy.tsv')
    parser.add_argument('--hierarchy_out_file', default='/hg190/corpora/GeoNames/combined_hierarchy.gz')
    args = parser.parse_args()
    hierarchy_out_file = args.hierarchy_out_file
    logger = get_logger('../../output/collect_geonames_hierarchy.txt')
    
    ## load data
    geonames_full = load_full_geonames_data()
    hierarchy = load_geonames_hierarchy()
    country_data = load_country_info()
    admin1_data = load_admin1_data()
    admin2_data = load_admin2_data()
    admin1_lookup = dict(zip(admin1_data.loc[:, 'admin1_code'].values.tolist(), 
                             admin1_data.loc[:, 'geonames_ID'].values.tolist()))
    
    country_lookup = dict(zip(country_data.loc[:, 'ISO'].values, country_data.loc[:, 'geonameid'].astype(int)))
    geonames_full.loc[:, 'admin1_code_combined'] = geonames_full.loc[:, 'country_code'] + '.' + geonames_full.loc[:, 'admin1_code'].astype(str)
    # need feature codes to eliminate duplicate parents
    feature_class_ordered = ['CONT', 'RGN', 'ZN', 'TERR', 'AREA', 'SEA', 'PCL', 'PCLI', 'PCLS', 'PCLF', 'PCLD', 'PCLIX', 'ADMD', 'ADM1', 'ADM2', 'ADM3', 'ADM4', 'ADM5', 'PEN', 'MTS', 'VAL', 'ISLS', 'ISL', 'CST', 'PPLC', 'PPLA', 'PPLA2', 'PPLA4', 'PPL', 'PPLL', 'PPLX', 'PRK', 'LK', 'STM', 'RDJCT', 'REST', 'HTL', 'AIRP']
    feature_class_lookup = dict(zip(feature_class_ordered, range(1, len(feature_class_ordered)+1)))
    
    ## connect to admin info
    admin1_hierarchy_data = []
    for a1_code, a1_group in geonames_full.groupby('admin1_code_combined'):
        if(a1_code in admin1_lookup):
            a1_id = admin1_lookup[a1_code]
            for c_id in a1_group.loc[:, 'geonames_ID'].values:
                if(a1_id != c_id):
                    admin1_hierarchy_data.append([a1_id, c_id, ''])
    logger.debug('%d admin1 hierarchy relationships'%(len(admin1_hierarchy_data)))
    admin2_hierarchy_data = []
    for a2_code, a2_id in admin2_data.loc[:, ['admin2_code', 'geonames_ID']].values.tolist():
        a1_code = '.'.join(a2_code.split('.')[:2])
        if(a1_code in admin1_lookup):
            a1_id = admin1_lookup[a1_code]
            admin2_hierarchy_data.append([a1_id, a2_id, ''])
    full_hierarchy = hierarchy.copy()
    admin1_hierarchy_df = pd.DataFrame(admin1_hierarchy_data, columns=['parent', 'child', 'admin_code'])
    admin2_hierarchy_df = pd.DataFrame(admin2_hierarchy_data, columns=['parent', 'child', 'admin_code'])
    # eliminate duplicate parents
    full_hierarchy = full_hierarchy[~full_hierarchy.loc[:, 'child'].isin(set(admin1_hierarchy_df.loc[:, 'child'].values))]
    full_hierarchy = full_hierarchy[~full_hierarchy.loc[:, 'child'].isin(set(admin2_hierarchy_df.loc[:, 'child'].values))]
    admin1_hierarchy_df = admin1_hierarchy_df[~admin1_hierarchy_df.loc[:, 'child'].isin(set(admin2_hierarchy_df.loc[:, 'child'].values))]
    # for remaining full hierarchy duplicates: get rid of higher admin codes
    full_hierarchy_dedup = pd.DataFrame()
    full_hierarchy_children_counts = full_hierarchy.loc[:, 'child'].value_counts()
    full_hierarchy_children_dups = full_hierarchy_children_counts[full_hierarchy_children_counts > 1].index.tolist()
    full_hierarchy_dups = full_hierarchy[full_hierarchy.loc[:, 'child'].isin(full_hierarchy_children_dups)]
    for c, c_group in full_hierarchy_dups.groupby('child'):
        c_group.loc[:, 'feature_code_val'] = c_group.loc[:, 'admin_code'].apply(feature_class_lookup.get)
        c_group.sort_values('feature_code_val', inplace=True, ascending=True)
        c_group = c_group.iloc[[0], :].drop('feature_code_val', axis=1, inplace=False)
        full_hierarchy_dedup = full_hierarchy_dedup.append(c_group)
    full_hierarchy_dedup = full_hierarchy_dedup.append(full_hierarchy[~full_hierarchy.loc[:, 'child'].isin(full_hierarchy_children_dups)])
    full_hierarchy = full_hierarchy_dedup
    logger.debug('%d entries in hierarchy'%(full_hierarchy.shape[0]))
    
    ## cover the remaining data by
    ## assigning each one to its parent country
    geonames_non_covered = geonames_full[~geonames_full.loc[:, 'geonames_ID'].isin(full_hierarchy.loc[:, 'child'].values)]
    geonames_non_covered.loc[:, 'country_ID'] = geonames_non_covered.loc[:, 'country_code'].apply(country_lookup.get)
    # eliminate nulls
    geonames_non_covered = geonames_non_covered[~geonames_non_covered.loc[:, 'country_ID'].isna()]
    geonames_non_covered.loc[:, 'country_ID'] = geonames_non_covered.loc[:, 'country_ID'].astype(int)
    logger.debug('%d non-covered samples'%(geonames_non_covered.shape[0]))
    # combine
    non_covered_hierarchy = geonames_non_covered.loc[:, ['country_ID', 'geonames_ID']]
    non_covered_hierarchy.loc[:, 'admin_code'] = ''
    non_covered_hierarchy.rename(columns={'country_ID' : 'parent', 'geonames_ID' : 'child'}, inplace=True)
    combined_hierarchy = full_hierarchy.append(non_covered_hierarchy)
    # eliminate bad nodes
    combined_hierarchy.index = pd.np.arange(0, combined_hierarchy.shape[0])
    combined_hierarchy = combined_hierarchy[combined_hierarchy.loc[:, 'child'].apply(lambda x: type(x) is int or x.isdigit())]
    combined_hierarchy.loc[:, 'child'] = combined_hierarchy.loc[:, 'child'].astype(int)
    logger.debug('%d combined samples'%(combined_hierarchy.shape[0]))
    
    ## convert to network for better representation
    hierarchy_graph = nx.DiGraph()
    for n1, n2 in combined_hierarchy.loc[:, ['parent', 'child']].values.tolist():
        hierarchy_graph.add_edge(n1, n2)
    logger.debug('%d nodes and %d edges'%(len(hierarchy_graph.nodes()), len(hierarchy_graph.edges())))
    
    ## remove cycles!
    hierarchy_graph_acyclic = nx.minimum_spanning_tree(hierarchy_graph.to_undirected())
    hierarchy_edges = hierarchy_graph_acyclic.edges()
    valid_edges = filter(lambda x: x in hierarchy_edges or reverse(x) in hierarchy_edges, hierarchy_graph.edges())
    hierarchy_graph = nx.DiGraph()
    hierarchy_graph.add_edges_from(valid_edges)
    
    ## write!!
    nx.write_gpickle(hierarchy_graph, path=hierarchy_out_file)
    
if __name__ == '__main__':
    main()