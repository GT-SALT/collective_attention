"""
Extract location entities and text data from 
NER data sample.
"""
import pandas as pd
from data_helpers import collect_entities_from_txt, load_simplified_geonames_data, process_query, get_logger, load_name_data, query_norm, load_flat_name_data
from argparse import ArgumentParser
from zipfile import ZipFile
import os, re
from multiprocessing import Pool
from itertools import izip, repeat

LOGGER=get_logger('../../output/extract_loc_text_data_from_ner_sample_parallel.txt')
def query_all(q, flat_name_data):
#     location_query_results = []
#     for i, q in enumerate(qs):
    LOGGER.debug('processing query %s'%(q))
    if(type(q) is str):
        q = unicode(q.decode('utf-8'))
#             q_exact, q_approx = process_query(q, geonames, name_ids, name_regexes)
    q_exact, q_approx = process_query(q, flat_name_data)
    q_combined = q_exact + q_approx
    location_query_results = q_combined
#         if(i % 100 == 0):
#             LOGGER.debug('processed %d queries'%(i))
    return location_query_results

def query_all_star(args):
    return query_all(*args)

def collect_locations(ner_files):
    locations_to_query = set()
    for f in ner_files:
        with ZipFile(f) as f_zip:
            f_txt = f_zip.filelist[0]
            LOGGER.debug('processing %s'%(f_txt.filename))
            for i, l in enumerate(f_zip.open(f_txt)):
                l_txt = l.strip()
                if(l_txt != ''):
                    l_ents = collect_entities_from_txt(l_txt, include_type=True, delim='/', use_bio=True, use_POS_tag=True)
                    l_ent_pairs = map(lambda x: x.split('/'), l_ents)
                    l_ent_loc_pairs = filter(lambda x: x[1] == 'geo-loc' or x[1] == 'facility', l_ent_pairs)
                    if(len(l_ent_loc_pairs) > 0):
                        l_ent_locs, _ = zip(*l_ent_loc_pairs)
                        locations_to_query.update(l_ent_locs)
                if(i % 10000 == 0):
                    LOGGER.debug('extracted %d unique locs from %d lines in %s'%(len(locations_to_query), i, f_txt.filename))
    return locations_to_query

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='../../data/mined_tweets/sample_ner_tweets')
    parser.add_argument('--ner_files', default=['tweets-Jan-01-16-03-57_ner.zip',
                                                'tweets-Feb-01-16-05-20_ner.zip',
                                                'tweets-Mar-01-16-04-10_ner.zip',
                                                'tweets-Apr-01-16-04-00_ner.zip',
                                                'tweets-May-01-16-03-41_ner.zip',
                                                'tweets-Jun-01-16-04-21_ner.zip',
                                                'tweets-Jul-01-16-03-43_ner.zip', 
                                                'tweets-Aug-01-16-04-13_ner.zip',
                                                'tweets-Sep-01-16-03-30_ner.zip',
                                                'tweets-Oct-01-16-03-50_ner.zip',
                                                'tweets-Nov-01-16-03-22_ner.zip',
                                                'tweets-Dec-01-16-04-01_ner.zip'])
#     parser.add_argument('--ner_files', default=['tweets-Jan-01-16-03-57_ner.zip']) # test run
    parser.add_argument('--valid_loc_file', default='ner_valid_loc_data.tsv')
    args = parser.parse_args()
    data_dir = args.data_dir
    ner_files = args.ner_files
    valid_loc_file = args.valid_loc_file
    ner_files = map(lambda f: os.path.join(data_dir, f), ner_files)
    valid_loc_file = os.path.join(data_dir, valid_loc_file)
    
    ## load data
    geonames = load_simplified_geonames_data()
    geonames.loc[:, 'name_norm'] = geonames.loc[:, 'name'].apply(lambda x: query_norm(x))
    
    ## process files!
    ## find location entities, attempt to disambiguate,
    ## THEN go back, identify lines with valid locations, write 
    ## each line with location, GeoNames ID, coords
    if(not os.path.exists(valid_loc_file)):
        ## collect locations
        locations_to_query = set()
        locations_to_query = collect_locations(ner_files)
        
        ## query locations
        location_query_results = []
        # load regex data for matching
#         name_data = load_name_data()
        flat_name_data = load_flat_name_data()
        # normalize names
        flat_name_data.loc[:, 'name'] = flat_name_data.loc[:, 'name'].apply(query_norm)
        # old code: regexes for serial code
#         name_ids = name_data.loc[:, 'geonames_ID'].values.tolist()
#         name_regexes = name_data.loc[:, 'name_regex']
#         name_regexes.fillna('', inplace=True)
#         # fix regexes with anchors
#         name_regexes = name_regexes.apply(lambda x: re.compile('^(%s)$'%(x)))

        # parallel query
        WORKERS=10
        pool = Pool(processes=WORKERS)
        locations_to_query = list(locations_to_query) # start small
        
        location_query_results = pool.map(query_all_star, izip(locations_to_query, repeat(flat_name_data, len(locations_to_query))))
        location_query_results = reduce(lambda x,y: x+y, location_query_results)
        # serial query
#         for i, q in enumerate(locations_to_query):
#             if(type(q) is str):
#                 q = unicode(q.decode('utf-8'))
#             LOGGER.debug('processing query %s'%(q))
# #             q_exact, q_approx = process_query(q, geonames, name_ids, name_regexes)
#             q_exact, q_approx = process_query(q, flat_name_data)
#             q_combined = q_exact + q_approx
#             location_query_results += q_combined
#             if(i % 1000 == 0):
#                 LOGGER.debug('processed %d/%d queries'%(i, len(locations_to_query)))

        # organize data
        valid_loc_data = pd.DataFrame(location_query_results)
        valid_loc_data.columns = ['query', 'geonames_ID']
        valid_loc_data.sort_values('query', inplace=True, ascending=True)
        valid_loc_data.to_csv(valid_loc_file, sep='\t', index=False, encoding='utf-8')
    else:
        valid_loc_data = pd.read_csv(valid_loc_file, sep='\t', index_col=False, encoding='utf-8')

    ## now! pair the locations with the text
    loc_text_data = pd.DataFrame()
    for f in ner_files:
        with ZipFile(f) as f_zip:
            f_txt = f_zip.filelist[0]
            LOGGER.debug('pairing locations/text in %s'%(f_txt))
            for i, l in enumerate(f_zip.open(f_txt)):
                l_txt = l.strip()
                if(l_txt != ''):
                    l_ents = collect_entities_from_txt(l_txt, include_type=True, delim='/', use_bio=True, use_POS_tag=True)
                    l_ent_pairs = map(lambda x: x.split('/'), l_ents)
                    l_ent_loc_pairs = filter(lambda x: x[1] == 'geo-loc' or x[1] == 'facility', l_ent_pairs)
                    if(len(l_ent_loc_pairs) > 0):
                        l_ent_locs, _ = zip(*l_ent_loc_pairs)
                        l_ent_locs = set(l_ent_locs)
                        # look up the locations!
                        valid_loc_data_l = valid_loc_data[valid_loc_data.loc[:, 'query'].isin(l_ent_locs)]
                        if(valid_loc_data_l.shape[0] > 0):
                            # add lat/lon
                            valid_loc_data_l = pd.merge(valid_loc_data_l, geonames.loc[:, ['geonames_ID', 'latitude', 'longitude']], how='inner', on='geonames_ID')
                            valid_loc_data_l.loc[:, 'text'] = l_txt
                            loc_text_data = loc_text_data.append(valid_loc_data_l)
                    if(i % 10000 == 0):
                        LOGGER.debug('extracted %d data samples from %d lines in %s'%(loc_text_data.shape[0], i, f_txt))
    
    ## write to file
    out_file = os.path.join(data_dir, 'ner_loc_text_data.tsv')
    loc_text_data.to_csv(out_file, sep='\t', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()