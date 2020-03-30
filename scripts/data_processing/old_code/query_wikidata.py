"""
Query Wikidata for population and language statistics.
"""
from argparse import ArgumentParser
import requests
import json
import pandas as pd
from data_helpers import decode_dict
import os

WIKI_URL = "https://query.wikidata.org/sparql?query=%s&format=JSON"
HEADERS = {"Accept" : "application/json"}
def query_wikidata_pop(wikidata_ID):
    """
    Query Wikidata for population.
    
    Parameters:
    -----------
    wikidata_ID : str
    
    Returns:
    --------
    pop : int
    """
    wiki_query_pop = """
    SELECT ?pop
        WHERE {
            wd:%s wdt:P1082 ?pop
        }
        """%(wikidata_ID)
    wiki_query_pop = WIKI_URL%(wiki_query_pop)
    req = requests.get(wiki_query_pop, headers=HEADERS)
    result = req.json()['results']['bindings']
    if(len(result) > 0):
        pop = int(result[0]['pop']['value'])
    else:
        pop = -1
    return pop

def query_wikidata_lang(wikidata_ID):
    """
    Query Wikidata for languages.
    
    Parameters:
    -----------
    wikidata_ID : str
    
    Returns:
    --------
    lang_count : int
    """
    wiki_query_lang = """
    SELECT (count(?lang) as ?numLang) 
        WHERE { 
            wd:%s rdfs:label ?label . 
            filter(!langmatches(lang(?label), 'en')) bind(lang(?label) as ?lang) 
        }
    """%(wikidata_ID)
    wiki_query_lang = WIKI_URL%(wiki_query_lang)
    req = requests.get(wiki_query_lang, headers=HEADERS)
    result = req.json()['results']['bindings']
    if(len(result) > 0):
        lang_count = int(result[0]['numLang']['value'])
    return lang_count
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--query_data', default='../../data/mined_tweets/GeoCorpora/geocorpora_opencage_query_results_temp.tsv')
    parser.add_argument('--out_dir', default='../../data/mined_tweets/GeoCorpora/')
    args = parser.parse_args()
    query_data_file = args.query_data
    out_dir = args.out_dir
    
    ## get all wikidata IDs
    query_data = pd.read_csv(query_data_file, sep='\t', index_col=False)
    wikidata_IDs = set()
    for i, q_row in query_data.iterrows():
        annotations = decode_dict(q_row.loc['annotations'])
        if(len(annotations) > 0 and annotations.get('wikidata') is not None):
            wikidata_ID = annotations['wikidata']
            if(wikidata_ID not in wikidata_IDs):
                wikidata_IDs.add(wikidata_ID)
        if(i % 1000 == 0):
            print('processed %d query results'%(i))
    wikidata_IDs = list(wikidata_IDs)
    
    ## mine all wikidata IDs
    wikidata_df = pd.DataFrame()
    wiki_cols = ['wikidata_ID', 'pop', 'lang']
    for wikidata_ID in wikidata_IDs:
        pop = query_wikidata_pop(wikidata_ID)
        lang = query_wikidata_lang(wikidata_ID)
        wikidata_df = wikidata_df.append(pd.DataFrame([wikidata_ID, pop, lang]).transpose())
    wikidata_df.columns = wiki_cols
    
    ## write!!
    out_file = os.path.join(out_dir, 'wikidata_mined_data.tsv')
    wikidata_df.to_csv(out_file, sep='\t', index=False)

if __name__ == '__main__':
    main()