"""
Extract all location URLs from DBPedia to 
help whittle down our string mention lexicon.
"""
from SPARQLWrapper import SPARQLWrapper, JSON
from argparse import ArgumentParser
from time import sleep
import os
import gzip

SLEEP_TIME=5
def attempt_query(sparql, query):
    sparql.setQuery(query)
    success = False
    while(not success):
        try:
            query_result = sparql.query().convert()
            success = True
        except Exception, e:
            print('stalling for exception %s'%(e))
            sleep(SLEEP_TIME)
    return query_result

def main():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='/hg190/corpora/crosswikis-data.tar.bz2')
    args = parser.parse_args()
    out_dir = args.out_dir
    
    ## build query
    LIMIT = 1000
    # "Place" (which is the same as "Location")
    query = """ 
SELECT DISTINCT ?loc
WHERE {
?loc rdf:type <http://dbpedia.org/ontology/Place>}
LIMIT %d OFFSET %d
"""

    ## query DB and write to file!!
    dbpedia_url = 'http://dbpedia.org/sparql'
    dbpedia_resource_base_url = 'http://dbpedia.org/resource/'
    sparql = SPARQLWrapper(dbpedia_url)
    sparql.setReturnFormat(JSON)
    done_query = False
    offset = 0
    out_file_name = os.path.join(out_dir, 'location_urls.gz')    
    with gzip.open(out_file_name, 'w') as out_file:
        while(not done_query):
            query_updated = query%(LIMIT, offset)
            results = attempt_query(sparql, query_updated)
            results_list = results['results']['bindings']
            results_names = [r['loc']['value'].replace(dbpedia_resource_base_url, '') for r in results_list]
            result_str = ('\n'.join(results_names)+'\n').encode('utf-8')
            out_file.write(result_str)
            if(len(results_list) < LIMIT):
                done_query = True
            else:
                offset += LIMIT
                if(offset % 1000000 == 0):
                    print('processed %d results'%(offset))

if __name__ == '__main__':
    main()
