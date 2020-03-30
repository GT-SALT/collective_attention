"""
For all GeoNames entities with a Wikipedia page p,
count the number of other Wiki pages that reference 
page p ("backlinks").
"""
from multiprocessing import Pool
import requests
import pandas as pd
from argparse import ArgumentParser
import re, os
from time import time, sleep
from itertools import izip, repeat

BACKLINK_URL='https://dispenser.info.tm/~dispenser/cgi-bin/backlinkscount.py?title=%s'
SLEEP_TIME=10
RATE_LIMIT_CODE=1226
OK_CODE=200
class DummyCounter:
    def __init__(self):
        self.ctr = 0

def get_backlink_count(title, counter):
    """
    Count number of backlinks to article using
    this hack: https://dispenser.info.tm/~dispenser/cgi-bin/backlinkscount.py?title=Title
    """
    title_url = BACKLINK_URL%(title)
    success = False
    while(not success):
        title_request = requests.get(title_url)
        if(title_request.status_code != OK_CODE):
            sleep(SLEEP_TIME)
        else:
            try:
                title_count = int(title_request.text.strip())
            except Exception, e:
                print('error at title %s due to text %s'%(title, title_request.text))
                title_count = 0
            success = True
    counter.ctr += 1
    if(counter.ctr % 1000 == 0):
        print('processed %d titles'%(counter.ctr))
    return (title, title_count)

def get_backlink_count_star(args):
    return get_backlink_count(*args)

MAX_PROCESSES=4
def main():
    parser = ArgumentParser()
    parser.add_argument('--alternate_names_file', default='/hg190/corpora/GeoNames/alternateNames.txt')
    parser.add_argument('--out_dir', default='/hg190/corpora/GeoNames/')
    args = parser.parse_args()
    alternate_names_file = args.alternate_names_file
    out_dir = args.out_dir
    
    ## load data
    alternate_names = pd.read_csv(alternate_names_file, sep='\t', index_col=False, encoding='utf-8')
    alternate_names.fillna('', inplace=True)
    alternate_names.columns = ['alternateNameId', 'geonameid', 'isolanguage', 
                               'alternate name', 'isPreferredName', 'isShortName', 
                               'isColloquial', 'isHistoric']
    
    ## extract wiki titles
    alternate_names_with_isolanguage = alternate_names[alternate_names.loc[:, 'isolanguage'] != '']
    wiki_matcher = re.compile('http://.*wikipedia.org.*/wiki/.*')
    alternate_names_with_wiki = alternate_names_with_isolanguage.loc[:, 'alternate name'].apply(lambda x: wiki_matcher.match(x) is not None)
    alternate_names_with_wiki = alternate_names_with_isolanguage[alternate_names_with_wiki]
    print('%d/%d names with wiki link'%(alternate_names_with_wiki.shape[0],
                                        alternate_names.shape[0]))
    wiki_title_matcher = re.compile('.*(?<=wiki/)(.*)')
    wiki_titles = alternate_names_with_wiki.loc[:, 'alternate name'].apply(lambda x: wiki_title_matcher.match(x).group(1))
    # need wiki title => ID lookup for later
    wiki_IDs = pd.DataFrame([alternate_names_with_wiki.loc[:, 'geonameid'].values.tolist(), wiki_titles.tolist()]).transpose()
    wiki_IDs.columns = ['geoname_ID', 'wiki_title']
    wiki_titles = wiki_titles.unique().tolist()
    
    print('%d unique wiki titles'%(len(wiki_titles)))
#     cutoff = 100
#     wiki_titles = wiki_titles[:cutoff]
    
    ## send backlink request for each title
    start_time = time()
    
    # serial requests
#     backlink_counts = []
#     for t in wiki_titles:
#         backlink_count = get_backlink_count(t)
#         backlink_counts.append([t, backlink_count])
#     backlink_counts = pd.DataFrame(backlink_counts)
#     backlink_counts.columns = ['name', 'backlinks']    
        
    # parallel requests
    pool = Pool(processes=MAX_PROCESSES)
    counter = DummyCounter()
    results = pool.map_async(get_backlink_count_star, izip(wiki_titles, repeat(counter)))
    backlink_counts = results.get()
        
    end_time = time()
    time_elapsed = end_time - start_time
    print('mining %d titles took %d seconds'%(len(backlink_counts), time_elapsed))
    backlink_count_df = pd.DataFrame(backlink_counts)
    backlink_count_df.columns = ['wiki_title', 'backlink_count']
    # also add Geonames IDs
    backlink_count_df = pd.merge(backlink_count_df, wiki_IDs, on='wiki_title', how='inner')
    backlink_count_df.drop_duplicates('wiki_title', inplace=True)    
    print(backlink_count_df.shape)
    print(backlink_count_df.head())

    ## write to file!!
    out_file = os.path.join(out_dir, 'wiki_backlink_counts.tsv')
    backlink_count_df.to_csv(out_file, sep='\t', index=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()