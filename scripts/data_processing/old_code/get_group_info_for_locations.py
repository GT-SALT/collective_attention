"""
Query Facebook for group information on
a given location. 
e.g., query "Huracan Maria Aguadillas" and 
look for all valid groups returned.
"""
from argparse import ArgumentParser
from data_helpers import load_facebook_auth
import time
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request
from facebook import GraphAPI
import pandas as pd
import os
    
def request_until_succeed(url):
    req = Request(url)
    success = False
    while success is False:
        try:
            response = urlopen(req)
            if response.getcode() == 200:
                success = True
        except Exception as e:
            print(e)
            time.sleep(5)

            print("Error for URL {}: {}".format(url, datetime.datetime.now()))
            print("Retrying.")

    return response.read()

#####
# HOW TO GET A LONG-TERM ACCESS TOKEN
# (-1) make FB App
# (0) login
# (1) go to https://developers.facebook.com/tools/explorer/
# (2) enter following query
# GET /oauth/access_token?  
#     grant_type=fb_exchange_token&           
#     client_id={app-id}&
#     client_secret={app-secret}&
#     fb_exchange_token={short-lived-token} 
#####

def main():
  parser = ArgumentParser()
  parser.add_argument('--location_file', default='../../data/facebook-maria/PR_municipalities.tsv')
  parser.add_argument('--search_query', default='Huracan Maria')
  parser.add_argument('--out_dir', default='../../data/facebook-maria/')
  args = parser.parse_args()
  location_file = args.location_file
  search_query = args.search_query
  out_dir = args.out_dir 
  
  ## load location names
  location_data = pd.read_csv(location_file, sep='\t', index_col=None)
  location_names = location_data.loc[:, 'Municipality'].values.tolist()
  
  ## connect to API
  app_id, app_secret, access_token = load_facebook_auth()
  graph = GraphAPI(access_token=access_token, version='2.10')
  
  # search for groups by location name!!
  location_data = []
  loc_index = ['location_name', 'group_id', 'group_name']
  for location_name in location_names:
    print('processing location %s'%(location_name))
    location_search_name = '%s %s'%(search_query, location_name)
    groups = graph.search(type='group', q=location_search_name)['data']
    open_groups = filter(lambda x: x['privacy'] == 'OPEN', groups)
    if(len(open_groups) > 0):
      group_data = [[location_name, g['id'], g['name']] for g in open_groups]
      location_data += group_data
  location_data = pd.DataFrame(location_data, columns=loc_index)
  
  ## write to file
  out_file = os.path.join(out_dir, 'location_group_data.tsv')
  location_data.to_csv(out_file, sep='\t', index=False, encoding='utf-8')
  
  ## get group objects => get group popularity
#   group_objects = graph.get_objects(ids=group_ids, type='group')
#   print(group_objects)
  
  # old query code that DOESN'T WORK 
  # I think I have the wrong query structure but don't know how to fix
#   ## build query
#   access_token = '%s|%s'%(app_id, app_secret)
#   location_name = 'Coamo Huracan Maria'
#   base = "https://graph.facebook.com/v2.9"
#   node = '/search/top/?q=%s&type=group'%(location_name)
#   parameters = "/?limit={}&access_token={}".format(100, access_token)
  
#   ## run query
#   base_url = base + node + parameters
#   print(base_url)
#   response = request_until_succeed(base_url)
#   print(response)
  
if __name__ == '__main__':
  main()