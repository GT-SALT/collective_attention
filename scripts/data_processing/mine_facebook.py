"""
Mine public Facebook page using API credentials.
"""
import sys
if('facebook-page-post-scraper' not in sys.path):
  sys.path.append('facebook-page-post-scraper')
from get_fb_posts_fb_group import scrapeFacebookPageFeedStatus
from argparse import ArgumentParser
from dateparser import parse as dparse
from datetime import datetime
from data_helpers import load_facebook_auth
import os
from fb_scrape_public import scrape_fb

# new code with fb_scrape_public
def main():
  parser = ArgumentParser()
  parser.add_argument('--group_id', default='127217134598253')
  parser.add_argument('--auth_file', default='../../data/facebook_auth.csv')
  parser.add_argument('--out_dir', default='../../data/facebook-maria-tmp')
  parser.add_argument('--start_date', default='2017-09-20')
  parser.add_argument('--end_date', default='2017-10-20')
  args = parser.parse_args()
  group_id = args.group_id
  auth_file_name = args.auth_file
  out_dir = args.out_dir
  # substitute // for space in case of more complicated date strings
  # start_date = args.start_date.replace('//', ' ')
  end_date = args.end_date.replace('//', ' ')
  # load token
  access_token, _ = [l.strip() for l in open(auth_file_name)][0].split(',')
  out_file = os.path.join(out_dir, '%s_posts_%s'%(group_id, end_date))
  
  ## collect posts!
  group_posts = scrape_fb(token=access_token, ids=group_id, outfile=out_file, version="3.0")


## old code with facebook-page-post-scraper
# def main():
#   parser = ArgumentParser()
#   parser.add_argument('--group_id', default="127217134598253")
#   parser.add_argument('--auth_file', default='../../data/facebook_auth.csv')
#   parser.add_argument('--out_dir', default='../../data/facebook-maria')
#   parser.add_argument('--start_date', default='2017-09-20')
#   parser.add_argument('--end_date', default='2017-10-20')
#   args = parser.parse_args()
#   group_id = args.group_id
#   auth_file_name = args.auth_file
#   out_dir = args.out_dir
#   # substitute // for space in case of more complicated date strings
#   start_date = args.start_date.replace('//', ' ')
#   end_date = args.end_date.replace('//', ' ')
  
#   ## load auth data
#   app_id, app_secret, _ = load_facebook_auth()
#   access_token = '%s|%s'%(app_id, app_secret)
  
#   ## convert dates to seconds
#   START = datetime(1970,1,1)
#   start_seconds = '%d'%((dparse(start_date) - START).total_seconds())
#   end_seconds = '%d'%((dparse(end_date) - START).total_seconds())
  
#   ## mine
#   out_file_name = os.path.join(out_dir, '%s_%s_%s_facebook_posts.tsv'%(group_id, start_date, end_date))
#   # start/end date
# #   scrapeFacebookPageFeedStatus(group_id, access_token, start_date, end_date, out_file_name=out_file_name)
#   # start/end seconds
#   scrapeFacebookPageFeedStatus(group_id, access_token, start_seconds, end_seconds, out_file_name=out_file_name)
#   print('finished scraping, writing statuses to %s'%(out_file_name))
  
if __name__ == '__main__':
  main()
