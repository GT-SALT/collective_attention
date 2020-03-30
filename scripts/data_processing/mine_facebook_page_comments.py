"""
Mine comments from public Facebook page using API credentials.
Requires you to have already mined for statuses and stored
in file with format $OUT_DIR/$GROUPID_facebook_statuses.csv
"""
import sys
if('facebook-page-post-scraper' not in sys.path):
  sys.path.append('facebook-page-post-scraper')
from get_fb_comments_from_fb import scrapeFacebookPageFeedComments
from argparse import ArgumentParser
import os

# candidate groups
# 127217134598253 (Barranquitas)
# 866250103534243 (Huracan Maria 2017 UPDATES)
# 171400810082867 (Huracan Maria OESTE)
# 324112331384312 (Huracan maria mayaguez puerto rico)
# huracanmariayabucoa (Huracan Maria Yabucoa)
def main():
  parser = ArgumentParser()
  parser.add_argument('--group_id', default="866250103534243")
  parser.add_argument('--auth_file', default='../../data/facebook_auth.csv')
  parser.add_argument('--out_dir', default='../../data/facebook-maria')
  parser.add_argument('--start_date', default='2017-09-20')
  parser.add_argument('--end_date', default='2017-10-20')
  args = parser.parse_args()
  group_id = args.group_id
  auth_file_name = args.auth_file
  out_dir = args.out_dir
  start_date = args.start_date
  end_date = args.end_date
  
  # load auth data
  auth = dict([l.strip().split(',') for l in open(auth_file_name)])
  app_id = auth['app_id']
  app_secret = auth['app_secret']
  access_token = '%s|%s'%(app_id, app_secret)
  
  # mine
  out_file_name = os.path.join(out_dir, '%s_%s_%s_facebook_comments.tsv'%(group_id, start_date, end_date))
  status_file_name = os.path.join(out_dir, '%s_%s_%s_facebook_posts.tsv'%(group_id, start_date, end_date))
  scrapeFacebookPageFeedComments(group_id, access_token, status_file_name, out_file_name)
  print('finished scraping, writing comments to %s'%(out_file_name))
  
if __name__ == '__main__':
  main()