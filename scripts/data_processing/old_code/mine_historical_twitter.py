"""
Mine Twitter for hashtags of days past.
"""
import sys, getopt, datetime, codecs, gzip, os
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got
from argparse import ArgumentParser
    
def main():
  parser = ArgumentParser()
  parser.add_argument('--hashtags', nargs='+')
  parser.add_argument('--start_date')
  parser.add_argument('--end_date')
  parser.add_argument('--out_file')
  parser.add_argument('--out_dir')
  parser.add_argument('--location', nargs='+')
  parser.add_argument('--within')
  args = parser.parse_args()
  hashtags = args.hashtags
  start_date = args.start_date
  end_date = args.end_date
  out_dir = args.out_dir
  location = args.location
  within = args.within
  
  ## set up miner criteria
  criteria = got.manager.TweetCriteria()
  criteria.setSince(start_date)
  criteria.setUntil(end_date)
  hashtag_flag = hashtags is not None and len(hashtags) > 0 and hashtags != ''
  location_flag = location is not None and location != ''
  within_flag = within is not None
  if(hashtag_flag):
    hashtag_str = ' OR '.join(hashtags)
    criteria.setQuerySearch(hashtag_str)
  if(location_flag):
    location = ' '.join(location)
    criteria.setNear(location)
    if(within_flag):
      criteria.setWithin(within)
    
  ## write to file
  data_format_list = ['%s', '%s', '%d', '%d', '"%s"', '%s', '%s', '%s', '"%s"', '%s']
  write_str = '\t'.join(data_format_list)+'\n'
  data_name_list = ['username', 'date', 'retweets', 'favorites', 'text', 'geo', 'mentions', 'hashtags', 'id', 'permalink']
  out_file_str = ''
  if(hashtag_flag):
    out_file_str += ','.join(hashtags) + '_'
  if(location_flag):
    out_file_str += '%s_'%(location)
    if(within_flag):
      out_file_str += '%s_'%(within)
  # always end file name with start/end date
  out_file_str += '%s_%s.gz'%(start_date, end_date)
  out_file = os.path.join(out_dir, out_file_str)
  with gzip.open(out_file, 'w') as output:
    output.write('%s\n'%('\t'.join(data_name_list)))
    def receiveBuffer(tweets):
      for t in tweets:
        data = (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink)
        data_str = (write_str%(data)).encode('utf-8')
        output.write(data_str)
      output.flush()
      print('More %d saved on file...\n'%(len(tweets)))
    got.manager.TweetManager.getTweets(criteria, receiveBuffer)
  
if __name__ == '__main__':
  main()