"""
Mine archive of Twitter data for specified
phrase or location.
"""
from datetime import datetime
import gzip
import json
from argparse import ArgumentParser
import os, re
from unidecode import unidecode
from functools import reduce

def contains_geo(geo, location_box):
    """
    Determine if location box contains geo point.Determine
    
    Parameters:
    -----------
    geo : [float, float]
    Latitude, longitude.
    location_box : [[float, float], [float, float]]
    Latitude 1 (W), latitude 2 (E), longitude 1 (S), longitude 2 (N).
    """
    lat, lon = geo
    lat1, lat2 = location_box[0]
    lon1, lon2 = location_box[1]
    contains = (lat >= lat1 and lat <= lat2 and lon >= lon1 and lon <= lon2)
    return contains

def mine_tweets(archive_file, output, phrases=None, location_box=None, match_hashtags=False, user_loc_phrase=None):
    """
    Extract tweets from archive according to matching
    phrases in the main text or location, and write to file.
    
    Parameters:
    -----------
    archive_file : str
    output : file 
    Writeable output file.
    phrases : [str]
    location_box : [[float]]
    Latitude and longitude values for bounding box.
    match_hashtags : bool
    Treat phrases as hashtags (different value in tweet JSON).
    user_loc_phrase : str
    User location matching phrase.
    """
    if(phrases is not None):
        # print('phrases %s'%(','.join(phrases)))
        if(match_hashtags):
            phrase_tags = set([unidecode(x.replace('#','').lower()) for x in phrases])
            print('phrase tags %s'%(phrase_tags))
        else:
#             phrase_matcher = re.compile('|'.join(phrases).lower())
            # remove spaces when needed
            phrases_fixed = phrases + [x.replace(' ', '') for x in phrases if ' ' in x]
            phrases_fixed_combined = '|'.join(phrases_fixed).lower()
            phrases_fixed_str = '(?<=[,\.! #])(%s)(?=[,\.! #])|(?<=[,\.! #])(%s)$|^(%s)(?=[,\.! #])'%(phrases_fixed_combined, phrases_fixed_combined, phrases_fixed_combined)
#             phrase_matcher = re.compile('|'.join(phrases_fixed).lower())
            phrase_matcher = re.compile(phrases_fixed_str)
            print('phrase match pattern %s'%(phrase_matcher.pattern))
    else:
        phrase_matcher = None
    if(user_loc_phrase is not None):
        user_loc_phrase_matcher = re.compile(user_loc_phrase)
    match_ctr = 0
    error_ctr = 0
    non_delete_ctr = 0
    with gzip.open(archive_file, 'r') as archive:
        for l in archive:
            try:
                if(type(l) is bytes):
                    l = l.decode('utf-8').strip()
                j = json.loads(l.strip())
                if('delete' not in j and 'status_withheld' not in j):
                    non_delete_ctr += 1
                    # print(sorted(j.keys()))
                    # phrase matching for hashtag/text
                    txt_match = True
                    phrase_match_j = []
                    if(phrases is not None):
                        if(match_hashtags):
                            if(j.get('entities') is not None and j['entities'].get('hashtags') is not None):
                                j_tags = set([unidecode(x['text'].lower()) for x in j['entities']['hashtags']])
                                phrase_match_j = j_tags
                                txt_match = (len(phrase_tags & j_tags) > 0)
                        else:
                            j_text = unidecode(j['text'].lower())
                            phrase_match_j = phrase_matcher.findall(j_text)
                            if(len(phrase_match_j) > 0):
                                phrase_match_j = list(reduce(lambda a,b: a+b, [[y for y in x if y!=''] for x in phrase_match_j]))
                            txt_match = (len(phrase_match_j) > 0)
                    j_geo = j.get('geo')
                    # location matching
                    loc_match = True
                    if(location_box is not None):
                        if(j_geo is not None and contains_geo(j_geo, location_box)):
                            loc_match = True
                        else:
                            loc_match = False
                    # user bio location matching
                    user_loc_match = True
                    if(user_loc_phrase is not None):
                        if(j['user'].get('location') is not None):
                            j_user_loc = j['user']['location']
                            user_loc_match = user_loc_phrase_matcher.search(j_user_loc) is not None
                        else:
                            user_loc_match = False

                    # if(phrase_matcher.search(j_text) is not None):
                    if(txt_match and loc_match and user_loc_match):
                        match_ctr += 1
#                         print(j_text)
                        # add info on matching phrase/s!
                        j['phrase_match'] = phrase_match_j
                        j_dump = json.dumps(j).replace('\n','')
                        try:
                            output.write('%s\n'%(j_dump))
                        except Exception as e:
                            print('write exception %s'%(e))
#                     tweets.append(j)
            except Exception as e:
                print(e)
                # handling broken tweets
                # print(e)
                # print(type(l))
                # print(l)
                error_ctr += 1
                pass
            # tmp debugging
#             if(non_delete_ctr > 100000):
#                 break
    print('non-delete %d tweets'%(non_delete_ctr))
    print('matched %d tweets'%(match_ctr))
    print('errored %d tweets'%(error_ctr))

def build_out_file(out_dir, phrases, phrase_file, location_box, user_loc_phrase, add_dates_from_files_to_out_file, archive_files):
    out_file_str = ''
    if(phrases is not None):
        if(phrase_file is not None):
            phrase_file_name_matcher = re.compile('(\w+)_location_phrases.txt')
            phrase_str = phrase_file_name_matcher.search(os.path.basename(phrase_file)).group(1)
            phrase_str = '_%s_location_phrases'%(phrase_str)
        else:
            phrase_str = '_%s'%(','.join(phrases))
        out_file_str += phrase_str
    if(location_box is not None):
        location_str = '_%s'%(','.join(map(lambda x: '%.3f'%(x), [location_box[0][0], location_box[1][0], location_box[0][1], location_box[1][1]])))
        # location_str = '_%s'%(','.join(map(lambda x: '%.3f'%(x), location_box)))
        out_file_str += location_str
    if(user_loc_phrase is not None):
        out_file_str += "_%s"%(user_loc_phrase)
        out_file_str += '_%s_%s'%(start_date, end_date)
    if(add_dates_from_files_to_out_file):
        archive_files = sorted(archive_files)
        date_matcher = re.compile('(?<=tweets-)\w{3}-\d{2}-\d{2}')
        date_str_1 = date_matcher.search(archive_files[0]).group(0)
        date_str_2 = date_matcher.search(archive_files[-1]).group(0)
        out_file_str += '_%s-%s'%(date_str_1, date_str_2)
    out_file = os.path.join(out_dir, 'archive%s.gz'%(out_file_str))
    return out_file
    
def main():
    parser = ArgumentParser()
    parser.add_argument('archive_files', nargs='+')
    parser.add_argument('--phrases', default="")
    # parser.add_argument('--phrases', default="#MariaPR,#PuertoRicoSeLevanta,#HuracanMariaPR,#HelpPuertoRico,#PuertoRicoDepie")
    parser.add_argument('--phrase_file', default='../../data/mined_tweets/maria_location_phrases.txt')
#     parser.add_argument('--match_hashtags', default=True)
    parser.add_argument('--match_hashtags', default=False)
    parser.add_argument('--location_box', nargs='+', default=None)
    # parser.add_argument('--location_box', nargs='+', default=[17.5, 18.5, -67, -65])
    parser.add_argument('--user_loc_phrase', default=None)
    # parser.add_argument('--user_loc_phrase', default="")
    parser.add_argument('--out_dir', default='../../data/mined_tweets/')
    parser.add_argument('--add_dates_from_files_to_out_file', type=bool, default=True)
    args = parser.parse_args()
    archive_files = args.archive_files
    phrases = args.phrases
    phrase_file = args.phrase_file
    match_hashtags = args.match_hashtags
    location_box = args.location_box
    user_loc_phrase = args.user_loc_phrase
    out_dir = args.out_dir
    add_dates_from_files_to_out_file = args.add_dates_from_files_to_out_file

    # extract phrases
    phrases = phrases.split(',')
    if(len(phrases) == 0):
        phrases = None
    if(os.path.exists(phrase_file)):
        phrases = sorted(set([l.strip().lower() for l in open(phrase_file, 'r')]) - set(['']))
    print('got phrases %s'%(','.join(phrases)))
    # load locations if they exist
    if(location_box is not None and len(location_box) > 0 and location_box != ''):
        location_box = list(map(float, location_box))
        location_box = [location_box[:2], location_box[2:]]
    else:
        location_box = None
#     print('user loc phrase "%s"'%(user_loc_phrase))
    if(user_loc_phrase == ''):
        user_loc_phrase = None

    # sort archive files by date (month-day-year)
    date_fmt = '%b-%d-%y'
    matcher = re.compile('[A-Z][a-z]+-[0-3][0-9]-1[0-9]')
    archive_file_dates = map(lambda x: datetime.strptime(matcher.findall(os.path.basename(x))[0], date_fmt), archive_files)
    archive_files, archive_file_dates = zip(*sorted(zip(archive_files, archive_file_dates), key=lambda x: x[1]))
    start_date = datetime.strftime(archive_file_dates[0], date_fmt)
    end_date = datetime.strftime(archive_file_dates[-1], date_fmt)
    # out file
    out_file = build_out_file(out_dir, phrases, phrase_file, location_box, user_loc_phrase, add_dates_from_files_to_out_file, archive_files)
    print('got out file %s'%(out_file))
    
    # find tweets with matching phrase and/or location
    # and write to file
    with gzip.open(out_file, 'wt') as output:
        for archive_file in archive_files:
            print('mining archive file %s'%(archive_file))
            mine_tweets(archive_file, output, phrases, location_box, match_hashtags=match_hashtags, user_loc_phrase=user_loc_phrase)

if __name__ == '__main__':
    main()