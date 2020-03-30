# -*- encoding: utf-8 -*-
"""
Helper methods for data processing.

TODO: divide these among multiple util files organized by function.
"""
from __future__ import division
import sys
## handling Python 2 versus 3 discrepancies
if(sys.version_info[0] == 3):    
    ## in Python 3 we need to add the library manually
    ## for the obscure packages like alphabet_detector
    if('../../lib/python3.4/site-packages/' not in sys.path):
        sys.path.append('../../lib/python3.4/site-packages/')
    izip = zip
    from functools import reduce
else:
    from itertools import izip
    
import re
import gzip
from bz2 import BZ2File
# try:
#     from bz2 import BZ2File
# except Exception, e:
#     print('import error %s'%(e))
from collections import defaultdict, Counter
import codecs
import pandas as pd
import numpy as np
import os
import geojson
from shapely.geometry import shape, Point, Polygon, MultiPolygon
import shapefile
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.nist import NISTTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.tokens import Doc
from sklearn.feature_extraction.text import CountVectorizer
from geopy.distance import great_circle
# import editdistance
from unidecode import unidecode
import json
from dateutil import parser as date_parser
from datetime import datetime, timedelta
from alphabet_detector.alphabet_detector import AlphabetDetector
import logging
import networkx as nx
import xlrd
#import torch
import dateutil
from math import floor

## entity annotation functions
def collect_entities_from_txt(txt, include_type=False, outside_tag='O', delim='/', use_bio=False, use_POS_tag=False):
    """
    Collect entity strings from tagged text. NOTE: we assume
    that all adjacent strings tagged as "ENTITY" form
    part of the same entity, which is obviously not true
    for some cases, e.g., "the hurricane hit Puerto/ENT Rico/ENT, Guadelupe/ENT..."
    
    Parameters
    ----------
    txt : str
    include_type : bool
    Include toponym type.
    outside_tag : str
    delim : str
    use_bio : bool
    Using B/I/O notation (default is I/O notation).
    use_POS_tag : bool
    Use POS tag after entity (token/BIO/POS_tag)
    
    Returns:
    --------
    entity_list : list
    """
    tokens = txt.strip().split(' ')
    # delim used to join multiple tokens in same entity
    multi_token_delim = ' '
    entity_list = []
    entity_str = []
    inside_tag = 'I-[a-zA-Z\-]+' # BI+type
    before_tag = 'B-[a-zA-Z\-]+'
    if(use_POS_tag):
        tag_position = -2
    else:
        tag_position = -1
    inside_tag_matcher = re.compile(inside_tag)
    before_tag_matcher = re.compile(before_tag)
    
    for t in tokens:
        # split into text and tag
        t_pieces = t.split(delim)
        t_txt = delim.join(t_pieces[:tag_position])
        # if text is non-Unicode then the delimiter breaks ;_;
        #if(abs(tag_position) >= len(t_pieces)):
        #    print('error with tag position %d, pieces %s, string %s'%(tag_position, str(t_pieces), txt))
        t_tag = t_pieces[tag_position]
        # build up entity string
        if(t_tag != outside_tag and (not use_bio or inside_tag_matcher.match(t_tag) or (before_tag_matcher.match(t_tag) and len(entity_str)==0))):
            # write everything
            if(include_type):
                entity_str.append(t)
            # write token + POS
            elif(use_POS_tag):
                t_POS = delim.join([t_pieces[0], t_pieces[-1]])
                entity_str.append(t_POS)
            # write token
            else:
                entity_str.append(t_txt)
        # store entity as multi-token string and reset entity string
        elif(len(entity_str) > 0):
            # if using type, attach the type of final token to multi-token string
            if(include_type):
                entity_type = entity_str[-1].split(delim)[tag_position]
                # get rid of "I-" prefix
                if(use_bio):
                    entity_type = entity_type[2:]
                # split entity string and tag
                entity_str = multi_token_delim.join(map(lambda x: x.split(delim)[0], entity_str))
                entity_list.append('%s%s%s'%(entity_str, delim, entity_type))
            # otherwise, just combine into multi-token string
            else:
                entity_list.append(multi_token_delim.join(entity_str))
            entity_str = []
            # start new entity with current string
            if(use_bio and t_tag != outside_tag):
                if(include_type):
                    entity_str.append(t)
                elif(use_POS_tag):
                    t_POS = delim.join([t_pieces[0], t_pieces[-1]])
                    entity_str.append(t_POS)
                else:
                    entity_str.append(t_txt)
    # take care of leftover entity string
    if(len(entity_str) > 0):
        if(include_type):
            entity_type = entity_str[-1].split(delim)[tag_position]
            # get rid of "I-" prefix
            if(use_bio):
                entity_type = entity_type[2:]
            # split entity string and tag
            entity_str = multi_token_delim.join(map(lambda x: x.split(delim)[0], entity_str))
            entity_list.append('%s%s%s'%(entity_str, delim, entity_type))
        # otherwise, just combine into multi-token string
        else:
            entity_list.append(multi_token_delim.join(entity_str))
    return entity_list

def split_tokens_types(entity_list, delim='/'):
    """
    For each entity token-type pair in list, 
    split by delimiter.

    Parameters:
    -----------
    entity_list : list
    token-delim-type strings.
    delim : str

    Returns:
    --------
    entity_list_split : list
    List of (token,entity-type) pairs.
    """
    entity_list_split = map(lambda x: tuple(x.split(delim)), entity_list)
    return entity_list_split

def filter_by_tag_type(tag_list, tag_type):
    """
    Filter list of entity-type pairs by specified type
    and return only the entities.
    
    Parameters:
    -----------
    tag_list : list
    (entity, type) pairs.
    tag_type = str
    
    Returns:
    --------
    tag_list : list
    Entity strings.
    """
    tag_list = filter(lambda x: x[1]==tag_type, tag_list)
    tag_list = map(lambda x: x[0], tag_list)
    return tag_list

def collect_bracketed_entities_from_txt(txt):
    """
    Collect entities from text based on bracket
    format: "blah blah [[ENTITY]] blah".
    
    Parameters:
    -----------
    txt : str

    Returns:
    --------
    entity_list : list
    """
    entity_matcher = re.compile('(?<=\[\[)[^\]]+(?=\]\])')
    entity_list = entity_matcher.findall(txt)
    return entity_list

DOUBLE_BRACKET_MATCHER = re.compile(r'(\[\[[^\]]+\]\])(\{[A-Z]+\})')
def collect_double_bracketed_entities(txt):
    """
    Collect entities in the form [[NAME]]{TYPE}.
    
    Parameters:
    -----------
    txt : str
    
    Returns:
    --------
    entity_list : list
    """
    entity_list = [(x[2:-2], y[1:-1]) for (x,y) in DOUBLE_BRACKET_MATCHER.findall(txt)]
    return entity_list

def write_txt_to_file(txt_series, file_name):
    """
    Write each line of text and its index to file.
    
    Parameters:
    -----------
    txt_series : pandas.Series
    file_name : str
    """
    N = txt_series.index.max()
    txt_series.sort_index(inplace=True)
    with codecs.open(file_name, 'w', encoding='utf-8') as file_output:
        for i, l in izip(txt_series.index, txt_series):
            # need to include index to original dataframe
            l_clean = l.decode('utf-8').replace('\n', '')
            file_output.write('INDEX%s\t%s'%(i, l_clean))
            # add line break for every line except the last
            if(i < N):
                file_output.write('\n')

TWEET_NUM_MATCHER=re.compile('[\d\.]+')
TWEET_AT_MATCHER=re.compile('@\w+')
TWEET_HASH_MATCHER=re.compile('#[^ #]+')
TWEET_FINAL_HASH_MATCHERS=re.compile('#\w+\s?(?=#\w+)$')
TWEET_URL_MATCHER=re.compile('http://[\w\.#//-]+|t.co/[\w\.#//-]+')
TWEET_MATCHERS=[TWEET_NUM_MATCHER, TWEET_AT_MATCHER, TWEET_HASH_MATCHER, TWEET_URL_MATCHER]
# TWEET_MATCHERS=[TWEET_AT_MATCHER, TWEET_URL_MATCHER]
TWEET_MATCHERS_SUBS = [[TWEET_NUM_MATCHER, '<NUM>'], [TWEET_AT_MATCHER, '@USER'], [TWEET_HASH_MATCHER, '#HASH'], [TWEET_URL_MATCHER, '<URL>']]
def clean_tweet_txt(txt):
    """
    Replace @, #, numbers and URL with generic mentions.
    
    Parameters:
    -----------
    txt : str
    
    Returns:
    --------
    txt_clean : str
    """
    txt_clean = txt
    # replace all matches with whitespace to prevent indexing problems
    for m, s in TWEET_MATCHERS_SUBS:
        for x in m.findall(txt_clean):
            txt_clean = txt_clean.replace(x, s)
    return txt_clean

PUNCT_MATCHER = re.compile('[-,]')
TWEET_MATCHER = re.compile('<URL>|@USER|#HASH')
def clean_data_for_spacy(x):
    """
    Clean data before parsing with spacy.
    """
    x = PUNCT_MATCHER.sub(' ', x)
    x = TWEET_MATCHER.sub('', x)
    return x

def clean_tagged_txt(txt_lines, delim='(?<=INDEX)[0-9_]+'):
    """
    Clean the irregular spacing in text lines (for tagged text)
    by using the index markers as line anchors.
    
    Parameters:
    -----------
    txt_lines : [str]
    delim = str
    Delimiter to mark line split.
    
    Returns:
    --------
    fixed_txt : [str]
    """
    curr_txt = []
    fixed_txt = []
    index_matcher = re.compile(delim)
    for i, l in enumerate(txt_lines):
        l = l.replace('\t', ' ')
        for t in l.split(' '):
            index_match = index_matcher.findall(t)
            if(len(index_match) > 0):
      #          index_match = index_match[0]
                if(len(curr_txt) > 0):
                    fixed_txt.append(' '.join(curr_txt))
                    curr_txt = []
      #              curr_txt = index_match + '\t'
      #          else:
      #              curr_txt = index_match + '\t'
            else:
                curr_txt.append(t)
        if(i % 10000 == 0):
            print('processed %d lines'%(i))
    # cleanup
    fixed_txt.append(' '.join(curr_txt))
    return fixed_txt

def clean_raw_txt(txt, lower=True):
    """
    Clean raw text before extracting
    lexicon toponyms.
    
    Parameters:
    -----------
    txt : list
    lower : bool
    Whether to lowercase text.
    
    Returns:
    --------
    txt : list
    """
    inter_word_sub = lambda x: '(?<=[\s,\.;!?])%s(?=[\s,\.;!?])|$%s(?=[\s,\.;!?])'%(x,x)
    replacers = [[re.compile('( urb\.?)|(^urb\.? )'), ' urbanización '],
                 [re.compile('( Urb\.?)|(^Urb\.? )'), ' Urbanización '],
                 [re.compile('( av\.?)|(^avda\.? )'), ' avenida '],
                 [re.compile('( Av\.?)|(^Avda\.? )'), ' Avenida '],
                 [re.compile('( bo\.?)|(^bo\.? )'), ' barrio '],
                 [re.compile('( Bo\.?)|(^Bo\.? )'), ' Barrio '],
                 [re.compile('((?<=)c/|^c/)'), 'calle '],
                 [re.compile('((?<=)C/|^C/)'), 'Calle '],
                 [re.compile(inter_word_sub('carr.')), ' carretera '],
                 [re.compile(inter_word_sub('Carr.')), ' Carretera ']
                 ]
    spanish_nums = ['uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez']
    replacers += [[re.compile(spanish_nums[i-1]), '%d'%(i)] for i in range(1,11)]
    replacers += [[re.compile('\s+'), ' ']]
    replacers += [[re.compile('https?://[^\s]+'), '<URL>']]
    def txt_sub(t):
#         t = t.encode('utf-8')
        # expand names
        for e, sub in replacers:
            t = e.sub(sub, t)
        return t
    # lowercase
    if(lower):
        txt = list(map(lambda x: x.lower(), txt))
    # substitute
    txt = list(map(txt_sub, txt))
    return txt

def collect_lexicon_toponyms_from_txt(txt, lexicon):
    """
    Collect all n-grams in text that match the provided
    lexicon.

    Parameters:
    -----------
    txt : list
    lexicon : list
    
    Returns:
    --------
    toponym_list : [list]
    """
    # lowercase lexicon
    lexicon_lower = map(lambda x: x.lower(), lexicon)
    txt = clean_raw_txt(txt)
    ngram_range = (1,5)
    vocab = lexicon
    tokenizer = TweetTokenizer()
    min_df = 1
    # convert to DTM
    cv = CountVectorizer(min_df=min_df, ngram_range=ngram_range, lowercase=True, tokenizer=tokenizer.tokenize, vocabulary=vocab)
    dtm = cv.fit_transform(txt)
    toponyms_per_post = {i : list() for i in range(dtm.shape[0])}
    dtm_nonzero = dtm.nonzero()
    ivoc = {v:k for k,v in cv.vocabulary_.iteritems()}
    # convert to list of lists
    for r, c in izip(dtm_nonzero[0], dtm_nonzero[1]):
        toponyms_per_post[r].append(ivoc[c])
    toponym_list = pd.Series(toponyms_per_post).values.tolist()
    return toponym_list

def annotate_lexicon_toponyms_in_txt(txt, lexicon):
    """
    For each line in text, annotate all n-grams that
    match lexicon.
    Start with max-length n-gram and work down: 
    in text = "vive en la iglesia de san juan",
    the n-gram "iglesia de san juan" should supercede
    "san juan" as a toponym if it exists in the lexicon.
    TODO: fuzzy matching based on edit distance...even if this takes 5ever.

    Parameters:
    -----------
    txt : list
    lexicon : list
    
    Returns:
    --------
    annotated_txt : list
    """
    # lowercase lexicon
    lexicon_lower = map(lambda x: x.lower(), lexicon)
    txt = clean_raw_txt(txt)
    annotated_txt = []
    # get maximum-length n-grams that match lexicon
    ngram_range = list(reversed(range(1,6)))
    index_matcher = re.compile('(?<=_)[0-9]+')
    index_matcher_2 = re.compile('_[0-9]+')
    annotate_brackets = '[[%s]]'
    get_min_index = lambda x: min(map(int, index_matcher.findall(x)))
    for l in txt:
        l_tokens = l.split(' ')
        l_tokens_num = map(lambda x: '%s_%d'%(x[1], x[0]), enumerate(l_tokens))
        l_num = ' '.join(l_tokens_num)
        l_extracted = list()
        L = len(l_tokens)
        for n in ngram_range:
            for i in range(L-n):
                l_gram = ' '.join(l_tokens[i:i+n])
                l_gram_num = ' '.join(l_tokens_num[i:i+n])
                if(l_gram in lexicon):
                    l_gram_already_counted = len(filter(lambda x: l_gram_num in x, l_extracted))
                    if(l_gram_already_counted == 0):
                        l_extracted.append(l_gram_num)
        # sort by index
        l_extracted = sorted(l_extracted, key=lambda x: get_min_index(x))
        # remove indexes
        # l_extracted = map(lambda x: index_matcher_2.sub('', x), l_extracted)
        # replace in text
        l_annotated = l_num
        for e in l_extracted:
            l_annotated = l_annotated.replace(e, annotate_brackets%(e))
        
        # remove indexes
        l_annotated = index_matcher_2.sub('', l_annotated)
        annotated_txt.append(l_annotated)
    return annotated_txt

## Facebook API auth
def load_facebook_auth(auth_file_name='../../data/facebook_auth.csv'):
    """
    Load Facebook API auth credentials from file.

    Parameters:
    -----------
    auth_file_name : str

    Returns:
    --------
    app_id : str
    app_secret : str
    access_token : str
    Long-term user access token.
    """
    auth = dict([l.strip().split(',') for l in open(auth_file_name)])
    app_id = auth['app_id']
    app_secret = auth['app_secret']
    access_token = auth['access_token']
    return app_id, app_secret, access_token

## Twitter API auth
def load_twitter_auth(auth_file_name='../../data/auth.csv'):
    """
    Load Twitter API auth credentials from file.
    
    Parameters:
    -----------
    auth_file_name : str
    
    Returns:
    --------
    consumer_key : str
    consumer_secret : str
    access_token : str
    access_secret : str
    """
    auth_lines = [x.strip().split(',') for x in open(auth_file_name, 'r')]
    auth_info = {k : v for k,v in auth_lines}
    consumer_key = auth_info['consumer_key']
    consumer_secret = auth_info['consumer_secret']
    access_token = auth_info['access_token']
    access_secret = auth_info['access_secret']
    return consumer_key, consumer_secret, access_token, access_secret

## entity lexicon filter functions
def test_unicode(x):
    try:
        x.encode('utf-8')
        return True
    # Python 2
#     except Exception, e:
#         return False
    # Python 3
    except(Exception, e):
        return False
MAX_TOKEN_LENGTH=5
MIN_CHAR_LENGTH=2
def test_length(x):
    x_tokens = x.split(' ')
    return len(x_tokens) <= MAX_TOKEN_LENGTH and len(x) >= MIN_CHAR_LENGTH
noise_chars = ['\|', '\(', '\)', '\+', '\*', '?', '\[', '\]', '\^', '\$']
noise_finder = re.compile('[%s]'%(noise_chars))
def clean_mention(x):
    x_clean = x.strip()
    x_noise = noise_finder.findall(x_clean)
    for n in x_noise:
        x_clean = x_clean.replace(n, '\%s'%(n))
    x_clean = x_clean.replace('_', ' ')
    return x_clean
URL_MATCHER = re.compile('https?://.*|[a-z]{2}.wikipedia.org/.*')
def test_url(x):
    return not URL_MATCHER.match(x)
  
def get_mention_entity_lists(dict_file='/hg190/corpora/crosswikis-data.tar.bz2/dictionary.bz2', verbose=False):
    """
    Extract mention-entity mapping from lexicon of Wiki cross-link data.
    
    Parameters:
    -----------
    dict_file : str
    Dict file stored with tab-separated mention-entity pairs (plus probability if you wanna get fancy).
    
    Returns:
    --------
    mention_entity_lists = {str : list}
    For each mention in lexicon, store list of possible entity redirects.
    """
    mention_entity_lists = defaultdict(list)
    for i, l in enumerate(BZ2File(dict_file, 'r')):
        l_split = l.split('\t')
        if(len(l_split) > 1):
            mention, entity_info = l_split
            mention = clean_mention(mention)
            entity_info_split = entity_info.split(' ')
            entity_name = entity_info_split[1]
            if(test_unicode(mention) and test_length(mention) and test_url(mention)):
                mention_entity_lists[mention].append(entity_name)
        if(i % 1000000 == 0):
            print('processed %d lines'%(i))
    return mention_entity_lists

def extract_annotations(annotate_file):
    """
    Extract annotations from each line of text in file.
    
    Parameters:
    -----------
    annotate_file : str
    
    Returns:
    --------
    annotation_list : list
    """
    annotate_matcher = re.compile('<<([\w\s\,\.]+)>>\[\[(\S+)\]\]')
    annotated_txt = [l.strip() for i, l in enumerate(codecs.open(annotate_file, 'r', encoding='utf-8'))]
    annotation_list = [annotate_matcher.findall(l) for l in annotated_txt]
    return annotation_list

def extract_topo_annotations(annotation_txt, include_coords=False):
    """
    Extract toponym annotations and labels from text.

    Parameters:
    -----------
    annotation_txt : list
    
    Returns:
    --------
    topo_mentions : list
    """
    topo_mentions = []
    if(include_coords):
        annotate_coord_matcher = re.compile(r'(\[\[[^\]]+\]\])(\{[A-Z]+\})(\{[^\}]+\})')
        annotate_matcher = re.compile(r'(\[\[[^\]]+\]\])(\{[A-Z]+\})(?!\{)')
    else:
        annotate_matcher = re.compile(r'(\[\[[^\]]+\]\])(\{[A-Z]+\})')
    for t in annotation_txt:
        annotations = []
        if(include_coords):
            annotations += [(x[2:-2], y[1:-1], z[1:-1]) for (x,y,z) in annotate_coord_matcher.findall(t)]
        annotations += [(x[2:-2], y[1:-1]) for (x,y) in annotate_matcher.findall(t)]
        topo_mentions.append(annotations)
    return topo_mentions

def extract_topo_extras(topo_data_str):
    """
    Extract additional toponym information from
    string.

    Parameters:
    -----------
    topo_data_str = str
    String containing ID, lat, lon values.

    Returns:
    --------
    topo_data_extra : dict
    ID, lat, lon values.
    """
    topo_data_extra_split = topo_data_str.split(',')
    topo_data_extra = {'id' : 0, 'lat' : 0, 'lon' : 0}
    if(topo_data_extra_split[0] != ''):
        topo_data_extra['id'] = int(topo_data_extra_split[0])
    if(topo_data_extra_split[1] != ''):
        topo_data_coords = map(float, topo_data_extra_split[1:])
        topo_data_extra['lat'] = topo_data_coords[0]
        topo_data_extra['lon'] = topo_data_coords[1]
    return topo_data_extra

def organize_topo_annotations(topo_annotations, status_ids=None):
    """
    Organize topo annotations extracted from
    extract_topo_annotations.
    
    Parameters:
    -----------
    topo_annotations : list
    status_ids : list
    Optional IDs to include with each status.
    
    Returns:
    --------
    topo_data_df : pandas.DataFrame
    Rows = samples, columns = topo mention, topo id, topo lat, topo lon, status id.
    """
    topo_data_df = []
    topo_matcher = re.compile('TOPO')
    for i, annotation_set in enumerate(topo_annotations):
        if(status_ids is not None):
            status_id = status_ids[i]
        topo_set = filter(lambda x: len(topo_matcher.findall(x[1])) > 0, annotation_set)
        for topo_info in topo_set:
            topo_data = {}
            topo_mention = topo_info[0]
            topo_mention = expand_mention(topo_mention)
            topo_data['topo'] = topo_mention
            topo_data.update(extract_topo_extras(topo_info[2]))
            if(status_ids is not None):
                topo_data['status_id'] = status_id
            topo_data = pd.Series(topo_data)
            topo_data_df.append(topo_data)
    topo_data_df = pd.concat(topo_data_df, axis=1).transpose()
    return topo_data_df

def organize_topo_city_annotations(topo_annotations, geo_data, geo_lexicon):
    """
    Organize topo and city annotations extracted from
    extract_topo_annotations.
    
    Parameters:
    -----------
    topo_annotations : list
    geo_data : pandas.DataFrame
    geo_lexicon : list
    
    Returns:
    --------
    topo_data_df : pandas.DataFrame
    Rows = samples, columns = topo mention, topo id, topo lat, topo lon, city lat/lon list.
    """
    topo_data_df = []
    for annotation_set in topo_annotations:
        topo_set = filter(lambda x: x[1] == 'TOPO', annotation_set)
        city_set = filter(lambda x: x[1] == 'CITY', annotation_set)
        city_names = map(lambda x: x[0], city_set)
        # convert city mentions to best guess for city
        city_data = convert_city_mentions(city_names, geo_data, geo_lexicon)
        city_lat_lons = city_data.loc[:, ['lat', 'lon']].values.tolist()
        for topo_info in topo_set:
            topo_data = {}
            topo_mention = topo_info[0]
            topo_mention = expand_mention(topo_mention)
            topo_data['topo'] = topo_mention
            topo_data['cities'] = city_names
            topo_data['city_lat_lons'] = city_lat_lons
            topo_data.update(extract_topo_extras(topo_info[2]))
            topo_data = pd.Series(topo_data)
            topo_data_df.append(topo_data)
    topo_data_df = pd.concat(topo_data_df, axis=1).transpose()
    return topo_data_df

def extract_status_ids(annotation_lines, original_lines):
    """
    Extract all IDs from statuses separated by line breaks.
    This involves removing the extra topo annotations and 
    searching for the matching status line in the original
    status file.
    
    Parameters:
    -----------
    annotation_lines : list
    original_lines : list
    
    Returns:
    --------
    status_id_list : list
    """
    topo_matcher = re.compile('(\[\[[^\]]+\]\]\{TOPO[/CITY]*\})(\{[^\}]+\})')
    status_id_list = []
    for l in annotation_lines:
        topo_matches = topo_matcher.findall(l)
        for short_annotation, extra_info in topo_matches:
            full_str = '%s%s'%(short_annotation, extra_info)
            l = l.replace(full_str, short_annotation)
        original_annotation_l = list(filter(lambda x: l in x, original_lines))
        if(len(original_annotation_l) > 0):
            id_l = original_annotation_l[0].split('\t')[0]
            status_id_list.append(id_l)
    return status_id_list

def convert_city_mentions(city_mentions, geo_data, geo_lexicon):
    """
    Convert list of city mentions to most likely
    candidates.
    
    Parameters:
    -----------
    city_mentions : list
    geo_data : pandas.DataFrame
    geo_lexicon : list
    
    Returns:
    --------
    city_data : pandas.DataFrame
    """
    city_mentions_lower = map(lambda x: x.lower(), city_mentions)
    city_feature_codes = ['first-order_administrative_division', 'second-order_administrative_division', 'seat_of_a_first-order_administrative_division', 'populated_place']
    is_a_city = geo_data.loc[:, 'feature_code'].isin(city_feature_codes)
    city_top_k = 5
    city_data_list = []
    for city_mention in city_mentions_lower:
        # do approximate matching again fml
        city_name_matches = geo_lookup(city_mention, geo_data, geo_lexicon, word_char=True, k=city_top_k)
        city_data_m = geo_data[is_a_city & geo_data.loc[:, 'name_lower_no_diacritic'].isin(city_name_matches.loc[:, 'name'])]
        city_data_m = city_data_m.iloc[0, :]
        city_data_list.append(city_data_m)
    city_data = pd.concat(city_data_list, axis=1).transpose()
    return city_data

## extraction metrics

def test_precision_recall(tagged_entity_list, gold_entity_list):
    """
    Compute the precision and recall of entity recognition 
    based on the number of entities recovered out of
    all possible entities.
    
    Parameters:
    -----------
    tagged_entity_list : list
    gold_entity_list : list
    
    Returns:
    --------
    false_positives : list
    false_negatives : list
    precision : float
    recall : float
    """
    TP = 0
    FP = 0
    FN = 0
    false_positives = []
    false_negatives = []
    for tagged, gold in izip(tagged_entity_list, gold_entity_list):
        # TODO: same test but without awkward set notation
        tagged_counts = pd.Series(Counter(tagged))
        gold_counts = pd.Series(Counter(gold))
        vocab = tagged_counts.index | gold_counts.index
        V = len(vocab)
        if(V > 0):
            dummy_counts = pd.Series(np.zeros(V), index=vocab)
            if(len(tagged_counts) == 0):
                tagged_counts = dummy_counts.copy()
            if(len(gold_counts) == 0):
                gold_counts = dummy_counts.copy()
            #try:
            #    tagged_counts.loc[vocab].fillna(0, inplace=False)
            #    gold_counts.loc[vocab].fillna(0, inplace=False)
            #except Exception, e:
            #    print('bad vocab %s'%(str(vocab)))
            tagged_gold_diff = (tagged_counts.loc[vocab].fillna(0, inplace=False) - 
                                gold_counts.loc[vocab].fillna(0, inplace=False)).astype(int)
            true_tags = tagged_gold_diff[tagged_gold_diff == 0].index.tolist()
            false_positive_tags = tagged_gold_diff[tagged_gold_diff > 0].index.tolist()
            false_negative_tags = tagged_gold_diff[tagged_gold_diff < 0].index.tolist()
            # multiply by actual counts
            false_positive_tags = [i for i in false_positive_tags for c in range(tagged_gold_diff.loc[i]) ]
            false_negative_tags = [i for i in false_negative_tags for c in range(abs(tagged_gold_diff.loc[i])) ]
            #true_tags = set(tagged) & set(gold)
            #false_positive_tags = set(tagged) - set(gold)
            #false_negative_tags = set(gold) - set(tagged)
            TP += len(true_tags)
            FP += len(false_positive_tags)
            FN += len(false_negative_tags)
        else:
            false_positive_tags = []
            false_negative_tags = []
        false_positives.append(false_positive_tags)
        false_negatives.append(false_negative_tags)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return false_positives, false_negatives, precision, recall

def get_precision_at_k(topo_data_list, gold_labels):
    """
    Compute precision at K over list of toponym candidates and gold labels.
    
    Parameters:
    -----------
    topo_data_list : [pandas.DataFrame]
    gold_labels : [int]
    
    Returns:
    --------
    precision : float
    """
    true_positive = 0
    N = len(gold_labels)
    for topo_data, gold_label in izip(topo_data_list, gold_labels):
        match = int(gold_label in topo_data.loc[:, 'id'])
        true_positive += match
    precision = true_positive / N
    return precision

def get_mean_dist(topo_data_list, gold_locs):
    """
    Compute mean distance between the top toponym's location
    and the gold location.
    
    Parameters:
    -----------
    topo_data_list : [pandas.Series]
    gold_locs : [(float, float)]

    Returns:
    --------
    mean_dist : float
    std_dist : float
    """
    dists = []
    for topo_data, gold_loc in izip(topo_data_list, gold_locs):
        t_loc = topo_data.loc[['lat', 'lon']].values.tolist()
        dist = great_circle(t_loc, gold_loc).miles
        dists.append(dist)
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    return mean_dist, std_dist

## group data organization

def get_all_group_files(data_dir='../../data/facebook-maria/', 
                        location_info_file='../../data/facebook-maria/location_group_data.tsv'):
    """
    Extract all location group post files from data directory.

    Parameters:
    -----------
    data_dir : str
    location_info_file : str
    
    Returns:
    --------
    group_files : list
    """
    location_info = pd.read_csv(location_info_file, sep='\t', index_col=False)
    group_ids = location_info.loc[:, 'group_id'].values.tolist()
    file_matcher = re.compile('.*facebook_posts.tsv')
    # group_files = [file_base_str%(g, date_str) for g in group_ids]
    group_files = filter(file_matcher.match, os.listdir(data_dir))
    group_files = filter(lambda x: int(x.split('_')[0]) in group_ids, group_files)
    group_files = map(lambda x: os.path.join(data_dir, x), group_files)
    return group_files

def get_all_group_data():
    """
    Extract all group data from files.

    Returns:
    --------
    group_data : pandas.DataFrame
    Rows = posts, columns = data.
    """
    group_files = get_all_group_files()
    group_data_list = []
    for f in group_files:
        group_df = pd.read_csv(f, sep='\t', index_col=False)
        # add group ID
        group_id = int(os.path.basename(f).split('_')[0])
        group_df.loc[:, 'group_id'] = group_id
        group_data_list.append(group_df)
    group_data = pd.concat(group_data_list, axis=0)
    return group_data

def load_combined_group_data(file_name='../../data/facebook-maria/combined_group_data.tsv'):
    """
    Assume that we've already written combined data;
    now let's load it.
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    post_data : pandas.DataFrame
    """
    post_data = pd.read_csv(file_name, sep='\t', index_col=False, encoding='utf-8')
    # convert times
    post_data.loc[:, 'status_published'] = post_data.loc[:, 'status_published'].apply(lambda x: date_parser.parse(x))
    return post_data

def load_group_location_data(file_name='../../data/facebook-maria/location_group_data.tsv'):
    """
    Load group location data (mapping each group ID to municipality).
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    location_data : pandas.DataFrame
    """
    location_data = pd.read_csv(file_name, sep='\t', index_col=False, encoding='utf-8')
    return location_data

## geo data manipulation

def containment_test(lon_lat, municipality_shapes):
    """
    Test which municipality is most likely to contain
    specified lat/lon point.

    Parameters:
    -----------
    lon_lat : (float, float)
    municipality_shapes : list
    Pair of municipality names and Shapes.
    Returns:
    --------
    municipality_name : str
    """
    point = Point(lon_lat)
    for municipality_name, municipality_shape in municipality_shapes:
        if(municipality_shape.contains(point)):
            return municipality_name
    return None

def load_osm_data(osm_file='../../data/geo_files/PR_OSM/combined_shp_xml_data.tsv'):
    """
    Load OSM data as data frame.

    Parameters:
    -----------
    osm_file : str
    
    Returns:
    --------
    osm_data : pandas.DataFrame
    """
    # need to exclude "all_points" column because it's TOO BIG
    use_cols = ['name', 'geometry_type', 'feature_class', 'osm_id', 'lon', 'lat', 'centroid', 'shp_type']
    osm_data = pd.read_csv(osm_file, sep='\t', usecols=use_cols)
    return osm_data

def load_geonames_data(geonames_file='../../data/geo_files/PR_geonames_gazetteer/geonames_data.tsv'):
    """
    Load GeoNames data as data frame.
    
    Parameters:
    -----------
    geonames_file : str
    
    Returns:
    --------
    geonames_data : pandas.DataFrame
    """
    geonames_data = pd.read_csv(geonames_file, sep='\t', index_col=False)
    return geonames_data

def load_gazetteer(gazetteer_file='/hg190/corpora/GeoNames/allCountriesSimplified.tsv'):
    """
    Load GeoNames gazetteer.
    
    :param gazetteer_file: .tsv with gazetteer data
    :returns gazetteer_data: pandas.DataFrame with gazetteer
    """
    gazetteer_data = pd.read_csv(gazetteer_file, sep='\t', index_col=False)
    return gazetteer_data

def generate_coord(lon1, lon_range, lat1, lat_range):
    """
    Generate random lat/lon point within specified ranges.
    
    Parameters:
    -----------
    lon1 : float
    Min longitude.
    lon_range : float
    Max - min longitude.
    lat1 : float
    Min latitude.
    lat_range : float
    Max - min latitude.
    
    Returns:
    --------
    coord : geometry.Point
    """
    coord = Point([np.random.rand(1)[0] * (lon_range) + lon1,
                   np.random.rand(1)[0] * (lat_range) + lat1])
    return coord

def sample_points(shape, N, bbox=None, pos_sample=True):
    """
    Sample N points from within the specified shape.
    
    Parameters:
    -----------
    shape : geometry.Polygon, geometry.MultiPolygon
    N : int
    bbox : list
    [lon1, lat1, lon2, lat2] = lower-left, upper-right corner of box.
    pos_sample : bool
    If we want a positive sample (versus a negative sample).
    """
    if(isinstance(shape, shapefile._Shape)):
        # convert to Polygon if necessary
        shape = Polygon(shape.points)
    if(bbox is None):
        bbox = shape.bounds
    lon1 = bbox[0]
    lon2 = bbox[2]
    lat1 = bbox[1]
    lat2 = bbox[3]
    lon_range = lon2 - lon1
    lat_range = lat2 - lat1
    sample = []
    for i in range(N):
        coord = generate_coord(lon1, lon_range, lat1, lat_range)
        while((not shape.contains(coord) and pos_sample) or
              (shape.contains(coord) and not pos_sample)):
            coord = generate_coord(lon1, lon_range, lat1, lat_range)
        sample.append(coord)
    return sample

def collect_all_children(hierarchy, parent_node=6295630):
    """
    Collect all of the parent node's (leaf) children in hierarchy.
    
    Parameters:
    -----------
    hierarchy : networkx.DiGraph
    parent_node : int
    
    Returns:
    --------
    children : list
    """
    children = []
    parents = [parent_node]
    while(len(parents) > 0):
        p = parents.pop()
#         q_children = map(lambda x: x[1], hierarchy.out_edges(p))
        p_children = list(hierarchy.neighbors(p))
        if(len(p_children) == 0):
            children.append(p)
        else:
            parents += p_children
    return children

def sample_from_hierarchy(hierarchy, parent_node=6295630, sample_size=100, replace=True):
    """
    Sample points from GeoNames hierarchy: starting at
    the parent node, go down the tree until
    hitting a leaf node, and take that as the sample.
    
    Parameters:
    -----------
    hierarchy : networkx.Digraph
    parent_node : int
    sample_size : int
    replace : bool
    
    Returns:
    --------
    sample : list
    """
    sample = []
    ctr = 0
    if(not replace):
        hierarchy = hierarchy.copy()
    p = parent_node
    for i in range(sample_size):
        s = None
        while(s is None):
            p_children = map(lambda x: x[1], hierarchy.out_edges(p))
#             p_children = list(hierarchy.neighbors(p))
            if(len(p_children) == 0):
                s = p
            else:
                p = np.random.choice(p_children, 1, replace=False)[0]
        if(not replace):
            hierarchy.remove_node(s)
        sample.append(s)
        p = parent_node
    return sample

def sample_from_hierarchy_graph_weighted(hierarchy, parent_node=6295630, sample_size=100, child_weight=1, replace=True, weight_smooth=1):
    """
    Sample points from GeoNames hierarchy and weight
    children based on out-degree: children with higher out-degree
    are more likely to be chosen.
    
    Parameters:
    -----------
    hierarchy : networkx.Digraph
    parent_node : int
    sample_size : int
    child_weight : int
    Extra weight to adjust children probabilities:
    P(child) ~ out_deg(child)**weight / sum(out_deg(children)**child_weight)
    replace : bool
    weight_smooth : int
    Count smoother for children => gives 0-degree children some probability of being picked.
    
    Returns:
    --------
    sample : int
    """
    sample = []
    h_degree = hierarchy.out_degree()
    if(not replace):
        hierarchy = hierarchy.copy()
    for i in range(sample_size):
        s = None
        p = parent_node
        while(s is None):
            p_children = list(hierarchy.neighbors(p))
            if(len(p_children) == 0):
                s = p
            else:
                p_children_weights = map(lambda x: (h_degree[x] + weight_smooth)**child_weight, p_children)
                p_children_weight_sum = sum(p_children_weights)
                # if all children have 0 weight, use uniform distribution
                if(p_children_weight_sum == 0):
                    p_children_weights = [1,]*len(p_children)
                    p_children_weight_sum = len(p_children)
                p_children_probs = map(lambda x: x / p_children_weight_sum, p_children_weights)
                p_children_data = pd.Series(p_children_probs, index=p_children)
                p_idx = np.where(np.random.multinomial(1, p_children_probs) == 1)[0][0]
                p = p_children[p_idx]
        sample.append(s)
        if(not replace):
            hierarchy.remove_node(s)
    return sample

def sample_by_feat(geonames, sample_size, feat_name, replace=True):
    """
    Sample for coordinates from GeoNames based on feature value (e.g. population).
    
    Parameters:
    -----------
    geonames_data : pandas.DataFrame
    sample_size : int
    feat_name : str
    replace : bool
    
    Returns:
    --------
    coords : numpy.array
    """
    geo_lookup = dict(zip(geonames.loc[:, 'geonames_ID'].values, 
                          geonames.loc[:, ['latitude', 'longitude']].values))
    geo_feat = geonames.loc[:, feat_name]
    min_feat_count = geo_feat[geo_feat > 0].min()
    smooth_counts = geo_feat + min_feat_count * 0.1
    pvals = geo_feat / geo_feat.sum()
    ## if we are replacing, we can do sampling in one fell swoop
    if(replace):
        samp_counts = np.random.multinomial(n=sample_size, pvals=pvals)
        samp_idx_counts = zip(range(geonames.shape[0]), samp_counts)
        samp_idx_counts = filter(lambda x: x[1] > 0, samp_idx_counts)
        samp = [[geonames.iloc[idx, :].loc['geonames_ID'],]*i for idx, i in samp_idx_counts]
        samp = list(reduce(lambda x,y: x+y, samp))
    ## if we are not replacing, we have to recompute the distribution every iteration
    else:
        geo_distribution = geonames.copy()
        geo_distribution.index = geo_distribution.loc[:, 'geonames_ID'].values
        samp = np.random.choice(geo_feat.index, size=sample_size, replace=replace, p=pvals)
    coords = np.array(list(map(geo_lookup.get, samp)))
    
    ## scramble!!
    np.random.shuffle(coords)
    return coords

def load_geonames_hierarchy_graph(file_name='/hg190/corpora/GeoNames/combined_hierarchy.gz'):
    """
    Load GeoNames hierarchy.
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    hierarchy : networkx.Digraph
    """
    hierarchy = nx.read_gpickle(file_name)
    return hierarchy

def extract_bounds(geo_boundaries):
    """
    Extract geo bounds from JSON object.
    
    Parameters:
    -----------
    geo_boundaries : dict
    
    Returns:
    --------
    geo_bounds : pandas.DataFrame
    Rows = countries, columns = geonames_ID, boundary as MultiPolygon.
    """
    geo_bounds = []
    for f in geo_boundaries['features']:
        f_id = int(f['properties']['geoNameId'])
        f_bounds = shape(f['geometry'])
        if(type(f_bounds) is MultiPolygon):
            f_bounds = f_bounds.buffer(0)
        geo_bounds.append([f_id, f_bounds])
    geo_bounds = pd.DataFrame(geo_bounds)
    geo_bounds.columns = ['geonames_ID', 'bounds']
    return geo_bounds

def load_country_boundaries(boundary_file='/hg190/corpora/GeoNames/shapes_simplified_low.json'):
    """
    Load boundaries for all countries in GeoNames.
    
    Parameters:
    -----------
    boundary_file : str
    
    Returns:
    --------
    geo_bounds : pandas.DataFrame
    Rows = countries, columns = geonames_ID, boundary as MultiPolygon.
    """
    with open(boundary_file) as bf:
        boundaries = geojson.load(bf)
    geo_bounds = extract_bounds(boundaries)
    return geo_bounds

def load_country_info(file_name='/hg190/corpora/GeoNames/countryInfo.txt'):
    """
    Load basic country info such as country codes, IDs, languages, etc.
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    country_info : pandas.DataFrame
    """
    country_info = pd.read_csv(file_name, sep='\t', index_col=False, skiprows=50)
    country_info.fillna('', inplace=True)
    return country_info

def load_admin1_data(file_name='/hg190/corpora/GeoNames/admin1CodesASCII.txt'):
    """
    Load data about admin1 regions (countries).
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    admin1_data : pandas.DataFrame
    """
    admin1_data = pd.read_csv(file_name, sep='\t', index_col=False, header=None)
    admin1_data.columns = ['admin1_code', 'name', 'name_norm', 'geonames_ID']
    return admin1_data

def load_admin2_data(file_name='/hg190/corpora/GeoNames/admin2Codes.txt'):
    """
    Load data about admin2 regions (states).
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    admin2_data : pandas.DataFrame
    """
    admin2_data = pd.read_csv(file_name, sep='\t', index_col=False, header=None)
    admin2_data.columns = ['admin2_code', 'name', 'name_norm', 'geonames_ID']
    return admin2_data

def load_geonames_hierarchy(file_name='/hg190/corpora/GeoNames/hierarchy.txt'):
    """
    Load GeoNames hierarchy of parent-child location pairs.
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    hierarchy : pandas.DataFrame
    Parent, child, admin code.
    """
    hierarchy = pd.read_csv(file_name, sep='\t', index_col=False, header=None)
    hierarchy.columns = ['parent', 'child', 'admin_code']
    return hierarchy

def load_municipality_data(municipality_file='../../data/geo_files/geo_data_municipalities.tsv'):
    """
    Load municipality data (the containing municipality for each ID in GeoNames/OSM data).Load
    
    Parameters:
    -----------
    municipality_file : str
    
    Returns:
    --------
    municipality_data : pandas.DataFrame
    """
    municipality_data = pd.read_csv(municipality_file, sep='\t', index_col=False)
    return municipality_data

def expand_mention(x):
    """
    Expand abbreviations in mention to standard
    forms.
    E.g., "urb" -> "urbanización".
    
    Parameters:
    -----------
    x : str
    
    Returns:
    --------
    x : str
    """
    replacers = [
        (re.compile('^urb[ \.]'), 'urbanización '),
        (re.compile('^Urb[ \.]'), 'Urbanización '),
        (re.compile('^carr[ \.]'), 'carretera '),
        (re.compile('^Carr[ \.]'), 'Carretera '),
        (re.compile('^CR '), 'Carretera '),
        (re.compile('^bda[ \.]'), 'barriada '),
        (re.compile('^Bda[ \.]'), 'Barriada '),
        (re.compile('^bo[ \.]'), 'barrio '),
        (re.compile('^Bo[ \.]'), 'Barrio '),
                ]
    for r, sub in replacers:
        x = r.sub(sub, x)
    return x

def load_combined_geo_data():
    """
    Load OSM and geonames data into combined data frame,
    and generate normalized names (e.g., lower, non-diacritic).
    
    Returns:
    --------
    geo_data : pandas.DataFrame
    """
    # may have combined data earlier
    combined_geo_file = '../../data/geo_files/combined_geo_data.tsv'
    if(os.path.exists(combined_geo_file)):
        geo_data = pd.read_csv(combined_geo_file, sep='\t', index_col=False, low_memory=False)
    else:
        geonames_data = load_geonames_data()
        osm_data = load_osm_data()
        geo_data = pd.concat([geonames_data, osm_data], axis=0)
        # get rid of null names
        geo_data.fillna('', inplace=True)
        geo_data = geo_data[geo_data.loc[:, 'name'] != '']
        geo_data = geo_data[geo_data.loc[:, 'name'].apply(lambda x: type(x) is str)]
        # get combined ID column
        geo_data.loc[:, 'id'] = (geo_data.loc[:, 'geoname_id'].map(str) + geo_data.loc[:, 'osm_id'].map(str)).apply(lambda x: int(float(x)))
        # get rid of duplicate IDs! 
        # (we already checked for possible collisions and found none)
        geo_data.drop_duplicates('id', inplace=True)
        # get name variants
        geo_data.loc[:, 'name_lower'] = geo_data.loc[:, 'name'].apply(lambda x: x.lower())
        geo_data.loc[:, 'name_lower_no_diacritic'] = geo_data.loc[:, 'name_lower'].apply(lambda x: unidecode(x.decode('utf-8')))
        # correct names
        geo_data.loc[:, 'name_expanded'] = geo_data.loc[:, 'name'].apply(lambda x: expand_mention(x))

    return geo_data

def generate_lexicon(geo_data):
    """
    Generate geo lexicon from combined data.
    
    Parameters:
    -----------
    geo_data : pandas.DataFrame
    """
    geo_lexicon = map(lambda x: str(x).strip(), geo_data.loc[:, 'name_lower_no_diacritic'].values.tolist())
    # get rid of duplicates
    geo_lexicon = list(set(geo_lexicon))
    return geo_lexicon

def load_lexicon():
    """
    Load toponym lexicon from OSM and GeoNames data.
    
    Returns:
    --------
    toponym_lexicon : list
    """
    osm_data = load_osm_data()
    print('loaded OSM')
    geonames_data = load_geonames_data()
    print('loaded geonames')
    # drop nan values
    osm_data.dropna(subset=['name'], inplace=True)
    geonames_data.dropna(subset=['name'], inplace=True)
    # collect unique names
    toponym_lexicon = list(set(osm_data.loc[:, 'name'].unique().tolist() + geonames_data.loc[:, 'name'].unique().tolist()))
    # lowercase because we're really good at this
    toponym_lexicon = list(set(map(lambda x: x.lower(), toponym_lexicon)))
    # TODO: principled way of removing common nouns and noise
    # such as "calle", "farmacia", "supermercado", "laura"
    # first remove short stuff
    NUM_MATCHER = re.compile('\d+')
    MIN_CHAR_LEN = 3
    # clunky: list of common banned words
    BANNED_WORDS = set(['calle', 'iglesia', 'parque', 'la familia'])
    def toponym_filter(t):
        return len(t) >= MIN_CHAR_LEN and NUM_MATCHER.match(t) is None and t not in BANNED_WORDS
    toponym_lexicon = filter(toponym_filter, toponym_lexicon)
    return toponym_lexicon

def load_filtered_lexicon():
    """
    Load pre-filtered lexicon.

    Returns:
    --------
    lexicon : list
    """
    lexicon_file = '../../data/geo_files/toponym_lexicon_filtered.txt'
    lexicon = [l.strip() for l in codecs.open(lexicon_file)]
    # add extras
    lexicon_lower = map(lambda x: x.lower(), lexicon)
    lexicon_lower_no_diacritic = map(lambda x: unidecode(x.decode('utf-8')), lexicon_lower)
    lexicon += lexicon_lower
    lexicon += lexicon_lower_no_diacritic
    # deduplicate
    lexicon = list(set(lexicon))
    return lexicon

## edit distance

# def get_edit_dist(mention, lexicon, word_level=False):
#     # word-level distance
#     if(word_level):
#         mention = mention.split(' ')
#         lexicon_split = map(lambda x: x.split(' '), lexicon)
#         dists = pd.Series(map(lambda x: int(editdistance.eval(x, mention)), lexicon_split))
#     else:
#         dists = pd.Series(map(lambda x: int(editdistance.eval(x, mention)), lexicon))
#     dists.index = lexicon
#     return dists

# def get_min_edit_dist_of_ngrams(match_string, candidate, word_level=False):
#     """
#     Compute minimum edit distance over all possible n-grams
#     in the candidate string and the match string.
#     This will help candidate strings that contain the match
#     string.
    
#     Parameters:
#     -----------
#     match_string : str
#     candidate : str
#     word_level : bool
    
#     Returns:
#     --------
#     min_edit : int
#     """
#     min_edit = np.inf
#     match_split = match_string.split(' ')
#     candidate_split = candidate.split(' ')
#     N = len(match_split)
#     M = len(candidate_split)
#     ngram_range = range(1, N+1)
#     for n in ngram_range:
#         for i in range(0, M-n+1):
#             ngram_split = candidate_split[i:i+n]
#             ngram = ' '.join(ngram_split)
#             if(word_level):
#                 ngram_dist = editdistance.eval(match_split, ngram_split)
#             else:
#                 ngram_dist = editdistance.eval(match_string, ngram)
#             min_edit = min(ngram_dist, min_edit)
#     return min_edit

# def get_norm_edit_dist(x, lexicon, word_level=False):
#     """
#     Compute normalized edit distance between a word and
#     all words in lexicon according to formula:
#     dist = edit_dist(x,y) / min(length(x), length(y))
    
#     Parameters:
#     -----------
#     x : str
#     lexicon : list
#     word_level : bool
    
#     Returns:
#     --------
#     edit_dists : pandas.Series
#     """
#     edit_dists = get_edit_dist(x, lexicon, word_level=word_level)
#     if(word_level):
#         edit_dists /= np.array(map(lambda y: min(len(x.split(' ')), len(y.split(' '))), lexicon))
#     else:
#         edit_dists /= np.array(map(lambda y: min(len(x), len(y)), lexicon))
#     return edit_dists

## geo data queries

def geo_lookup(mention, geo_data, geo_lexicon, name_col='name_lower_no_diacritic', k=5, word_char=False):
    """
    Return k most likely candidates for
    mention in geo data based on edit distance.
    
    Parameters:
    -----------
    mention : str
    geo_data : pandas.DataFrame
    geo_lexicon : list
    name_col : str
    Name column in geo_data to use when linking lexicon names => IDs.
    k : int
    word_char : bool
    Whether to measure edit distance as the sum of word and character level edit distance.
    If false, just use character level edit distance.
    
    Returns:
    --------
    candidates : pandas.DataFrame
    Rows = candidate, columns = name, ID, lat, lon.
    """
#     if(word_char):
#         dists_1 = get_edit_dist(mention, geo_lexicon, word_level=True)
#         dists_2 = get_edit_dist(mention, geo_lexicon, word_level=False)
#         dists = dists_1 + dists_2
#     else:
#         dists = get_edit_dist(mention, geo_lexicon, word_level=False)
    if(word_char):
        dists_1 = get_norm_edit_dist(mention, geo_lexicon, word_level=True)
        dists_2 = get_norm_edit_dist(mention, geo_lexicon, word_level=False)
        dists = dists_1 + dists_2
    else:
        dists = get_norm_edit_dist(mention, geo_lexicon, word_level=False)
    dists = dists.sort_values(inplace=False, ascending=True)
    dists = dists[~dists.index.duplicated(keep='first')]
    top_k_dists = dists[:k]
    candidate_names = set(top_k_dists.index.tolist())
    # join with full data
    candidates = geo_data[geo_data.loc[:, 'name_lower_no_diacritic'].apply(lambda x: x in candidate_names)].loc[:, ['name_lower_no_diacritic', 'id', 'lat', 'lon']]
    candidates.rename(columns={'name_lower_no_diacritic' : 'name'}, inplace=True)
    # sort by distance
    top_k_dists = pd.DataFrame([top_k_dists.index, top_k_dists], index=['name', 'dist']).transpose()
    candidates = pd.merge(candidates, top_k_dists, on='name')
    candidates.sort_values('dist', inplace=True, ascending=True)
    return candidates

## geo distance calculations
def lat_lon_dist(point_1, point_2):
    """
    Compute distance in miles between point 1 and point 2.
    
    Parameters:
    -----------
    point_1 : [lat, lon]
    point_2 : [lat, lon]
    
    Returns:
    --------
    dist : float
    """
    dist = great_circle(point_1, point_2).miles
    return dist

def compute_distances(topo_data, city_data, geo_data):
    """
    Compute mean distance between the toponym 
    ID and the city data.
    
    Parameters:
    -----------
    topo_data : pandas.Series
    city_data : list
    List of lat/lon pairs.
    geo_data : pandas.DataFrame
    
    Returns:
    --------
    dist : pandas.Series
    """
    topo_lat_lon = topo_data.loc[['lat', 'lon']]
    dist = sum(map(lambda x: lat_lon_dist(x, topo_lat_lon), city_data))
    dist = pd.Series([dist, topo_data.loc['name'], topo_data.loc['id']], index=['dist', 'name', 'id'])
    return dist

# converting string to dict
REPLACE_PAIRS = [
    (" u'", '"'),
    ("{u", "{"),
    ("[u", "["),
    ("':", '":'),
    ("',", '",'),
    ("{'", '{"'),
    ("'}", '"}'),
    ("['", '["'),
    ("']", '"]'),
    ("\\", '')
]
def clean_dict_str(x):
    for old, new in REPLACE_PAIRS :
        x = x.replace(old, new)
    return(x)
def decode_dict(x):
    # cleanup first
    x_clean = clean_dict_str(str(x))
    try:
        x_dict = json.loads(x_clean)
    # Python 2
#     except Exception, e:
#         x_dict = {}
    # Python 3
    except(Exception, e):
        x_dict = {}
    return(x_dict)

## normalize query text
# determining if text is Roman chars

AD = AlphabetDetector()
def is_latin(x):
    return AD.is_latin(x)

PUNCT_MATCHER = re.compile('[\.,\*\?\+"\']+')
REPLACEMENTS = [(re.compile('&'), ' and '), 
                (re.compile('$I-(?=[0-9])'), 'interstate '),
                (re.compile(' st$| st '), ' street '), 
                (re.compile(' rd$| rd '), ' road '),
                (re.compile(' hgwy$| hgwy '), ' highway '),
                (re.compile(' pl$| pl '), ' place '),
                (re.compile('^the '), ''),
                (re.compile(' hs$| hs '), ' high school '),
                (re.compile(' uni$| uni '), ' university '),
                ]
def normalize_str(s, regex=False):
    s = s.lower()
    # only decode text with roman chars
#     if((type(s) is unicode and is_latin(s)) or 
#        (type(s) is not unicode and is_latin(unicode(s.decode('utf-8'))))):
    if(is_latin(s)):
        s = unidecode(s)
    for r1, r2 in REPLACEMENTS:
        s = r1.sub(r2, s)
    # might have leftover space from replacements
    s = s.strip()
    s = PUNCT_MATCHER.sub('', s)
    # if converting string to regex, you need to escape special chars
    if(regex):
        s = re.escape(s)
        # fix space escaping
        s = s.replace('\ ', ' ')
    return s

CAMEL_MATCHER = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
# TODO: probabilistic word segmentation > camelcase splitting
def fix_camels(query):
    camels = CAMEL_MATCHER.findall(query)
    query_fixed = ' '.join(camels)
    return query_fixed

def query_norm(query, regex=False):
    # fix camelcase before lowering, escaping, etc.
    query = fix_camels(query)
    query = normalize_str(query, regex=regex)
    return query

ABBREVIATIONS = [(re.compile('$I-(?<=[0-9])| I-(?<=[0-9])'), 'Interstate'), 
                 (re.compile('Hosp.'), 'Hospital'),
                 (re.compile(' [Uu]ni$|^[Uu]ni '), 'University'),
                 (re.compile(' [Rr]d\.?'), 'Road'),
                 (re.compile(' [Ss]t\.?$'), 'Street'),
                 (re.compile(' [Aa]ve\.?'), 'Avenue'), 
                 (re.compile('Hgwy'), 'Highway')]
def expand_abbreviations(query):
    """
    Expand common English abbreviations
    to their full form, e.g. "I-" => "Interstate".
    
    Parameters:
    -----------
    query : str
    
    Returns:
    --------
    query : str
    """
    for a1, a2 in ABBREVIATIONS:
        a1_matches = a1.findall(query)
        for a1_match in a1_matches:
            query = query.replace(a1_match, a2)
    return query

NUM_MATCHER=re.compile('^\d+ ')
def strip_numbers(query):
    """
    Strip initial numbers from query.
    """
    query = NUM_MATCHER.sub('', query)
    return query

CARD_DIRECTIONS=re.compile('^north[ern]+ | north[ern]+ |^south[ern]+ | south[ern]+ |^east[ern]+ | east[ern]+ |^west[ern]+ | west[ern]+ ')
def strip_cardinal_directions(query):
    """
    Remove cardinal directions from query.
    """
    query = CARD_DIRECTIONS.sub('', query)
    return query

## query method
def process_query(q, flat_name_data):
    """
    Process query and generate exact, approximate
    matches for query in GeoNames database.
    
    Parameters:
    -----------
    q : str
    flat_name_data : pandas.DataFrame
    
    Returns:
    --------
    q_exact_results : list
    q_approx_results : list
    """
    # TODO: use string matching instead of regexes!!
    # normalize first
    q_norm = query_norm(q)
    
    # separate original, alt names
    original_names = flat_name_data[flat_name_data.loc[:, 'type'] == 'name']
    alt_names = flat_name_data[flat_name_data.loc[:, 'type'] == 'alternate_name']
    # exact match
    q_name_match = np.where(original_names.loc[:, 'name'] == q_norm)[0]
    q_exact_results = []
    if(len(q_name_match) > 0):
        for q_id in flat_name_data.loc[q_name_match, 'geonames_ID'].values.tolist():
            q_exact_results.append([q, q_id])
    # alternate match
    q_alt_match = np.where(alt_names.loc[:, 'name'] == q_norm)[0]
    q_approx_results = []
    if(len(q_alt_match) > 0):
        for q_id in flat_name_data.loc[q_alt_match, 'geonames_ID'].values.tolist():
            q_approx_results.append([q, q_id])
    if(len(q_approx_results) == 0):
        q_norm_stripped = strip_numbers(q_norm)
        q_norm_stripped = strip_cardinal_directions(q_norm_stripped)
        q_name_match = np.where(original_names.loc[:, 'name'] == q_norm_stripped)[0]
        q_alt_match = np.where(alt_names.loc[:, 'name'] == q_norm_stripped)[0]
        q_strip_match = list(q_name_match) + list(q_alt_match)
        if(len(q_strip_match) > 0):
            for q_id in flat_name_data.loc[q_strip_match, 'geonames_ID']:
                q_approx_results.append([q, q_id])
    
    ## old regex stuff
#     q_name_match = np.where(geonames.loc[:, 'name_norm'] == q_norm)[0]
#     q_exact_results = []
#     q_approx_results = []
#     if(len(q_name_match) > 0):
#         q_geonames = geonames.loc[q_name_match, :]
#         for q_id in q_geonames.loc[:, 'geonames_ID'].values.tolist():            
#             q_exact_results.append([q, q_id])
#     # approx match
#     # include alternate names by default: TODO determine impact on downstream disambiguation
#     #         else:
#     for name_id, name_regex in izip(name_ids, name_regexes):
#         n_match = name_regex.match(q_norm)
#         if(n_match is not None and n_match.group(0) == q_norm):
#             q_approx_results.append([q, name_id])
#     # rough match: remove numbers and cardinality
#     if(len(q_approx_results) == 0):
#         q_norm_stripped = strip_numbers(q_norm)
#         q_norm_stripped = strip_cardinal_directions(q_norm_stripped)
#         for name_id, name_regex in izip(name_ids, name_regexes):
#             if(name_regex.match(q_norm_stripped)):
#                 q_approx_results.append([q, name_id])
    return q_exact_results, q_approx_results

## GeoNames data
def load_simplified_geonames_data(file_name='/hg190/corpora/GeoNames/allCountriesSimplified.tsv'):
    """
    Load the simplified GeoNames data for baseline resolution system.
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    geonames_data : pandas.DataFrame
    """
    geonames_data = pd.read_csv(file_name, sep='\t', index_col=False, encoding='utf-8')
    # make sure ID column is integer
    geonames_data.loc[:, 'geonames_ID'] = geonames_data.loc[:, 'geonames_ID'].astype(int)
    return geonames_data

def load_full_geonames_data(file_name='/hg190/corpora/GeoNames/allCountries.zip'):
    """
    Load full GeoNames database for most complete geographic representation.
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    geonames_data : pandas.DataFrame
    """
    col_names = ['geonames_ID', 'name', 'ascii_name', 'alternate_names', 'latitude', 'longitude', 'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation', 'dem', 'timezone', 'mod_date']
    geonames_data = pd.read_csv(file_name, sep='\t', index_col=False, encoding='utf-8', compression='zip')
    geonames_data.columns = col_names
    return geonames_data
    
# names
def load_name_data(file_name='/hg190/corpora/GeoNames/geonames_clean_combined_names.tsv'):
    """
    Load clean combined names + alternate names for use
    in name matching.
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    name_data : pandas.DataFrame
    """
    name_data = pd.read_csv(file_name, sep='\t', index_col=False, encoding='utf-8')
    return name_data

# flat names
def load_flat_name_data(file_name='/hg190/corpora/GeoNames/alternate_names.tsv'):
    """
    One row per name => corresponds to GeoNames ID and name.
    
    Parameters:
    -----------
    file_name : str
    File containing GeoNames ID, name and each alternate name.
    
    Returns:
    --------
    flat_name_data : pandas.DataFrame
    Row = GeoNames ID, name.
    """
    alternate_names = pd.read_csv(file_name, sep='\t', index_col=False, encoding='utf-8')
    # flatten
    alternate_names_1 = alternate_names.loc[:, ['geonames_ID', 'name']].drop_duplicates(['geonames_ID', 'name'], inplace=False)
    alternate_names_1.loc[:, 'type'] = 'name'
    alternate_names_2 = alternate_names.loc[:, ['geonames_ID', 'alternate_name']].rename(columns={'alternate_name' : 'name'}, inplace=False)
    alternate_names_2.loc[:, 'type'] = 'alternate_name'
    flat_name_data = pd.concat([alternate_names_1, alternate_names_2], axis=0)
    # also include entries with no alternate names!
    geonames_data = load_simplified_geonames_data()
    geonames_data.fillna('', inplace=True)    
    geonames_data_no_alternate_names = geonames_data[geonames_data.loc[:, 'alternate_names'] == ''].loc[:, ['geonames_ID', 'name']]
    geonames_data_no_alternate_names.loc[:, 'type'] = 'name'
    flat_name_data = flat_name_data.append(geonames_data_no_alternate_names)
    flat_name_data.loc[:, 'name'] = flat_name_data.loc[:, 'name'].apply(lambda x: str(x) if (type(x) is not str and type(x) is not unicode) else x)
    return flat_name_data

## GeoCorpora data
def load_geocorpora_data(file_name='../../data/mined_tweets/GeoCorpora/geocorpora.tsv'):
    """
    Load basic GeoCorpora data (no tweet context).
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    geocorpora : pandas.DataFrame
    """
    geocorpora = pd.read_csv(file_name, sep='\t', index_col=0, encoding='utf-8')
    # make sure ID column is integer
    geocorpora.loc[:, 'geoNameId'] = geocorpora.loc[:, 'geoNameId'].replace('noGeonameId', 0, inplace=False).fillna(0, inplace=False)
    geocorpora.loc[:, 'geoNameId'] = geocorpora.loc[:, 'geoNameId'].astype(int)
    # get rid of invalid rows
    geocorpora = geocorpora[geocorpora.loc[:, 'geoNameId'] != 0]
    return geocorpora

## logging
def get_logger(log_file_name):
    """
    Set up basic logger with default output 
    settings for debugging.
    
    Parameters:
    -----------
    log_file_name : str
    
    Returns:
    --------
    logger : logging.Logger
    """
    if(os.path.exists(log_file_name)):
        os.remove(log_file_name)
    logger=logging.getLogger('basiclogger')
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.DEBUG)
    fmtr = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmtr)
    logger.addHandler(fh)
    return logger

## reading Excel files
def read_excel_as_dataframe(file_name):
    """
    Read Excel file and convert to DataFrame.
    
    Parameters:
    -----------
    file_name : str
    
    Returns:
    --------
    data : list
    List of DataFrame objects, one DataFrame per sheet.
    """
    xl_data = xlrd.open_workbook(file_name)
    data = []
    for s in xl_data.sheets():
        full_data_s = [pd.DataFrame([s.cell(r,c).value for c in range(s.ncols)]).transpose() for r in range(s.nrows)]
        data_s = pd.concat(full_data_s[1:])
        data_s.columns = full_data_s[0].values[0]
        data_s.index = pd.np.arange(data_s.shape[0])
        data.append(data_s)
    return data

## convert int to positional encoding
def positional(x, dim=64, norm_const=10000.):
    """
    Convert integer value to periodic/positional encoding.
    
    :param x: integer value
    :param dim: dimensions
    :param norm_const: normalizing constant
    :return pos_x:: positional encoding
    """
    pos_x = np.repeat([x], int(dim/2))
    norm_const = np.repeat([norm_const], int(dim/2))
    i_tensor = np.arange(0, int(dim/2))
    pos_x = pos_x / (norm_const ** (2.*i_tensor/dim))
    pos_x[0::2] = np.sin(pos_x[0::2])
    pos_x[1::2] = np.cos(pos_x[1::2])
    if(any(np.isnan(pos_x))):
        print('error with positional data %s'%(x))
    return pos_x

# context extraction
def extract_context_from_txt(entity_str, entity_start_char, entity_end_char, txt, context_window=5, TKNZR=None):
    """
    Find entity in text and iteratively extract tokens 
    from before and after the entity.
    
    :param entity_str: Entity string
    :param entity_start_char: Entity start char
    :param entity_end_char: Entity end char
    :param txt: Text containing entity
    :param context_window: context window size
    :param TKNZR: Tokenizer
    :returns start_context:: start context
    :returns end_context:: end context
    """
    if(TKNZR is None):
        TKNZR = NISTTokenizer()
    start_context = []
    end_context = []
    start_idx = entity_start_char - 1
    end_idx = entity_end_char + 1
    # collect start context
#     print('entity %s: start=%d end=%d'%(entity_str, entity_start_char, entity_end_char))
    while(len(start_context) < context_window+1):
        if(start_idx < 0):
            start_context = ['<START>'] + start_context
            break
        else:
            start_txt = txt[start_idx:entity_start_char]
            start_context = TKNZR.tokenize(start_txt)
            start_idx -= 1
    # remove first token if extra
    if(len(start_context) == context_window+1):
        start_context = start_context[1:]
    # collect end context
    while(len(end_context) < context_window+1):
        if(end_idx >= len(txt)):
            end_context = end_context + ['<END>']
            break
        else:
            end_txt = txt[(entity_end_char+1):end_idx]
            end_context = TKNZR.tokenize(end_txt)
            end_idx += 1
    # remove last token if extra
    if(len(end_context) == context_window+1):
        end_context = end_context[:-1]
    return start_context, end_context

def new_context_generation(text, entity_string, entity_start_ix, entity_end_ix, max_context_window=50, remove_stopwords=False, pre_post_split=False, sent_window=2, TKNZR=None):
    if(TKNZR is None):
        TKNZR = NISTTokenizer()
    text = text.lower()
    entity_string = entity_string.lower()
    pre_text = text[:entity_start_ix]
    post_text = text[entity_end_ix:]
    pre_match_text = sent_tokenize(pre_text)
    post_match_text = sent_tokenize(post_text)
    relevant_pre_match_text = ""
    for elem in pre_match_text[-sent_window:]:
        relevant_pre_match_text += elem
        relevant_pre_match_text += ' '
    relevant_post_match_text = ""
    for elem in post_match_text[:sent_window]:
        relevant_post_match_text += elem
        relevant_post_match_text += ' '
    pre_context_list = process_context_word(relevant_pre_match_text, if_join=False, rs=remove_stopwords, TKNZR=TKNZR)
    post_context_list = process_context_word(relevant_post_match_text, if_join=False, rs=remove_stopwords, TKNZR=TKNZR)
    pre_context = pre_context_list[-max_context_window:]
    post_context = post_context_list[:max_context_window]
    if pre_post_split:
        return pre_context, post_context
    return (pre_context + [entity_string] + post_context, len(pre_context))

# easy context update
def update_context_data(data, context_window, TKNZR=None, doc_col=None, sent_params=None):
    """
    Update context data with new window size.
    
    :param data: DataFrame to update
    :param context_window: Size of start/end context
    :param TKNZR: tokenizer
    :param doc_col: column with unique document ID (optional)
    :param sent_params: optional extra params ex. sentence context
    :returns updated_data:: updated DataFrame
    """
    if(doc_col is not None):
        updated_data = data.drop_duplicates(doc_col, inplace=False)
    else:
        updated_data = data.copy()
    if(TKNZR is None):
        TKNZR = NISTTokenizer()
    if(sent_params is not None):
        new_context_data = updated_data.apply(lambda x: sent_context_generation(x.loc['text'], x.loc['entity_string'], int(x.loc['char_start']), int(x.loc['char_end']), max_context_window=sent_params['max_context_window'], remove_stopwords=False, pre_post_split=True, sent_window=sent_params['sent_window'], TKNZR=TKNZR), axis=1)
    else:
        new_context_data = updated_data.apply(lambda x: extract_context_from_txt(x.loc['entity_string'], int(x.loc['char_start']), int(x.loc['char_end']), x.loc['text'], context_window=context_window, TKNZR=TKNZR), axis=1)
    updated_data.loc[:, 'entity_start_context'] = new_context_data.apply(lambda x: x[0])
    updated_data.loc[:, 'entity_end_context'] = new_context_data.apply(lambda x: x[1])
    updated_data.loc[:, 'context'] = updated_data.apply(lambda x: list(x.loc['entity_start_context']) + list(x.loc['entity_end_context']), axis=1)
    # recombine context with original data
    if(doc_col is not None):
        context_cols = ['context', 'entity_start_context', 'entity_end_context']
        updated_data = pd.merge(data.drop(context_cols, axis=1), updated_data.loc[:, context_cols+[doc_col]], on=doc_col, how='inner')
    return updated_data

def process_context_word(cw, if_join, rs, TKNZR=None):
    """
    Remove punctuations and stopwords from context but keep the approximate shape of the phrase.
    
    :param cw: text
    :param if_join: join tokens
    :param rs: remove stopwords
    :returns nonstop_pw_tokens: cleaned context
    """
    pw = cw.lower()
    pw = re.sub(r'[^\w\s]', ' ', pw)
    pw_tokens = TKNZR.tokenize(pw)
    if rs:
        nonstop_pw_tokens = [tok for tok in pw_tokens if tok not in stopwordset]
    else:
        nonstop_pw_tokens = pw_tokens
    if if_join:
        return ' '.join(nonstop_pw_tokens)
    else:
        return nonstop_pw_tokens

def sent_context_generation(text, entity_string, entity_start_ix, entity_end_ix, max_context_window=50, remove_stopwords=False, pre_post_split=False, sent_window=2, TKNZR=None):
    """
    Generate the context before and after 
    the entity string using the sentence 
    surrounding the entity. Useful for 
    ELMO which assumes sentential context!
    
    :param text: text containing entity string
    :param entity_string: entity string to extract
    :param entity_start_ix: start index to entity
    :param entity_end_ix: end index to entity
    :param max_context_window: maximum context length before/after entity
    :param remove_stopwords: remove stopwords from context
    :param pre_post_split: return pre-context and post-context separately
    :param sent_window: surrounding sentences as "context"
    :returns context: surrounding context
    """
    text = text.lower()
    entity_string = entity_string.lower()
    pre_text = text[:entity_start_ix]
    post_text = text[entity_end_ix:]
    pre_match_text = sent_tokenize(pre_text)
    post_match_text = sent_tokenize(post_text)
    relevant_pre_match_text = ""
    for elem in pre_match_text[-sent_window:]:
        relevant_pre_match_text += elem
        relevant_pre_match_text += ' '
    relevant_post_match_text = ""
    for elem in post_match_text[:sent_window]:
        relevant_post_match_text += elem
        relevant_post_match_text += ' '
    pre_context_list = process_context_word(relevant_pre_match_text, if_join=False, rs=remove_stopwords, TKNZR=TKNZR)
    post_context_list = process_context_word(relevant_post_match_text, if_join=False, rs=remove_stopwords, TKNZR=TKNZR)
    pre_context = pre_context_list[-max_context_window:]
    post_context = post_context_list[:max_context_window]
    if pre_post_split:
        return pre_context, post_context
    return (pre_context + [entity_string] + post_context, len(pre_context))

def norm_coords(data_list, coord_cols=['lat', 'lon'], norm_coord_cols=['lat_norm', 'lon_norm']):
    """
    Normalize coordinates according to the combined mean/sd:
    X_hat = (X - mu(X)) / sd(X)
    
    :param data_list: list of DataFrames
    :param coord_cols: coordinate columns
    :param norm_coord_cols: norm coordinate columns
    :returns data_list_normed: list of DataFrames with norm coordinates
    """
    data_combined = pd.concat(data_list, axis=0)
    coord_mean = data_combined.loc[:, coord_cols].mean(axis=0).values
    coord_sd = data_combined.loc[:, coord_cols].std(axis=0).values
    data_list_normed = []
    for data in data_list:
        data_norm_coords = (data.loc[:, coord_cols].values - coord_mean) / np.repeat(np.reshape(coord_sd, (1,2)), data.shape[0], axis=0)
        data_norm_coords = pd.DataFrame(data_norm_coords, columns=norm_coord_cols, index=data.index)
        # drop norm coord cols before re-adding
        data.drop(norm_coord_cols, axis=1, inplace=True)
        data = pd.concat([data, data_norm_coords], axis=1)
        data_list_normed.append(data)
    return data_list_normed

# def log_sum_exp(value, dim=None, keepdim=False):
#     """Numerically stable implementation of the operation
#     value.exp().sum(dim, keepdim).log()
#     stolen from https://github.com/jxhe/vae-lagging-encoder/blob/master/modules/utils.py
#     """
#     if dim is not None:
#         m, _ = torch.max(value, dim=dim, keepdim=True)
#         value0 = value - m
#         if keepdim is False:
#             m = m.squeeze(dim)
#         m_norm = m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
#         return m_norm
#     else:
#         m = torch.max(value)
#         sum_exp = torch.sum(torch.exp(value - m))
#         m_norm = m + torch.log(sum_exp)
#         return m_norm
    
## NE code
def extract_NEs(token_tags):
    NE_list = []
    NE_curr = []
    tag_curr = ''
    for i, token_tag in enumerate(token_tags):
        if(len(token_tag) == 2):
            token, tag = token_tag
            extra = []
        else:
            token, tag = token_tag[:2]
            extra = list(token_tag[2:])
        if(tag != 'O'):
            if(tag == tag_curr or tag_curr == ''):
                NE_curr.append(token)
        if(tag != tag_curr and tag_curr != ''):
            if(len(NE_curr) > 0):
                NE_curr = (' '.join(NE_curr), tag_curr)
                if(len(extra) > 0):
                    NE_curr = tuple(list(NE_curr) + extra)
                NE_list.append(NE_curr)
                NE_curr = []
            if(tag != 'O'):
                NE_curr.append(token)
        tag_curr = tag
    if(len(NE_curr) > 0):
        NE_curr = (' '.join(NE_curr), tag_curr)
        if(len(extra) > 0):
            NE_curr = tuple(list(NE_curr) + extra)
        NE_list.append(NE_curr)
    return NE_list

def extract_NEs_from_tags(token_tags, keep_all_tokens=False):
    """
    Extract NEs from text. If we're using BIO then 
    that requires a different set of rules than just IO.
    
    :param token_tags: list of token/tag tuples
    :param keep_all_tokens: keep all tokens, including O
    :returns NE_list: extracted NEs
    """
    use_bio = len(set([tag.split('-')[0] for token, tag in token_tags]) & set(['B','I','U','E'])) > 0
    NE_list = []
    if(use_bio):
        previous_state = "O"
        entity_start = 0
        for i, (token, tag) in enumerate(token_tags):
            state = tag.split('-')[0]
            if(state in ("B", "U") or
               (state in ("I", "E") and previous_state not in ("B", "I"))):
                entity_start = i
            if(state in ("E", "U") or
               (state in ("B", "I") and (i == len(token_tags) - 1 or token_tags[i + 1][1].split('-')[0] not in ("I", "E")))):
                entity_type = tag.split('-')[1]
                NE_combined_token = ' '.join([x[0] for x in token_tags[entity_start:i+1]])
                if entity_type is not None:
                    NE_list.append((NE_combined_token, entity_type))
                entity_start = None
            if(state == 'O' and keep_all_tokens):
                NE_list.append((token, tag))
            previous_state = state
    else:
        NE_curr = []
        tag_curr = ''
        for i, (token, tag) in enumerate(token_tags):
            if(tag != 'O'):
                if(tag == tag_curr or tag_curr == ''):
                    NE_curr.append(token)
            if(tag != tag_curr and tag_curr != ''):
                if(len(NE_curr) > 0):
                    NE_curr_str = ' '.join(NE_curr)
                    NE_curr = (NE_curr_str, tag_curr)
                    NE_list.append(NE_curr)
                    NE_curr = []
                if(tag != 'O'):
                    NE_curr.append(token)
            tag_curr = tag
            if(tag == 'O' and keep_all_tokens):
                NE_list.append((token, tag))
        if(len(NE_curr) > 0):
            NE_curr = (' '.join(NE_curr), tag_curr)
            NE_list.append(NE_curr)
    return NE_list

HANGING_UNDERSCORE_MATCHER = re.compile('^[_ ]+|[_ ]+$')
TOKEN_TAG_MATCHER = re.compile('([^ ]+)/([^ ]+)')
QUOTE_MATCHER = re.compile('^[\'"]+|[\'"]+$')
def extract_NEs_from_file(f):
    """
    Extract list of NE lists from file.
    Assume that each line is formatted with word/tag pairs.
    e.g. "ID    This/O is/O America/I-geo-loc"
    
    :param f: input tag file
    :returns NE_data:: pandas.DataFrame with NE-list and ID for each status
    """
    NE_data = []
    for l in gzip.open(f, 'r'):
        l = l.decode('utf-8').strip()
        l_id, l_txt = l.split('\t')
        l_id = int(l_id)
        l_token_tags = TOKEN_TAG_MATCHER.findall(l_txt)
        NE_l = extract_NEs_from_tags(l_token_tags)
        # replace space with underscore for easier handling later
        # and replace hanging quotes!
        NE_l = [(x[0].replace(' ', '_'), x[1]) for x in NE_l]
        # replace hanging underscores??
        NE_l = [(HANGING_UNDERSCORE_MATCHER.sub('', x[0].strip()), x[1]) for x in NE_l]
        # replace hanging quotes
        NE_l = [(QUOTE_MATCHER.sub('', x[0]), x[1]) for x in NE_l]
        NE_data.append([l_id, NE_l])
    NE_data = pd.DataFrame(NE_data, columns=['id', 'NE_list'])
    return NE_data

## parsing code
def get_subtree(dep_idx, graph):
    """
    Get indices of all nodes below specified node.
    
    :param dep_idx: int
    :param graph: directed graph
    :returns dep_subtree: indices of all subtree nodes
    """
    dep_subtree = nx.descendants(graph, dep_idx)
    return dep_subtree

def extract_all_NE_subtrees(status_message_tags, status_parse_tree, status_parse_tree_graphs, NE_types=None):
    """
    Collect contiguous location NEs from tags, 
    find matching string in tree, then collect
    biggest subtree in sequence for each NE.
    
    :param status_message_tags: all tags in status
    :param status_parse_tree: all parse trees (as list of dependencies)
    :param status_parse_tree_graphs: all parse trees (as directed graph)
    :param NE_types: NE types allowed (e.g. only LOCATION)
    :returns NE_subtrees: NE and subtrees
    """
    NE_subtrees = []
    NEs = extract_NEs(status_message_tags)
    if(len(NEs) > 0):
        if(NE_types is not None):
            NEs = [x for x in NEs if x[1] in NE_types]
        tree_idx = 0
        tree_token_idx = 0
        # add idx to parse trees
        status_parse_tree = [[y + [i,] for i,y in enumerate(x)] for x in status_parse_tree]
        for NE in NEs:
            NE_str = NE[0]
            NE_tree, NE_subtree, NE_tree_idx, tree_idx, tree_token_idx = extract_NE_subtree(NE_str, status_parse_tree, status_parse_tree_graphs, tree_idx, tree_token_idx)
            NE_subtrees.append((NE, NE_subtree))
    return NE_subtrees

def align_parses_txt(txt, parse_trees, parse_trees_dep, sent_tokenizer, tokenizer):
    """
    Align parse trees with text. 
    
    :param txt: original text
    :param parse_trees: parse trees
    :param parse_trees_dep: parse tree dependency graphs
    :returns parse_trees:: aligned parse trees
    :returns parse_trees_dep:: aligned parse tree dependency graphs
    """
    # hack: tokenize text, find sentence with max overlap
    txt_sents = [tokenizer.tokenize(x) for x in sent_tokenizer.tokenize(txt)]
    txt_sent_counts = [pd.Series(x).value_counts() for x in txt_sents]
    parse_idx_ordered = []
    for parse_tree in parse_trees:
        parse_tokens = [x[0] for x in parse_tree]
        # clean tokens
        parse_tokens = [clean_raw_txt([x])[0] for x in parse_tokens]
        parse_token_counts = pd.Series(parse_tokens).value_counts().astype(float)
        txt_diff_pcts = [(i, (x - parse_token_counts).fillna(1).sum() / (x.sum() + parse_token_counts.sum())) for i, x in enumerate(txt_sent_counts)]
        parse_idx = min(txt_diff_pcts, key=lambda x: x[0])[0]
        parse_idx = np.argmin(txt_diff_pcts)
        bad_parse_idx = []
        while(parse_idx >= len(parse_trees)):
            bad_parse_idx.append(parse_idx)
            txt_diff_pcts = [(i, (x - parse_token_counts).fillna(1).sum() / (x.sum() + parse_token_counts.sum())) for i, x in enumerate(txt_sent_counts) if i not in bad_parse_idx+parse_idx_ordered]
            parse_idx = min(txt_diff_pcts, key=lambda x:x[0])[0]
        parse_idx_ordered.append(parse_idx)
    if(max(parse_idx_ordered) >= len(parse_trees)):
        print('bad parse idx %s'%(str(parse_idx_ordered)))
    parse_trees = [parse_trees[i] for i in parse_idx_ordered]
    parse_trees_dep = [parse_trees_dep[i] for i in parse_idx_ordered]
    return parse_trees, parse_trees_dep

def build_parse(parse_result, parse_type='spacy'):
    """
    Build parse tuples from raw parse results.
    Supports spacy, stanford and google parse results.
    
    :param parse_result: generic parse result
    :param parse_type: parse type
    :returns parse_tuples:: list of (token, POS, head_index, dep_label)
    """
    parse_tuples = [] # token, POS, head_index, dependency label
    # get tokens
    if(parse_type == 'spacy'):
        parse_tokens = parse_result
    elif(parse_type == 'stanford'):
        parse_tokens = sorted(parse_result.nodes.items(), key=lambda x: x[0])[1:] # skip the first token
    elif(parse_type == 'google'):
        parse_tokens = parse_result.__getattribute__('tokens')
    # get tuple for each token
    google_pos_matcher = re.compile('(?<=tag: ).+')
    google_proper_matcher = re.compile('(?<=proper: ).+')
    google_dep_label_matcher = re.compile('(?<=label: ).+')
    for i, token in enumerate(parse_tokens):
        if(parse_type == 'spacy'):
            parse_tuple = [
                token.text, 
                token.pos_,
                token.head.i,
                token.dep_,
                i
            ]
        elif(parse_type == 'stanford'):
            idx, tok = token
            parse_tuple = [
                tok['word'],
                tok['tag'],
                tok['head'],
                tok['rel'],
                i
            ]
        elif(parse_type == 'google'):
#             print(token)
#             print(token.__getattribute__('part_of_speech'))
#             print(token.__getattribute__('dependency_edge'))
            pos_tag = google_pos_matcher.search(str(token.__getattribute__('part_of_speech'))).group(0)
            proper_tag = google_proper_matcher.search(str(token.__getattribute__('part_of_speech')))
            if(proper_tag is not None and proper_tag.group(0) == 'PROPER'):
                pos_tag = 'PROPN'
            parse_tuple = [
                token.__getattribute__('text').__getattribute__('content'), 
                pos_tag,
                token.__getattribute__('dependency_edge').__getattribute__('head_token_index'),
                google_dep_label_matcher.search(str(token.__getattribute__('dependency_edge'))).group(0).lower(),
                i
            ]
        parse_tuples.append(parse_tuple)
    return parse_tuples

def extract_dep(tree, dedup_subtrees=False, noun_types = ['NOUN', 'PROPN']):
    """
    Extract NPs and their dependent children.
    
    :param tree: list of (token, POS, head_index, dep_label)
    :param dedup_subtrees: deduplicate subtrees among children list
    :param noun_types: list of allowed NP tags
    :returns np_descendants: list of list of (token, POS, head_index, dep_label) (each list = separate set of descendants)
    """
    # identify PROPN, then identify any attached appositive/PP phrases
    tree = [x + [i] for i,x in enumerate(tree)] # add index for bookkeeping
    np_heads = list(filter(lambda x: x[1] in noun_types, tree))
    np_descendants = []
    tree_graph = nx.DiGraph()
    for i, dep in enumerate(tree):
        tree_graph.add_edge(dep[2], i, pos=dep[1], name='/'.join(map(str, dep)), dep=dep)
    for np_head_i in np_heads:
        idx = np_head_i[4]
        # collect descendents
        desc_i = nx.descendants(tree_graph, idx)
        desc_i = [np_head_i] + [tree[j] for j in desc_i]
        # sort by sentence order
        desc_i = sorted(desc_i, key=lambda x: x[4])
        np_descendants.append(desc_i)
    # get rid of subordinate (duplicate) trees
    if(dedup_subtrees):
        np_descendants_dedup = []
        np_descendants_dedup_str = []
        np_descendants_str = list(map(lambda x: list(map(lambda y: '/'.join(map(str, y)), x)), np_descendants))
        for desc_i in sorted(np_descendants, key=lambda x: len(x), reverse=True):
            desc_str = list(map(lambda x: '/'.join(map(str, x)), desc_i))
            prev_match = False
            for np_descendants_dedup_str_i in np_descendants_dedup_str:
                if(len(set(np_descendants_dedup_str_i) - set(desc_str)) < len(np_descendants_dedup_str_i)):
                    prev_match = True
                    break
            if(not prev_match):
                np_descendants_dedup_str.append(desc_str)
                np_descendants_dedup.append(desc_i)
        np_descendants = np_descendants_dedup
    return np_descendants

## let's do this the right way
## backoff = start with all contiguous PROPN near head
## if no match, try just head PROPN
def collect_NP(children):
    """
    Collect head PROPN NP from full children list.
    
    :param children: list of (token, POS, head_index, dep_label)
    :returns NP:: NP list of (token, POS, head_index, dep_label)
    """
    head_idx, head = min([[j, x] for j, x in enumerate(children)], key=lambda x: x[1][2])
    NP_complete_left = False
    NP_complete_right = False
    NP = []
    left_idx = head_idx
    right_idx = head_idx
    while(not (NP_complete_left or NP_complete_right)):
#         print('left=%d,right=%d'%(left_idx, right_idx))
        if(right_idx >= len(children)-1 or children[right_idx+1][1] != 'PROPN'):
            NP_complete_right = True
        else:
            right_idx += 1
        if(left_idx == 0 or children[left_idx-1][1] != 'PROPN'):
            NP_complete_left = True
        else:
            left_idx -= 1
    NP = children[left_idx:right_idx+1]
    return NP

def collect_match_NP(children_list, geo_data):
    """
    Collect all head NNPs from children
    and try to match each one on geo data.
    
    :param children_list: list of head NNPs/NPs with children
    :param geo_data: geo gazetteer
    :returns NP_match_list:: list of head NNPs/NPs with children, matched with geo-data (lat/lon)
    """
    NP_match_list = []
    for children_i in children_list:
        NP_i = collect_NP(children_i)
#         print('collected NP %s'%(NP_i))
        NP_str_i = ' '.join([x[0] for x in NP_i])
        NP_str_i_fixed = unidecode(NP_str_i.lower())
#         print('collected NP fixed %s'%(NP_str_i_fixed))
        # try to match!!
        geo_data_i = geo_data[geo_data.loc[:, 'alternate_names_fixed_regex'].apply(lambda x: x.search(NP_str_i_fixed) is not None)]
        if(geo_data_i.shape[0] > 0):
            geo_coord_i = list(geo_data_i.iloc[0].loc[['latitude', 'longitude']].values)
            NP_match_i = [NP_str_i, geo_coord_i, children_i]
            NP_match_list.append(NP_match_i)
    return NP_match_list

## improved parsing code that matches on location name
def extract_full_tree(parse_list):
    """
    Extract tree from parse list.
    
    :param parse_list: list of dependency data (token, POS, head_idx, dep_type)
    :returns tree_graph: directed graph
    """
    # add index for bookkeeping
    if(len(parse_list[0]) < 5):
        parse_list = [x + [i,] for i,x in enumerate(parse_list)]
    tree = nx.DiGraph()
    for i, dep in enumerate(parse_list):
        tree.add_edge(dep[2], dep[4], pos=dep[1], name='/'.join(map(str, dep)), dep=dep)
    return tree

# def extract_NEs(token_tags):
#     NE_list = []
#     NE_curr = []
#     tag_curr = ''
#     for i, (token, tag) in enumerate(token_tags):
#         if(tag != 'O'):
#             if(tag == tag_curr or tag_curr == ''):
#                 NE_curr.append(token)
#         if(tag != tag_curr and tag_curr != ''):
#             if(len(NE_curr) > 0):
#                 NE_curr = (' '.join(NE_curr), tag_curr)
#                 NE_list.append(NE_curr)
#                 NE_curr = []
#             if(tag != 'O'):
#                 NE_curr.append(token)
#         tag_curr = tag
#     if(len(NE_curr) > 0):
#         NE_curr = (' '.join(NE_curr), tag_curr)
#         NE_list.append(NE_curr)
#     return NE_list

def get_subtree(dep_idx, graph):
    """
    Get indices of all nodes below specified node.
    
    :param dep_idx: int
    :param graph: directed graph
    """
    dep_subtree = nx.descendants(graph, dep_idx)
    return dep_subtree

def extract_NE_subtree(status_NE, trees, tree_graphs, tree_ctr, tree_token_ctr, verbose=False):
    """
    Extract subtree associated with NE provided, matching
    on string and extracting the largest subtree from the NE 
    if multiple tokens.
    
    :param status_NE: NE string
    :param trees: list of tree tuples
    :param tree_graphs: list of DiGraph objects
    :param tree_ctr: counter for current tree
    :param tree_token_ctr: counter for current tree token
    :param verbose: print output
    :returns status_NE_tokens_tree::: tree tokens associated with NE
    :returns status_NE_tokens_subtree:: subtree tokens associated with NE
    :returns status_NE_tree_ctr:: counter for current status NE
    :returns tree_ctr:: counter for current tree (may have advanced past status NE)
    :returns tree_token_ctr:: counter for current tree token
    """
    status_NE_tokens = status_NE.split(' ')
#     print('processing status NE tokens %s'%(str(status_NE_tokens)))
    status_NE_tokens_tree = []
    status_NE_token_ctr = 0
    tree_ctr_0 = tree_ctr
    tree_token_ctr_0 = tree_token_ctr
    status_NE_tree_ctr = tree_ctr
#     print('extracting status NE tokens %s'%(str(status_NE_tokens)))
    while(len(status_NE_tokens_tree) < len(status_NE_tokens) and tree_ctr < len(trees)):
        curr_tree = trees[tree_ctr]
        curr_tree_graph = tree_graphs[tree_ctr]
        if(tree_token_ctr == len(curr_tree)):
            print('error with status NE %s'%(str(status_NE_tokens)))
            print('error at curr tree %s'%(str(curr_tree)))
            print('prev tree %s'%(str(trees[tree_ctr-1])))
            print('next tree %s'%(str(trees[tree_ctr+1])))
            print('original tree %s'%(trees[tree_ctr_0]))
        curr_tree_dep = curr_tree[tree_token_ctr]
        if(verbose):
            print('tree=%d,tree_token=%d,curr_dep=%s'%(tree_ctr, tree_token_ctr, str(curr_tree_dep)))
        ## if string match NE, then add dependency to NE tree
        if(curr_tree_dep[0] == status_NE_tokens[status_NE_token_ctr]):
#             print('adding curr dep %s at tree=%d, token=%d'%(str(curr_tree_dep), tree_ctr, tree_token_ctr))
            status_NE_tokens_tree.append(curr_tree_dep)
            status_NE_token_ctr += 1
            status_NE_tree_ctr = tree_ctr
        ## if string doesn't match, then reset NE tree
        elif(len(status_NE_tokens_tree) > 0):
            status_NE_tokens_tree = []
            status_NE_token_ctr = 0
        tree_token_ctr += 1
        ## if we're at the end of the tree, then advance the search
        if(tree_token_ctr == len(trees[tree_ctr])):
#             if(len(status_NE_tokens_tree) > 0):
#                 if(verbose):
#                     print('breaking early with NE tokens %s'%(str(status_NE_tokens)))
#                 break
            # if we still don't have NE tokens collected
            # then advance the tree
#             else:
#             tree_ctr = min(tree_ctr+1, len(trees)-1)
            tree_ctr += 1
            tree_token_ctr = 0
    # sanity check: reset tree_ctr to max allowed value
    tree_ctr = min(tree_ctr, len(trees)-1)
    # if we haven't found any subtrees,
    # reset tree/token counter for next round
    if(len(status_NE_tokens_tree) == 0):
        tree_ctr = tree_ctr_0
        tree_token_ctr = tree_token_ctr_0
        status_NE_tokens_subtree = []
    else:
        # now that we have all NE tokens accounted for,
        # get subtrees for each token and
        # choose minimal subtree (i.e. doesn't contain NE)
    #     curr_tree_graph = tree_graphs[tree_ctr]
    #     print('extracted status NE tokens tree %s from tree %s'%(str(status_NE_tokens_tree), str(curr_tree)))
    #     print('status NE_tokens_tree %s'%(str(status_NE_tokens_tree)))
#         try:
#         if(verbose):
#             print('extracting subtrees for NE tree=%s'%(str(status_NE_tokens_tree)))
        status_NE_tokens_dep_subtrees = [get_subtree(x[4], curr_tree_graph) for x in status_NE_tokens_tree]
#         except Exception as e:
#             print('graph error %s'%(e))
#             print('curr_tree_graph %s'%(list(curr_tree_graph.edges())))
#             print('previous tree %s'%(trees[tree_ctr-1]))
#             print('previous tree graph %s'%(list(tree_graphs[tree_ctr-1].edges())))
#             print('status_NE_tokens_tree %s'%(str(status_NE_tokens_tree)))
#         if(len(status_NE_tokens_dep_subtrees) == 0):
#             print('no subtrees for status NE tokens %s, status NE tree %s'%(str(status_NE_tokens), str(status_NE_tokens_tree)))
    #     print('got subtrees %s'%(str(status_NE_tokens_dep_subtrees)))
        status_NE_tokens_tree_idx = max(status_NE_tokens_dep_subtrees, key=lambda x: len(x))
        status_NE_tokens_subtree = [x for x in curr_tree if x[4] in status_NE_tokens_tree_idx]
        # remove NE from subtree
        status_NE_tokens_tree_idx = [x[4] for x in status_NE_tokens_tree]
    #     print('status NE idx %s'%(str(status_NE_tokens_tree_idx)))
        status_NE_tokens_subtree = [x for x in status_NE_tokens_subtree if x[4] not in status_NE_tokens_tree_idx]
        # fix tree/token counter if we've hit the end
        if(tree_token_ctr >= len(trees[tree_ctr])):
            tree_ctr += 1
            tree_token_ctr = 0
    if(verbose):
        print('extracted NE tree=%s'%(status_NE_tokens_tree))
    return status_NE_tokens_tree, status_NE_tokens_subtree, status_NE_tree_ctr, tree_ctr, tree_token_ctr

def extract_NE_supertree(status_NE, trees, tree_graphs, tree_ctr, tree_token_ctr):
    """
    Extract supertree associated with NE provided, matching
    on string and extracting the largest subtree from the NE 
    if multiple tokens.
    
    :param status_NE: NE string
    :param trees: list of tree tuples
    :param tree_graphs: list of DiGraph objects
    :param tree_ctr: counter for current tree
    :param tree_token_ctr: counter for current tree token
    :returns status_NE_tokens_subtree:: subtree tokens associated with NE
    :return tree_ctr:: counter for current tree
    :param tree_token_ctr: counter for current tree token
    """
    status_NE_tokens = status_NE.split(' ')
    status_NE_tokens_tree = []
    status_NE_token_ctr = 0
    tree_ctr_0 = tree_ctr
    tree_token_ctr_0 = tree_token_ctr
    while(len(status_NE_tokens_tree) < len(status_NE_tokens) and tree_ctr < len(trees)):
        curr_tree = trees[tree_ctr]
        curr_tree_graph = tree_graphs[tree_ctr]
        if(tree_token_ctr == len(curr_tree)):
            print('error with status NE %s'%(str(status_NE_tokens)))
            print('error at curr tree %s'%(str(curr_tree)))
            print('prev tree %s'%(str(trees[tree_ctr-1])))
            print('next tree %s'%(str(trees[tree_ctr+1])))
            print('original tree %s'%(trees[tree_ctr_0]))
        curr_tree_dep = curr_tree[tree_token_ctr]
        if(curr_tree_dep[0] == status_NE_tokens[status_NE_token_ctr]):
            status_NE_tokens_tree.append(curr_tree_dep)
            status_NE_token_ctr += 1
        elif(len(status_NE_tokens_tree) > 0):
            status_NE_tokens_tree = []
            status_NE_token_ctr = 0
        tree_token_ctr += 1
        # if we already have some NE tokens collected
        # then just break
        if(tree_token_ctr == len(trees[tree_ctr])):
            if(len(status_NE_tokens) > 0):
                break
            # if we still don't have NE tokens collected
            # then advance the tree
            else:
                tree_ctr += 1
                tree_token_ctr = 0
    # if we haven't found any subtrees,
    # reset tree/token counter
    if(len(status_NE_tokens_tree) == 0):
        tree_ctr = tree_ctr_0
        tree_token_ctr = tree_token_ctr_0
        status_NE_tokens_subtree = []
    else:
        # now that we have all NE tokens accounted for,
        # get subtrees for each token and
        # choose minimal supertree
        status_NE_tokens_dep_subtrees = [get_supertree(x[4], curr_tree_graph) for x in status_NE_tokens_tree]
        status_NE_tokens_tree_idx = max(status_NE_tokens_dep_subtrees, key=lambda x: len(x))
        status_NE_tokens_subtree = [x for x in curr_tree if x[4] in status_NE_tokens_tree_idx]
        # remove NE from subtree
        status_NE_tokens_tree_idx = [x[4] for x in status_NE_tokens_tree]
        status_NE_tokens_subtree = [x for x in status_NE_tokens_subtree if x[4] not in status_NE_tokens_tree_idx]
        if(tree_token_ctr >= len(trees[tree_ctr])):
            tree_ctr += 1
            tree_token_ctr = 0
    return status_NE_tokens_subtree, tree_ctr, tree_token_ctr

def extract_all_NE_supertrees(status_message_tags, status_parse_tree, status_parse_tree_graphs, NE_types=['LOCATION']):
    """
    Collect contiguous location NEs from tags, 
    find matching string in tree, then collect
    biggest supertree in sequence for each NE.
    
    :param status_message_tags: all tags in status
    :param status_parse_tree: all parse trees (as list of dependencies)
    :param status_parse_tree_graphs: all parse trees (as directed graph)
    :param NE_types: NE types allowed (e.g. only LOCATION)
    :returns NE_subtrees: NE and subtrees
    """
    NE_subtrees = []
    status_NEs = extract_NEs(status_message_tags)
    if(len(status_NEs) > 0):
        if(NE_types is not None):
            status_NEs = [x for x in status_NEs if x[1] in NE_types]
        tree_ctr = 0
        tree_token_ctr = 0
        # add idx to parse trees
        status_parse_tree = [[y + [i,] for i,y in enumerate(x)] for x in status_parse_tree]
        for status_NE in status_NEs:
            status_NE_str = status_NE[0]
            NE_subtree, tree_ctr, tree_token_ctr = extract_NE_supertree(status_NE_str, status_parse_tree, status_parse_tree_graphs, tree_ctr, tree_token_ctr)
            NE_subtrees.append((status_NE, NE_subtree))
    return NE_subtrees

def get_supertree(dep_idx, graph):
    """
    Get indices of all nodes above specified node.
    
    :param dep_idx: int
    :param graph: directed graph
    :returns dep_subtree: indices of all supertree nodes
    """
    dep_subtree = nx.ancestors(graph, dep_idx)
    return dep_subtree

## extracting status info, tags from Twitter file
def process_status_line(x):
    """
    Process tab-separated status line.
    
    :param x: string containing status information
    :returns x_id::
    :returns x_username::
    :returns x_date::
    :returns x_retweets::
    :returns x_favorites::
    :returns x_lang::
    """
    x_line = x.strip().split('\t')
    x_id, x_username, x_date, x_retweets, x_favorites, x_lang = x_line
    x_id = int(x_id)
    x_date = dateutil.parser.parse(x_date)
    x_retweets = int(x_retweets)
    x_favorites = int(x_favorites)
    return x_id, x_username, x_date, x_retweets, x_favorites, x_lang

def process_status_txt_tag_files(f, verbose=False):
    """
    Extract status, text and tag information from files.
    Expects existence of files in format:
    filename_status.gz, filename_txt.txt, filename_txt_tags.gz
    """
    f_name = os.path.basename(f).replace('_txt_tags.gz', '')
    f_name_fixed = f_name.split('_')[-1]
    if(verbose):
        logging.debug('processing file %s'%(f_name))
    f_status = f.replace('_txt_tags.gz', '_status.gz')
    f_txt = f.replace('_txt_tags.gz', '_txt.txt')
    NE_data = extract_NEs_from_file(f)
    f_status = pd.DataFrame([process_status_line(x) for x in gzip.open(f_status, 'rt')], columns=['id', 'username', 'date', 'retweets', 'favorites', 'lang'])
    f_txt = pd.DataFrame([x for x in open(f_txt)], columns=['txt'])
    f_combined = pd.concat([f_status, f_txt], axis=1)
    # join NE list on IDs to be more careful!!
#     f_combined.loc[:, 'NE_list'] = pd.Series(NE_list)
    f_combined = pd.merge(f_combined, NE_data, on='id')
    f_combined.loc[:, 'data_name'] = f_name
    f_combined.loc[:, 'data_name_fixed'] = f_name_fixed
#     print('combined data shape %s'%(str(f_combined.shape)))
#     display(f_combined.head())
    return f_combined

def extract_freq_NEs(data, NE_var='NE_fixed', dep_var='has_descriptor', min_count_0=10, min_count_1=10):
    """
    Extract frequent NEs from data, s.t. frequent NE
    occur at least X times with dep_var==0 and
    Y times with dep_var==1. Assumes binary dep_var.
    
    :param data: pandas.DataFrame
    :param NE_var: NE column name
    :param dep_var: dependent variable to determine "frequent"
    :param min_count_0: min count of dep_var==0
    :param min_count_1: min count of dep_var==1
    :returns freq_NEs:: frequent NEs
    """
    data_dep_0 = data[data.loc[:, dep_var]==0]
    data_dep_1 = data[data.loc[:, dep_var]==1]
    NE_count_0 = data_dep_0.loc[:, NE_var].value_counts()
    NE_count_1 = data_dep_1.loc[:, NE_var].value_counts()
    freq_NEs = NE_count_0[NE_count_0 >= min_count_0].index & NE_count_1[NE_count_1 >= min_count_1].index
    freq_NEs = list(freq_NEs)
#     print('%d freq NEs'%(len(freq_NEs)))
    return freq_NEs

## anchor detection
STATES_SHORT_FULL_LOOKUP = {
        'FL' : 'Florida', 'NC' : 'North Carolina', 'SC' : 'South Carolina', 
        'VA' : 'Virginia', 'GA' : 'Georgia', 'PR' : 'Puerto Rico',
        'LA' : 'Louisiana', 'TX' : 'Texas',
    }
DATA_NAME_STATES_SHORT_LOOKUP = {
    'florence' : ['FL', 'NC', 'SC', 'VA', 'GA'],
    'irma' : ['FL', 'GA', 'SC'],
    'harvey' : ['TX', 'LA'],
    'maria' : ['PR', 'NC'],
    'michael' : ['FL', 'GA', 'NC', 'SC']
}
DATA_NAME_STATES_LONG_LOOKUP = {k : [STATES_SHORT_FULL_LOOKUP[v] for v in vs] for k, vs in DATA_NAME_STATES_SHORT_LOOKUP.items()}
DATA_NAME_STATES_LOOKUP = {k : DATA_NAME_STATES_SHORT_LOOKUP[k]+DATA_NAME_STATES_LONG_LOOKUP[k] for k in DATA_NAME_STATES_SHORT_LOOKUP.keys()}
# need to add Maria FB data
# collect municipalities (ADM1) from GeoNames
DATA_NAME_STATES_LOOKUP['maria_fb'] = ["Adjuntas", "Aguada", "Aguadilla", "Aguas Buenas", "Aibonito", "Añasco", "Arecibo", "Arroyo", "Barceloneta", "Barranquitas", "Bayamón", "Cabo Rojo", "Caguas", "Camuy", "Canóvanas", "Carolina", "Cataño", "Cayey", "Ceiba", "Ciales", "Cidra", "Coamo", "Comerío", "Corozal", "Culebra", "Dorado", "Fajardo", "Florida", "Guánica", "Guayama", "Guayanilla", "Guaynabo", "Gurabo", "Hatillo", "Hormigueros", "Humacao", "Isabela", "Jayuya", "Juana Díaz", "Juncos", "Lajas", "Lares", "Las Marías", "Las Piedras", "Loíza", "Luquillo", "Manatí", "Maricao", "Maunabo", "Mayagüez", "Moca", "Morovis", "Naguabo", "Naranjito", "Orocovis", "Patillas", "Peñuelas", "Ponce", "Rincón", "Quebradillas", "Río Grande", "Sabana Grande", "Salinas", "San Germán", "San Juan", "San Lorenzo", "San Sebastián", "Santa Isabel", "Toa Alta", "Toa Baja", "Trujillo Alto", "Utuado", "Vega Alta", "Vega Baja", "Villalba", "Yabucoa", "Yauco", "Vieques", "PR", "Puerto Rico"]
DATA_NAME_STATES_MATCHERS = {k : re.compile('|'.join([' %s |^%s | %s$|^%s$'%((unidecode(v1.lower()),)*4) for v1 in v])) for k,v in DATA_NAME_STATES_LOOKUP.items()}

def detect_anchor_by_type(data, valid_var='valid_loc', 
                          anchor_var='max_population', 
                          NE_var='NE', 
                          parse_var='parse',
                          data_name_var='data_name_fixed',
                          child_dep_types=['acl', 'appos', 'prep', 'nummod'],
                          parent_dep_types=['nmod', 'compound', 'appos'],
                          parent_state_dep='nmod',
                          parent_compound_dep='compound',
                          txt_var='txt',
                          verbose=False):
    """
    Detect NE anchor with parse data.
    
    :param data: pandas.DataFrame with all NEs, NE data (ex. importance), parse data
    :param valid_var: whether NE is valid
    :param anchor_var: anchor variable (ex. max_population)
    :param NE_var: NE variable
    :param data_name_var: data name variable
    :param child_dep_types: allowed child dependency types
    :param parent_dep_types: allowed parent dependency types
    :param parent_state_dep: allowed parent dependency type for states
    :param parent_compound_dep: allowed parent dependency type for compounds
    :param verbose: print output
    :returns anchor_state:: data anchor labels with state subclause
    :returns anchor_descriptor:: data anchor labels with descriptor subclause
    """
    ## TODO: generalize into two-part process
    ## 1. string match LOC, ENCLOSING_LOC (ENCLOSING_LOC = {STATE, MUNICIPALITY})
    ## 2. parse match LOC + DESCRIPTOR (ANCHOR_LOC)
    data_NEs = data.loc[:, NE_var].values
    # fix format to match parse
    data_NEs = [x.replace('_', ' ') for x in data_NEs]
    data_valid = data.loc[:, valid_var].values
    data_NEs_valid = [x for x,y in zip(data_NEs, data_valid) if y==1]
    data_anchor_vals = data.loc[:, anchor_var].values
    data_txt = data.loc[:, txt_var].iloc[0]
    data_txt_clean = data_txt.lower()
    anchor_descriptor = []
    anchor_state = []
    anchor_compound = []
    anchor_list = []
    trees = data.loc[:, parse_var].iloc[0]
    if(verbose):
        print('full parse %s'%(str(trees)))
    tree_graphs = [extract_full_tree(t) for t in trees]
    tree_ctr = 0
    tree_token_ctr = 0
    data_name = data.loc[:, data_name_var].iloc[0]
    states = DATA_NAME_STATES_LOOKUP[data_name]
    state_matcher = DATA_NAME_STATES_MATCHERS[data_name]
    anchor_phrases = []
    for i, data_NE in enumerate(data_NEs_valid):
        if(verbose):
            print('candidate %d=%s'%(i, data_NE))
        anchor_val_i = data_anchor_vals[i]
        anchor_NE_candidates = [x for x,y in zip(data_NEs, data_anchor_vals) if y > anchor_val_i]
        if(verbose):
            print('anchor NE candidates = %s'%(','.join(anchor_NE_candidates)))
        anchor_state_i = 0
        anchor_descriptor_i = 0
        anchor_compound_i = 0
        anchor_list_i = 0
        anchor_phrase_i = ''
        ## easiest case: LOC, STATE pattern
        data_NE_state_matcher = re.compile('|'.join(['%s\s?,?\s?%s[^a-zA-Z]'%(data_NE, state,) for state in states]).lower())
        anchor_state_i = int(data_NE_state_matcher.search(data_txt_clean) is not None)
        if(anchor_state_i==1):
            anchor_phrase_i = data_NE_state_matcher.search(data_txt_clean).group(0)
        if(len(anchor_NE_candidates) > 0):
            # add whitespace to candidates to avoid matching substrings e.g. "FL" doesn't match "FLOW"
            anchor_NE_candidate_matcher = re.compile('|'.join([' %s |^%s | %s$|^%s$'%((x,)*4) for x in anchor_NE_candidates]))
            data_NE_tree, data_NE_subtree, NE_tree_ctr, tree_ctr, tree_token_ctr = extract_NE_subtree(data_NE, trees, tree_graphs, tree_ctr, tree_token_ctr)
            ## parent test: test if sibling contains state
            if(len(data_NE_tree) > 0):
                if(verbose):
                    print('data NE tree=%s'%(str(data_NE_tree)))
                    print('tree_ctr=%d/%d, tree_token_ctr=%d'%(NE_tree_ctr, len(trees), tree_token_ctr))
                # find parent node for noun, preposition tests
                # parent node => index that is not included in NE indices
                data_NE_tree_idx = [x[4] for x in data_NE_tree]
                NE_child_nodes = [x for x in data_NE_tree if x[2] not in data_NE_tree_idx]
                if(verbose):
                    print('child nodes %s'%(str(NE_child_nodes)))
                if(len(NE_child_nodes) > 0):
                    highest_child_node = NE_child_nodes[0]
                    parent_node_idx = highest_child_node[2]
                    parent_node_dep = highest_child_node[3]
                    parent_node = trees[NE_tree_ctr][parent_node_idx]
                    if(verbose):
                        print('NE parse token at tree=%d, token=%d:'%(NE_tree_ctr, tree_token_ctr))
                        print(str(trees[NE_tree_ctr][tree_token_ctr-1]))
                        print('NE parent token:')
                        print(str(parent_node))
                    # if parent is connected by valid dep, find children and look for state mention
                    if(parent_node_dep in parent_dep_types):
                        parent_node_subtree_idx = get_subtree(parent_node[4], tree_graphs[NE_tree_ctr])
                        # include parent in subtree
                        parent_node_subtree = [x for x in trees[NE_tree_ctr] if x[4] in parent_node_subtree_idx or x[4]==parent_node_idx]
                        parent_node_subtree_str = ' '.join([x[0] for x in parent_node_subtree]).lower()
                        if(verbose):
                            print('parent node subtree %s'%(str(parent_node_subtree)))
                            print('parent node subtree str "%s"'%(parent_node_subtree_str))
#                         if(verbose):
#                             print('matching states %s on tree %s'%(state_matcher.pattern, parent_node_subtree_str.lower()))
                        ## collapse compound/nmod because they're arbitrary
#                         if(parent_node_dep in parent_dep_types):
#                         anchor_state_i = int(state_matcher.search(parent_node_subtree_str) is not None) 
                        if(parent_node_dep == parent_compound_dep):
                            if(verbose):
                                print('testing to find anchor in parent subtree => compound')
#                                 print('pattern = %s'%(anchor_NE_candidate_matcher.pattern))
#                             anchor_compound_i = int(state_matcher.search(parent_node_subtree_str) is not None) 
                                anchor_compound_i = int(anchor_NE_candidate_matcher.search(parent_node_subtree_str) is not None)
                                if(anchor_compound_i == 1):
                                    anchor_phrase_i = parent_node_subtree_str
#                         parent_anchor_anchor_i = int(state_matcher.search(parent_node_subtree_str) is not None)
            
            ## child test: test if child contains anchor
            if(len(data_NE_subtree) > 0):
                if(verbose):
                    print('NE=%s subtree=%s'%(data_NE, str(data_NE_subtree)))
                ## filter for allowed dependency trees
                ## find dep type for highest node (lowest parent index) in subtree
                min_node_dep_idx = min(data_NE_subtree, key=lambda x: x[2])
                min_node_deps = [x[3] for x in data_NE_subtree if x[2]==min_node_dep_idx[2]]
                if(verbose):
                    print('min node deps %s'%(str(min_node_deps)))
                subtree_str = ' '.join([x[0] for x in data_NE_subtree])
                if(len(set(min_node_deps) & set(child_dep_types)) > 0):
                    ## look for NE in phrase
                    if(verbose):
                        print('subtree = %s'%(subtree_str))
                    anchor_descriptor_i = int(anchor_NE_candidate_matcher.search(subtree_str) is not None)
                    if(anchor_descriptor_i == 1):
                        anchor_phrase_i = anchor_NE_candidate_matcher.search(subtree_str).group(0)
                    # sometimes the state comes as an appositive
#                     anchor_state_i = int(state_matcher.search(subtree_str.lower()) is not None)
                    ## conjuction is a special case: anchor NE must occur 
                    ## in format NE CONJ NE, state
                elif('conj' in min_node_deps):
                    anchor_list_i = int(state_matcher.search(subtree_str.lower()) is not None)
                    if(anchor_list_i == 1):
                        anchor_phrase_i = state_matcher.search(subtree_str.lower())
        anchor_state.append(anchor_state_i)
        anchor_descriptor.append(anchor_descriptor_i)
        anchor_compound.append(anchor_compound_i)
        anchor_list.append(anchor_list_i)
        anchor_phrases.append(anchor_phrase_i)
    anchor_state = pd.Series(anchor_state, index=data_NEs_valid)
    anchor_descriptor = pd.Series(anchor_descriptor, index=data_NEs_valid)
    anchor_compound = pd.Series(anchor_compound, index=data_NEs_valid)
    anchor_list = pd.Series(anchor_list, index=data_NEs_valid)
    anchor_phrases = pd.Series(anchor_phrases, index=data_NEs_valid)
    return anchor_state, anchor_descriptor, anchor_compound, anchor_list, anchor_phrases
#     return anchor_state, anchor_descriptor

def fix_timezone(x, date_fmt='%Y-%m-%d %H:%M:%S', timezone_str='+0000'):
    # add timezone offset for "naive" dates
    if(x.utcoffset() is None):
        x = datetime.strptime('%s%s'%(x.strftime(date_fmt), timezone_str), '%s%%z'%(date_fmt))
    return x

def assign_relative_peak_time_vars(data, peak_date_buffer):
    """
    Assign relative peak time variables:
    pre-peak, post-peak, during-peak, since-peak, since-start.
    
    :param data: per-NE data
    :param peak_date_buffer: days before/after peak time to consider pre/post peak
    :returns data:: per-NE data with relative peak time
    """
    peak_date_buffer = timedelta(days=peak_date_buffer)
    peak_date_var = 'peak_date'
    round_date_var = 'date_day'
    data_name_var = 'data_name_fixed'
    data = data.assign(**{
        'pre_peak' : (data.loc[:, round_date_var] <= data.loc[:, peak_date_var] - peak_date_buffer).astype(int),
        'post_peak' : (data.loc[:, round_date_var] >= data.loc[:, peak_date_var] + peak_date_buffer).astype(int),
        'during_peak' : ((data.loc[:, round_date_var] >= data.loc[:, peak_date_var] - peak_date_buffer) & (data.loc[:, round_date_var] <= data.loc[:, peak_date_var] + peak_date_buffer)).astype(int)
    })
    # add time since peak; time since start
    since_start_var = 'since_start'
    data_start_var = 'data_start'
    data_date_mins = data.groupby(data_name_var).apply(lambda x: x.loc[:, round_date_var].min()).reset_index().rename(columns={0 : data_start_var})
    logging.debug('data date mins =\n%s'%(data_date_mins))
    data = pd.merge(data_date_mins, data, on=data_name_var)
    data = data.assign(**{
        'since_peak' : data.loc[:, 'post_peak'] * (data.loc[:, round_date_var] - data.loc[:, peak_date_var]).apply(lambda x: x.days),
        since_start_var : (data.loc[:, round_date_var] - data.loc[:, data_start_var]).apply(lambda x: x.days)
    })
    return data

# def round_to_day(x):
#     x_day = datetime(day=x.day, month=x.month, year=x.year)
#     return x_day

def round_to_day(date, round_day=1):
    day_round = (floor(date.timetuple().tm_yday / round_day)*round_day)
    date_round = datetime.strptime('%d-%d'%(day_round, date.year), '%j-%Y')
    return date_round

def assign_peak_date(data, count_var='NE_count', date_var='date_day'):
    """
    Assign peak date for given NE based on 
    date of maximum raw frequency.
    """
    max_count = data.loc[:, count_var].max()
    peak_date = data[data.loc[:, count_var] == max_count].loc[:, date_var].iloc[0]
    return peak_date

def compute_post_length(data, bins=11):
    """
    Compute length of post without context phrase:
    compute phrase length, log-transform,
    normalize [0,1] for each dataset,
    bin.
    
    :param data:: data with context phrases marked
    :param bins: number of bins to split text length
    :return data_norm:: data with binned normalized length 
    """
    txt_var = 'txt'
    clean_txt_var = 'txt_clean'
    anchor_txt_var = 'anchor_phrase'
    data = data.assign(**{anchor_txt_var : data.loc[:, anchor_txt_var].fillna('', inplace=False)})
    data = data.assign(**{clean_txt_var : data.apply(lambda x: x.loc[txt_var].lower().replace(x.loc[anchor_txt_var], ''), axis=1)})
    len_var = '%s_len'%(txt_var)
    txt_len_smooth = 1
    data = data.assign(**{len_var : np.log(data.loc[:, clean_txt_var].apply(lambda x: len(x)+txt_len_smooth))})
    max_char_len = 280
    data = data[data.loc[:, len_var] <= np.log(max_char_len)]
    ## normalize per-dataset
    data_norm = []
    data_name_var = 'data_name_fixed'
    len_var = 'txt_len'
    len_smooth = 1e-3
    for name_i, data_i in data.groupby(data_name_var):
        max_len = np.exp(data_i.loc[:, len_var].max())
        min_len = np.exp(data_i.loc[:, len_var].min())
        len_range = max_len - min_len
        data_i = data_i.assign(**{
            '%s_norm'%(len_var) : (np.exp(data_i.loc[:, len_var]) - min_len) / len_range
        })
        data_norm.append(data_i)
    data_norm = pd.concat(data_norm, axis=0)
    ## bin 
    norm_var = 'txt_len_norm'
    norm_bin_var = '%s_bin'%(norm_var)
    norm_len_bins = np.linspace(data_norm.loc[:, norm_var].min(), data_norm.loc[:, norm_var].max(), num=bins)
    data_norm = data_norm.assign(**{
        norm_bin_var : np.digitize(data_norm.loc[:, norm_var], norm_len_bins)
    })
    ## combine N-1 and N bins
    len_bin_max = data_norm.loc[:, norm_bin_var].max()
    data_norm = data_norm.assign(**{
        norm_bin_var : data_norm.loc[:, norm_bin_var].apply(lambda x: min(x, len_bin_max-1))
    })
    return data_norm

def shift_dates(data, date_var='date_day', date_shift=1, verbose=False):
    unique_dates = pd.DataFrame(pd.Series(data.loc[:, date_var].unique()).sort_values(inplace=False, ascending=True)).rename(columns={0 : date_var})
    shift_date_var = '%s_shift'%(date_var)
    # rewrite list of dates: 0, 0, ... T-date_shift
    shifted_dates = unique_dates.loc[:, date_var].iloc[:date_shift].values.tolist() + unique_dates.loc[:, date_var].iloc[:-date_shift].values.tolist()
    if(verbose):
        logging.debug('shifted dates\n%s'%(str(shifted_dates)))
    shifted_dates = np.array([pd.Timestamp(x) for x in shifted_dates])
    unique_dates = unique_dates.assign(**{shift_date_var : shifted_dates})
    data = pd.merge(data, unique_dates, on=date_var)
    return data

def add_post_length(data):
    """
    Compute length of post without context phrase:
    compute phrase length, log-transform,
    normalize [0,1] for each dataset,
    bin into N=10.
    
    :param data:: data with context phrases marked
    :return data_norm:: data with binned normalized length 
    """
    txt_var = 'txt'
    clean_txt_var = 'txt_clean'
    anchor_txt_var = 'anchor_phrase'
    data = data.assign(**{anchor_txt_var : data.loc[:, anchor_txt_var].fillna('', inplace=False)})
    # remove anchor text from full text
    data = data.assign(**{clean_txt_var : data.apply(lambda x: x.loc[txt_var].lower().replace(x.loc[anchor_txt_var], ''), axis=1)})
    len_var = '%s_len'%(txt_var)
    data = data.assign(**{len_var : np.log(data.loc[:, clean_txt_var].apply(lambda x: len(x)))})
    max_char_len = 280
    data = data[data.loc[:, len_var] <= np.log(max_char_len)]
    ## normalize per-dataset
    data_norm = []
    data_name_var = 'data_name_fixed'
    len_var = 'txt_len'
    len_smooth = 1e-3
    for name_i, data_i in data.groupby(data_name_var):
        max_len = np.exp(data_i.loc[:, len_var].max())
        min_len = np.exp(data_i.loc[:, len_var].min())
        len_range = max_len - min_len
        data_i = data_i.assign(**{
            '%s_norm'%(len_var) : (np.exp(data_i.loc[:, len_var]) - min_len) / len_range
        })
        data_norm.append(data_i)
    data_norm = pd.concat(data_norm, axis=0)
    ## bin 
    N_bins = 11
    norm_var = 'txt_len_norm'
    norm_bin_var = '%s_bin'%(norm_var)
    norm_len_bins = np.linspace(data_norm.loc[:, norm_var].min(), data_norm.loc[:, norm_var].max(), num=N_bins)
    data_norm = data_norm.assign(**{
        norm_bin_var : np.digitize(data_norm.loc[:, norm_var], norm_len_bins)
    })
    ## combine N-1 and N bins
    len_bin_max = data_norm.loc[:, norm_bin_var].max()
    data_norm = data_norm.assign(**{
        norm_bin_var : data_norm.loc[:, norm_bin_var].apply(lambda x: min(x, len_bin_max-1))
    })
    return data_norm

def compute_shift_counts(data, date_var='date_day', prior_shift=1, count_var='post_count_prior', aggregate_var=None, null_val=0.):
    """
    Compute and shift date counts (ex. number of posts at t-1).
    
    :param data: post DataFrame
    :param date_var: date variable
    :param prior_shift: N days to shift
    :param count_var: count variable
    :param aggregate_var: optional variable to aggregate at each date (ex. mean likes at t-1)
    :return data: shifted data
    """
    data.sort_values(date_var, ascending=True, inplace=True)
    if(aggregate_var is None):
        date_counts = data.loc[:, date_var].value_counts().sort_index()
    else:
        date_counts = data.groupby(date_var).apply(lambda x: x.loc[:, aggregate_var].mean())
    date_range = (data.loc[:, date_var].max() - data.loc[:, date_var].min()).days
    date_range = [data.loc[:, date_var].min() + timedelta(days=x) for x in range(date_range)]
    date_counts = date_counts.loc[date_range].fillna(0, inplace=False)
    # shift dates forward
    date_counts_idx = date_counts.index[prior_shift:]
    date_counts_shift = date_counts.iloc[:-prior_shift]
    date_counts_shift.index = date_counts_idx
    # add values for null dates (the first N dates)
    date_counts_shift = date_counts_shift.append(pd.Series([null_val]*prior_shift, index=date_counts.index[:prior_shift]))
    date_counts_shift.sort_index(inplace=True)
    # reorganize for merge
    date_counts_shift = pd.DataFrame(date_counts_shift).reset_index().rename(columns={'index' : date_var, 0 : count_var})
    # add back to original data
    data = pd.merge(data, date_counts_shift, on=date_var)
    return data

def get_text_with_no_context(data, id_var='status_id', txt_var='status_message', subtree_var='subtree', tree_var='tree', context_var='anchor'):
    """
    Replace context from text and return clean text.
    
    :param data: data
    :param id_var: status ID
    :param txt_var: text
    :param subtree_var: subtree (connected to location)
    :param tree_var: super-tree (connected to location)
    :param context_var: whether location has context
    :returns data_no_context_txt:: data with text minus context
    """
    no_context_txt_var = '%s_no_context'%(txt_var)
    data_no_context_txt = []
    for id_i, data_i in data.groupby(id_var):
        txt_i = data_i.loc[:, txt_var].iloc[0]
        # replace context in txt_i
        txt_i_clean = txt_i
        for idx_j, NE_data_j in data_i.iterrows():
            if(NE_data_j.loc[context_var]==1):
    #             print('clean txt before: %s'%(txt_i_clean))
                tree_txt_j = ' '.join([token[0] for token in NE_data_j.loc[tree_var]])
                subtree_txt_j = ' '.join([token[0] for token in NE_data_j.loc[subtree_var]])
                txt_i_clean = txt_i_clean.replace(tree_txt_j, '')
                txt_i_clean = txt_i_clean.replace(subtree_txt_j, '')
        data_i = data_i.assign(**{
            no_context_txt_var : data_i.apply(lambda x: txt_i_clean if x.loc[context_var]==1 else txt_i, axis=1)
        })
        data_no_context_txt.append(data_i)
    data_no_context_txt = pd.concat(data_no_context_txt, axis=0)
    return data_no_context_txt

class BasicTokenizer:
    """
    Wrapper class for nltk.tokenize.word_tokenize.
    """
    def __init__(self, lang):
        self.lang = lang
        
    def tokenize(self, txt):
        return word_tokenize(txt, self.lang)
    
class NLTKTokenizerSpacy(object):
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __call__(self, text):
        words = self.tokenizer.tokenize(text)
        # All tokens 'own' a subsequent space character in this tokenizer
#         spaces = [True] * len(words)
        return Doc(self.vocab, words=words)