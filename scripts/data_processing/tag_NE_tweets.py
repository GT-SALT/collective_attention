"""
Tag NEs in a collection of tweets.
This requires filtering for language first.

1. Filter for language (EN/ES)
2. Write tweets to 2 files: status ID etc. AND raw text
3. Tag raw text file
4. Recombine status ID etc. tags
"""
from zipfile import ZipFile
import gzip
from langid import langid
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json
from dateutil.parser import parse
import re
import sys
## "noisy" English NER tagger
if('TwitterNER/NoisyNLP/' not in sys.path):
    sys.path.append('TwitterNER/NoisyNLP/')
## Ritter English Twitter tagger
if('twitter_nlp/' not in sys.path):
    sys.path.append('twitter_nlp/')
    sys.path.append('twitter_nlp/python/')
    sys.path.append('twitter_nlp/python/pos_tag/')
    sys.path.append('twitter_nlp/python/cap/')
    sys.path.append('twitter_nlp/hbc/python/')
# from ner.extractEntitiesDummy import GetNer, GetLLda
# from ner import Features
# from LdaFeatures import LdaFeatures
# import cap_classifier
# import pos_tagger_stdin
# import chunk_tagger_stdin
# from Dictionaries import Dictionaries
# from Vocab import Vocab
## NoisyNLP English tagger
from run_ner import TwitterNER
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
# Stanford Spanish NER tagger (TODO: find a Twitter-trained NER tagger)
from nltk.parse.corenlp import CoreNLPParser
import os
from ast import literal_eval
import logging

def load_json(x):
    """
    Try to load with literal_eval first
    then json.loads if that fails. UGH
    """
    try:
        x_json = literal_eval(x)
    except Exception as e:    
        x_json = json.loads(x)
    return x_json

X_KEYS = ['id', 'username', 'date', 'retweets', 'favorites', 'text']
QUOTE_MATCHER=re.compile('^"|"$')
def process_line(x, input_file_json=False, verbose=False):
    """
    Convert .tsv/.json line to pandas.Series.
    
    ID | user | date | retweets | favorites | text
    
    :param x: string data
    :param input_file_json: whether input is json or not
    :returns x_data:: pandas.Series containing line data
    """
    x = x.decode('utf-8').strip()
    if(input_file_json):
        x_data = load_json(x)
#         x_data = json.loads(x)
#         x_data = literal_eval(x)
        x_data['username'] = x_data['user']['screen_name']
        x_data['date'] = x_data['created_at']
        x_data['retweets'] = x_data['retweet_count']
        x_data['favorites'] = x_data['favorite_count']
        # fix text if retweet
        x_data = pd.Series({x_key:x_data.get(x_key) for x_key in X_KEYS})
    else:
        x_split = x.split('\t')
        if(verbose):
            logging.debug('split data %s'%(str(x_split)))
        if(len(x_split) == 10):
            x_user, x_date, x_retweets, x_favorites, x_text, x_geo, x_mentions, x_hashtags, x_id, x_permalink = x_split
        elif(len(x_split) == 6):
            x_lat, x_lon, x_id, x_user, x_text, x_date = x_split
            x_retweets = -1
            x_favorites = -1
        x_date = parse(x_date)
        x_retweets = int(x_retweets)
        x_favorites = int(x_favorites)
        x_id = int(x_id.replace('"', ''))
        x_text = QUOTE_MATCHER.sub('', x_text)
        x_data = pd.Series({'username':x_user, 'date':x_date, 'retweets':x_retweets, 'favorites':x_favorites, 'id':x_id, 'text':x_text})
        x_data.fillna('', inplace=True)
    return x_data

class StanfordTaggerWrapper:
    """
    Wrapper class for Stanford NER tagger.
    """
    def __init__(self, parser):
        self.parser = parser
        
    def tag(self, tokens):
        token_tags = self.parser.tag(tokens)
        return token_tags

class NoisyTaggerWrapper:
    """
    Wrapper class for NoisyNLP NER tagger.
    Borrowing from here: https://github.com/napsternxg/TwitterNER/blob/master/NoisyNLP/run_ner.py
    """
    def __init__(self, tagger):
        self.tagger = tagger
        
    def tag(self, tokens):
        tokens_features = self.tagger.get_features(tokens)
        tags = self.tagger.model.predict([tokens_features])[0]
        token_tags = list(zip(tokens, tags))
        return token_tags

class TwitterNLPTaggerWrapper:
    """
    Wrapper class for TwitterNLP tagger.
    Borrowing from here: https://github.com/aritter/twitter_nlp/blob/master/python/ner/extractEntities.py
    
    TODO: make this work with Python 3!! would be a huge boon for Github
    """
    def __init__(self, twitter_nlp_dir='twitter_nlp/'):
        ## set up vocab, NER, etc.
        self.llda = GetLLda()
        ner_model_type = 'ner.model'
        mallet_memory = '256m'
        self.ner = GetNer(ner_model_type, memory=mallet_memory)
        self.feature_extractor = Features.FeatureExtractor('%s/data/dictionaries' % (twitter_nlp_dir))
        self.capClassifier = cap_classifier.CapClassifier()
        self.vocab = Vocab('%s/hbc/data/vocab' % (twitter_nlp_dir))
        self.dictMap = {}
        i = 1
        for line in open('%s/hbc/data/dictionaries' % (twitter_nlp_dir)):
            dictionary = line.rstrip('\n')
            self.dictMap[i] = dictionary
            i += 1

        self.dict2index = {v : k for k,v in self.dictMap.items()}

        self.dictionaries = Dictionaries('%s/data/LabeledLDA_dictionaries3' % (twitter_nlp_dir), self.dict2index)
        self.entityMap = {}
        i = 0
        for line in open('%s/hbc/data/entities' % (twitter_nlp_dir)):
            entity = line.rstrip('\n')
            self.entityMap[entity] = i
            i += 1
        self.dict2label = {}
        for line in open('%s/hbc/data/dict-label3' % (twitter_nlp_dir)):
            (dictionary, label) = line.rstrip('\n').split(' ')
            self.dict2label[dictionary] = label
            
        ## POS tagger
        self.posTagger = pos_tagger_stdin.PosTagger()
        
        ## chunk tagger
        self.chunkTagger = chunk_tagger_stdin.ChunkTagger()
    
    def tag(self, tokens):
        """
        Copied from extractEntities.py
        """
        tags = []
        seq_features = []
        
        ## TODO: do we actually need to classify? what is goodCap?
#         goodCap = self.capClassifier.Classify(tokens) > 0.9
        goodCap = True
        pos = self.posTagger.TagSentence(tokens)
        pos = [re.sub(r':[^:]*$', '', p) for p in pos]  # remove weights

        # Chunking the tweet
        word_pos = zip(tokens, [p.split(':')[0] for p in pos])
        chunk = self.chunkTagger.TagSentence(word_pos)
        chunk = [c.split(':')[0] for c in chunk]  # remove weights  

        quotes = Features.GetQuotes(tokens)
        for i in range(len(tokens)):
            features = self.feature_extractor.Extract(tokens, pos, chunk, i, goodCap) + ['DOMAIN=Twitter']
            if quotes[i]:
                features.append("QUOTED")
            seq_features.append(" ".join(features))
        self.ner.stdin.write(("\t".join(seq_features) + "\n").encode('utf8'))
            
        for i in range(len(tokens)):
            tags.append(self.ner.stdout.readline().rstrip('\n').strip(' '))
        
        features = LdaFeatures(tokens, tags)

        #Extract and classify entities
        for i in range(len(features.entities)):
            type = None
            wids = [str(self.vocab.GetID(x.lower())) for x in features.features[i] if self.vocab.HasWord(x.lower())]
            if llda and len(wids) > 0:
                entityid = "-1"
                if self.entityMap.has_key(features.entityStrings[i].lower()):
                    entityid = str(self.entityMap[features.entityStrings[i].lower()])
                labels = self.dictionaries.GetDictVector(features.entityStrings[i])

                if sum(labels) == 0:
                    labels = [1 for x in labels]
                self.llda.stdin.write("\t".join([entityid, " ".join(wids), " ".join([str(x) for x in labels])]) + "\n")
                sample = self.llda.stdout.readline().rstrip('\n')
                labels = [self.dict2label[self.dictMap[int(x)]] for x in sample[4:len(sample)-8].split(' ')]

                count = {}
                for label in labels:
                    count[label] = count.get(label, 0.0) + 1.0
                maxL = None
                maxP = 0.0
                for label in count.keys():
                    p = count[label] / float(len(count))
                    if p > maxP or maxL == None:
                        maxL = label
                        maxP = p

                if maxL != 'None':
                    tags[features.entities[i][0]] = "B-%s" % (maxL)
                    for j in range(features.entities[i][0]+1,features.entities[i][1]):
                        tags[j] = "I-%s" % (maxL)
                else:
                    tags[features.entities[i][0]] = "O"
                    for j in range(features.entities[i][0]+1,features.entities[i][1]):
                        tags[j] = "O"
            else:
                tags[features.entities[i][0]] = "B-ENTITY"
                for j in range(features.entities[i][0]+1,features.entities[i][1]):
                    tags[j] = "I-ENTITY"
        token_tags = list(zip(tokens, tags))
        return token_tags
        
def load_spanish_tagger(port='8893'):
    """
    Load Spanish NER tagger.
    Default to Stanford tagger because we can't
    find Twitter pre-trained tagger. FML
    Assumes that server is already running as follows:
    cd /hg190/corpora/StanfordCoreNLP/stanford-corenlp-full-2018-02-27/tmp/stanford-corenlp-full-2018-02-27/
    java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-spanish.properties -preload tokenize,ssplit,pos,ner,parse -status_port 9003  -port 9003 -timeout 15000
    
    :returns tagger:: Spanish NER tagger
    """
    parser = CoreNLPParser('http://localhost:9003', tagtype='ner')
    tagger = StanfordTaggerWrapper(parser)
    return tagger

def load_english_tagger():
    """
    Load English Twitter NER tagger.
    downloaded/stolen from https://github.com/napsternxg/TwitterNER
    
    :returns tagger:: English Twitter tagger
    """
    tagger_full = TwitterNER()
    tagger = NoisyTaggerWrapper(tagger_full)
    return tagger

# remove URLs, include possible spaces ;_;
URL_MATCHER = re.compile('https?://.+|twitter.com/.+|pic\.twitter\.com/.+')
RETURN_MATCHER = re.compile('\n|\r')
def clean_txt(txt):
    txt = RETURN_MATCHER.sub(' ', txt)
    txt = URL_MATCHER.sub('<URL>', txt)
    return txt

def write_status_txt(tweet_file, out_file_status, out_file_txt, langs=['en']):
    """
    Write status info and text to file.
    
    :param tweet_file: tweet input file
    :param out_file_status: status output file
    :param out_file_txt: text output file
    """
    status_idx = [x for x in X_KEYS if x != 'text']
    with gzip.open(out_file_status, 'wb') as out_file_status_output, open(out_file_txt, 'wb') as out_file_txt_output:
        with gzip.open(tweet_file, 'r') as tweet_file_input:
#             try:
            for i, x in enumerate(tweet_file_input):
#             for archive_file in archive_dir.filelist:
#                 for i, x in enumerate(archive_dir.open(archive_file)):
                # check for JSON
                if(i == 0):
                    x_str = x.decode('utf-8').strip()
                    try:
                        x_test = load_json(x_str)
#                         x_test = literal_eval(x_str)
#                         x_test = json.loads(x_str)
#                             json.loads(x.decode('utf-8').strip())
                        input_file_json = type(x_test) is dict
                    except Exception as e:
                        logging.debug('json error %s with tweet %s'%(e, str(x)))
                        input_file_json = False
                # check for dummy .tsv first line
                if(not input_file_json and x.decode('utf-8').split('\t')[0]=='username'):
                    continue
                else:
                    x_data = process_line(x, input_file_json)
                    x_data_status = x_data.loc[status_idx].values
                    x_data_txt = clean_txt(x_data.loc['text'])
                    if(x_data_txt == ''):
                        logging.debug('empty text at status %s'%(x))
                    ## TODO: include original status text in status data file
                    ## only tag valid statuses
                    if(x_data_txt != ''):
                        ## filter for language
                        x_lang = langid.classify(x_data_txt)
                        if(x_lang[0] in langs):
                            x_data_status = np.append(x_data_status, [x_lang[0]])
                            x_data_status_str = [str(y) for y in x_data_status]
                            out_file_status_output.write(('%s\n'%('\t'.join(x_data_status_str))).encode('utf-8'))
                            out_file_txt_output.write(('%s\n'%(x_data_txt)).encode('utf-8'))
                if(i % 1000 == 0):
                    logging.debug('processed %d tweets'%(i))
                    # tmp debugging
#                     if(i > 10000):
#                         break
#             except Exception as e:
#                 logging.debug('exception %s'%(e))
#                 logging.debug('error opening file after %d tweets; shutting down now'%(i))

def tag_txt(status_file, txt_file, tags_file):
    """
    Tag all text for NE and NE type.
    
    
    """
    # load taggers
    # Spanish tagger = Stanford
    spanish_tagger = load_spanish_tagger()
    # English tagger = NoisyNLP
#     english_tagger = load_english_tagger()
#     taggers = {'en':english_tagger, 'es':spanish_tagger}
    tokenizers = {'en':TweetTokenizer(), 'es':ToktokTokenizer()}
    langs = tokenizers.keys()
    tag_output_files = {l:txt_file.replace('.txt', '_%s_tags.txt'%(l)) for l in langs}
    tag_outputs = {l:open(f, 'w') for l, f in tag_output_files.items()}
        
    ## new non-sequential tagging code
    ## 1. write tokens to file by language
    ## 2. run tagger on each file 
    ## 3. realign tagged tweets by status ID
    status_ids = []
    with gzip.open(status_file) as status_input, open(txt_file, 'r') as txt_input:
        for i, (status_i, txt_i) in enumerate(zip(status_input, txt_input)):
            status_data_i = status_i.decode('utf-8').strip().split('\t')
            status_id_i = status_data_i[0]
            # tmp debugging 
            lang_i = status_data_i[-1]
            tokenizer = tokenizers[lang_i]
#             tagger = taggers[lang_i]
            tokens_i = tokenizer.tokenize(txt_i)
            # write status ID to pre-tag file to re-join later
            tag_outputs[lang_i].write(('%s\t%s\n'%(status_id_i, ' '.join(tokens_i))))
            status_ids.append(status_id_i)
    # close files
    for _, tag_output in tag_outputs.items():
        tag_output.close()
    ## tmp: check tag lines
#     for tag_output_file in tag_output_files.values():
#         line_ctr = 0
#         for l in open(tag_output_file, 'r'):
#             line_ctr += 1
#         logging.debug('%d lines in %s file'%(line_ctr, tag_output_file))
    
    # run taggers 
    # English
    tag_output_file_en = tag_output_files['en'].replace('.txt', '_out.txt')
    if(os.path.exists(tag_output_files['en'])):
        logging.debug('tagging English data')
        tagger_en_command = 'export TWITTER_NLP=twitter_nlp/; python2 twitter_nlp/python/ner/extractEntities.py --classify --text-pos 1 %s -o %s'%(tag_output_files['en'], tag_output_file_en)
        os.system(tagger_en_command)
        # cleanup 
        os.rename(tag_output_file_en, tag_output_files['en'])
        tag_line_count_en = 0
        for x in open(tag_output_files['en'], 'r'):
            tag_line_count_en += 1
        logging.debug('%d English lines tagged'%(tag_line_count_en))
        ## tmp: count lines processed
#         line_ctr = 0
#         for l in open(tag_output_files['en'], 'r'):
#             line_ctr += 1
#         logging.debug('%d tag lines processed'%(line_ctr))
    # Spanish
    tag_output_file_es = tag_output_files['es'].replace('.txt', '_out.txt')
    if(os.path.exists(tag_output_files['es'])):
        logging.debug('tagging Spanish data')
        with open(tag_output_file_es, 'w') as tag_output_es:
            for l in open(tag_output_files['es'], 'r'):
#                 logging.debug(l)
                if(len(l.strip().split('\t')) > 1):
                    l_id, l_str = l.strip().split('\t')
                    l_tokens = l_str.split(' ')
                    l_tags = spanish_tagger.tag(l_tokens)
                    tag_output_es.write('%s\t%s\n'%(l_id, ' '.join(['/'.join(t) for t in l_tags])))
        # cleanup
        os.rename(tag_output_file_es, tag_output_files['es'])
        tag_line_count_es = 0
        for x in open(tag_output_files['es'], 'r'):
            tag_line_count_es += 1
        logging.debug('%d Spanish lines tagged'%(tag_line_count_es))
    
    # realign with status ID
    tag_data_combined = []
    for lang, tag_output_file in tag_output_files.items():
#         for l in open(tag_output_file):
#             l_txt = l.strip().split('\t')
#             if(len(l_txt) < 2):
#                 logging.debug('bad line %s'%(l))
        tag_data = [l.strip().split('\t') for l in open(tag_output_file, 'r')]
        # get rid of hanging line
#         tag_data = [l for l in tag_data if len(l) > 0]
#         for idx, l in enumerate(tag_data):
#             if(len(l) == 0):
#                 logging.debug('bad tag data %d/%d %s'%(idx, len(tag_data), str(l)))
        # split statuses
        status_tag_data = [l[0] for l in tag_data]
        tag_data = [l[1] for l in tag_data]
        tag_data_series = pd.Series(tag_data)
        tag_data_series.index = status_tag_data
        tag_data_combined.append(tag_data_series)
    tag_data_combined = pd.concat(tag_data_combined, axis=0)
#     logging.debug(tag_data_combined.head())
    # align
    tag_data_combined = tag_data_combined.loc[status_ids].dropna(inplace=False)
    # drop duplicate statuses
    tag_data_combined = tag_data_combined.loc[tag_data_combined.index.drop_duplicates()]
    
    logging.debug('%d tag lines %d status IDs'%(len(tag_data_combined), len(status_ids)))
    # write to file
    tag_output_file_final = txt_file.replace('.txt', '_tags.gz')
    tag_data_combined.to_csv(tag_output_file_final, sep='\t', compression='gzip', index=True)
    
    ## remove separate language tag files
    for lang, tag_output_file in tag_output_files.items():
        os.remove(tag_output_file)
    
    # tag_output_file_combined = txt_file.replace('.txt', '_%s_tags.txt'%(l)
    
#     ## write tags to gzip file
#     tag_output_files_txt = {l : f.replace('.txt', '_out.txt') for l, f in tag_output_files.items()}
#     tag_output_files_zip = {l : f.replace('.txt', '.gz') for l,f in tag_output_files.items()}
#     for l, tag_output_file_txt in tag_output_files_txt.items():
#         if(os.path.exists(tag_output_file_txt)):
#             tag_output_file_zip = tag_output_files_zip[l]
#             with open(tag_output_file_txt, 'r') as tag_output_txt, gzip.open(tag_output_file_zip, 'w') as tag_output_zip:
#                 shutil.copyfileobj(tag_output_txt, tag_output_zip)
    
    ## delete old files
    
            
            ## old sequential tagging code that doesn't work with TwitterNLP
#             if(len(tokens_i) > 0):
# #                 logging.debug('about to tag status %s'%(str(status_i)))
# #                 logging.debug('about to tag tokens %s'%(str(tokens_i)))
#                 try:
#                     token_tags_i = tagger.tag(tokens_i)
#                     token_tags_i_str = ' '.join(['%s/%s'%(x[0], x[1]) for x in token_tags_i])
#                 except Exception as e:
#                     logging.debug('bad text %s'%(txt_i))
#                     token_tags_i_str = ''
#             else:
#                 logging.debug('empty tokenized text %s'%(txt_i))
#                 token_tags_i_str = ''
#             tags_output.write(('%s\n'%(token_tags_i_str)).encode('utf-8'))
#             if(i % 1000 == 0):
#                 logging.debug('tagged %d tweets'%(i))
                    
def main():
    parser = ArgumentParser()
    # archive files => JSON
    # Maria
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/stream_maria.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/archive_maria.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/historical_maria.gz')
    # Harvey
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/stream_harvey.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/archive_harvey.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/historical_harvey.gz')
    # Irma
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/stream_irma.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/archive_irma.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/historical_irma.gz')
    # Florence
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/east_coast_geo_twitter_2018/geo_stream_florence.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/archive_florence.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/historical_florence.gz')
    # Michael
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/east_coast_geo_twitter_2018/geo_stream_michael.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/archive_michael.gz')
#     parser.add_argument('--tweet_file', default='../../data/mined_tweets/historical_michael.gz')
    # power users
    parser.add_argument('--tweet_file', default='../../data/mined_tweets/combined_data_power_user_tweets.gz')
    # allowed languages
    # Maria
    parser.add_argument('--langs', default=['en', 'es'])
    # others
#     parser.add_argument('--langs', default=['en'])
    args = vars(parser.parse_args())
    args['langs'] = set(args['langs'])
    logging_file = '../../output/tag_NE_tweets.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    out_file_status = args['tweet_file'].replace('.gz', '_status.gz')
    out_file_txt = args['tweet_file'].replace('.gz', '_txt.txt')
    
    ## write tweet data to files
#     input_file_json = args['tweet_file'].endswith('.json')
    # shortcut to replace old files: messy
    replace_old_files = True
#     replace_old_files = False
    if(replace_old_files or not os.path.exists(out_file_status) or not os.path.exists(out_file_txt)):
        write_status_txt(args['tweet_file'], out_file_status, out_file_txt, langs=args['langs'])

    ## tag for NE!
    out_file_tags = args['tweet_file'].replace('.gz', '_txt_tags.gz')
    if(not os.path.exists(out_file_tags)):
        tag_txt(out_file_status, out_file_txt, out_file_tags)
    
if __name__ == '__main__':
    main()