"""
Tag all NEs in FB data using the Stanford NER tool.

ASSUMES that the Stanford server is running locally
using the following command:

```
cd /hg190/corpora/StanfordCoreNLP/stanford-corenlp-full-2018-02-27
java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-spanish.properties -preload tokenize,ssplit,pos,ner,parse -status_port 9003  -port 9003 -timeout 15000
```
"""
import pandas as pd
import numpy as np
import nltk
from nltk.parse.corenlp import CoreNLPParser
from nltk.stem import SnowballStemmer
from argparse import ArgumentParser
import re
import os

ABBREVIATIONS=[('sra', 'Señora'), ('bo', 'Barrio'), ('sr', 'Señor'), ('carr', 'Carretera'), ('urb', 'Urbanización'), ('ave', 'Avenida'), ('esc', 'Escuela'), ('esq', 'Esquina')]
ABBREVIATIONS=[(' [%s%s]%s[\. ]'%(x[0].capitalize(), x[0], x[1:]), ' %s'%(y)) for x,y in ABBREVIATIONS]
ABBREVIATIONS=[(re.compile(x), y) for x,y in ABBREVIATIONS]
URL_MATCHER=re.compile('https?://[^ ]+|www.[^ ]+')
NUM_MATCHER=re.compile('[0-9\-]+')
SPACE_MATCHER=re.compile('\s{2,}')
CAMEL_MATCHER=re.compile('(?<=[a-z])(?=[A-Z])')
PUNCT_MATCHER=re.compile('(?<=\s)[\.\?!]')
PUNCT=['.', '!', '?', ',', ':']
STEMMER=SnowballStemmer('spanish')

def clean_txt(txt, tokenizer):
    txt_clean = URL_MATCHER.sub('URL', txt)
    txt_clean = CAMEL_MATCHER.sub(' ', txt_clean)
#    for x in tokenizer.tokenize(txt):
    for a_matcher, a_sub in ABBREVIATIONS:
        txt_clean = a_matcher.sub(a_sub, txt_clean)
    txt_clean = SPACE_MATCHER.sub(' ', txt_clean)
    for p in PUNCT:
        txt_clean = txt_clean.replace(' %s'%(p), p)
    return txt_clean

def extract_NEs(tags):
    curr_tag = ''
    curr_ne = []
    ne_list = []
    for word, tag in tags:
        if(tag != 'O'):
            if(curr_tag != '' and tag != curr_tag and len(curr_ne) > 0):
                curr_ne = (' '.join([x for x,y in curr_ne]), curr_ne[0][1]) # save (word,tag)
                ne_list.append(curr_ne)
                curr_ne = []
            curr_ne.append((word, tag))
        elif(tag != curr_tag and curr_tag != ''):
            curr_ne = (' '.join([x for x,y in curr_ne]), curr_ne[0][1]) # save (word,tag)
            ne_list.append(curr_ne)
            curr_ne = []
        curr_tag = tag
    return ne_list

def mark_NE_in_status(tags, keep_tag_type=False):
    """
    Add NE tag to the end of all
    NEs in status and join into
    combined string.
    """
    status_list = []
    curr_tag = ''
    curr_ne = []
    for word, tag in tags:
        if(tag != 'O'):
            if(curr_tag != '' and tag != curr_tag and len(curr_ne) > 0):
                curr_ne = ('_'.join([x for x,y in curr_ne]), curr_ne[0][1]) # save (word,tag)
                status_list.append(curr_ne)
                curr_ne = []
            curr_ne.append((word, tag))
        elif(tag != curr_tag and curr_tag != ''):
            curr_ne = ('_'.join([x for x,y in curr_ne]), curr_ne[0][1]) # save (word,tag)
            status_list.append(curr_ne)
            curr_ne = []
        if(tag == 'O'):
            status_list.append((word, tag))
        curr_tag = tag
    # catch any straggler NEs
    if(len(curr_ne) > 0):
        curr_ne = ('_'.join([x for x,y in curr_ne]), curr_ne[0][1]) # save (word,tag)
        status_list.append(curr_ne)
    if(keep_tag_type):
        status_word_list = ['%s.<NE.%s>'%(word, tag) if tag != 'O' else word for word, tag in status_list]
    else:
        status_word_list = ['%s.<NE>'%(word) if tag != 'O' else word for word, tag in status_list]
    status_str = ' '.join(status_word_list)
    return status_str

def process_status_tags(tags, stemmer=None):
    """
    Join NEs in status as single words,
    replace numbers/URLs with markers, and
    (optional) stem words.
    
    :param tags: (word, tag) list
    :param stemmer: optional SnowballStemmer
    :returns status_str: processed status
    """
    status_str = mark_NE_in_status(tags, keep_tag_type=True)
    if(stemmer is not None):
        status_str = ' '.join([stemmer.stem(x) for x in status_str.split(' ')])
    for p in PUNCT:
        status_str = status_str.replace(' %s'%(p), p)
    # clean up status str
    status_str = NUM_MATCHER.sub('NUM', status_str)
    return status_str

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_file', default='../../data/facebook-maria/combined_group_data.tsv')
    args = vars(parser.parse_args())

    ## load data
    combined_data = pd.read_csv(args['data_file'], sep='\t', index_col=False)
    # only Spanish
    combined_data = combined_data[combined_data.loc[:, 'status_lang']=='es']
    # tmp debugging
    # combined_data = combined_data.head(100)
    
    ## clean data
    parser = CoreNLPParser(url='http://localhost:9003', tagtype='ner')
    combined_data.loc[:, 'status_message_clean'] = combined_data.loc[:, 'status_message'].apply(lambda x: clean_txt(x, parser))
    combined_data = combined_data[combined_data.loc[:, 'status_message_clean'].apply(lambda x: x != '')]
    
    ## tag NEs
    combined_data_tags = []
    for i, combined_data_i in combined_data.iterrows():
        x = combined_data_i.loc['status_message_clean']
        try:
            x_tags = parser.tag(parser.tokenize(x))
        except Exception as e:
            print('problem with status %s'%(x))
            print('original status %s'%(combined_data_i.loc['status_message']))
            x_tags = []
        combined_data_tags.append(x_tags)
    combined_data_tags = pd.Series(combined_data_tags, index=combined_data.index)
#     combined_data_tags = pd.Series(combined_data.loc[:, 'status_message_clean'].apply(lambda x: parser.tag(parser.tokenize(x))))
    combined_data_tags_ne = combined_data_tags.apply(extract_NEs)
    combined_data.loc[:, 'status_message_tags'] = combined_data_tags
    combined_data.loc[:, 'status_message_tags_ne'] = combined_data_tags_ne
    # check for float??
    for i, combined_data_i in combined_data.iterrows():
        if(type(combined_data_i.loc['status_message_tags']) is float):
            print('error with data %s'%(combined_data_i))
    
    ## generate tagged/stemmed statuses
    combined_data.loc[:, 'status_message_ne_tagged_stemmed'] = combined_data.loc[:, 'status_message_tags'].apply(lambda x: process_status_tags(x, STEMMER))
    
    ## write to file
    out_file = args['data_file'].replace('.tsv', '_es_tagged.tsv')
    combined_data.to_csv(out_file, sep='\t', index=False)    

if __name__ == '__main__':
    main()
