"""
Parse Facebook data with SyntaxNet.
"""
from argparse import ArgumentParser
import logging
import os
from google.cloud import language
import time
import nltk
# tmp debugging
from ast import literal_eval
from data_helpers import build_parse, BasicTokenizer, NLTKTokenizerSpacy, clean_data_for_spacy
import pandas as pd
import spacy

# sleeping for Syntaxnet in case of rate limiting
SLEEP_TIME = 30.
SLEEP_CTR = 1000
def get_parse(sent, parse_type='syntaxnet', parse_client=None, lang='es'):
    global SLEEP_CTR
    if(parse_type == 'syntaxnet'):
        google_doc = language.types.Document(content=sent, type=language.enums.Document.Type.PLAIN_TEXT, language=lang)
        parse_success = False
        while(not parse_success):
            try:
                google_parse = parse_client.analyze_syntax(google_doc, retry=False)
                parse_success = True
            except Exception as e:
                # avoid rate limiting
                logging.debug('exception %s'%(e))
                logging.debug('sleeping %.3f'%(SLEEP_TIME))
                time.sleep(SLEEP_TIME)
                SLEEP_CTR -= 1
                # too many errors => rate limit
                if(SLEEP_CTR <= 0):
                    parse_success = True
                    break
        sent_parsed = build_parse(google_parse, 'google')
    else:
        spacy_parse = parse_client(sent)
        sent_parsed = build_parse(spacy_parse, 'spacy')
    return sent_parsed

def parse_sents(sent_ids, sents, out_file_name, parse_client=None, parse_type='syntaxnet', lang='es'):
    """
    Parse sentences using either (1) spacy or (2) Google SyntaxNet ($$$).
    
    :param sents: tokenized sentences
    
    """
    sent_ctr = 0
    with open(out_file_name, 'w') as out_file:
        for sent_id, sent in zip(sent_ids, sents):
            parse_tree = get_parse(sent, parse_type=parse_type, parse_client=parse_client, lang=lang)
            parse_str = ' '.join(['/'.join(map(str, x)) for x in parse_tree])
            if(parse_str != ''):
                out_file.write('%s\n'%('\t'.join(map(str, [sent_id, parse_str]))))
            if(sent_ctr % 100 == 0):
                logging.debug('processed %d sents'%(sent_ctr))
            sent_ctr += 1

def main():
    parser = ArgumentParser()
    parser.add_argument('--tagged_data_file', default='../../data/facebook-maria/combined_group_data_es_tagged.tsv')
    parser.add_argument('--out_dir', default='../../data/facebook-maria/')
    parser.add_argument('--cred_file', default='../../data/Google_cloud/TestProject_creds.json')
    parser.add_argument('--lang', default='es')
    parser.add_argument('--parse_type', default='spacy')
    args = vars(parser.parse_args())
    logging_file = '../../output/parse_FB_data.txt'
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    ## load data
    data_tagged = pd.read_csv(args['tagged_data_file'], sep='\t', index_col=False, converters={'status_message_tags' : literal_eval, 'status_message_tags_ne' : literal_eval})
    logging.debug('%d original statuses'%(data_tagged.shape[0]))
    # credentials
    ## WARNING: make sure you calculate how much this will cost before running
    ## https://cloud.google.com/natural-language/pricing
    lang = args['lang']
    long_lang_lookup = {'en' : 'english', 'es' : 'spanish'}
    long_lang = long_lang_lookup[lang]
    parse_type = args['parse_type']
    if(parse_type == 'syntaxnet'):
        parse_client = language.LanguageServiceClient.from_service_account_json(args['cred_file'])
    else:
        if(lang == 'es'):
            parse_client = spacy.load('es_core_news_sm', disable=['ner', 'textcat'])
        elif(lang == 'en'):
            parse_client = spacy.load('en_core_web_md', disable=['ner', 'textcat'])
        parse_client.tokenizer = NLTKTokenizerSpacy(parse_client.vocab, BasicTokenizer(lang=long_lang))
    # get rid of extra-long statuses because of parsing issues
    max_status_len = 150
    data_tagged = data_tagged[data_tagged.loc[:, 'status_message_clean'].apply(lambda x: len(x.split(' '))) < max_status_len]
    # split posts into sentences
    data_tagged_sents = data_tagged.loc[:, 'status_message_clean']
#     print('status example: %s'%(data_tagged_sents.iloc[0]))
    data_ids = data_tagged.loc[:, 'status_id']
    sent_tokenizer = nltk.data.load(f'tokenizers/punkt/{long_lang}.pickle')
    ids_flat = []
    sents_flat = []
    for id_i, sent_i in zip(data_ids, data_tagged_sents):
        split_sents_i = sent_tokenizer.tokenize(sent_i)
        for split_sent_j in split_sents_i:
            split_sent_j = clean_data_for_spacy(split_sent_j)
            sents_flat.append(split_sent_j)
            ids_flat.append(id_i)
            
    ## parse!!
    # parse sentences and write at same time, in case of internet break
    out_file_name = os.path.join(args['out_dir'], args['tagged_data_file'].replace('.tsv', '_parsed_%s.txt'%(parse_type)))
    # tmp safeguard
#     if(not os.path.exists(out_file_name)):
    parse_sents(ids_flat, sents_flat, out_file_name, parse_type=parse_type, parse_client=parse_client)

if __name__ == '__main__':
    main()