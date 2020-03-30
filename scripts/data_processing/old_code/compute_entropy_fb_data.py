"""
Compute per-post and per-context entropy 
for all FB posts. 
ASSUMES:
- all NEs in posts have already been tagged in `tag_ne_in_fb_data.py`
"""
import pandas as pd
import numpy as np
import nltk
from nltk.util import ngrams
from nltk.lm import Lidstone
from nltk.tokenize.toktok import ToktokTokenizer
from argparse import ArgumentParser
from functools import reduce
import re
from multiprocessing import Pool
from itertools import repeat

def generate_ngrams(sent, tokenizer, n=3):
    """
    Generate ngrams from sentence tokens.
    
    :param sent: string sentence
    :param tokenizer: word tokenizer
    :returns ngram_iter: ngram generator
    """
    ngram_iter = ngrams(tokenizer.tokenize(sent), n, pad_left = True, pad_right = True, right_pad_symbol='EOS', left_pad_symbol="BOS")
    return ngram_iter

def compute_entropy_sent(sents, lm, tokenizer, ngram_order):
    """
    Compute entropy for all sentences.
    
    :param sents: list of sentences
    :param lm: language model
    :param tokenizer: tokenizer
    :param ngram_order: ngram order for entropy
    :returns entropy_vals:: entropy for all sentences
    """
    entropy_vals = [lm.entropy(generate_ngrams(sent, tokenizer, n=ngram_order)) for sent in sents]
    return entropy_vals

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_file', default='../../data/facebook-maria/combined_group_data_es_tagged.tsv')
    parser.add_argument('--ngram_order', default=3)
    args = vars(parser.parse_args())
    
    ## load data
    combined_data = pd.read_csv(args['data_file'], sep='\t', index_col=False)
    # remove URL-only statuses
    min_status_len = 3
    combined_data = combined_data[combined_data.loc[:, 'status_message_ne_tagged_stemmed'].apply(lambda x: len(str(x).split(' '))) > min_status_len]
    # tmp debugging
#     combined_data = combined_data.head(100)
    
    ## compute entropy
    ## (1) per-post
    ## (2) per-mention (within sentence)
    ## (3) per-mention (within fixed window?)
    ## train language model
    ngram_order = args['ngram_order']
    tokenizer = ToktokTokenizer() # use TokTok for tokens because it's multilingual
    sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
    sent_tokenize = lambda x: sent_tokenizer.tokenize(x)
    combined_data.loc[:, 'status_message_sents'] = combined_data.loc[:, 'status_message_ne_tagged_stemmed'].apply(sent_tokenize)
    combined_data_sents = reduce(lambda x,y: x+y, combined_data.loc[:, 'status_message_sents'].values)
    combined_data_ngrams = ((ngram for ngram in generate_ngrams(sent, tokenizer, n=ngram_order)) for sent in combined_data_sents)
    # train Lidstone language model
    gamma = 0.1
    combined_data_lm = Lidstone(order=ngram_order, gamma=gamma)
    vocab = set(reduce(lambda x,y: x+y, [tokenizer.tokenize(txt) for txt in combined_data.loc[:, 'status_message_ne_tagged_stemmed'].values]))
    print('about to fit LM with order=%d and vocab=%d'%(ngram_order, len(vocab)))
    combined_data_lm.fit(combined_data_ngrams, vocabulary_text=vocab)

    ## split data to per-sentence for easier handling
    print('about to compute entropy for all sentences')
    # multi-threading for slightly faster performance
    MAX_THREADS = 10
    pool = Pool(MAX_THREADS)
    combined_data_sents_ordered = combined_data.loc[:, 'status_message_sents'].values
    combined_data_entropy = pool.starmap(compute_entropy_sent, zip(combined_data_sents_ordered, repeat(combined_data_lm), repeat(tokenizer), repeat(ngram_order)))
    print('combined data entropy shape %d'%(len(combined_data_entropy)))
    combined_data.loc[:, 'sent_entropy'] = combined_data_entropy
    # serial threading for LOSERS
#     combined_data.loc[:, 'sent_entropy'] = combined_data.loc[:, 'status_message_sents'].apply(lambda x: [combined_data_lm.entropy(generate_ngrams(sent, tokenizer, n=ngram_order)) for sent in x])
    ## compute entropy for each (1) post (2) entity
    ## format: status ID, publish time, status message, entity name, post entropy, sentence entropy
    combined_data.loc[:, 'post_entropy'] = combined_data.loc[:, 'sent_entropy'].apply(lambda x: np.mean(x))
    combined_data_flat = []
    ne_matcher = re.compile('\w+\.<ne\.\w+>')
    ne_null = 'NULL_ENTITY.<ne>'
    for i, combined_data_i in combined_data.iterrows():
        status_id_i = combined_data_i.loc['status_id']
        status_time_i = combined_data_i.loc['status_published']
        status_message_i = combined_data_i.loc['status_message_ne_tagged_stemmed']
        entropy_i = combined_data_i.loc['post_entropy']
        for sent_j, entropy_j in zip(*combined_data_i.loc[['status_message_sents', 'sent_entropy']].values):
            print('processing sent %s'%(sent_j))
            sent_tokens = tokenizer.tokenize(sent_j)
            sent_tokens_ne = list(filter(lambda x: ne_matcher.search(x) is not None, sent_tokens))
            if(len(sent_tokens_ne) > 0):
                data_j = pd.DataFrame([[status_id_i, status_time_i, status_message_i, sent_j, entropy_i, sent_token_ne, entropy_j] for sent_token_ne in sent_tokens_ne])
            else:
                data_j = pd.DataFrame([[status_id_i, status_time_i, status_message_i, sent_j, entropy_i, ne_null, entropy_j]])
            print('adding data with shape %s'%(len(data_j)))
            combined_data_flat.append(data_j)
    combined_data_flat_cols = ['status_id', 'status_published', 'status_message', 'sent', 'post_entropy', 'entity', 'sent_entropy']
    combined_data_flat = pd.concat(combined_data_flat, axis=0)
    combined_data_flat.columns = combined_data_flat_cols
    
    ## get examples of low/high entropy statuses
    combined_data_flat.sort_values('sent_entropy', inplace=True, ascending=False)
    print(combined_data_flat.loc[:, 'status_message'].head(5))
    print(combined_data_flat.loc[:, 'status_message'].tail(5))
    
    ## save flat data
    out_file = args['data_file'].replace('.tsv', '_entropy.tsv')
    combined_data_flat.to_csv(out_file, sep='\t', index=False)

if __name__ == '__main__':
    main()