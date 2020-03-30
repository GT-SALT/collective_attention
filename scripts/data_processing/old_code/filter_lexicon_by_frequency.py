"""
Filter toponym lexicon for high-frequency items
like "amigo", "museo".
We make exceptions for municipality names like "ponce"
because these could serve as key anchor locations.
"""
from __future__ import division
from data_helpers import load_lexicon, load_geonames_data
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import codecs

def main():
    parser = ArgumentParser()
    parser.add_argument('--tf_file', default='../../data/Sep-20-17_tf.tsv')
    parser.add_argument('--out_file', default='../../data/geo_files/toponym_lexicon_filtered.txt')
    args = parser.parse_args()
    tf_file = args.tf_file
    out_file = args.out_file
    
    ## load data
    tf = pd.read_csv(tf_file, sep='\t', header=None, index_col=0).iloc[:, 0]
    print('loaded %d items'%(len(tf)))
    # normalize
    tf /= tf.sum()
    print('normalized tf')
    lexicon = load_lexicon()
    print('loaded %d words in lexicon'%(len(lexicon)))
    assert 'ponce' in lexicon
    # need geonames data for municipality names
    geonames_data = load_geonames_data()
    municipality_names = geonames_data[geonames_data.loc[:, 'feature_code'] == 'first-order_administrative_division'].loc[:, 'name'].values.tolist()
    municipality_names = map(lambda x: x.replace('Municipio', ''), municipality_names)
    # add lowercase just in case
    municipality_names += set(map(lambda x: x.lower(), municipality_names))
    print('got municipality_names')
    print(municipality_names[:10])

    ## filter
    cutoff_pct = 99
    cutoff = np.percentile(tf, cutoff_pct)
    tf_cutoff = tf[tf >= cutoff]
    # overlap
    lexicon_overlap = list(set(lexicon) & set(tf_cutoff.index))
    print('%d mentions in lexicon over cutoff %.3E'%(len(lexicon_overlap), cutoff))
    cutoff_lexicon = set(lexicon_overlap)
    lexicon_final = set(lexicon) - cutoff_lexicon
    # add municipality names for completeness 
    # e.g., might have cut off "ponce" because common
    lexicon_final.update(set(municipality_names))
    lexicon_final = sorted(lexicon_final)
    # decode
    lexicon_final = map(lambda x: x.decode('utf-8'), lexicon_final)

    ## write to file
    with codecs.open(out_file, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(lexicon_final))

if __name__ == '__main__':
    main()
