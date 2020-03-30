"""
For each mention string, find the 
entity with the maximum probability.
This is our baseline for entity linking!!
"""
from argparse import ArgumentParser
import os
from bz2 import BZ2File
from collections import defaultdict
import pandas as pd

def main():
  parser = ArgumentParser()
  parser.add_argument('--prob_file', default='/hg190/corpora/crosswikis-data.tar.bz2/dictionary.bz2')
  args = parser.parse_args()
  prob_file = args.prob_file
  out_dir = os.path.dirname(prob_file)
  
  ## iterate through prob file and collect highest probabilities
  max_probs = defaultdict(float)
  max_prob_entities = defaultdict(str)
  for i, l in enumerate(BZ2File(prob_file, 'r')):
    l_split = l.strip().split('\t')
    if(len(l_split) == 2):
      mention, entity_info = l_split
      entity_info = entity_info.split(' ')
      prob = float(entity_info[0])
      entity = entity_info[1]
      if(prob > max_probs[mention]):
        max_probs[mention] = prob
        max_prob_entities[mention] = entity
    if(i % 1000000 == 0):
      print('processed %d dict lines'%(i))
  # combine data
  # NOPE too much memory
#   max_prob_cols = ['prob', 'entity']
#   max_prob_df = pd.concat([pd.Series(max_probs), pd.Series(max_prob_entities)], axis=1)
#   max_prob_df.columns = max_prob_cols
  
  ## write to file!!
  out_file_name = os.path.join(out_dir, 'max_prob_entities.bz2')
#   max_prob_df.to_csv(out_file_name, sep='\t', compression='bz2')
  col_names = ['mention', 'prob', 'entity']
  with BZ2File(out_file_name, 'w') as out_file:
    out_file.write('%s\n'%('\t'.join(col_names)))
    for m, entity in max_prob_entities.iteritems():
        prob_str = str(max_probs[m])
        out_str = '\t'.join([m, prob_str, entity])
        out_file.write('%s\n'%(out_str))
        
if __name__ == '__main__':
  main()