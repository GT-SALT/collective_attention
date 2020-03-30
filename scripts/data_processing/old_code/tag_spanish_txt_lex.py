"""
Tag all toponyms in sample text using
the OSM+GeoName data.
"""
from data_helpers import annotate_lexicon_toponyms_in_txt, load_lexicon, load_filtered_lexicon
from argparse import ArgumentParser
import codecs

def main():
    parser = ArgumentParser()
    parser.add_argument('--sample_txt_file', default='../../data/facebook-maria/all_group_sample_statuses.txt')
    args = parser.parse_args()
    sample_txt_file = args.sample_txt_file
    
    ## load data
    lexicon = load_filtered_lexicon()
    sample_txt = [l.strip() for l in codecs.open(sample_txt_file, 'r', encoding='utf-8')]
    annotated_txt = annotate_lexicon_toponyms_in_txt(sample_txt, lexicon)
    
    ## write to file
    out_file_name = sample_txt_file.replace('.txt', '_lex.txt')
    print(out_file_name)
    with codecs.open(out_file_name, 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(annotated_txt).decode('utf-8'))
    
if __name__ == '__main__':
    main()
