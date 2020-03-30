"""
After tagging the text data for locations using
automated and manual systems, let's figure out the 
relative precision/recall of each system in terms of
how well it extracts legit and relevant toponyms.
"""
import pandas as pd
import os
from data_helpers import clean_tagged_txt, collect_entities_from_txt, collect_bracketed_entities_from_txt, test_precision_recall, collect_double_bracketed_entities, split_tokens_types, filter_by_tag_type
from argparse import ArgumentParser
import codecs
from itertools import izip
from collections import Counter
import logging

LOG_FILE='../../output/compare_toponym_retrieval_methods.txt'
def main():
    parser = ArgumentParser()
    parser.add_argument('--ner_tagged_file', default='../../data/facebook-maria/all_group_sample_statuses_ner.txt')
    parser.add_argument('--neural_ner_tagged_file', default='../../data/facebook-maria/all_group_sample_statuses_neural_ner.txt')
    parser.add_argument('--lex_tagged_file', default='../../data/facebook-maria/all_group_sample_statuses_lex.txt')
    parser.add_argument('--gold_data_file', default='../../data/facebook-maria/all_group_sample_statuses_annotated.txt')
    parser.add_argument('--accuracy_output_file', default='../../output/toponym_retrieval_accuracy.tsv')
    # parser.add_argument('--accuracy_output_file', default='../../output/toponym_retrieval_accuracy_banned_phrases.tsv')
    args = parser.parse_args()
    ner_tagged_file = args.ner_tagged_file
    neural_ner_tagged_file = args.neural_ner_tagged_file
    lex_tagged_file = args.lex_tagged_file
    gold_data_file = args.gold_data_file
    accuracy_output_file = args.accuracy_output_file
    tag_files = [ner_tagged_file, ner_tagged_file, neural_ner_tagged_file, neural_ner_tagged_file, lex_tagged_file]
    tag_names = ['ner', 'ner_loc', 'neural_ner', 'neural_ner_loc', 'lex']
    logging.basicConfig(filename=LOG_FILE,format='%(message)s', level=logging.INFO, filemode='w')

    ## load data
    tag_line_list = [[l.strip() for l in codecs.open(f, 'r', encoding='utf-8')] for f in tag_files]
    # clean NER data
    ner_delim = '[0-9]{10,}_[0-9]{10,}'
    tag_line_list[0] = clean_tagged_txt(tag_line_list[0], delim=ner_delim)
    tag_line_list[1] = clean_tagged_txt(tag_line_list[1], delim=ner_delim)
    # load gold data
    gold_annotation_lines = [l.strip() for l in codecs.open(gold_data_file, 'r', encoding='utf-8')]
    gold_annotations = map(collect_double_bracketed_entities, gold_annotation_lines)
    gold_annotations_known = map(lambda x: filter(lambda y: y[1]!='UNK', x), gold_annotations)
    # remove type
    gold_annotations_clean = map(lambda x: map(lambda y: y[0], x), gold_annotations_known)
    # lowercase for lexicon matching
    gold_annotations_lower = map(lambda x: map(lambda y: y.lower(), x), gold_annotations_clean)

    ## debugging: entity extraction
    # test_lines = tag_line_list[0]
    # test_line = test_lines[15]
    # print(test_line)
    # neural NER
    # tag_data = collect_entities_from_txt(test_line, include_type=True, outside_tag='O', delim='__', use_bio=True)
    # NER
    # tag_data = collect_entities_from_txt(test_line, include_type=True, outside_tag='O', delim='/', use_bio=False)
    # print(tag_data)
    
    ## extract annotations
    tag_data_list = []
    banned_phrases = []
    # banned_phrases = ['Puerto Rico']
    for tag_lines, tag_name in izip(tag_line_list, tag_names):
        if(tag_name == 'ner'):
            tag_data = map(lambda x: collect_entities_from_txt(x, include_type=False, outside_tag='O', delim='/', use_bio=False), tag_lines)
        # filter by LUG type
        elif(tag_name == 'ner_loc'):
            tag_data = map(lambda x: collect_entities_from_txt(x, include_type=True, outside_tag='O', delim='/', use_bio=False), tag_lines)
            tag_data = map(lambda x: split_tokens_types(x, delim='/'), tag_data)
            tag_data = map(lambda x: filter_by_tag_type(x, 'LUG'), tag_data)
        elif(tag_name == 'neural_ner'):
            tag_data = map(lambda x: collect_entities_from_txt(x, include_type=False, outside_tag='O', delim='__', use_bio=True), tag_lines)
        # filter by LOC type
        elif(tag_name == 'neural_ner_loc'):
            tag_data = map(lambda x: collect_entities_from_txt(x, include_type=True, outside_tag='O', delim='__', use_bio=True), tag_lines)
            tag_data = map(lambda x: split_tokens_types(x, delim='__'), tag_data)
            tag_data = map(lambda x: filter_by_tag_type(x, 'LOC'), tag_data)
        elif(tag_name == 'lex'):
            tag_data = map(collect_bracketed_entities_from_txt, tag_lines)
        # TEMP: remove banned phrases that shouldn't count against score
        tag_data = map(lambda x: filter(lambda y: y not in banned_phrases, x), tag_data)
        tag_data_list.append(tag_data)

    ## compute precision/recall 
    toponym_accuracy_results = []
    results_index = ['precision', 'recall', 'f1']
    top_k = 30
    for tag_data, tag_name in izip(tag_data_list, tag_names):
        logging.info('testing retrieval method %s'%(tag_name))
        if(tag_name == 'lex'):
            false_positives, false_negatives, precision, recall = test_precision_recall(tag_data, gold_annotations_lower)
        else:
            false_positives, false_negatives, precision, recall = test_precision_recall(tag_data, gold_annotations_clean)
        f1 = 2 * (precision*recall) / (precision + recall)
        logging.info('precision = %.3f, recall = %.3f'%(precision, recall))
        false_positives_flat = reduce(lambda x,y: x+y, false_positives)
        false_negatives_flat = reduce(lambda x,y: x+y, false_negatives)
        false_positive_counts = Counter(false_positives_flat)
        false_negative_counts = Counter(false_negatives_flat)
        logging.info('most common false positives:\n%s'%('\n'.join(map(str, false_positive_counts.most_common(top_k)))))
        logging.info('most common false negatives:\n%s'%('\n'.join(map(str, false_negative_counts.most_common(top_k)))))
        results = pd.Series([precision, recall, f1], index=results_index)
        toponym_accuracy_results.append(results)
    toponym_accuracy_results = pd.concat(toponym_accuracy_results, axis=1).transpose()
    toponym_accuracy_results.index = tag_names
    

    ## write to file
    toponym_accuracy_results.to_csv(accuracy_output_file, sep='\t')

if __name__ == '__main__':
    main()
