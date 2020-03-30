"""
Extract status IDs from broadly-annotated posts,
but restrict to only the fine-grained-annotated posts.
"""
from argparse import ArgumentParser
from data_helpers import extract_status_ids
import codecs

def main():
    parser = ArgumentParser()
    parser.add_argument('--topo_annotate_file', default='../../data/facebook-maria/all_group_sample_statuses_annotated_topo.txt')
    parser.add_argument('--raw_annotate_file', default='../../data/facebook-maria/all_group_sample_statuses_annotated.txt')
    args = parser.parse_args()
    topo_annotate_file = args.topo_annotate_file
    raw_annotate_file = args.raw_annotate_file
    
    ## load data
    topo_annotate_lines = [l.strip() for l in codecs.open(topo_annotate_file, 'r', encoding='utf-8')]
    annotate_lines = [l.strip() for l in codecs.open(raw_annotate_file, 'r', encoding='utf-8')]
    
    ## extract
    status_id_list = extract_status_ids(topo_annotate_lines, annotate_lines)
    
    ## write to file
    out_file = topo_annotate_file.replace('.txt', '_ids.txt')
    with open(out_file, 'w') as output:
        output.write('\n'.join(status_id_list))

if __name__ == '__main__':
    main()