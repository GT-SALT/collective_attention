"""
Convert osm.pbf file to XML format, using
a parser I stole from here: http://pbf.raggedred.net/
"""
from parsepbf.parsepbf import PBFParser
from argparse import ArgumentParser
import os

def convert_pbf_to_xml(pbf_file_name, xml_file_name):
    with open(pbf_file_name, 'r') as f_input, open(f_name, 'w') as f_output:
        pbf_parser = parsepbf.PBFParser(f_input, f_output)
        if(not pbf_parser.init('pbfparser.py 1.3')):
            print('header trouble')
        pbf_parser.outputxmlhead()
        pbf_parser.parse()
        pbf_parser.outputxmltrail()

def main():
    parser = ArgumentParser()
    parser.add_argument('--pbf_file', default='../../data/geo_files/PR_OSM/puerto-rico-latest.osm.pbf')
    parser.add_argument('--out_dir', default='../../data/geo_files/PR_OSM/xml_files')
    args = parser.parse_args()
    pbf_file_name = args.pbf_file
    out_dir = args.out_dir
    xml_file_name = os.path.join(out_dir, pbf_file_name.replace('.osm.pbf', '.xml'))
    
    ## convert
    convert_pbf_to_xml(pbf_file_name, xml_file_name)
