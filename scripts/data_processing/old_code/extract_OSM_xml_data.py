"""
Collect OSM data from XML file (downloaded here: http://download.geofabrik.de/north-america/us/puerto-rico-latest.osm.bz2).
We use this data instead of the SHP file data because the SHP files are incomplete.
"""
from xml.etree import ElementTree
from argparse import ArgumentParser
from collections import defaultdict
import logging
import pandas as pd
import os

logging.basicConfig(filemode='w', filename='../../output/extract_OSM_xml_data.txt', format='%(message)s', level=logging.INFO)
def extract_all_nodes(xml_file):
    node_dict = defaultdict(list)
    way_dict = defaultdict(list)
    relation_dict = defaultdict(list)
    valid_keys = ['id', 'lat', 'lon', 'name']
#     cutoff = 5000000
#     valid_node_keys = ['addr:city', 'addr:street', 'place']
    with open(xml_file, 'r') as xml_input:
        ctr = 0
        tree = ElementTree.iterparse(xml_input)
        curr_data = []
        for items in tree:
            for item in items:
                if(item != 'end' and item.tag != 'bounds' and item.tag != 'osm'):
                    attrib_i = item.attrib
                    items = item.items()
                    tag_i = item.tag
                    if(tag_i != 'node' and tag_i != 'way' and tag_i != 'relation'):
                        if(tag_i == 'tag' and items[0][0]=='k' and items[1][0]=='v'):
                            k = items[0][1]
                            v = items[1][1]
                            curr_data.append((k,v))
                        # handling relations
                        elif(tag_i == 'member'):
                            curr_data.append(items[1])
                        else:
                            curr_data += items
                    # if we've hit a node/way/relation, update dict and flush data
                    else:
                        # restrict to important data
                        valid_items = filter(lambda x: x[0] in valid_keys, items)
                        curr_data += valid_items
                        if(tag_i == 'node'):
                            node_id = attrib_i['id']
                            node_dict[node_id] = list(curr_data)
                            curr_data = []
                        elif(tag_i == 'way'):
                            way_id = attrib_i['id']
                            way_dict[way_id] = list(curr_data)
                            curr_data = []
                        elif(tag_i == 'relation'):
                            relation_id = attrib_i['id']
                            relation_dict[relation_id] = list(curr_data)
                            curr_data = []
                    ctr += 1
                    if(ctr % 1000000 == 0):
                        print('processed %d items'%(ctr))
#                     if(ctr >= cutoff):
#                         break
#                 if(ctr >= cutoff):
#                     break
    # connect ways to lat/lon points using ref numbers
    for way_id, way_data in way_dict.iteritems():
        ref_ids = map(lambda x: x[1], filter(lambda y: y[0]=='ref', way_data))
        non_ref_data = filter(lambda x: x[0]!='ref', way_data)
        ref_lats = [[x[1] for x in node_dict[ref_id] if x[0]=='lat'] for ref_id in ref_ids]
        ref_lats = filter(lambda x: len(x) > 0, ref_lats)
        ref_lons = [[x[1] for x in node_dict[ref_id] if x[0]=='lon'] for ref_id in ref_ids]
        ref_lons = filter(lambda x: len(x) > 0, ref_lons)
        ref_lat_lons = zip(ref_lats, ref_lons)
        ref_lat_lons = [(x[0], y[0]) for x,y in ref_lat_lons]
        non_ref_data.append(('lat_lons', ref_lat_lons))
        way_dict[way_id] = non_ref_data
    for relation_id, relation_data in relation_dict.iteritems():
        ref_ids = map(lambda x: x[1], filter(lambda y: y[0]=='ref', relation_data))
        non_ref_data = filter(lambda x: x[0]!='ref', relation_data)
        ref_lats = [[x[1] for x in node_dict[ref_id] if x[0]=='lat'] for ref_id in ref_ids]
        ref_lats = filter(lambda x: len(x) > 0, ref_lats)
        ref_lons = [[x[1] for x in node_dict[ref_id] if x[0]=='lon'] for ref_id in ref_ids]
        ref_lons = filter(lambda x: len(x) > 0, ref_lons)
        ref_lat_lons = zip(ref_lats, ref_lons)
        ref_lat_lons = [(x[0], y[0]) for x,y in ref_lat_lons]
        non_ref_data.append(('lat_lons', ref_lat_lons))
        relation_dict[relation_id] = non_ref_data
    # remove all nameless nodes, ways and relations
    node_dict = {k : v for k,v in node_dict.iteritems() if 'name' in map(lambda x: x[0], v)}
    way_dict = {k : v for k,v in way_dict.iteritems() if 'name' in map(lambda x: x[0], v)}
    relation_dict = {k : v for k,v in relation_dict.iteritems() if 'name' in map(lambda x: x[0], v)}
    # restrict to name, id, lat, lon
    final_valid_keys = ['id', 'name', 'lat', 'lon', 'lat_lons']
    filter_for_valid_keys = lambda x: filter(lambda y: y[0] in final_valid_keys, x)
    node_dict = {k : filter_for_valid_keys(v) for k,v in node_dict.iteritems()}
    way_dict = {k : filter_for_valid_keys(v) for k,v in way_dict.iteritems()}
    relation_dict = {k : filter_for_valid_keys(v) for k,v in relation_dict.iteritems()}
    # convert to rows
    if(len(node_dict) > 0):
        node_df = pd.concat(map(lambda x: pd.Series(dict(x)), node_dict.values()), axis=1).transpose()
    else:
        node_df = pd.DataFrame()
    if(len(way_dict) > 0):
        way_df = pd.concat(map(lambda x: pd.Series(dict(x)), way_dict.values()), axis=1).transpose()
    else:
        way_df = pd.DataFrame()
    if(len(relation_dict) > 0):
        relation_df = pd.concat(map(lambda x: pd.Series(dict(x)), relation_dict.values()), axis=1).transpose()
    else:
        relation_df = pd.DataFrame()
#     return node_dict, way_dict, relation_dict
    return node_df, way_df, relation_df

def main():
    parser = ArgumentParser()
    parser.add_argument('--xml_file', default='../../data/geo_files/PR_OSM/xml_files/puerto-rico-latest.osm')
    parser.add_argument('--out_dir', default='../../data/geo_files/PR_OSM/xml_files/')
    args = parser.parse_args()
    xml_file = args.xml_file
    out_dir = args.out_dir
    
    ## load data
#     node_dict, way_dict, relation_dict = extract_all_nodes(xml_file)
    node_df, way_df, relation_df = extract_all_nodes(xml_file)
    logging.info('collected %d nodes, %d ways, %d relations'%(len(node_df), len(way_df), len(relation_df)))
    
    ## write to file
    node_df.to_csv(os.path.join(out_dir, 'xml_node_data.tsv'), sep='\t', index=False, encoding='utf-8')
    way_df.to_csv(os.path.join(out_dir, 'xml_way_data.tsv'), sep='\t', index=False, encoding='utf-8')
    relation_df.to_csv(os.path.join(out_dir, 'xml_relation_data.tsv'), sep='\t', index=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()