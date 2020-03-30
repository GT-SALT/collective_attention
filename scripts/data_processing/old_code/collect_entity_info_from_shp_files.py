"""
Collect OSM entity information
from raw shape files.
"""
from __future__ import division
from itertools import izip
import pandas as pd
import numpy as np
import fiona
import json
import os
from argparse import ArgumentParser
from itertools import cycle
from shapely.geometry import LineString, Polygon

def area_of_polygon(x, y):
    """Calculates the signed area of an arbitrary polygon given its verticies
    http://stackoverflow.com/a/4682656/190597 (Joe Kington)
    http://softsurfer.com/Archive/algorithm_0101/algorithm_0101.htm#2D%20Polygons
    """
    area = 0.0
    for i in xrange(-1, len(x) - 1):
        area += x[i] * (y[i + 1] - y[i - 1])
    return area / 2.0

def centroid_of_polygon(points):
    """
    http://stackoverflow.com/a/14115494/190597 (mgamba)
    """
    area = area_of_polygon(*zip(*points))
    result_x = 0
    result_y = 0
    N = len(points)
    points = cycle(points)
    x1, y1 = next(points)
    for i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)
    return (result_x, result_y)

def collect_tuples(coord_list):
    """
    Collect all coordinate tuples in list by flattening list.
    """
    full_coord_list = []
    if(type(coord_list[0]) is not list):
        return coord_list
    else:
        for l in coord_list:
            full_coord_list += collect_tuples(l)
    return full_coord_list

def get_geotag_info(f_name):
    """
    Extract name, location, and feature class of 
    OSM items in file.
    
    Parameters:
    -----------
    f_name : str
    
    Returns:
    --------
    geotag_info : pandas.DataFrame
    """
    print('processing file %s'%(f_name))
    shape_collection = fiona.open(f_name)
    geotag_info = []
    for i, item in enumerate(shape_collection):
        geometry = item['geometry']
        geometry_type = geometry['type']
        coords = geometry['coordinates']
        if(geometry_type == 'Point'):
            # get point estimate
            lat, lon = coords
            lat_lon_tuples = [coords]
            centroid = coords
        elif(geometry_type == 'LineString' or geometry_type == 'Polygon'):
            # compute average of coord list (NOT PERFECT BUT WE'LL FIX LATER)
            lat_lon_tuples = collect_tuples(coords)
            lat_vals, lon_vals = zip(*lat_lon_tuples)
            lat = np.mean(lat_vals)
            lon = np.mean(lon_vals)
            if(geometry_type == 'Polygon'):
                centroid = Polygon(lat_lon_tuples).centroid
                centroid = (centroid.x, centroid.y)
                # centroid = centroid_of_polygon(lat_lon_tuples)
            else:
                centroid = LineString(lat_lon_tuples).centroid
                centroid = (centroid.x, centroid.y)
        properties = item['properties']
        name = properties['name']
        osm_id = properties['osm_id']
        feature_class = properties['fclass']
        geo_item = [name, geometry_type, feature_class, osm_id, lat, lon, centroid, lat_lon_tuples]
        geotag_info.append(geo_item)
    geotag_cols = ['name', 'geometry_type', 'feature_class', 'osm_id', 'lat', 'lon', 'centroid', 'all_points']
    geotag_info = pd.DataFrame(geotag_info, columns=geotag_cols)
    return geotag_info

def main():
    parser = ArgumentParser()
    parser.add_argument('--shp_file_dir', default='../../data/geo_files/PR_OSM/shp_files/')
    args = parser.parse_args()
    shp_file_dir = args.shp_file_dir

    # extract all shape files and filter to relevant ones
    all_shp_files = [os.path.join(shp_file_dir, f) for f in os.listdir(shp_file_dir) if '.shp' in f]
    all_shp_types = [os.path.basename(f).split('_')[1] for f in all_shp_files]
    relevant_shp_types = ['buildings', 'landuse', 'natural', 'places', 'pofw', 'pois', 'roads', 'traffic']
    relevant_shp_files = [f for t,f in izip(all_shp_types, all_shp_files) if t in relevant_shp_types]
    relevant_shp_types = [os.path.basename(f).split('_')[1] for f in relevant_shp_files]

    # collect all the data
    entity_df_list = [get_geotag_info(f) for f in relevant_shp_files]
    for t, df in izip(relevant_shp_types, entity_df_list):
        df.loc[:, 'shp_type'] = t
    entity_df_combined = pd.concat(entity_df_list, axis=0)
    
    # write to file
    out_file_name = os.path.join(shp_file_dir, 'combined_shp_data.tsv')
    entity_df_combined.to_csv(out_file_name, sep='\t', index=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()
