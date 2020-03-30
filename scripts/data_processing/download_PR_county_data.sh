DATA_DIR=../../data/geo_files
COUNTY_DATA_DIR=$DATA_DIR/county_shape_files
mkdir $COUNTY_DATA_DIR
cd $COUNTY_DATA_DIR
PR_county_file=https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/tl_2018_us_county.zip
wget https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/tl_2018_us_county.zip .
unzip tl_2018_us_county.zip