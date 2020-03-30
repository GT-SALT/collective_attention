# extract all entities marked as "location" from Wikipedia and write URLs to file
OUT_DIR=/hg190/corpora/crosswikis-data.tar.bz2
OUTPUT=../../output/extract_all_locations_from_dbpedia.txt
(python extract_all_locations_from_dbpedia.py --out_dir $OUT_DIR > $OUTPUT)&