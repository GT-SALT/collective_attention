# split Twitter data into chunks and parse chunks
# concurrently
DATA_DIR=../../data/mined_tweets
# DATA_FILE=$DATA_DIR/combined_tweet_NE_flat_data.gz
# DATA_FILE=$DATA_DIR/combined_tweet_tag_data_NE_flat.gz
DATA_FILE=$DATA_DIR/combined_data_power_user_NE_flat.gz

CHUNKS=10
CHUNK_IDX=$(seq 1 $CHUNKS)
TOTAL_TWEETS=$(zcat $DATA_FILE | wc -l)
JOBS=1
parallel --jobs $JOBS --bar --verbose python parse_twitter_data.py --tweet_file $DATA_FILE --chunk_idx {} --chunk_count $CHUNKS --total_tweets $TOTAL_TWEETS ::: "${CHUNK_IDX[@]}"

## recombine once finished
CHUNK_DATA_FILES=$(echo $DATA_FILE | sed 's/.gz/_parsed_*.gz/')
echo $CHUNK_DATA_FILES
python combine_data_frames.py $CHUNK_DATA_FILES
# remove all evidence
rm $CHUNK_DATA_FILES