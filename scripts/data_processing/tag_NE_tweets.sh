# tag NE tweets
DATA_DIR=../../data/mined_tweets
# spanish
# ES_TWEET_FILES=($DATA_DIR/stream_maria.gz $DATA_DIR/archive_maria.gz $DATA_DIR/historical_maria.gz)
# ES_TWEET_FILES=($DATA_DIR/archive_maria_location_phrases_combined.gz)
# english
# EN_TWEET_FILES=($DATA_DIR/stream_harvey.gz $DATA_DIR/archive_harvey.gz $DATA_DIR/historical_harvey.gz $DATA_DIR/stream_irma.gz $DATA_DIR/archive_irma.gz $DATA_DIR/historical_irma.gz $DATA_DIR/east_coast_geo_twitter_2018/geo_stream_florence.gz $DATA_DIR/archive_florence.gz $DATA_DIR/historical_florence.gz $DATA_DIR/east_coast_geo_twitter_2018/geo_stream_michael.gz $DATA_DIR/archive_michael.gz $DATA_DIR/historical_michael.gz)
# EN_TWEET_FILES=($DATA_DIR/archive_florence_location_phrases_combined.gz $DATA_DIR/archive_harvey_location_phrases_combined.gz $DATA_DIR/archive_irma_location_phrases_combined.gz $DATA_DIR/archive_michael_location_phrases_combined.gz)
# power users
TWEET_FILES=($DATA_DIR/combined_data_power_user_florence.gz $DATA_DIR/combined_data_power_user_harvey.gz $DATA_DIR/combined_data_power_user_irma.gz $DATA_DIR/combined_data_power_user_maria.gz $DATA_DIR/combined_data_power_user_michael.gz)

## run in parallel
JOBS=1
# ls "${ES_TWEET_FILES[@]}" | xargs echo
# ls "${EN_TWEET_FILES[@]}" | xargs echo
ls "${TWEET_FILES[@]}" | xargs echo
# parallel --jobs $JOBS --bar --verbose python tag_NE_tweets.py --tweet_file {} ::: "${ES_TWEET_FILES[@]}"
# parallel --jobs $JOBS --bar --verbose python tag_NE_tweets.py --tweet_file {} ::: "${EN_TWEET_FILES[@]}"
parallel --jobs $JOBS --bar --verbose python tag_NE_tweets.py --tweet_file {} ::: "${TWEET_FILES[@]}"

## run in serial??
# for ES_TWEET_FILE in "${ES_TWEET_FILES[@]}"; do
#     echo $ES_TWEET_FILE
#     (python tag_NE_tweets.py --tweet_file $ES_TWEET_FILE)&
# done