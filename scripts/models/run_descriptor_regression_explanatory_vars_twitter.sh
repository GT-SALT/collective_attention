# run regression on all non-temporal explanatory variables (Twitter data)
DATA_FILE=../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor_NE_peak_times_consistent_authors.gz
OUT_DIR=../../output/twitter_data/
if [ ! -d $OUT_DIR ];
then
    mkdir $OUT_DIR
fi
DEP_VAR='anchor'
CAT_VARS=(data_name_fixed NE_fixed)
BINARY_VARS=(has_URL image_video_URL organization is_local)
CLEAN_VARS="('is_local',-1) ('organization',-1)"
SCALAR_VARS=(since_start NE_count_prior)
REGRESSION_TYPE="regularized_logit"
python run_descriptor_regression.py --clean_data $DATA_FILE --out_dir $OUT_DIR --dep_var $DEP_VAR --cat_vars "${CAT_VARS[@]}" --binary_vars "${BINARY_VARS[@]}" --clean_vars $CLEAN_VARS --scalar_vars "${SCALAR_VARS[@]}" --regression_type $REGRESSION_TYPE