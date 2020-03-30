# Twitter mined data
ORG_USER_DATA_FILE=../../data/mined_tweets/tweet_user_data/user_data_twitter_mine.gz
# Twitter archive data
# ORG_USER_DATA_FILE=../../data/mined_tweets/tweet_user_data/user_data_archive.gz
# clean input data
python clean_user_data_for_classification.py --user_data_file "$ORG_USER_DATA_FILE"
ORG_DETECT_INPUT=${ORG_USER_DATA_FILE/.gz/_json.txt}
# convert to absolute path because we're changing directories a few times
ORG_DETECT_INPUT=$(readlink -f "$ORG_DETECT_INPUT")
ORG_DETECT_OUTPUT="$ORG_DETECT_INPUT".labelled
# echo $ORG_DETECT_INPUT
# echo $ORG_DETECT_OUTPUT
cd demographer/
python -m demographer.cli.process_tweets --classifier organization --model simple --input $ORG_DETECT_INPUT --output $ORG_DETECT_OUTPUT
# # clean output data
cd ../
python clean_user_data_from_classification.py --user_label_file "$ORG_DETECT_OUTPUT"