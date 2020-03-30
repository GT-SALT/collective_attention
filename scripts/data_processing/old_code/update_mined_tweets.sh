# move the new mined tweets
# to a separate file
#DATA_DIR=../../data/mined_tweets
# need absolute file path for crontab
DATA_DIR=/hg190/istewart6/crisis_language/data/mined_tweets
CURRENT_TIME=$(date +'%b-%d-%g-%H-%M')
echo $CURRENT_TIME
TWEET_FILE_NAME=crisis_maria_tweets.gz
OLD_TWEET_FILE=$DATA_DIR/$TWEET_FILE_NAME
NEW_TWEET_FILE=${TWEET_FILE_NAME/.gz/_"$CURRENT_TIME".gz}
NEW_TWEET_FILE=$DATA_DIR/$NEW_TWEET_FILE
echo $NEW_TWEET_FILE
# move to new file
mv $OLD_TWEET_FILE $NEW_TWEET_FILE