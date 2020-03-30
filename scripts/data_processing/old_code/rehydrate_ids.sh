# rehydrate old tweet IDs
DATA_DIR=../../data
ID_FILE=$DATA_DIR/mined_tweets/HurricaneHarvey_ids.txt.gz
AUTH_FILE=$DATA_DIR/auth.csv
OUT_FILE=${ID_FILE/.txt.gz/_rehydrated.json.gz}
echo $OUT_FILE
OUTPUT_FILE=../../output/rehydrate_ids.txt
(python rehydrate_ids.py $ID_FILE $AUTH_FILE $OUT_FILE > $OUTPUT_FILE)&