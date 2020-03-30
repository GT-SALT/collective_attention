# mine twitter for specified hashtags
# HASHTAGS=("Harvey" "Irma" "HurricaneHarvey" "HurricaneIrma" )
# HASHTAGS=("HurricaneMaria" "Maria" "PuertoRico")
HASHTAGS=("Ophelia" "HurricaneOphelia" "StormOphelia")
DATA_DIR=../../data
AUTH_FILE=$DATA_DIR/auth.csv
OUT_FILE=$DATA_DIR/mined_tweets/crisis_maria_tweets.gz
OUTPUT_DIR=$DATA_DIR/output
OUTPUT=$OUTPUT_DIR/mine_twitter.txt
(python mine_twitter.py "${HASHTAGS[@]}" --auth_file $AUTH_FILE --out_file $OUT_FILE)