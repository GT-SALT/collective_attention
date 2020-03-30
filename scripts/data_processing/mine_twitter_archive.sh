# mine Twitter archive for specified hashtags or locations
# phrases in file
# PHRASE_FILE=../../data/hurricane_data/location_phrases_florence.txt
# PHRASE_FILE=../../data/hurricane_data/location_phrases_harvey.txt
# PHRASE_FILE=../../data/hurricane_data/location_phrases_irma.txt
# PHRASE_FILE=../../data/hurricane_data/location_phrases_maria.txt
PHRASE_FILE=../../data/hurricane_data/location_phrases_michael.txt
# comma-separated phrases
# PHRASES="#Maria,#HurricaneMaria,#HuracanMaria,#PuertoRico"
# PHRASES='"#HurricaneMichael,#Michael"'
# PHRASES='"#HurricaneFlorence,#Florence"'
# PHRASES='""'
# location: lat1 lat2 lon1 lon2
# LOCATION_BOX=(24 26 70 71)
DATA_DIR=/hg190/corpora/twitter-crawl/new-archive
OUT_DIR=../../data
OUTPUT=$OUT_DIR/mine_twitter_archive.txt
if [ -f $OUTPUT ]; then
  rm $OUTPUT
fi
OUT_DIR_TWEETS=$OUT_DIR/mined_tweets
USER_LOC_PHRASE='""'
#USER_LOC_PHRASE='", PR|, Puerto Rico|^PR$|^Puerto Rico$"'
#USER_LOC_PHRASE=", TX|, Texas|^TX$|^Texas$"
# USER_LOC_PHRASE='", FL|, Florida|^FL$|^Florida$"'

## iteratively collect archive files
# we do this because dateseq doesn't work for month names
ARCHIVE_FILES=()
# Florence dates
# MONTHS=(Aug Sep)
# START_DAYS=(30 1)
# END_DAYS=(31 26)
# YEAR=18
# Harvey dates
# MONTHS=(Aug Sep)
# START_DAYS=(15 1)
# END_DAYS=(31 9)
# YEAR=17
# Irma dates
# MONTHS=(Aug Sep)
# START_DAYS=(30 1)
# END_DAYS=(31 20)
# YEAR=17
# Maria dates
# MONTHS=(Sep Oct)
# START_DAYS=(18 1)
# END_DAYS=(30 31)
# YEAR=17
# Michael dates
MONTHS=(Oct)
START_DAYS=(7)
END_DAYS=(23)
YEAR=18

MONTH_COUNT=$(expr "${#MONTHS[@]}" - 1)
for i in $(seq 0 "$MONTH_COUNT");
do
  MONTH="${MONTHS[$i]}"
  START_DAY="${START_DAYS[$i]}"
  END_DAY="${END_DAYS[$i]}"
  for DAY in $(seq -f "%02g" "$START_DAY" "$END_DAY");
  do
    RELEVANT_FILE=$DATA_DIR/tweets-"$MONTH"-"$DAY"-"$YEAR"-[0-9][0-9]-[0-9][0-9].gz
    ARCHIVE_FILES+=($RELEVANT_FILE)
  done
done
# echo "archive files:${ARCHIVE_FILES[@]}"

## mine files
# hashtags and location
# (python mine_twitter_archive.py "${ARCHIVE_FILES[@]}" --hashtags "${PHRASES[@]}" --location $LOCATION --out_dir $OUT_DIR > $OUTPUT)&
# only hashtag
# NOT GOOD! can overload system with simultaneous queries
#for ARCHIVE_FILE in "${ARCHIVE_FILES[@]}";
#do
#  echo $ARCHIVE_FILE

  # (python mine_twitter_archive.py $ARCHIVE_FILE --phrases "${PHRASES[@]}" --out_dir $OUT_DIR_TWEETS >> $OUTPUT)&
#done

# mine safely in parallel
JOBS=10
# parallel --jobs $JOBS --bar --verbose python mine_twitter_archive.py {} --phrases $PHRASES --user_loc_phrase $USER_LOC_PHRASE --out_dir $OUT_DIR_TWEETS ::: $(ls "${ARCHIVE_FILES[@]}")
parallel --jobs $JOBS --bar --verbose python mine_twitter_archive.py {} --phrase_file $PHRASE_FILE --user_loc_phrase $USER_LOC_PHRASE --out_dir $OUT_DIR_TWEETS ::: $(ls "${ARCHIVE_FILES[@]}")
# ls "${ARCHIVE_FILES[@]}" | parallel --dryrun --jobs $JOBS --bar python "mine_twitter_archive.py --archive_files {} --phrases ${PHRASES[@]} --out_dir $OUT_DIR_TWEETS" ::: "${ARCHIVE_FILES[@]}"

## combine files after collecting
## only works when using PHRASE_FILE...yikes
PHRASE_FILE_BASE=$(basename $PHRASE_FILE | sed 's/.txt//g')
COMBINED_OUT_FILE=archive_"$PHRASE_FILE_BASE".gz
COMBINED_OUT_FILE=$OUT_DIR_TWEETS/$COMBINED_OUT_FILE
if [ -f $COMBINED_OUT_FILE ]; then
    rm $COMBINED_OUT_FILE
fi
MINED_FILES=$(ls $OUT_DIR_TWEETS/archive_$PHRASE_FILE_BASE*.gz)
zcat $MINED_FILES | gzip > $COMBINED_OUT_FILE
rm $MINED_FILES

# (python mine_twitter_archive.py "${ARCHIVE_FILES[@]}" --phrases "${PHRASES[@]}" --out_dir $OUT_DIR_TWEETS > $OUTPUT)&
# only location
# (python mine_twitter_archive.py "${ARCHIVE_FILES[@]}" --out_dir $OUT_DIR_TWEETS --location $LOCATION > $OUTPUT)&