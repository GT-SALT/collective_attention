# mine Twitter for old hashtags/location data

## hashtags
# HASHTAGS=("#Irma" "#HurricaneIrma" "#Harvey" "#HurricaneHarvey")
# HASHTAGS=("#Maria" "#HurricaneMaria" "#PuertoRico")
# HASHTAGS=("#Nate" "#HurricaneNate" "#nateupdates" "#NateHurricane")
HASHTAGS=("#Ophelia" "#HurricaneOphelia" "#StormOphelia")

## location
# LOCATION='Miami'
# LOCATION="\"San Juan, Puerto Rico\""
# optional location-bounding region
# WITHIN="100mi"

OUT_DIR=../../data
# OUT_FILE=$OUT_DIR/mined_tweets/hurricane_data_"$START"_"$END".gz
OUTPUT=$OUT_DIR/mine_historical_twitter.txt
OUT_DIR_TWEETS=$OUT_DIR/mined_tweets

# old code: mine over range of dates
# START="2017-09-13"
# END="2017-09-26"
# hashtags and location
# (python mine_historical_twitter.py --hashtags "${HASHTAGS[@]}" --start_date $START --end_date $END --location $LOCATION --out_dir $OUT_DIR > $OUTPUT)&
# only hashtags
# (python mine_historical_twitter.py --hashtags "${HASHTAGS[@]}" --start_date $START --end_date $END --out_dir $OUT_DIR_TWEETS > $OUTPUT)&

# more aggressive strategy: mine historical data per-day since that seems to be less messy than multi-day mining
# iteratively build date list for specified months and start/end days
# Harvey/Irma dates
# MONTHS=(08 09)
# START_DAYS=(20 01)
# END_DAYS=(31 10)

# Maria dates
# MONTHS=(09 10)
# START_DAYS=(19 01)
# END_DAYS=(30 04)

# Ophelia dates
MONTHS=(10)
START_DAYS=(09)
END_DAYS=(18)

YEAR=2017
ALL_DATES=()
for ((i=0; i<"${#MONTHS[@]}"; i++));
do
  MONTH="${MONTHS[$i]}"
  START_DAY="${START_DAYS[$i]}"
  END_DAY="${END_DAYS[$i]}"
  for DAY in $(seq -f '%02g' $START_DAY $END_DAY);
  do
    CURR_DATE="$YEAR-$MONTH-$DAY"
    ALL_DATES+=($CURR_DATE)
  done
done
for ((i=0; i<$(expr "${#ALL_DATES[@]}" - 1); i++));
do
  j=$(expr $i + 1)
  START_DATE="${ALL_DATES[$i]}"
  END_DATE="${ALL_DATES[$j]}"
# hashtags
  (python mine_historical_twitter.py --hashtags "${HASHTAGS[@]}" --start_date $START_DATE --end_date $END_DATE --out_dir $OUT_DIR_TWEETS > $OUTPUT)&  
# location
#   (python mine_historical_twitter.py --location $LOCATION --start_date $START_DATE --end_date $END_DATE --out_dir $OUT_DIR_TWEETS > $OUTPUT)&  
# location with location radius
#   (python mine_historical_twitter.py --location $LOCATION --within $WITHIN --start_date $START_DATE --end_date $END_DATE --out_dir $OUT_DIR_TWEETS > $OUTPUT)&  
done