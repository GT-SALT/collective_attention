## custom group IDs
#GROUP_IDS=(866250103534243 171400810082867 130913387550000 296720367472342 130913387550000 486819048360070 721270821393500 1979604895658060 866250103534243 891721064308258)
# BAD_IDS=(324112331384312)
## location group IDs
LOCATION_DATA_FILE=../../data/facebook-maria/location_group_data.tsv
GROUP_IDS=$(cut -f 2 $LOCATION_DATA_FILE)
IFS=' ' read -r -a GROUP_IDS <<< $GROUP_IDS
# skip the column string
GROUP_IDS=("${GROUP_IDS[@]:1}")
AUTH_FILE=../../data/facebook_auth.csv
OUT_DIR=../../data/facebook-maria
START_DATE="2017-09-20"
END_DATE="2017-10-20"
for GROUP_ID in "${GROUP_IDS[@]}";
do
  OUT_FILE=../../output/mine_facebook_comments_"$GROUP_ID".txt
  (python mine_facebook_page_comments.py --group_id $GROUP_ID --auth_file $AUTH_FILE --out_dir $OUT_DIR --start_date $START_DATE --end_date $END_DATE > $OUT_FILE)&
done