CONTEXT_DATA=../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor.gz
DATA_NAMES_TO_PLOT=('maria' 'florence')
NES_TO_PLOT=('san juan' 'myrtle beach')
OUT_DIR=../../output

python plot_frequency_descriptor_information_examples.py $CONTEXT_DATA --data_names_to_plot "${DATA_NAMES_TO_PLOT[@]}" --NEs_to_plot "${NES_TO_PLOT[@]}" --out_dir $OUT_DIR