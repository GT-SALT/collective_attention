# kill process by name
#PROCESS="kernel-8df385b2"
#PROCESS="CRFClassifier"
# PROCESS="python mine_facebook_page_comments.py"
#PROCESS="python mine_facebook.py"
#PROCESS="python get_geonames_wiki_backlink_count.py"
#PROCESS="python twitter_nlp/python/ner/extractEntities.py"
#PROCESS="python test_multiprocessing.py"
#PROCESS="python optimize_VAE_hyperparameters.py"
#PROCESS="python extract_loc_text_data_from_ner_sample.py"
#PROCESS="python train_GPS_vae.py"
#PROCESS="mine_historical_geotagged_tweets.py"
#PROCESS="generate_toponym_candidates_simple.*"
PROCESS="mine_twitter_archive.py"
PROCESSES=$(ps aux | grep "$PROCESS" | awk '{print $2}')
echo $PROCESSES
kill $PROCESSES
