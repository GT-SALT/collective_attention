# extract entities from raw text file
# export TWITTER_NLP=./
BASE_DIR=/hg190/istewart6/crisis_language
TWITTER_NLP_DIR=$BASE_DIR/lib/twitter_nlp/
export TWITTER_NLP=$TWITTER_NLP_DIR
DATA_DIR=../../data/mined_tweets
# TXT_FILE=$DATA_DIR/\#Irma,\#Harvey,\#HurricaneIrma,\#HurricaneHarvey_combined_data_txt.tsv
TXT_FILE_ZIP=$DATA_DIR/HurricaneHarvey_ids_rehydrated_clean.txt.gz
TXT_FILE="${TXT_FILE_ZIP/.gz/}"
if [ ! -f $TXT_FILE ] ; then
  gunzip $TXT_FILE_ZIP -c > $TXT_FILE
fi
# OUT_FILE="${TXT_FILE/.tsv/_ner_classified.tsv}"
OUT_FILE="${TXT_FILE/.txt/_ner.txt}"
OUT_FILE_ZIP="${OUT_FILE/.txt/.txt.gz}"
LOG_FILE=../../output/extract_entities.txt
NER_SCRIPT=$TWITTER_NLP_DIR/python/ner/extractEntities.py
# extract entity class too?? no because classes are too noisy (option = --classify)
# extract entities, rezip entities file and delete unzipped text 
(python $NER_SCRIPT $TXT_FILE -o $OUT_FILE > $LOG_FILE && gzip -f $OUT_FILE > $OUT_FILE_ZIP && rm $TXT_FILE)&