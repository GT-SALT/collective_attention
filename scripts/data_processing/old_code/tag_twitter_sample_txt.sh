# tag sampled Twitter data using twitter_nlp NER system
export TWITTER_NLP=./twitter_nlp/
#DATA_DIR=../../data/mined_tweets/sample_ner_tweets
#CORPORA=$DATA_DIR/*.txt
CORPORA=(../../data/mined_tweets/maria_combined_tweets.txt)
for CORPUS in $CORPORA;
do
    OUT_FILE=${CORPUS/.txt/_ner.txt}
    echo $OUT_FILE
    if [ -f $OUT_FILE ];
    then
        rm $OUT_FILE
    fi
    (python twitter_nlp/python/ner/extractEntities.py $CORPUS --classify --pos -o $OUT_FILE)&
done