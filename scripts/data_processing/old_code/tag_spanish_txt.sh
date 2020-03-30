# need to add java to path and NER to classpath
CURR_DIR=$(pwd .)
PATH=$PATH:/usr/lib/jvm/java-1.8.0-ibm-1.8.0.4.5-1jpp.1.el7_3.x86_64/
# need to add directory storing Stanford files => class path
CLASSPATH=$CLASSPATH:$CURR_DIR
#DATA_DIR=../../../data
TXT_FILE=$1
# add one more level because we're working in the classifier folder
TXT_FILE=../$TXT_FILE
#TXT_FILE=$DATA_DIR/facebook-maria/combined_group_statuses.txt
#TXT_FILE=$DATA_DIR/mined_tweets/maria_combined_tweets_es.txt
NER_OUTPUT_FILE="${TXT_FILE/.txt/_ner.txt}"

# run NER
cd stanford-ner-2017-06-09
# java -mx600m -cp "*:lib\*" edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier classifiers/english.all.3class.distsim.crf.ser.gz -textFile sample.txt
java -mx900m -cp "*:lib\*" edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier classifiers/spanish.ancora.distsim.s512.crf.ser.gz -textFile $TXT_FILE > $NER_OUTPUT_FILE