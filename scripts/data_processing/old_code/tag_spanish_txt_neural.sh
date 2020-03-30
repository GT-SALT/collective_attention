# tag Spanish text with neural model downloaded from here: https://github.com/glample/tagger
DATA_DIR=../../data/facebook-maria/
TXT=$DATA_DIR/combined_group_statuses.txt
TXT_PATH=$(readlink -e $TXT)
python neural_tagger/tagger.py --model neural_tagger/models/