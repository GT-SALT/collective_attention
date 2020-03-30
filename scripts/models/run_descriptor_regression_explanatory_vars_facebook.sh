# run regression on all non-temporal explanatory variables (Facebook data)
DATA_FILE=../../data/facebook-maria/combined_group_data_clean_regression.tsv
# DATA_FILE=../../data/facebook-maria/combined_group_data_spacy_clean_regression.tsv
OUT_DIR=../../output/facebook_data/
# OUT_DIR=../../output/facebook_data/spacy/
if [ ! -d $OUT_DIR ];
then
    mkdir -p $OUT_DIR
fi
DEP_VAR='anchor'
CAT_VARS=(NE_fixed_cap status_author_id_cap group_name_cap)
SCALAR_VARS=(NE_count txt_len_norm author_group_count group_size)
BINARY_VARS=(group_contains_NE)
REGRESSION_TYPE="regularized_logit"
python run_descriptor_regression.py --clean_data $DATA_FILE --out_dir $OUT_DIR --dep_var $DEP_VAR --cat_vars "${CAT_VARS[@]}" --binary_vars "${BINARY_VARS[@]}" --clean_vars "" --scalar_vars "${SCALAR_VARS[@]}" --regression_type $REGRESSION_TYPE