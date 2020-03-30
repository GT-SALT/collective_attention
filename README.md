# Collective attention

This repository contains code and data associated with the ICWSM 2020 paper: 

I. Stewart, D. Yang, J. Eisenstein. 
2020.
Characterizing Collective Attention via Descriptor Context: A Case Study of Public Discussions of Crisis Events. 
ICWSM.

In this study we investigate how collective attention is reflected in the use of descriptive context information, e.g. if an author writes "San Juan, Puerto Rico" instead of "San Juan" then they may assume that their audience needs extra context to understand the location.
In an analysis of crisis-related posts on Facebook and Twitter, the study finds that intuitive audience-centric factors such as whether a location is "local" can explain the use of descriptor information and validates theories around information status in public discourse (i.e. what it takes for a name to be considered "new" information).

I have done my best to remove all private data from the data and notebooks, but if you find any please notify me immediately.

## How to replicate the paper

The code is split into data processing and analysis.
The workflow proceeds as follows.

### Data Collection

We have to collect the data first!

For the Facebook data:

- obtain authentication credentials (sign up as developer [here](https://developers.facebook.com/docs/facebook-login/access-tokens/))
- identify groups of interest by browsing Facebook, store in `../../data/facebook-maria/location_group_data.tsv
    - format: `region | group ID | group name`
- mine all available posts in Facebook groups: `bash mine_facebook.sh`
- mine all available comments in Facebook group posts: `bash mine_facebook_comments.sh`

For the Twitter data:

- Stream data:
    - obtain authentication credentials (sign up [here](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html))
    - identify hashtags and timeframe of interest
    - stream Twitter data: `bash mine_twitter.sh`
- Mine archive data (from pre-collected 1% sample):
    - identify hashtags, timeframe, languages of interest
    - mine archive data: `bash mine_twitter_archive.sh`
- Scrape historical data:
    - download GetOldTweets [here](https://github.com/Jefferson-Henrique/GetOldTweets-python) to same directory `scripts/data_processing`
    - identify hashtags, timeframe, languages of interest
    - mine historical data: `python mine_old_tweets_GOT.py`
- Collect Twitter author metadata:
    - mine corresponding time in archive: `python mine_user_data_from_twitter_archive.py`
    - mine from Twitter: `python mine_user_data_from_twitter.py`

For the geographic data:
- GeoNames gazetteer:
    - download [here](http://download.geonames.org/export/dump/allCountries.zip)
    - simplify gazetteer: `python simplify_geonames_data.py`
    - get alternate names: `python get_geonames_alternate_names.py`
    - convert to dictionary for faster lookup: `python convert_gazetteer_to_dict.py`
- PR county data (location containment tests):
    - download data: `bash download_PR_county_data.sh`

The compressed social media data required several GB of space. The gazetteer required ~ 1 GB.

### Data processing

Next, we process the data.

1. Tag all named entities (NE).

- (English) download Twitter NER tagger [here](https://github.com/aritter/twitter_nlp) and run setup
- (Spanish) download Stanford CoreNLP suite [here](https://stanfordnlp.github.io/CoreNLP/)
    - before tagging, navigate to CoreNLP folder and set up the tagger server:
        - `java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-spanish.properties -preload tokenize,ssplit,pos,ner,parse -status_port 9003  -port 9003 -timeout 15000`
- tag Facebook data: `python tag_NE_FB_data.py`
- tag Twitter data: `bash tag_NE_tweets.sh`

2. Parse all sentences.

- (English) download spacy [here](https://spacy.io/usage)
- (Spanish) install Google cloud wrapper [here](https://cloud.google.com/python/)
    - before tagging, get API credentials [here](https://console.cloud.google.com/apis/credentials)
- parse Twitter data: `bash parse_twitter_data.sh`
- parse Facebook data:
    - test different parsing techniques: `jupyter-notebook test_parsing_on_fb_data.ipynb`
    - having decided the SyntaxNet works the best, parse all Spanish Facebook data with SyntaxNet: `python parse_FB_data.py`
        - WARNING: make sure you know how much this will cost [here](https://cloud.google.com/natural-language/pricing); I spent about $50 on 30K queries but this may have been suboptimal

3. Validate location NEs based on occurrence within regions of interest.

- validate Twitter data: `python collect_validate_NEs_in_tweets.py`
- validate FB data: `python validate_NEs_in_FB_data.py`

4. Extract descriptor phrases for location NEs:

- extract phrases in Twitter data: `python extract_descriptors_in_twitter_data.py`
- extract phrases in FB data: `python extract_descriptors_in_FB_data.py`
    - determine group containment for locations: `python check_NE_group_containment.py`
- validate descriptor phrases: `jupyter-notebook test_descriptor_detection_on_annotated_twitter_data.ipynb`

5. Get data for Twitter power users.

- identify power users and mine historical tweets for users: `python mine_tweets_for_power_users.py`
- tag: `bash tag_NE_tweets.sh` (modify arguments)
- parse: `bash parse_twitter_data` (modify arguments)
- validate: `python validate_NEs_in_FB_data.py` (modify arguments)

6. Get descriptive stats.

- Stats for Twitter: `jupyter-notebook get_descriptive_stats_twitter.ipynb`
- Stats for Facebook: `jupyter-notebook get_descriptive_stats_FB_data.ipynb`

7. Classify Twitter user metadata, using location string matching and an organization classifier (download [here](https://bitbucket.org/mdredze/demographer/src/peoples2018/) and setup locally).

- Clean data: `python clean_user_data_for_classification.py`
- Organizations: `bash classify_org.sh`
- Locals: `python classify_locals.py`
- Combine: `python combine_author_metadata.py`
- Label/test organizations/locals: `python sample_label_author_metadata.py`

8. Clean data for regression.

- clean Twitter data: `python clean_twitter_descriptor_data_for_regression.py`
- clean Twitter power user data: `python clean_power_user_descriptor_data_for_regression.py`
- clean Twitter regular user data: `python clean_regular_author_descriptor_data_for_regression.py`
- clean Facebook data: `python clean_facebook_descriptor_data_for_regression.py`

### Analysis

1. Plot frequency/descriptor example.

- plot Twitter example: `python generate_example_frequency_context_plot.py`
    - prototyping: `jupyter-notebook plot_frequency_anchoring_information_examples.ipynb`

2. Predict descriptor use (in `models/`).

- Determine optimal L2 weight: `python test_L2_weights_fixed_effect_regression.py`
- Predict descriptors in Twitter:
    - full data, explanatory vars: `bash run_descriptor_regression_explanatory_vars_twitter.sh`
    - full data, time vars: `bash run_descriptor_regression_time_variables.sh`
    - power user data: `bash run_descriptor_regression_active_authors.sh`
    - regular author data: `bash run_descriptor_regression_regular_authors.sh`
- Predict descriptors in Facebook:
    - full data: `bash run_descriptor_regression_explanatory_vars_facebook.sh`

3. (optional) Examine examples of descriptor variation:

- Facebook data:
    - `jupyter-notebook compare_descriptor_probability_in_FB_data.ipynb`
- Twitter data:
    - power user data: `jupyter-notebook compare_descriptor_use_in_power_user_data.ipynb` (under `#Examine-authors-who-increase/decrease-context-use`)

## Old code

### Toponym resolution
This project also investigated the use of unsupervised machine learning to help toponym resolution.
The code and data for that endeavor are included under `toponym_resolution/`.
