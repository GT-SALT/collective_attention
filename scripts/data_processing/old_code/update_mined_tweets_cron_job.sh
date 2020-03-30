# default parameters to use in cron job
# instructions here: https://www.cyberciti.biz/faq/how-do-i-add-jobs-to-cron-under-linux-or-unix-oses/
HOUR=0
MIN=0
$MIN $HOUR * * * /hg190/istewart6/crisis_language/scripts/data_processing/update_mined_tweets.sh