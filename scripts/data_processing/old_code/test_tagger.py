import tag_NE_tweets
from tag_NE_tweets import TwitterNLPTaggerWrapper

w = TwitterNLPTaggerWrapper()

test_tokens = 'I want to go to America'.split(' ')
#test_tokens = 'I wanna go to america pls'
test_token_tags = w.tag(test_tokens)
