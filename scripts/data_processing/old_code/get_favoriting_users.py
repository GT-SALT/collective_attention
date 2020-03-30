import urllib2
from lxml.html import parse

#returns list(retweet users),list(favorite users) for a given screen_name and status_id
def get_twitter_user_rts_and_favs(screen_name, status_id):
    url = urllib2.urlopen('https://twitter.com/' + screen_name + '/status/' + status_id)
    root = parse(url).getroot()

    num_rts = 0
    num_favs = 0
    rt_users = []
    fav_users = []

    for ul in root.find_class('stats'):
        for li in ul.cssselect('li'):

            cls_name = li.attrib['class']

            if cls_name.find('retweet') >= 0:
                num_rts = int(li.cssselect('a')[0].attrib['data-tweet-stat-count'])

            elif cls_name.find('favorit') >= 0:
                num_favs = int(li.cssselect('a')[0].attrib['data-tweet-stat-count'])

            elif cls_name.find('avatar') >= 0 or cls_name.find('face-pile') >= 0:#else face-plant

                for users in li.cssselect('a'):
                    #apparently, favs are listed before retweets, but the retweet summary's listed before the fav summary
                    #if in doubt you can take the difference of returned uids here with retweet uids from the official api
                    if num_favs > 0:#num_rt > 0:
                        #num_rts -= 1
                        num_favs -= 1
                        #rt_users.append(users.attrib['data-user-id'])
                        fav_users.append(users.attrib['data-user-id'])
                    else:                        
                        #fav_users.append(users.attrib['data-user-id'])
                        rt_users.append(users.attrib['data-user-id'])

        return rt_users, fav_users


#example
if __name__ == '__main__':
    attempt_ctr = 1000
    for i in range(attempt_ctr):
        get_twitter_user_rts_and_favs('alien_merchant', '674104400013578240')
    print(get_twitter_user_rts_and_favs('alien_merchant', '674104400013578240'))
