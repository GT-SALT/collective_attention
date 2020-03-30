"""
Check (1) if Facebook groups exist,
which is an important consideration due
to the time-sensitive nature of the crisis, and
(2) if the mined comments still exist, which 
is important for replicability.
"""
from argparse import ArgumentParser
import requests
import json
import logging

FB_URL="https://graph.facebook.com/%s?fields=name&access_token=%s"
logging.basicConfig(filename='output/check_facebook_group_exist.txt',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')
def main():
    parser = ArgumentParser()
    parser.add_argument('--group_list_file', default='data/facebook-maria/group_ids.csv')
    parser.add_argument('--facebook_auth_file', default='data/facebook_auth_tmp.csv')
    args = parser.parse_args()
    group_list_file = args.group_list_file
    facebook_auth_file = args.facebook_auth_file

    access_token = [l.strip().split(',')[0] for l in open(facebook_auth_file)][0]

    groups = [l.strip().split(',')[0] for l in open(group_list_file)]
    # try querying each group
    success_ctr = 0
    for g in groups:
        try:
            res = requests.get(FB_URL%(g, access_token))
            res_dict = json.loads(res.text)
            logging.info('have data for group %s: name=%s'%(g, res_dict['name']))
            success_ctr += 1
        except Exception as e:
            logging.info('failed to recover group %s'%(g))
    logging.info('recovered %d/%d groups'%(success_ctr, len(groups)))

if __name__ == '__main__':
    main()
