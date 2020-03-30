"""
Mine historical tweets that match geotags.
"""
import gzip
import json
import os
from dateparser import parse
import re
from argparse import ArgumentParser
from math import ceil
from multiprocessing import Pool, Lock, Manager, Process, Queue
import itertools
import logging
import time

## convert state abbreviation to full name
STATE_LOOKUP = {
    'FL':'Florida',
    'TX':'Texas',
}
ABBREV_LOOKUP = {
    v : k for k,v in STATE_LOOKUP.items()
}

class FileCounter:
    def __init__(self):
        self.ctr = 0
    def up(self):
        self.ctr += 1

LOG_FILE='../../output/mine_historical_geotagged_tweets.txt'
if(os.path.exists(LOG_FILE)):
    os.remove(LOG_FILE)
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format='%(asctime)-15s: %(message)s')

def file_listener(q, out_filename):
    """
    Listen for message from queue then write to file.
    
    :param q: Queue for multiprocessing
    :param out_filename: output file name
    """
    out_file = gzip.open(out_filename, 'w')
    while True:
        try:
            file_type, file_message = q.get(block=True)
            if(file_message == 'kill'):
                logging.debug('DONE')
                out_file.close()
                break
            if(file_type == 'logging'):
                logging.debug(file_message)
            else:
                file_message_str = ('%s\n'%(json.dumps(file_message))).encode('utf-8')
    #             print(type(file_message_str))
                out_file.write(file_message_str)
                out_file.flush()
        except Exception as e:
            logging.debug('handling error in queue %s'%(e))
            pass
#     open_file.close()

def mine_tweet_files(data_files, mine_states, FILE_CTR, q, use_lock=True, QUEUE_DELAY=0.1):
    """
    Mine data files for specified states.
    
    :param data_files: list of data files
    :param mine_states: allowed states for geocoded tweet matching
    :param mine_state_filenames: output filename
    :param use_lock: use lock for multiprocessing
    """
#     mine_state_files = {s : gzip.open(f, 'a') for s, f in zip(mine_states, mine_state_filenames)}
    mine_state_names = [STATE_LOOKUP[s] for s in mine_states]
    mine_state_matcher = re.compile('|'.join(['%s$'%(x) for x in mine_states+mine_state_names]))
    tweet_keys = {'created_at', 'id', 'text', 'place'}
    tweet_ctr = 0
    write_ctr = 0
    all_results = []
    for f in data_files:
#         if(use_lock):
#             lock.acquire()
#         logging_msg = 'processing file %s'%(os.path.basename(f))
#         q.put('logging', logging_msg)
#         logging.debug('processing file %s'%(os.path.basename(f)))
#         if(use_lock):
#             lock.release()
        for i, l in enumerate(gzip.open(f, 'r')):
            try:
                j = json.loads(l.strip())
                if(j.get('place') is not None and j['place']['country_code']=='US' and j['lang'] == 'en'):
                    j_place_name = j['place']['full_name'].replace(', USA', '')
                    j_place_search = mine_state_matcher.search(j_place_name)
    #                 j_state = j_place_name.split(', ')[-1].strip()
                    if(j_place_search is not None):
                        j_state = j_place_search.group(0)
                        # normalize
                        if(j_state not in mine_states):
                            j_state = ABBREV_LOOKUP[j_state]
                        j_info = {k : j[k] for k in tweet_keys}
                        j_info['user_id'] = j['user']['id']
                        j_info['state'] = j_state
#                         j_str = '%s\n'%(json.dumps(j_info))
                        # lock before writing
#                         if(use_lock):
#                             lock.acquire()
                        q.put(('output', j_info))
                        time.sleep(QUEUE_DELAY)
#                         mine_state_files[j_state].write(j_str.encode())
#                         if(use_lock):
#                             lock.release()
                        write_ctr += 1
#                         all_results.append([j_state, j_info])        
#                         yield j_state, j_info
#                         print(j_info)
            except Exception as e:
#                 yield None
#                 print(e)
                pass
            tweet_ctr += 1
# #             if(tweet_ctr > 5000):
# #                 break
#             if(tweet_ctr % 100 == 0):
#                 print(tweet_ctr)
            if(tweet_ctr % 1000000 == 0):
                logging_msg = 'written %d/%d tweets for file %s'%(write_ctr, tweet_ctr, f)
                q.put(('logging', logging_msg))
#                 logging.debug(logging_msg)
#                 if(use_lock):
#                     lock.release()
#                 print('written %d/%d tweets'%(write_ctr, tweet_ctr))
        
#         if(use_lock):
#             lock.acquire()
        FILE_CTR.up()
#         if(use_lock):
#             lock.release()
        logging_msg = 'finished %s => %d files finished'%(f, FILE_CTR.ctr)
        q.put(('logging', logging_msg))
#         logging.debug(logging_msg)
#     return all_results

def mine_tweet_files_star(args):
#     print('processing %d args'%(len(args)))
    mine_tweet_files(*args)

# lock c/o https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes
def init_lock(l):
    global lock
    lock = l

def main():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='../../data/mined_tweets')
    parser.add_argument('--mine_states', default=['TX','FL'])
    parser.add_argument('--data_dir', default='/hg190/corpora/twitter-crawl/new-archive')
    parser.add_argument('--start_date', default='Aug 01 2016')
    parser.add_argument('--end_date', default='Aug 01 2017')
    args = parser.parse_args()
    out_dir = args.out_dir
    mine_states = args.mine_states
    data_dir = args.data_dir
    start_date = args.start_date
    end_date = args.end_date

    ## get relevant tweet files
    start_date_time = parse(start_date)
    end_date_time = parse(end_date)
    date_matcher = re.compile('(?<=tweets\-)([A-Za-z]+-[0-9]{2}-1[0-9])')
    # filter files
    def check_file(x, start_date, end_date, date_matcher):
        if(date_matcher.search(x) is not None):
            x_date = parse(date_matcher.search(x).group(0))
            return x_date >= start_date and x_date <= end_date
        else:
            return False
    data_files = list(filter(lambda x: check_file(x, start_date_time, end_date_time, date_matcher), os.listdir(data_dir)))
    # sort by date
    data_files = sorted(data_files, key=lambda x: parse(date_matcher.search(x).group(0)))
    data_files = list(map(lambda x: os.path.join(data_dir, x), data_files))
    print('%d valid data archive files'%(len(data_files)))

    ## mine!!
#     mine_state_filenames = [os.path.join(out_dir, '%s_%s_%s.gz'%(start_date.replace(' ','-'), end_date.replace(' ', '-'), x)) for x in mine_states]
    # write to shared file for easy processing
    combined_output_filename = os.path.join(out_dir, '%s_%s_%s.gz'%(start_date.replace(' ','-'), end_date.replace(' ','-'), '_'.join(mine_states)))
    # remove existing files
    if(os.path.exists(combined_output_filename)):
        os.remove(combined_output_filename)
#     for f in mine_state_filenames:
#         if(os.path.exists(f)):
#             os.remove(f)
#     mine_state_files = {s : gzip.open(f, 'w') for s, f in zip(mine_states, mine_state_filenames)}
    # debugging = use fewer files
#     data_files = data_files[:10]
    file_chunks = 10
    file_chunk_size = int(ceil(len(data_files) / file_chunks))
    data_file_chunks = [data_files[(i*file_chunk_size):((i+1)*file_chunk_size)] for i in range(file_chunks)]
#     use_lock = True
#     file_lock = Lock()
    FILE_CTR = FileCounter()
    manager = Manager()
#     pool = Pool(processes=file_chunks)
#     pool = Pool(processes=file_chunks, initializer=init_lock, initargs=(file_lock,))
    # start queue for files
#     q = Queue()
    q = manager.Queue()
    writer_process = Process(target=file_listener, args=(q, combined_output_filename))
    writer_process.start()
#     file_watcher = pool.map_async(file_listener, (q,combined_output_filename))
#     q.put(('testing', 'testing 123'))
    
    ## queue code
    ## separate jobs, async
    # then start jobs
#     jobs = []
#     for i, data_file_chunk in enumerate(data_file_chunks):
#         print('starting chunk %d'%(i))
# #         print('%d files'%(len(data_file_chunk)))
#         #mine_tweet_files(data_files, mine_states, mine_state_filenames, FILE_CTR, q, use_lock=True)
#         job = Process(target=mine_tweet_files, args=(data_file_chunk, mine_states, FILE_CTR, q))
#         job.start()
#         time.sleep(0.1)
#         jobs.append(job)
#     for job in jobs:
#         print('job joining')
#         job.join()
#         job = pool.apply_async(mine_tweet_files, (data_file_chunk, mine_states, FILE_CTR, q, use_lock))
#         jobs.append(job)
#     for job in jobs:
#         job.get()
    # combined map
    rep_arg = itertools.repeat
    pool = Pool(processes=file_chunks)
    pool.map(mine_tweet_files_star, zip(data_file_chunks, rep_arg(mine_states), rep_arg(FILE_CTR), rep_arg(q)))

    q.put('', 'kill')
    writer_process.join()

    ## combined map
#     rep_arg = itertools.repeat
#     pool.map(mine_tweet_files_star, zip(data_file_chunks, rep_arg(mine_states), rep_arg(mine_state_filenames), rep_arg(FILE_CTR), rep_arg(q), rep_arg(use_lock)))
#     pool.close()
#     pool.join()
#     q.put('', 'kill')
    
    # finish writing
#     mine_state_files = {s : f.close() for s,f in mine_state_files.iteritems()}

    ## write each result one at a time
#     rep_arg = itertools.repeat
#     mine_state_files = {s : gzip.open(f, 'w') for s, f in zip(mine_states, mine_state_filenames)}
#     i_ctr = 0
# #     for results in pool.map(mine_tweet_files_star, zip(data_files, rep_arg(mine_states), rep_arg(mine_state_filenames), rep_arg(FILE_CTR), rep_arg(use_lock))):
#     for results in pool.starmap(mine_tweet_files, zip(data_file_chunks, rep_arg(mine_states), rep_arg(mine_state_filenames), rep_arg(FILE_CTR), rep_arg(use_lock))):
#         for result in results:
# #         if(result is not None):
#             i_state, i_data = result
#             mine_state_files[i_state].write('%s\n'%(json.dumps(i_data)))
#             i_ctr += 1
# #         FILE_CTR.up()
# #         if(FILE)
#     print(i_ctr)
    
if __name__ == '__main__':
    main()
