"""
Using the sampled data (by volume/time),
generate the annotation sheets for each group-time pair
with one sample per line and one cell per potential
toponym and potential PII (1+10+20=31).
This assumes that we've already sampled the data
using sample_binned_posts_for_annotation.py.
"""
import pandas as pd
from httplib2 import Http
from urllib2 import HTTPError
import os
from apiclient import discovery
from apiclient.http import MediaFileUpload
from oauth2client import file, client, tools
from oauth2client.file import Storage
from argparse import ArgumentParser, Namespace
import logging
import re
from time import sleep

def write_bin_groups(data, data_dir, topo_count=10, PII_count=20):
    """
    Group data by bins and write to separate files.
    
    Parameters:
    -----------
    data : pandas.DataFrame
    data_dir : str
    topo_count : int
    Max number of toponyms to annotate.
    PII_count : int
    Max number of PII strings to annotate.
    """
    SPACE_MATCHER = re.compile('\s+')
    for (g_loc, g_time), g_group in data.groupby(['location_name', 'time_bin']):
        out_file = os.path.join(data_dir, '%s_%s.tsv'%(g_loc, g_time))
        if(not os.path.exists(out_file)):
            # just write ID and status
            g_data = g_group.loc[:, ['status_id', 'status_message']]
            # fix status messages
            g_data.loc[:, 'status_message'] = g_data.loc[:, 'status_message'].apply(lambda x: SPACE_MATCHER.sub(' ', x).strip())
            # add cells for toponyms, PII
            new_cols = ['TOPO_%d'%(t+1) for t in range(topo_count)] + ['PII_%d'%(p+1) for p in range(PII_count)] + ['notes']
            g_data = pd.concat([g_data, pd.DataFrame(columns=new_cols)], axis=1).fillna('', inplace=False)
            g_data.to_csv(out_file, sep='\t', index=False, encoding='utf-8')

def configure_creds(cred_file):
    """
    Convert credentials to service connection.
    
    Parameters:
    -----------
    cred_file : str
    
    Returns:
    --------
    service : apiclient.Service
    """
    store = file.Storage(cred_file)
    creds = store.get()
    if not creds or creds.invalid:
        SCOPES = 'https://www.googleapis.com/auth/drive.file'
        no_browser_flag = 'noauth_local_webserver'
        logging_flag = 'logging_level'
        flags = Namespace()
        flags.__setattr__(no_browser_flag, True)
        flags.__setattr__(logging_flag, 'DEBUG')
        flow = client.flow_from_clientsecrets(cred_file, SCOPES)
        creds = tools.run_flow(flow, store, flags)
    service = discovery.build('drive', 'v3', http=creds.authorize(Http()))
    return service

def upload_file(service, file_name, file_mime_type, upload_mime_type, parent_id):
    """
    Upload file to Drive.
    
    Parameters:
    -----------
    service : apiclient.Service
    file_name : str
    file_mime_type : str
    Type of file on disk (e.g. .tsv).
    upload_mime_type : str
    Type of file on Drive (e.g. spreadsheet)
    """
    media_body = MediaFileUpload(file_name, mimetype=file_mime_type, resumable=True)
    uploaded_file_name = os.path.splitext(os.path.basename(file_name))[0] # remove extension and directory
    body = {
        'name' : uploaded_file_name,
        'mime_type' : upload_mime_type,
        'parents' : [parent_id],         
    }
    UPLOAD_DELAY = 15 # delay between uploads to avoid (?) rate-limiting
    success = False
    while(not success):
        try:
            file_request = service.files().create(body=body, media_body=media_body)
            file_request.execute()
            success = True
        except HTTPError, e:
            if(e.code == 403):
                print('rate limit exceeded, sleeping for %d seconds'%(UPLOAD_DELAY))
                sleep(UPLOAD_DELAY)
            else:
                print('failed file upload because\n%s'%(e))
                success = True
        except Exception, e:
            print('failed file upload because\n%s'%(e))
            success = True

def main():
    parser = ArgumentParser()
    parser.add_argument('--google_auth_file', default='../../data/GoogleSheets/client_secret.json')
    parser.add_argument('--sample_data_file', default='../../data/facebook-maria/volume_time_binned_post_sample.tsv')
    args = parser.parse_args()
    google_auth_file = args.google_auth_file
    sample_data_file = args.sample_data_file
    
    ## load data
    sample_data = pd.read_csv(sample_data_file, sep='\t', index_col=False, encoding='utf-8')
    
    ## group by location and time, then write each to file
    sample_data_dir = os.path.dirname(sample_data_file)
    out_dir = os.path.join(sample_data_dir, 'volume_time_binned_data')
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    write_bin_groups(sample_data, out_dir)
    
    ## configure credentials
    cred_storage_file = 'cred_storage.json'
    DRIVE = configure_creds(cred_storage_file)
    
    ## upload all files to Drive
    file_mime_type = 'text/tab-separated-values'
    upload_mime_type = 'application/vnd.google-apps.spreadsheet'
#     parent_id = 'annotations'
    parent_id = '1_SqrvrP595UoFMLyH6WPuaOkzNPpwXvr' # manually copied from directory URL
    g_files = [os.path.join(out_dir, g_file) for g_file in os.listdir(out_dir)]
    for g_file in g_files:
        upload_file(DRIVE, g_file, file_mime_type, upload_mime_type, parent_id)
            
if __name__ == '__main__':
    main()