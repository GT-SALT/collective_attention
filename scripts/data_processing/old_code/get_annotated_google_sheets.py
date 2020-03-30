"""
Get data from an annotator's Google sheets (hosted at gatech.spanish.study@gmail.com).
THIS SHOULDN'T BE SO HARD.
"""
# import httplib2
import os
# from apiclient import discovery
# from oauth2client import client
# from oauth2client import tools
# from oauth2client.file import Storage
from oauth2client.service_account import ServiceAccountCredentials
# import codecs
from argparse import ArgumentParser
import gspread
import pandas as pd
import re

## old code for opening credentials from secret client key
# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/sheets.googleapis.com-python-quickstart.json
# SCOPES = 'https://www.googleapis.com/auth/spreadsheets.readonly'
# CLIENT_SECRET_FILE = '../../data/GoogleSheets/client_secret.json'
# CLIENT_SECRET_FILE = '../../data/GoogleSheets/'

# def get_credentials():
#     """Gets valid user credentials from storage.

#     If nothing has been stored, or if the stored credentials are invalid,
#     the OAuth2 flow is completed to obtain the new credentials.

#     Returns:
#         Credentials, the obtained credential.
#     """
#     credential_dir = os.path.dirname(CLIENT_SECRET_FILE)
#     credential_path = os.path.join(credential_dir,
#                                    'sheets.googleapis.com-python-quickstart.json')

#     store = Storage(credential_path)
#     credentials = store.get()
#     if not credentials or credentials.invalid:
#         flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
#         flow.user_agent = APPLICATION_NAME
#         if flags:
#             credentials = tools.run_flow(flow, store, flags)
#         else: # Needed only for compatibility with Python 2.6
#             credentials = tools.run(flow, store)
#         print('Storing credentials to ' + credential_path)
#     return credentials

def main():
    """
    Extract all lines from sheet and save to .csv file.
    """
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='../../data/facebook-maria/annotations/annotator_1')
    parser.add_argument('--service_key_file', default='../../data/GoogleSheets/service_client_creds.json') # download client credentials here: https://console.developers.google.com/apis/credentials/serviceaccountkey
    args = parser.parse_args()
    out_dir = args.out_dir
    service_key_file = args.service_key_file

    ## load client
    scope = ['https://spreadsheets.google.com/feeds']
    creds = ServiceAccountCredentials.from_json_keyfile_name(service_key_file, scope)
    client = gspread.authorize(creds)
    ## NOTE: in order for the spreadsheets to be accessible,
    ## you need to manually share the spreadsheets with the 
    ## client email specified in service_key_file
    ## there may be an automated way of doing this...TBD
    
## old form response code
    
#     credentials = get_credentials()
#     http = credentials.authorize(httplib2.Http())
#     discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
#                     'version=v4')
#     service = discovery.build('sheets', 'v4', http=http,
#                               discoveryServiceUrl=discoveryUrl)

#     spreadsheetId = '1897f7cqzDOraJ9n8eTv2Za1XmqfKVyF7ScIeAez430k' #annotator 1 form response

#     out_file = '../../data/facebook-maria/all_group_sample_statuses_annotated_topo_annotator_1.txt'
    
#     rangeName = 'Form Responses 1!C2:V2'
#     rangeName = 'Form Responses 1!A1:A100'

    spreadsheet_ids = ['1qvKn7Kj4mDyLXebiedoqYHxDWxsyBHNjI10makqHH74', # Guayama_Early 
                       '1KxoYmHz6LMK8UZSmTFkKN2Cpwzm7H45u4aq9cgo7yc8', # Guayama_Mid
                       '1qTv-UQaAkdyG0oVjhQwV3ZLZ5Q51fA97EkT0bHnOc3I', # Guayama_Late
                       ]
    out_file_bases = ['Guayama_Early', 'Guayama_Mid', 'Guayama_Late']
    
    ## download all spreadsheets
    ## assume one worksheet per sheet
    ID_matcher = re.compile('(?<=\()([0-9]+|VAGUE|UN)(?=\))')
    source_matcher = re.compile('(?<=\()[OG]')
    paren_matcher = re.compile('\(\w+\)$')
    def extract_id(x, ID_matcher, null_id=-1):
        x_id = ID_matcher.findall(x)
        if(len(x_id) > 0):
            x_id = x_id[0]
        else:
            x_id = null_id
        return x_id
    for i in range(len(spreadsheet_ids)):
        spreadsheet_id = spreadsheet_ids[i]
        out_file_base = out_file_bases[i]
        print('processing %s'%(spreadsheet_id))
        out_file = os.path.join(out_dir, '%s_annotated.tsv'%(out_file_base))
        sheet = client.open_by_key(spreadsheet_id)
        sheet_data = sheet.worksheets()[0].get_all_values()
        sheet_data_df = pd.DataFrame(sheet_data[1:], columns=sheet_data[0])
        # clean the data: separate TOPO, ID and source
        topo_cols = list(filter(lambda x: 'TOPO' in x, sheet_data_df.columns))
        for t in topo_cols:
            t_ids = sheet_data_df.loc[:, t].apply(lambda x: extract_id(x, ID_matcher, null_id=-1))
            t_sources = sheet_data_df.loc[:, t].apply(lambda x: extract_id(x, source_matcher, null_id='UNK'))
            sheet_data_df.loc[:, '%s_id'%(t)] = t_ids
            sheet_data_df.loc[:, '%s_source'%(t)] = t_sources
            sheet_data_df.loc[:, t] = sheet_data_df.loc[:, t].apply(lambda x: paren_matcher.sub('', x))
        # write to file
        sheet_data_df.to_csv(out_file, sep='\t', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
