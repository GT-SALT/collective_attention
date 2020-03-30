"""
Combine data frames.
Assume that files are in FILE_NAME_[0-9]+.gz format
and have no index.
"""
import pandas as pd
from argparse import ArgumentParser
import re

def combine_write_data(data_files, sep='\t'):
    """
    Combine data files and write
    to combined output.
    
    :param data_files: list of data files
    :param sep: data separator
    """
    ## combine
    if(data_files[0].endswith('gz')):
        compression = 'gzip'
    else:
        compression = None
    data = pd.concat([pd.read_csv(f, sep=sep, compression=compression, index_col=False) for f in data_files], axis=0)
    
    ## save
    NUM_REPLACER = re.compile('_[0-9]+(?=\.\w+$)')
    out_file_name = NUM_REPLACER.sub('', data_files[0])
    print('writing combined data to %s'%(out_file_name))
    data.to_csv(out_file_name, sep='\t', compression=compression, index=False)

def main():
    parser = ArgumentParser()
    parser.add_argument('data_files', nargs='+')
    parser.add_argument('--sep', default='\t')
    args = vars(parser.parse_args())
    data_files = args['data_files']
    sep = args['sep']
    combine_write_data(data_files, sep)
    
if __name__ == '__main__':
    main()