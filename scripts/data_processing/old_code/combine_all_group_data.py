"""
Combine all group data into single data frame
with text, group ID, language (identified with langid),
status time and author ID.
"""
from data_helpers import get_all_group_data
from langid import classify
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--out_file_name', default='../../data/facebook-maria/combined_group_data.tsv')
    args = parser.parse_args()
    out_file_name = args.out_file_name

    ## load, process data
    group_df = get_all_group_data()
    group_df.dropna(subset=['status_message'], inplace=True)
    lang_list = group_df.loc[:, 'status_message'].apply(lambda x: classify(x)[0])
    group_df.loc[:, 'status_lang'] = lang_list
    group_df.loc[:, 'group_id'] = group_df.loc[:, 'group_id'].astype(int)

    ## write to file
    var_names = ['group_id', 'status_author_id', 'status_message', 'status_lang', 'status_id', 'status_published']
    group_df_relevant = group_df.loc[:, var_names]
    group_df_relevant.to_csv(out_file_name, sep='\t', index=False, float_format='%d')

if __name__ == '__main__':
    main()
    
