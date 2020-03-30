"""
After copy-pasting the municipality data from Wiki
(here: https://en.wikipedia.org/wiki/Municipalities_of_Puerto_Rico#Demographics),
we need to clean it up to make it .tsv format.
"""
import re
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--municipality_data_file', default='../../data/geo_files/municipality_data/municipality_data.tsv')
    args = parser.parse_args()
    municipality_data_file = args.municipality_data_file
    
    ## fix spaces
    m_lines = [l.strip() for l in open(municipality_data_file, 'r')]
    space_fixer = re.compile('\s{2,}')
    m_lines_fixed = [space_fixer.sub('\t', l) for l in m_lines if l != '']
    # fix first line
    m_cols = ['municipality', 'FIPS', 'population', '%_population', 'area']
    m_lines_fixed[0] = '\t'.join(m_cols)
    municipality_fixed_file = municipality_data_file.replace('.tsv', '_fixed.tsv')
    with open(municipality_fixed_file, 'w') as data_output:
        data_output.write('\n'.join(m_lines_fixed))
    
if __name__ == '__main__':
    main()