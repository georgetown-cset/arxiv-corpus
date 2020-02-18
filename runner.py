"""
Preprocess the source data; label and partition it; and write out various train/dev/test sets.
"""
import argparse

import pandas as pd

from assign_labels import label
from preprocess import preprocess
from util import TODAY
from write_partitions import write


def main(input_path=None, read_processed=False, write_multi=False, date=TODAY):
    if read_processed:
        df = pd.read_pickle(input_path)
    else:
        df = preprocess(input_path)
        df.to_pickle(f'data/processed/paper_data_{date}.pickle.gz')
        df.to_json(f'data/processed/paper_data_{date}.jsonl', lines=True, orient='records')
    df = label(df)
    write(df, multi=write_multi, today=date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess the source data; label and partition it; and write out various train/dev/test sets. '
                    'For example:\n'
                    '$ python runner.py "data/source/arxiv-data-20200125-split*.jsonl.gz".')
    parser.add_argument('input_path', default='data/source/arxiv-data-20200125-split*.jsonl.gz',
                        help='Path to source data, or if --read_processed then path to processed data (e.g., '
                             '"data/processed/paper_data_202002125.pickle.gz")')
    parser.add_argument('--read_processed', '-r', action='store_true',
                        help='Read processed data from disk and skip preprocessing')
    parser.add_argument('--multi', '-m', action='store_true',
                        help='Also write multiclass and multilabel data')
    parser.add_argument('--date', '-d', default=TODAY,
                        help='ISO date to use in output filenames')
    args = parser.parse_args()
    main(input_path=args.input_path, read_processed=args.read_processed, write_multi=args.multi, date=args.date)
