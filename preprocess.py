"""
Preprocess the arXiv or WoS corpus.
"""
import argparse
import json
from typing import Optional

import apache_beam as beam
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions

from preprocessing_utils import replace_currency_symbols, replace_numbers, normalize_whitespace, \
    remove_inline_math, normalize_unicode, remove_accents, remove_punctuation
from preprocessing_utils.remove import remove_punctuation_keeping_periods, strip_copyright
from util import read_source


def preprocess(input_path='data/source/arxiv-data-20200125-split*.jsonl.gz') -> pd.DataFrame:
    """Read arXiv data from the disk, then join and preprocess the title and abstract text.

    :param input_path: Path to source data.
    :return: Dataframe including preprocessed text.
    """
    df = pd.DataFrame(read_source(input_path))
    assert df.shape[0]
    # Create columns 'text', 'clean_text', and 'lower_words'
    df['text'] = _join_text(df)
    df['year'] = df['created'].str[:4].astype(int)
    df['clean_text'] = df['text'].apply(clean_text, keep_periods=True)
    df['lower_words'] = df['clean_text'].str.lower()
    return df


def _join_text(df: pd.DataFrame) -> pd.Series:
    assert not (df['title'].str.strip() == '').any()
    assert not (df['abstract'].str.strip() == '').any()
    return df['title'] + '. ' + df['abstract']


def clean_text(value: Optional[str], keep_periods=False) -> Optional[str]:
    """Preprocess text.
    """
    if value is None or pd.isnull(value):
        return None
    value = remove_inline_math(value)
    value = normalize_unicode(value, form='NFKC')
    if keep_periods:
        # We keep periods in SciBERT training
        value = remove_punctuation_keeping_periods(value)
    else:
        value = remove_punctuation(value)
    value = replace_currency_symbols(value, '')
    value = remove_accents(value)
    value = replace_numbers(value, '')
    value = normalize_whitespace(value)
    return value


def beam_runner(input_path, output_path, pipeline_args):
    """Define and run a Beam pipeline for preprocessing corpus text.

    Written for a JSONL export of the WoS corpus.
    """
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        (p | "Load Data" >> beam.io.ReadFromText(input_path)
         | "Read as JSON" >> beam.Map(lambda x: json.loads(x))
         | "Clean" >> beam.Map(lambda x: clean_dict_text(x))
         | "Filter" >> beam.Filter(lambda x: x is not None)
         | "Write Data" >> beam.io.WriteToText(output_path))


def clean_dict_text(record: dict, text_cols=('title', 'abstract')) -> Optional[dict]:
    """Wrap text preprocessing for a source record as read from JSONL.

    Written for a JSONL export of the WoS corpus.
    """
    # Require non-missing text fields
    for col in text_cols:
        if (col not in record) or (record[col] is None) or (len(record[col].strip()) == 0):
            return None
    # Restrict to WoS Core Collection / Dimensions
    if not (record["id"].startswith("WOS:") or record["id"].startswith("pub")):
        return None
    # Unlike in arXiv, copyright notices at the end of abstracts are common in WoS
    record['abstract'] = strip_copyright(record['abstract'])
    for col in text_cols:
        record[col] = record[col].lower()
        record[col] = clean_text(record[col], keep_periods=True)
    record['text'] = ". ".join([record[r] for r in text_cols])
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('--beam', '-b', action='store_true')
    args, pipeline_args = parser.parse_known_args()
    if args.beam:
        beam_runner(args.input_path, args.output_path, pipeline_args)
    else:
        df = preprocess(args.input_path)
        df.to_json(args.output_path, lines=True, orient='records')
