"""
IO utilities.
"""
import gzip
import json
from datetime import date
from pathlib import Path

from sklearn.model_selection import train_test_split
from srsly import read_jsonl, write_jsonl
from srsly.util import force_path

TODAY = date.today().strftime('%Y%m%d')
FACTORS = [1, 2, 4, 8, 16, 32]
SUBJECTS = ['cs_AI', 'cs_CL', 'cs_CV', 'cs_LG', 'cs_MA', 'cs_RO']


def read_source(input_path='data/source/arxiv-data-20200125-split*.jsonl.gz'):
    """Read split JSONL files from source directory."""
    input_path = force_path(input_path, require_exists=False)
    fields = ['id', 'title', 'abstract', 'keywords', 'categories', 'created']
    paths = list(Path(input_path).parent.glob(input_path.name))
    if not paths:
        raise FileNotFoundError(input_path)
    for p in paths:
        for record in _read_gz_jsonl(p):
            yield {k: v for k, v in record.items() if k in fields}


def _read_gz_jsonl(filename: Path):
    if filename.suffix.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:
            for line in f:
                yield json.loads(line)
    else:
        for line in read_jsonl(filename):
            yield line


def write_partitions(df, output_dir, label_column, text_column='lower_words', split_column=None):
    """Write SciBERT-formatted train/dev/test partitions to disk.

    The result is "train.jsonl", "dev.jsonl", and "test.jsonl" in the ``output_dir``.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    meta_columns = ['id', 'year']
    if split_column is None:
        split_column = f"split_{label_column}"
    for split in ['train', 'dev', 'test']:
        path = Path(output_dir, f'{split}.jsonl')
        records = (_to_bert_record(row, label_column, text_column, meta_columns)
                   for i, row in df.query(f"{split_column} == '{split}'").iterrows())
        write_jsonl(path, records)


def _to_bert_record(record, label_column, text_column, meta_columns):
    label = record.pop(label_column)
    if isinstance(label, bool):
        label = str(label)
    return {
        'label': label,
        'text': record.pop(text_column),
        'meta': {k: record.get(k) for k in meta_columns},
    }


def sample_corpus(d, size, strata):
    """Sample from the full corpus for smaller experiments."""
    split, _ = train_test_split(d, train_size=size, stratify=strata, random_state=20200206)
    return split.copy()


def path(label, sample=None, factor=None, today=TODAY, train=True):
    """Construct directory paths for subject partitions, samples, and undersamples.
    """
    if sample is None:
        sample = 'complete'
    else:
        # Expecting sample='10-pct'
        sample += '-sample'
    if train:
        dir_ = 'train'
    else:
        dir_ = 'processed'
    p = f'data/{dir_}/arxiv-{today}-{label}-{sample}'
    if factor is not None:
        # Expecting e.g. factor=8
        p += f'-{factor}-1'
    if not train:
        p += '.jsonl'
    return p


def write_multi(d, split_col='split_Any_AI', sample=None, today=TODAY):
    """Wrap writing all the multiclass and multilabel partitions to disk."""
    write_partitions(d, path('multi', sample=sample, today=today), 'primary_label', split_column=split_col)
    write_partitions(d, path('multi-first', sample=sample, today=today), 'first_ai_label', split_column=split_col)
    write_partitions(d, path('multilabel', sample=sample, today=today), 'multilabel', split_column=split_col)


def test_path():
    assert path('ai-binary') == f'data/train/arxiv-{TODAY}-ai-binary-complete'
    assert path('bert', train=False) == f'data/processed/arxiv-{TODAY}-bert-complete.jsonl'
    assert path('ai-binary', factor=32) == f'data/train/arxiv-{TODAY}-ai-binary-complete-32-1'
    assert path('ai-binary', '10-pct', factor=32) == f'data/train/arxiv-{TODAY}-ai-binary-10-pct-sample-32-1'
