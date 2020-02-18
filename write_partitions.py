"""
Write training data to disk in various partitions and samples.
"""

import pandas as pd

from util import write_partitions, sample_corpus, path, write_multi, FACTORS, SUBJECTS, TODAY


def write(df: pd.DataFrame, multi=False, today=TODAY) -> None:
    """Write training data to disk in various partitions and samples.

    :param df: Preprocessed arXiv data, the result of running assign_labels.label().
    :param multi: If True, also write multiclass and multilabel data. Skip for faster runtime.
    :param today: ISO datestamp string used in output paths.
    """
    write_partitions(df, path('ai-binary', today=today), 'Any_AI')
    if multi:
        write_multi(df, today=today)

    # Take a 10% sample for experiments where the full corpus would be unwieldy
    sample_10 = sample_corpus(df, .1, strata=df[['Any_AI', 'split_Any_AI', 'year']])
    write_partitions(sample_10, path('Any_AI', sample='10-pct', today=today), 'Any_AI')
    if multi:
        write_multi(df, sample='10-pct-sample', today=today)

    # For very quick tests take a 1% sample
    sample_01 = sample_corpus(df, .01, strata=df[['Any_AI', 'split_Any_AI', 'year']])
    write_partitions(sample_01, path('Any_AI', sample='1-pct', today=today), 'Any_AI')
    if multi:
        write_multi(sample_01, label='Any_AI', sample='1-pct-sample', today=today)

    # Write out under-samples
    for f in FACTORS:
        write_partitions(df, path('ai-binary', factor=f, today=today), 'Any_AI', split_column=f'split_Any_AI_{f}_1')
        write_partitions(sample_10, path('ai-binary', '10-pct', factor=f, today=today), 'Any_AI',
                         split_column=f'split_Any_AI_{f}_1')

    for label in SUBJECTS:
        print(label)
        write_partitions(df, path(label, today=today), label)
        subj_sample = sample_corpus(df, .1, strata=df[[label, f'split_{label}', 'year']])
        write_partitions(subj_sample, path(label, '10-pct', today=today), label)
        for f in FACTORS:
            write_partitions(subj_sample, path(label, '10-pct', f, today=today), label,
                             split_column=f'split_{label}_{f}_1')
            write_partitions(df, path(label, factor=f, today=today), label, split_column=f'split_{label}_{f}_1')
