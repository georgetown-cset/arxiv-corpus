"""
Map arXiv subjects to class labels.
"""
from sklearn.model_selection import train_test_split

from write_partitions import FACTORS, SUBJECTS

RELEVANT_SUBJECTS = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.MA', 'cs.RO', 'stat.ML']


def label(df):
    # Create indicators for each relevant subject, named by arXiv abbreviation, e.g. 'cs_AI' (also creating
    # 'split_categories' and 'category_n' columns)
    df = indicate_subjects(df)
    df = label_any_ai(df)
    # We now have e.g. 'Any_AI' taking true/false and 'split_Any_AI' giving whether an article is train/dev/test/omit
    # set; similarly for each subject

    # Add a 'primary_label' column giving the primary label like 'cs_RO'
    df = label_multiclass(df)
    # Collapse MA, which is small, into LG for 'coarse multiclass' label (explored but not used in paper)
    df['coarse_multiclass'] = df['primary_label']
    df['coarse_multiclass'] = df['coarse_multiclass'].apply(lambda x: 'cs_LG' if x == 'cs_MA' else x)
    df['coarse_multiclass'].value_counts()

    # Add a 'first_ai_label' column giving the first AI-relevant label (used for descriptives)
    df = label_multiclass(df, primary=False)
    df['first_ai_label'] = df['first_ai_label'].str.replace('.', '_')
    df['first_ai_label'] = df['first_ai_label'].apply(lambda x: 'cs_LG' if x == 'cs_MA' else x)
    df['first_ai_label'].value_counts()

    # Add a 'multilabel' column giving an array of 0-K labels (explored but not used in paper)
    df['cs_LG_MA'] = df['cs_LG'] | df['cs_MA']
    df['cs_LG_MA'].value_counts()
    df = add_multilabel(df)

    # Add undersample columns (used in ongoing work, not in paper)
    for f in FACTORS:
        df = undersample(df, label_col='Any_AI', new_split_col=f'split_Any_AI_{f}_1', factor=f)
        for subj in SUBJECTS:
            print(subj)
            df = undersample(df, label_col=subj, factor=f)

    return df


def label_any_ai(df):
    """Assign the "Any_AI" label, positive if any of the AI-relevant subjects are in an article's categories."""
    df['Any_AI'] = df['split_categories'].apply(lambda x: any(subj in x for subj in RELEVANT_SUBJECTS))
    df = split_label(df, label_col='Any_AI')
    return df


def label_multiclass(d, labels=('cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.MA', 'cs.RO'), primary=True):
    """Create a column "primary_label" with the primary (first) subject of an article if relevant, else "Other"."""
    if primary:
        d['primary_label'] = d['split_categories'].apply(
            lambda x: x[0].replace('.', '_') if x[0] in labels else 'Other')
    else:
        d['first_ai_label'] = d['split_categories'].apply(lambda cats: [x for x in cats if x in labels])
        d['first_ai_label'] = d['first_ai_label'].apply(lambda cats: cats[0] if cats else 'Other')
    return d


def indicate_subjects(df):
    """Create indicator columns for each label of interest.

    We consider stat.ML and cs.LG interchangeable. Ostensibly (see https://arxiv.org/corr/subjectclasses) papers are
    automatically cross-posted between the two, but in the data this isn't true for a small number of papers.
    """
    # Quick solution: replace all 'cs.LG' with 'stat.ML' - sometimes results in repeated 'cs.LG', but we don't assume
    # uniqueness
    df['categories'] = df['categories'].str.replace('stat.ML', 'cs.LG')
    df['split_categories'] = df['categories'].str.split(' ')
    df['category_n'] = df['split_categories'].apply(lambda x: len(set(x)))
    for cat in [subj for subj in RELEVANT_SUBJECTS if subj != 'stat.ML']:
        # Period is an illegal character for labels or columns somewhere downstream
        col = cat.replace('.', '_')
        df[col] = df['categories'].str.contains(cat)
        df = split_label(df, col)
    return df


def add_multilabel(d, columns=('cs_AI', 'cs_CL', 'cs_CV', 'cs_LG_MA', 'cs_RO')):
    """Create a column "multilabel" giving an array of subject labels as strings."""
    d['multilabel'] = d.apply(lambda row: [k for k, v in row.items() if k in columns and v], axis=1)
    return d


def undersample(d, label_col='Any_AI', split_col=None, new_split_col=None, split_value='train', factor=4, to_test=True):
    """Undersample negative examples to adjust class balance.

    The result is a new split column. From e.g. "split_cs_AI", which gives the assignment of papers to train/dev/test
    stratified by year and the cs_AI label, we create "split_cs_AI_4_1" for factor=4, representing new assignments in
    which negative class examples have been undersampled to achieve a 4:1 imbalance.
    """
    if split_col is None:
        split_col = f'split_{label_col}'
    # We keep the original split_col and modify a copy
    if new_split_col is None:
        new_split_col = f'split_{label_col}_{factor}_1'
    d[new_split_col] = d[split_col]

    # Get the original ratio of negative to positive examples
    counts = d.loc[d[split_col] == split_value, label_col].value_counts()
    ratio = counts[False] / counts[True]
    print(f'Original class ratio is {ratio.round(1)}:1 neg:pos')
    if factor > ratio:
        # We only undersample
        print(f'Doing nothing for "factor" > {ratio.round(1)}')
        return d

    # Keep negative examples to produce the ratio indicated by factor
    n_keep = counts[False] - counts[True] * factor
    negative_idx = d.loc[(d[split_col] == split_value) & (~d[label_col])].index
    _, drop_idx = train_test_split(negative_idx, test_size=n_keep, random_state=20200212,
                                   stratify=d.loc[negative_idx, 'year'])

    # Either reassign the excluded examples to the test set or assign them to no set
    if to_test:
        d.loc[drop_idx, new_split_col] = 'test'
    else:
        d.loc[drop_idx, new_split_col] = None

    # Verify the new ratio of negative to positive examples
    new_counts = d.loc[d[new_split_col] == split_value, label_col].value_counts()
    new_ratio = new_counts[False] / new_counts[True]
    assert float(factor) == new_ratio
    print(f'New class ratio using "{new_split_col}" column is {new_ratio.round(1)}:1 neg:pos')
    return d


def split_label(df, label_col):
    """Create a column containing the assignment of papers into train/dev/test sets.

    The naming convention is for a label column like "cs_AI" to have an assignment column "split_cs_AI".
    """
    # Ensure we have a default numeric index
    df = df.reset_index(drop=True)
    # First an 80/20 train/not split -> 80/10/10 train/dev/test split
    train_i, temp_i = train_test_split(df.index, train_size=.8, stratify=df[['year', label_col]], random_state=20200126)
    dev_i, test_i = train_test_split(temp_i, train_size=.5, stratify=df.loc[temp_i, ['year', label_col]],
                                     random_state=20200126)
    split_col = f'split_{label_col}'
    df[split_col] = None
    df.loc[train_i, split_col] = 'train'
    df.loc[dev_i, split_col] = 'dev'
    df.loc[test_i, split_col] = 'test'
    # Check balance
    year_balance = df.loc[df[split_col] == 'train']. \
        groupby(['year', label_col]). \
        agg({'id': 'count'}). \
        rename(columns={'id': 'count'})
    print(year_balance)
    return df
