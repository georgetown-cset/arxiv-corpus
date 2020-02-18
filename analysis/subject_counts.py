"""
Tabulate counts of papers by arXiv subject.
"""
import pandas as pd

date = '20200215'
subject_labels = {
    'cs_AI': r'Artificial Intelligence (\texttt{cs.AI})',
    'cs_CL': r'Natural Language Processing (\texttt{cs.CL})',
    'cs_CV': r'Computer Vision (\texttt{cs.CV})',
    'cs_LG': r'Machine Learning (\texttt{cs.LG, stat.ML})',
    'cs_MA': r'Multiagent Systems (\texttt{cs.MA})',
    'cs_RO': r'Robotics (\texttt{cs.RO})',
    'Any_AI': 'Any of the Above',
}

df = pd.read_pickle(f'data/processed/paper_data_{date}.pickle.gz')
assert ((df['year'] >= 2010).all() & (df['year'] <= 2019).all())

# Get counts including cross-posts (these are non-exclusive)
subject_counts = df[['cs_AI', 'cs_CL', 'cs_CV', 'cs_LG', 'cs_MA', 'cs_RO', 'Any_AI']].agg(sum, result_type='expand')
subject_counts = subject_counts.reset_index(name='Papers Including Cross-posts').rename(columns={'index': 'Subject'})

# Get counts by primary category
df['primary'] = df['split_categories'].apply(lambda x: x[0])
paper_counts = df.loc[df['primary'].isin(['cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.MA', 'cs.RO'])]. \
    groupby('primary')['primary'].agg('count')
# Add count of unique papers across subjects
paper_counts['Any_AI'] = paper_counts.sum()
paper_counts = paper_counts.reset_index(name='Papers with Primary Subject').rename(columns={'primary': 'Subject'})

paper_counts['Subject'] = paper_counts['Subject'].str.replace('.', '_')
counts = pd.merge(paper_counts, subject_counts, on='Subject', how='right')

n_primary = counts.iloc[-1, 1]
n_post = counts.iloc[-1, 2]
counts.iloc[:, 1:] = counts.iloc[:, 1:].applymap(lambda x: f'{x:,.0f}')
table = counts.to_latex(index=False)

tex = fr"""\begin{{table}}[h]
\centering
\caption{{arXiv contains {n_primary:,} papers from 2010–2019 whose primary subject is one of the six we selected as
      relevant. {n_post:,} papers, or an additional {n_post - n_primary:,}, appeared in at least one of the six
      subjects. This includes cross-posts from other subjects.}}
{table}
\label{{tab:subject-counts}}
\end{{table}}"""
for subj, label in subject_labels.items():
    tex = tex.replace(subj.replace('_', '\\_'), label)
tex = tex.replace('{lll}', '{lrr}')
tex = tex.replace('Any of the Above', '\\midrule\n Any of the Above')

print(tex)
with open('analysis/subject_counts.tex', 'wt') as f:
    f.write(tex)

