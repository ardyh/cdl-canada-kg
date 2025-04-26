# %% [markdown]
# We want to:
# - Load the extracted information from X, Bluesky, and Reddit data
# - Cluster these to get entity membership

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import io
import json
import os
import pickle
import re
import uuid

from itertools import product 

from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from sklearn.cluster import AgglomerativeClustering, \
    MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# %%
pd.set_option('display.max_columns', 100)

# %% [markdown]
# # Load data
# From my and Ardy's extractions

# %%
def default_loads(text) -> dict:
    # json.loads with default value for empty values
    if len(text) == 0:
        return {}
    return json.loads(text)

# %%
data_dir = './data/consolidated'
fnames = os.listdir(data_dir)
for fname in fnames:
    new_object_name = '_'.join(fname.split('_')[:2])
    globals()[new_object_name] = pd.read_csv(os.path.join(data_dir, fname))
    print(new_object_name, ':', globals()[new_object_name].shape)

# %% [markdown]
# ## Reddit

# %%
path = './data/intermediate_kg/narrative_v2_ent_obj_deduplicated.json'
df_reddit = pd.read_json(path).T.reset_index()
df_reddit = df_reddit.rename(columns={'index': 'record_id'})
print(df_reddit.shape)
df_reddit[:3]

# %%
# Normalize the extractions
df_reddit = df_reddit.explode(['political_narratives']).reset_index(drop=True)
print('Exploded:', df_reddit.shape)
df_reddit = df_reddit.drop(columns=['political_narratives']).join(
    pd.json_normalize(df_reddit['political_narratives'])
)
print('Normalized:', df_reddit.shape)
df_reddit[:3]

# %%
# Clean up sentiment
def clean_sentiment(text) -> int:
    if text == 'negative':
        return -1
    elif text == 'neutral':
        return 0
    elif text == 'positive':
        return 1
    return text
df_reddit.loc[:, 'sentiment'] = df_reddit['sentiment'].apply(clean_sentiment)
df_reddit['sentiment'].value_counts(dropna=False)

# %%
# Rename in agreement with my extractions
df_reddit = df_reddit.rename(columns={
    'name': 'agent_norm',
    'action_or_role': 'action_or_event_norm',
    'affected_entities': 'object_norm',
    'narrative': 'narrative',
})

# %% [markdown]
# ## Bluesky

# %%
path = './data/processed/gpt4omini_posts_bluesky_20250301_20250314_extractions_zeroshot.pkl'
with open(path, 'rb') as f:
    id2processed_text = pickle.load(f)
id2processed_text = {id_: default_loads(text).get('events', []) for id_, text in id2processed_text.items()}
print(len(id2processed_text))

# %%
df_bluesky = pd.DataFrame({
    'record_id': id2processed_text.keys(),
    'political_narratives': id2processed_text.values(),
})
print(df_bluesky.shape)
df_bluesky[:3]

# %%
# Normalize the extractions
df_bluesky = df_bluesky.explode(['political_narratives']).reset_index(drop=True)
print('Exploded:', df_bluesky.shape)
df_bluesky = df_bluesky.drop(columns=['political_narratives']).join(
    pd.json_normalize(df_bluesky['political_narratives'])
)
print('Normalized:', df_bluesky.shape)
df_bluesky[:3]

# %%
# Replace generic "User" with user's name
temp = posts_bluesky.set_index(['did'])[['uri']].join(
    users_bluesky.set_index(['did'])[['name']]
).drop_duplicates()
record_id2user_name = {
    record_id: name for record_id, name in zip(
        temp['uri'], temp['name']
    )
}
len(record_id2user_name)

# %%
def impute_user_name(
    row: pd.Series, record_id2user_name, col_to_impute, record_id_col,
) -> str:
    # Impute user name
    val = row[col_to_impute]
    if val == 'User':
        user_name = record_id2user_name.get(row[record_id_col], 'User')
        return user_name
    return val

# %%
df_bluesky.loc[:,'agent_norm'] = df_bluesky.apply(
    impute_user_name,
    record_id2user_name=record_id2user_name,
    col_to_impute='agent_norm',
    record_id_col='record_id',
    axis=1
)
df_bluesky.loc[:,'object_norm'] = df_bluesky.apply(
    impute_user_name,
    record_id2user_name=record_id2user_name,
    col_to_impute='object_norm',
    record_id_col='record_id',
    axis=1
)

# %% [markdown]
# ## X
# TODO Load completed extractions when they're done

# %%
path = './data/processed/gpt4omini_posts_x_20250301_20250314_extractions_zeroshot.pkl'
with open(path, 'rb') as f:
    id2processed_text = pickle.load(f)
id2processed_text = {id_: default_loads(text).get('events', []) for id_, text in id2processed_text.items()}
print(len(id2processed_text))

# %%
df_x = pd.DataFrame({
    'record_id': id2processed_text.keys(),
    'political_narratives': id2processed_text.values(),
})
print(df_x.shape)
df_x[:3]

# %%
# Normalize the extractions
df_x = df_x.explode(['political_narratives']).reset_index(drop=True)
print('Exploded:', df_x.shape)
df_x = df_x.drop(columns=['political_narratives']).join(
    pd.json_normalize(df_x['political_narratives'])
)
print('Normalized:', df_x.shape)
df_x[:3]

# %%
# Replace generic "User" with user's name
temp = posts_x.set_index(['author_id'])[['tweet_id']].join(
    users_x.set_index(['author_id'])[['name']]
).drop_duplicates()
record_id2user_name = {
    record_id: name for record_id, name in zip(
        temp['tweet_id'], temp['name']
    )
}
len(record_id2user_name)

# %%
df_bluesky.loc[:,'agent_norm'] = df_bluesky.apply(
    impute_user_name,
    record_id2user_name=record_id2user_name,
    col_to_impute='agent_norm',
    record_id_col='record_id',
    axis=1
)
df_bluesky.loc[:,'object_norm'] = df_bluesky.apply(
    impute_user_name,
    record_id2user_name=record_id2user_name,
    col_to_impute='object_norm',
    record_id_col='record_id',
    axis=1
)

# %% [markdown]
# ## Preprocessing

# %%
# Indicate null extractions
null_extractions = {
    '', '.', '/', ':', ',', '>',
    'none', ': none', 'unknown',
}
def clean_nulls(text):
    if pd.isna(text):
        return None
    elif type(text) != str:
        return text
    elif text.lower().strip() in null_extractions:
        return None
    return text

df_bluesky.loc[:,'agent_norm'] = df_bluesky['agent_norm'].apply(clean_nulls)
df_bluesky.loc[:,'object_norm'] = df_bluesky['object_norm'].apply(clean_nulls)
df_x.loc[:,'agent_norm'] = df_x['agent_norm'].apply(clean_nulls)
df_x.loc[:,'object_norm'] = df_x['object_norm'].apply(clean_nulls)

# %%
# Clean up the text to reduce some of the unique examples for clustering
cols = [
    'agent_norm',
    'action_or_event_norm',
    'object_norm',
    'narrative'
]

def clean_text(text):
    if pd.isna(text):
        return None
    text = text.strip().lower()
    return text

for df in [df_reddit, df_bluesky, df_x]:
    for col in cols:
        print('Column:', col)
        print('Before cleaning:', df[col].nunique())
        df.loc[:,f'{col}_clean'] = df[col].apply(clean_text)
        print('After cleaning:', df[f'{col}_clean'].nunique())
        print('='*20)

# %% [markdown]
# # Cluster

# %%
# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda:3')

# %%
# Collect inputs to be clustered
cols_to_cluster = [
    'agent_norm_clean',
    'action_or_event_norm_clean',
    'object_norm_clean',
    'narrative_clean'
]

# Dataset name to dataframe
dataset2df = {
    'reddit': df_reddit,
    'bluesky': df_bluesky,
    'x': df_x,
}

# Mapping from dataset to column name to unique values
dataset2col2values = {}
for dataset, df in dataset2df.items():
    dataset2col2values[dataset] = {}
    for col in cols_to_cluster:
        values = df[col].unique()
        values = [v for v in values if not pd.isna(v)]
        dataset2col2values[dataset][col] = values
        print(f'Dataset: {dataset}\t\tColumn: {col}\t\t# of values: {len(values)}')

print('='*50)
# Get overall unique values per column
overall_col2values = {}
for col in cols_to_cluster:
    values = []
    for df in dataset2df.values():
        values.append(df[col])
    values = pd.concat(values).unique()
    values = [v for v in values if not pd.isna(v)]
    overall_col2values[col] = values
    print(f'Overall\t\tColumn: {col}\t\t# of values: {len(values)}')

# %% [markdown]
# ## Agglomerative
# Clustering the actions like this would take about 30 minutes

# %%
overall_col2values.keys()

# %%
range_n_clusters_short = list(range(100, 1000, 100)) + list(range(1000, 5001, 500))
range_n_clusters_med = list(range(100, 1000, 100)) + list(range(1000, 5001, 500)) + \
    list(range(5500, 10001, 500))
range_n_clusters_long = list(range(100, 1000, 100)) + list(range(1000, 5001, 500)) + \
    list(range(5500, 10001, 500)) + list(range(11000, 20001, 1000))
range_n_clusters_longest = list(range(100, 1000, 100)) + list(range(1000, 5001, 500)) + \
    list(range(5500, 10001, 500)) + list(range(11000, 20001, 1000)) + \
    list(range(21000, 30001, 1000))
col2range_n = {
    'agent_norm_clean': range_n_clusters_short,
    'action_or_event_norm_clean': range_n_clusters_long,
    'object_norm_clean': range_n_clusters_med,
    'narrative_clean': range_n_clusters_longest,
}

clustering_results = {
    'dataset': [],
    'col': [],
    'n_clusters': [],
    'ss': [],
    'labels': [],
}
for col, values in overall_col2values.items():
    print('Starting:', col)
    embeddings = model.encode(values, show_progress_bar=True)

    # Reduce embedding dimension to speed up distance computation
    pca = PCA(n_components=100, random_state=0)
    embeddings = pca.fit_transform(embeddings)

    silhouette_scores = []
    range_n = col2range_n[col]
    for n_clusters in tqdm(range_n):
        if n_clusters > len(values):
            # No point in clustering with more clusters than samples
            continue
        clustering = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=2048,
            random_state=0,
        )
        cluster_labels = clustering.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)

        clustering_results['dataset'].append('overall')
        clustering_results['col'].append(col)
        clustering_results['n_clusters'].append(n_clusters)
        clustering_results['ss'].append(score)
        clustering_results['labels'].append(cluster_labels)

        print(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
    
df_clustering_results = pd.DataFrame(clustering_results)
fname = './data/scoring/clustering_scores_kmeans_all.xlsx'
df_clustering_results.to_excel(fname, index=False)
print('Wrote to:', fname)
