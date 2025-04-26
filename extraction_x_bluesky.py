# %% [markdown]
# Perform narrative extraction on larger set of data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import io
import json
import os
import pickle
import re
import uuid

from itertools import product 

import openai
from openai import OpenAI
from pydantic import BaseModel
import tiktoken

from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

# %%
load_dotenv('./.env')

# %%
key_openai = os.getenv('KEY_OPENAI')

# %% [markdown]
# # Load data

# %% [markdown]
# ## Snapshots

# %%
data_dir = './data/consolidated'
fnames = os.listdir(data_dir)
for fname in fnames:
    new_object_name = '_'.join(fname.split('_')[:2])
    globals()[new_object_name] = pd.read_csv(os.path.join(data_dir, fname))
    print(new_object_name, ':', globals()[new_object_name].shape)

# %% [markdown]
# ## Annotations

# %%
fname = f'./data/annotated/df_x_sample_filtered_20250226_20250227_annotated.xlsx'
df_x = pd.read_excel(fname)
print(df_x.shape)
df_x = df_x[:52]
print(df_x.shape)

# %%
fname = f'./data/annotated/df_bluesky_sample_filtered_20250226_20250227_annotated.xlsx'
df_bluesky = pd.read_excel(fname)
print(df_bluesky.shape)
df_bluesky = df_bluesky[:114]
print(df_bluesky.shape)

# %% [markdown]
# # Extraction

# %%
client = OpenAI(api_key=key_openai)

# %%
model_name = 'gpt-4o-mini-2024-07-18'

# %% [markdown]
# ## Define json schema for extractions

# %%
class NarrativeEvent(BaseModel):
    agent: str
    agent_norm: str
    action_or_event: str
    action_or_event_norm: str
    object: str
    object_norm: str
    narrative: str

class NarrativeExtraction(BaseModel):
    events: List[NarrativeEvent]

# %% [markdown]
# ## Few-shot examples

# %%
cols = [
    'agent', 'agent_norm',
    'action_or_event', 'action_or_event_norm',
    'object', 'object_norm',
    'narrative', 'sentiment',
]

# %%
def join_if_list(value):
    if isinstance(value, (list, tuple)) or (hasattr(value, 'ndim') and value.ndim == 1):
        return ', '.join(str(x) for x in value)
    if pd.isna(value):
        return ""
    return value

def is_list_like(value):
    return isinstance(value, (list, tuple)) or (hasattr(value, 'ndim') and value.ndim == 1)

# %%
df_x_examples = df_x[df_x['good_example'] == 1].groupby(['data.text']).agg({
    col: list for col in cols
}).reset_index()

fewshot_inputs_x = []
fewshot_outputs_x = []
for index, row in df_x_examples.iterrows():
    fewshot_inputs_x.append(row['data.text'])
    prepared = {}
    max_len = 1  # minimum 1 event per row
    
    # Process each column value. If it's missing, assign a default list;
    # if list-like, use it as is; otherwise, wrap it in a one-element list.
    for col in cols:
        val = row[col]
        if is_list_like(val):
            prepared[col] = list(val)
            if len(prepared[col]) > max_len:
                max_len = len(prepared[col])
        else:
            prepared[col] = [val]
    
    # Pad any lists shorter than max_len with empty defaults
    for col in cols:
        if len(prepared[col]) < max_len:
            pad_val = 0 if col == "sentiment" else ""
            prepared[col].extend([pad_val] * (max_len - len(prepared[col])))
    
    # Create one event per index position and append to a list for that row
    events = []
    for i in range(max_len):
        event = {
            "agent": prepared["agent"][i] if not pd.isna(prepared["agent"][i]) else None,
            "agent_norm": prepared["agent_norm"][i] if not pd.isna(prepared["agent_norm"][i]) else None,
            "action_or_event": prepared["action_or_event"][i] if not pd.isna(prepared["action_or_event"][i]) else None,
            "action_or_event_norm": prepared["action_or_event_norm"][i] if not pd.isna(prepared["action_or_event_norm"][i]) else None,
            "object": prepared["object"][i] if not pd.isna(prepared["object"][i]) else None,
            "object_norm": prepared["object_norm"][i] if not pd.isna(prepared["object_norm"][i]) else None,
            "narrative": prepared["narrative"][i] if not pd.isna(prepared["narrative"][i]) else None,
            "sentiment": prepared["sentiment"][i] if not pd.isna(prepared["sentiment"][i]) else None,
        }
        events.append(event)
    
    fewshot_outputs_x.append(events)
fewshot_outputs_x[1] = []

# %%
df_bluesky_examples = df_bluesky[df_bluesky['good_example'] == 1].groupby(['commit.record.text']).agg({
    col: list for col in cols
}).reset_index()

fewshot_inputs_bluesky = []
fewshot_outputs_bluesky = []
for index, row in df_bluesky_examples.iterrows():
    fewshot_inputs_bluesky.append(row['commit.record.text'])
    prepared = {}
    max_len = 1  # minimum 1 event per row
    
    # Process each column value. If it's missing, assign a default list;
    # if list-like, use it as is; otherwise, wrap it in a one-element list.
    for col in cols:
        val = row[col]
        if is_list_like(val):
            prepared[col] = list(val)
            if len(prepared[col]) > max_len:
                max_len = len(prepared[col])
        else:
            prepared[col] = [val]
    
    # Pad any lists shorter than max_len with empty defaults
    for col in cols:
        if len(prepared[col]) < max_len:
            pad_val = 0 if col == "sentiment" else ""
            prepared[col].extend([pad_val] * (max_len - len(prepared[col])))
    
    # Create one event per index position and append to a list for that row
    events = []
    for i in range(max_len):
        event = {
            "agent": prepared["agent"][i] if not pd.isna(prepared["agent"][i]) else None,
            "agent_norm": prepared["agent_norm"][i] if not pd.isna(prepared["agent_norm"][i]) else None,
            "action_or_event": prepared["action_or_event"][i] if not pd.isna(prepared["action_or_event"][i]) else None,
            "action_or_event_norm": prepared["action_or_event_norm"][i] if not pd.isna(prepared["action_or_event_norm"][i]) else None,
            "object": prepared["object"][i] if not pd.isna(prepared["object"][i]) else None,
            "object_norm": prepared["object_norm"][i] if not pd.isna(prepared["object_norm"][i]) else None,
            "narrative": prepared["narrative"][i] if not pd.isna(prepared["narrative"][i]) else None,
            "sentiment": prepared["sentiment"][i] if not pd.isna(prepared["sentiment"][i]) else None,
        }
        events.append(event)
    
    fewshot_outputs_bluesky.append(events)
fewshot_outputs_bluesky[2] = []

# %% [markdown]
# ## Prompting

# %%
zeroshot_system_prompt = '''
You are an expert at structured data extraction and narrative understanding from social media data, specializing in the 2025 Canadian Presidential election. You will be given unstructured text from a social media post and should convert it into the given structure, a list of events where each event contains the following elements, focusing on figures in Canadian (and related international) politics:

agent: One who is/has done the event.
agent_norm: Normalized form of agent.
action_or_event: Action which the agent has taken.
action_or_event_norm: Normalized form of action_or_event.
object: One who is receiving the action or being acted upon.
object_norm: Normalized form of object.
narrative: Short, 1-sentence description of the larger narrative that this agent-action-object triple seems to be a part of.

A post may contain no events or multiple. Extract all identified events in the post. Elements may be explicitly found in the post or implicit. Elements that cannot be filled should be left as None. If the social media post's poster is extracted as an element, they should be referred to as "User". All other people can be identifed by their name and social media handle (if found in the post).
'''.strip()
print(zeroshot_system_prompt)

# %%
fewshot_system_prompt = '''
You are an expert at structured data extraction and narrative understanding from social media data, specializing in the 2025 Canadian Presidential election. You will be given unstructured text from a social media post and should convert it into the given structure, a list of events where each event contains the following elements, focusing on figures in Canadian (and related international) politics:

agent: One who is/has done the event.
agent_norm: Normalized form of agent.
action_or_event: Action which the agent has taken.
action_or_event_norm: Normalized form of action_or_event.
object: One who is receiving the action or being acted upon.
object_norm: Normalized form of object.
narrative: Short, 1-sentence description of the larger narrative that this agent-action-object triple seems to be a part of.

A post may contain no events or multiple. Extract all identified events in the post. Elements may be explicitly found in the post or implicit. Elements that cannot be filled should be left as None. If the social media post's poster is extracted as an element, they should be referred to as "User". All other people can be identifed by their name and social media handle (if found in the post).

Here are some examples of valid extractions:
'''.strip()
print(fewshot_system_prompt)

# %%
# Bluesky-specific few-shot prompt
fewshot_system_prompt_bluesky = f'''
{fewshot_system_prompt}

Example Input 1:
{fewshot_inputs_bluesky[0]}
Example Output 1:
{fewshot_outputs_bluesky[0]}

Example Input 2:
{fewshot_inputs_bluesky[1]}
Example Output 2:
{fewshot_outputs_bluesky[1]}

Example Input 3:
{fewshot_inputs_bluesky[2]}
Example Output 3:
{fewshot_outputs_bluesky[2]}
'''.strip()

# %%
# X-specific few-shot prompt
fewshot_system_prompt_x = f'''
{fewshot_system_prompt}

Example Input 1:
{fewshot_inputs_x[0]}
Example Output 1:
{fewshot_outputs_x[0]}

Example Input 2:
{fewshot_inputs_x[1]}
Example Output 2:
{fewshot_outputs_x[1]}

Example Input 3:
{fewshot_inputs_x[2]}
Example Output 3:
{fewshot_outputs_x[2]}
'''.strip()

# %% [markdown]
# # Run
# Based on evaluation on annotations:
# - Bluesky: 0-shot
# - X: fewshot

# %% [markdown]
# ## Cost estimation
# Estimate cost based on input prompts over the dataset

# %%
price_per_input_token  = 0.15 / 1_000_000
price_per_output_token = 0.60 / 1_000_000

enc = tiktoken.encoding_for_model(model_name)

# %%
def count_tokens_in_messages(messages):
    total = 0
    for msg in messages:
        total += len(enc.encode(msg["role"]))
        total += len(enc.encode(msg["content"]))
    return total

# %%
def estimate_chat_batch_cost(batched_messages, responses):
    cost = 0.0
    for msgs, reply in zip(batched_messages, responses):
        in_toks  = count_tokens_in_messages(msgs)
        out_toks = len(enc.encode(reply))
        cost += in_toks * price_per_input_token + out_toks * price_per_output_token
    return cost

# %%
pd.set_option('display.max_columns', 100)

# %%
# # Bluesky prompts
# prompts = []
# for text in posts_bluesky['record_text']:
#     prompts.append([
#         {'role': 'system', 'content': zeroshot_system_prompt},
#         {'role': 'user', 'content': text}
#     ])
# cost = estimate_chat_batch_cost(prompts, ['']*len(prompts))
# print('Estimated cost:', cost)

# %%
# # X prompts
# prompts = []
# for text in posts_x['full_text']:
#     prompts.append([
#         {'role': 'system', 'content': fewshot_system_prompt_x},
#         {'role': 'user', 'content': text}
#     ])
# cost = estimate_chat_batch_cost(prompts, ['']*len(prompts))
# print('Estimated cost:', cost)

# %% [markdown]
# The full datasets take about $80 just for input

# %%
# Just operate on first 2 weeks of March
posts_bluesky_filt = posts_bluesky[posts_bluesky['first_updated'] < '2025-03-08']\
    .drop_duplicates(['uri'], keep='last')
posts_x_filt = posts_x[posts_x['date_collected'] < '2025-03-08']\
    .drop_duplicates(['tweet_id'], keep='last')

print(posts_bluesky_filt.shape, posts_x_filt.shape)

# %%
# # Bluesky prompts
# prompts = []
# for text in posts_bluesky_filt['record_text']:
#     prompts.append([
#         {'role': 'system', 'content': zeroshot_system_prompt},
#         {'role': 'user', 'content': text}
#     ])
# cost = estimate_chat_batch_cost(prompts, ['']*len(prompts))
# print('Estimated cost:', cost)

# %%
# # X prompts
# prompts = []
# for text in posts_x_filt['full_text']:
#     prompts.append([
#         {'role': 'system', 'content': fewshot_system_prompt_x},
#         {'role': 'user', 'content': text}
#     ])
# cost = estimate_chat_batch_cost(prompts, ['']*len(prompts))
# print('Estimated cost:', cost)

# %% [markdown]
# The filtered datasets take about $14 for input

# %% [markdown]
# ## Bluesky

# %%
# processed_data_dir = os.path.join('./data', 'processed')
# fname = 'gpt4omini_posts_bluesky_20250301_20250314_extractions_zeroshot'
# fname_parsed_rounds_in_prog = os.path.join(
#     processed_data_dir, f'in_prog_{fname}.pkl'
# )
# save_every = 500

# # Use uri as id
# id2processed_text = {}

# # Whether to load existing
# resume_existing = True
# if resume_existing:
#     if os.path.exists(fname_parsed_rounds_in_prog):
#         with open(fname_parsed_rounds_in_prog, 'rb') as f:
#             id2processed_text = pickle.load(f)

# for idx, row in enumerate(posts_bluesky_filt.iterrows()):
#     row = row[1]
#     id_ = row['uri']
#     if id_ in id2processed_text:
#         continue
#     try:
#         completion = client.beta.chat.completions.parse(
#             model=model_name,
#             messages=[
#                 {"role": "system", "content": zeroshot_system_prompt},
#                 {"role": "user", "content": row['record_text']}
#             ],
#             response_format=NarrativeExtraction,
#         )
#         message = completion.choices[0].message
#         try:
#             parsed_round = message.content
#         except:
#             # Parsing error
#             print(f'error_parsing round: {round_n}')
#             parsed_round = message
#     except:
#         # Invalid JSON
#         parsed_round = {}
#     id2processed_text[id_] = parsed_round

#     # Save intermittently
#     if (idx + 1) % save_every == 0:
#         with open(fname_parsed_rounds_in_prog, 'wb') as f:
#             pickle.dump(id2processed_text, f)
#         print(f'# processed: {idx + 1}')
# print('DONE')

# processed_data_dir = os.path.join('./data', 'processed')
# fname_parsed_rounds = os.path.join(processed_data_dir, f'{fname}.pkl')

# with open(fname_parsed_rounds, 'wb') as f:
#     pickle.dump(id2processed_text, f)
# print('Final Bluesky export to:', fname)

# %% [markdown]
# ## X

# %%
processed_data_dir = os.path.join('./data', 'processed')
fname = 'gpt4omini_posts_x_20250301_20250314_extractions_zeroshot'
fname_parsed_rounds_in_prog = os.path.join(
    processed_data_dir, f'in_prog_{fname}.pkl'
)
save_every = 500

# Use tweet_id as id
id2processed_text = {}

# Whether to load existing
resume_existing = True
if resume_existing:
    if os.path.exists(fname_parsed_rounds_in_prog):
        with open(fname_parsed_rounds_in_prog, 'rb') as f:
            id2processed_text = pickle.load(f)

for idx, row in enumerate(posts_x_filt.iterrows()):
    row = row[1]
    id_ = row['tweet_id']
    if id_ in id2processed_text:
        continue
    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": fewshot_system_prompt_x},
                {"role": "user", "content": row['full_text']}
            ],
            response_format=NarrativeExtraction,
        )
        message = completion.choices[0].message
        try:
            parsed_round = message.content
        except:
            # Parsing error
            print(f'error_parsing round: {round_n}')
            parsed_round = message
    except:
        # Invalid JSON
        parsed_round = {}
    id2processed_text[id_] = parsed_round

    # Save intermittently
    if (idx + 1) % save_every == 0:
        with open(fname_parsed_rounds_in_prog, 'wb') as f:
            pickle.dump(id2processed_text, f)
        print(f'# processed: {idx + 1}')
print('DONE')

processed_data_dir = os.path.join('./data', 'processed')
fname_parsed_rounds = os.path.join(processed_data_dir, f'{fname}.pkl')

with open(fname_parsed_rounds, 'wb') as f:
    pickle.dump(id2processed_text, f)
print('Final X export to:', fname)

# %%
# Peek on the results
fname_parsed_rounds = '/nas/ckgfs/users/eboxer/complexdata/data/processed/in_prog_gpt4omini_posts_bluesky_20250301_20250314_extractions_zeroshot.pkl'
with open(fname_parsed_rounds, 'rb') as f:
    id2processed_text = pickle.load(f)
id2processed_text = {id_: json.loads(text)['events'] for id_, text in id2processed_text.items()}
print('Read from:', fname_parsed_rounds)
print(len(id2processed_text))

# %% [markdown]
# Check on runs. Expect a significant amount of progress on bluesky at least.
# 
# If not, interrupt, write some code to pick up from the last checkpoint (by input id) and add more filtering (just a week or less per dataset)
