# %%
import pandas
from charset_normalizer import models
import pandas as pd
import data
import os
import time
import soundfile as sf
import torch
import numpy as np
from functools import reduce
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from tqdm import tqdm
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils import audio_augment
import models
import pickle
import matplotlib.pyplot as plt

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
dataset_to_use = valid_dataset
dataset_to_use.df.head()
test_dataset.df.head()

act_obj_loc = {}
for dataset in [train_dataset, valid_dataset, test_dataset]:
    for _, row in dataset.df.iterrows():
        act, obj, loc = row['action'], row['object'], row['location']
        if (act, obj, loc) not in act_obj_loc:
            act_obj_loc[(act, obj, loc)] = len(act_obj_loc)
len(act_obj_loc)

# %%
device = torch.device('cpu')
df = pd.concat([train_dataset.df, test_dataset.df, valid_dataset.df])
# process data
new_train_df = deepcopy(df)
act_obj_loc_idxs = []
for _, row in new_train_df.iterrows():
    act, obj, loc = row['action'], row['object'], row['location']
    act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
new_train_df['cache'] = act_obj_loc_idxs
new_train_df.head()


def count_unique_words(sentences):
    # Concatenate all sentences into a single string
    text = ' '.join(sentences)
    # Split the string into words and create a set of unique words
    unique_words = set(word for word in text.split())
    return len(unique_words)

"""
# ------------------- SLURP ------------------

slurp_df = pd.read_csv('slurp.csv', usecols=['sentence_normalised'])
df1 = pd.DataFrame(columns=['#uniq_tscpt'])
df2 = pd.DataFrame(columns=['#uniq_words'])
for tscpt_len in list(range(1, 29)):
    sentences = set()
    for _, sentence in slurp_df['sentence_normalised'].iteritems():
        if len(sentence.split()) == tscpt_len:
            sentences.add(sentence)
    print(len(sentences))
    uniq_words = count_unique_words(sentences)
    df1_values = [len(sentences)]
    df2_values = [uniq_words]
    df1_values = pd.DataFrame([df1_values], columns=df1.columns)
    df2_values = pd.DataFrame([df2_values], columns=df2.columns)
    df1 = df1.append(df1_values, ignore_index=True)
    df2 = df2.append(df2_values, ignore_index=True)
df1.index = pd.RangeIndex(start=1, stop=len(df1) + 1, step=1)
df2.index = pd.RangeIndex(start=1, stop=len(df2) + 1, step=1)
df1.plot.bar(color='#FFA500')
plt.xlabel('tscpt_len')
plt.ylabel('#tscpt Count')
plt.title('tscpt len vs unique tscpt')
plt.savefig('tscpt_1.png')

df2.plot.bar()
plt.xlabel('tscpt_len')
plt.ylabel('#words Count')
plt.title('tscpt len vs unique words')
plt.savefig('tscpt_2.png')
"""
"""
slurp_df = pd.read_csv('slurp.csv', usecols=['sentence_normalised'])
all_sentences = set()
for tscpt_len in list(range(1, 29)):
    sentences = set()
    for _, sentence in slurp_df['sentence_normalised'].iteritems():
        if len(sentence.split()) == tscpt_len:
            sentences.add(sentence)
            all_sentences.add(sentence)
    print(len(sentences))
uniq_words = count_unique_words(all_sentences)
print(uniq_words)
"""
"""
slurp_df = pd.read_csv('slurp.csv', usecols=['sentence_normalised'])
sentences = []
for _, sentence in slurp_df['sentence_normalised'].iteritems():
    sentences.append(sentence)
print(len(sentences))

total_length = 0
total_words = 0

for tscpt in sentences:
    words = tscpt.split()
    total_words += len(words)

average_length = total_words / len(sentences)
print(average_length)
"""
# -------------------- FSC ----------------------
"""
speakers = np.unique(new_train_df['speakerId'])
df1 = pd.DataFrame(columns=['#uniq_tscpt'])
df2 = pd.DataFrame(columns=['#uniq_words'])
for tscpt_len in list(range(1, 11)):
    transcript_list = set()
    for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
        print('speaker %s' % speakerId)
        tmp = new_train_df[new_train_df['speakerId'] == speakerId]
        transcripts = np.unique(tmp['transcription'])
        for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
            # count #words
            if len(transcript.split()) == tscpt_len:
                transcript_list.add(transcript)
    print(len(transcript_list))
    uniq_words = count_unique_words(transcript_list)
    df1_values = [len(transcript_list)]
    df2_values = [uniq_words]
    df1_values = pd.DataFrame([df1_values], columns=df1.columns)
    df2_values = pd.DataFrame([df2_values], columns=df2.columns)
    df1 = df1.append(df1_values, ignore_index=True)
    df2 = df2.append(df2_values, ignore_index=True)
df1.index = pd.RangeIndex(start=1, stop=len(df1) + 1, step=1)
df2.index = pd.RangeIndex(start=1, stop=len(df2) + 1, step=1)
df1.plot.bar(color='#FFA500')
plt.xlabel('Sentence Length')
plt.ylabel('Count')
plt.title('Sentence Len vs Transcript')
plt.savefig('fsc-uniq-tscpt.png')

df2.plot.bar()
plt.xlabel('Sentence Length')
plt.ylabel('Count')
plt.title('Sentence Len vs #Words')
plt.savefig('fsc-uniq-words.png')

print()
"""
"""
speakers = np.unique(new_train_df['speakerId'])
all_transcripts = set()
for tscpt_len in list(range(1, 11)):
    transcript_list = set()
    for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
        print('speaker %s' % speakerId)
        tmp = new_train_df[new_train_df['speakerId'] == speakerId]
        transcripts = np.unique(tmp['transcription'])
        for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
            # count #words
            if len(transcript.split()) == tscpt_len:
                transcript_list.add(transcript)
                all_transcripts.add(transcript)
    print(len(transcript_list))
uniq_words = count_unique_words(all_transcripts)
print(uniq_words)
"""
"""
speakers = np.unique(new_train_df['speakerId'])
transcript_list = []
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    print('speaker %s' % speakerId)
    tmp = new_train_df[new_train_df['speakerId'] == speakerId]
    transcripts = np.unique(tmp['transcription'])
    for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
        transcript_list.append(transcript)

total_length = 0
total_words = 0

for tscpt in transcript_list:
    words = tscpt.split()
    total_words += len(words)

average_length = total_words / len(transcript_list)
print(average_length)
"""