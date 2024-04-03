import pandas as pd
import json
import os
import data
from copy import  deepcopy
import numpy as np
import statistics
from utils import get_audio_duration
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pwd = os.getcwd()

slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/SLURP/csv/slurp_mini_FE_MO_ME_FO_UNK.csv')
slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/SLURP/csv/slurp_headset.csv')
slurp_df = deepcopy(slurp_df)

def find_unique_words_from_sentence(sentence):
    # Tokenize the sentence into words (assuming words are separated by spaces)
    words = re.findall(r'\w+', sentence.lower())  # Convert all words to lowercase for case-insensitivity

    # Use a set to store unique words
    unique_words_set = set(words)

    # Convert the set back to a list to get unique words
    unique_words = list(unique_words_set)

    return unique_words

total_dur = 0
numbers, paths = [], []
full_text = ""
for _, row in slurp_df.iterrows():
    audio = '/Users/afsarabenazir/Downloads/speech_projects/speechcache/SLURP/slurp_real/' + row['recording_path']
    dur = get_audio_duration(audio)
    total_dur += dur

    full_text += row['sentence'] + " "
    numbers.append(dur)
    base_name, _ = os.path.splitext(row['recording_path'])
    paths.append(base_name)
    print(dur)


df = pd.DataFrame({'duration': numbers, 'recording_path': paths})
median = statistics.median(numbers)
print('total dur ', total_dur)
print('avg ', total_dur / len(slurp_df))
print('mdeian ', median)
print()
# sns.histplot(df['duration'], kde=True, bins=185)
# plt.title('Distribution of Durations')
# plt.xlabel('Duration (seconds)')
# plt.ylabel('Frequency')
# plt.grid(True)

df = df.sample(frac=0.10, random_state=42)
# df.to_csv('histogram.csv', index=False)
sns.histplot(df['duration'], kde=True, bins=185)
plt.title('Distribution of Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
# plt.savefig('subset_SLURP_data_distribution.png', dpi=300)

df = pd.read_csv('network_latency_LTE.csv')

# df = df.dropna(subset=['latency'])
# df['latency'] = df['latency'] * 1000
# df['duration'] = df['duration'].round(2)
# df.to_csv('network_latency_LTE.csv')

plt.figure(figsize=(10,6))
plt.scatter(df['duration'], df['latency'], color='blue', marker='o')
plt.title('LTE Duration vs Latency (w/o outliers)')
plt.xlabel('Duration(sec)')
plt.ylabel('Latency(ms) (LTE)')
plt.ylim(0, 2000)

ticks = np.arange(0, df['duration'].max() + 0.5, 0.5)
plt.xticks(ticks, rotation=45)  # rotation for better visibility

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()  # Adjusts subplot parameters for better layout

plt.savefig('network_latency_LTE.png')

words = find_unique_words_from_sentence(full_text)
median = statistics.median(numbers)
print('total dur ', total_dur)
print('avg ', total_dur / len(slurp_df))
print('mdeian ', median)
print()
get_audio_duration('example_fsc.wav')
device = "cpu"
config = data.read_config("experiments/no_unfreezing.cfg")
speechcache_config = data.read_config("experiments/speechcache.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
_, _, _ = data.get_SLU_datasets(speechcache_config)  # used to set config.num_phonemes
print('config load ok')

new_train_df = deepcopy(train_dataset.df)

total_dur = 0
numbers = []
full_text = ""
speakers_to_remove = ['4aGjX3AG5xcxeL7a', '5pa4DVyvN2fXpepb', '9Gmnwa5W9PIwaoKq', 'KLa5k73rZvSlv82X',
                      'LR5vdbQgp3tlMBzB', 'OepoQ9jWQztn5ZqL', 'X4vEl3glp9urv4GN', 'Ze7YenyZvxiB4MYZ',
                      'eL2w4ZBD7liA85wm', 'eLQ3mNg27GHLkDej', 'ldrknAmwYPcWzp4N', 'mzgVQ4Z5WvHqgNmY',
                      'nO2pPlZzv3IvOQoP2', 'oNOZxyvRe3Ikx3La', 'roOVZm7kYzS5d4q3', 'rwqzgZjbPaf5dmbL',
                      'wa3mwLV3ldIqnGnV', 'xPZw23VxroC3N34k', 'ywE435j4gVizvw3R', 'zwKdl7Z2VRudGj2L',
                      '35v28XaVEns4WXOv', 'YbmvamEWQ8faDPx2', 'neaPN7GbBEUex8rV', '9mYN2zmq7aTw4Blo']

for item in speakers_to_remove:
    new_train_df = new_train_df.drop(new_train_df[new_train_df['speakerId'] == item].index)
for _, row in new_train_df.iterrows():
    audio = os.path.join(pwd, 'fluent_speech_commands_dataset/' + row['path'])
    full_text += row['transcription'] + " "
    dur = get_audio_duration(audio)
    total_dur += dur
    numbers.append(dur)
    print(dur)
words = find_unique_words_from_sentence(full_text)
median = statistics.median(numbers)
print('total dur ', total_dur)
print('avg ', total_dur / len(new_train_df))
print('mdeian ', median)

print()

#
# sentences = set()
# slurp_df = pd.read_csv("/Users/afsarabenazir/Downloads/speech_projects/speechcache/SLURP/csv/slurp_headset.csv")
# slurp_df = deepcopy(slurp_df)
# for _, row in slurp_df.iterrows():
#         sentences.add(row['sentence'])
# uniq_words = count_unique_words(sentences)
# print()