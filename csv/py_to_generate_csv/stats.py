import pandas as pd
import data
import torch
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import csv
import matplotlib.pyplot as plt

device = torch.device('cpu')
config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')

act_obj_loc = {}
for dataset in [train_dataset, valid_dataset, test_dataset]:
    for _, row in dataset.df.iterrows():
        act, obj, loc = row['action'], row['object'], row['location']
        if (act, obj, loc) not in act_obj_loc:
            act_obj_loc[(act, obj, loc)] = len(act_obj_loc)
len(act_obj_loc)

# process data
new_train_df = deepcopy(train_dataset.df)
# new_train_df = deepcopy(test_dataset.df)
# new_train_df = deepcopy(valid_dataset.df)
act_obj_loc_idxs = []
for _, row in new_train_df.iterrows():
    act, obj, loc = row['action'], row['object'], row['location']
    act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
new_train_df['cache'] = act_obj_loc_idxs
new_train_df.head()

"""
# -------------- speaker stats on train set
speakers = np.unique(new_train_df['speakerId'])
spk_dets = []
curr_spk_dets = []
header = ['SpeakerID', 'distinct_tscpt', 'repeat']
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    tmp = new_train_df[new_train_df['speakerId'] == speakerId]
    # speakerID, train_count, test_count
    curr_spk_dets = [speakerId, np.unique(tmp['transcription']).size, tmp['transcription'].size - np.unique(tmp['transcription']).size]
    spk_dets.append(curr_spk_dets)

df = pd.DataFrame(spk_dets, columns=header)
df.to_csv('speakers_train_set.csv',  encoding='ascii', errors='replace', index=False)
df.plot(x="SpeakerID", y=["distinct_tscpt", "repeat"], kind="bar").legend(loc='center left',bbox_to_anchor=(1.0, 0.5));
plt.tight_layout(pad=0.5)
plt.savefig('speakers_train_set.png')
plt.show()

"""
# --------------  transcript stats
transcripts = np.unique(new_train_df['transcription'])
header = ['Transcription', 'occurs', 'spoken_by']
tscpt_dets = []
for tscptid, transcription in tqdm(enumerate(transcripts), total=len(transcripts)):
    tmp = new_train_df[new_train_df['transcription'] == transcription]
    speakers = np.unique(tmp['speakerId'])
    curr_tscpt = [transcription, len(tmp), len(speakers)]
    tscpt_dets.append(curr_tscpt)

df = pd.DataFrame(tscpt_dets, columns=header)
df = df.sort_values('occurs', ascending=False)
df.to_csv('transcripts_train_set.csv', encoding='ascii', errors='replace', index=False)
df.plot(x="Transcription", y=["occurs", "spoken_by"], kind="bar").legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
plt.tight_layout(pad=0.5)
plt.savefig('transcripts_train_set.png')
plt.show()


# plotting graph

