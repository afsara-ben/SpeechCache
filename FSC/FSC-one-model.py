# %%
import pandas as pd
from charset_normalizer import models
import data
import os
import time
import soundfile as sf
import torch
import numpy as np
import pandas as pd
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
from thop import profile
import ast

device = torch.device('cpu')
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

# new_test_df = deepcopy(test_dataset.df)
# act_obj_loc_idxs = []
# for _, row in new_test_df.iterrows():
#     act, obj, loc = row['action'], row['object'], row['location']
#     act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
# new_test_df['cache'] = act_obj_loc_idxs
# new_test_df.head()
# data = {
#     "text": new_test_df['transcription'],
#     "label":  new_test_df['cache']
# }
#
# new_test_df = pd.DataFrame(data)
# new_test_df.to_csv('test_data_cache.csv', index=None)

# process data
new_train_df = deepcopy(train_dataset.df)
act_obj_loc_idxs = []
for _, row in new_train_df.iterrows():
    act, obj, loc = row['action'], row['object'], row['location']
    act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
new_train_df['cache'] = act_obj_loc_idxs
new_train_df.head()
# new_train_df.to_csv('train_data_cache.csv', index=None)

with open('phoneme_list.txt', 'r') as file:
    id2phoneme = ast.literal_eval(file.read())
phoneme2id = {v: k for k, v in id2phoneme.items()}

pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models-FSC-phoneme-ctc')

# emulate an oracle cloud model
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()


num_nan_train, nan_nan_eval = 0, 0
speakers = np.unique(new_train_df['speakerId'])
speakers_to_remove = ['4aGjX3AG5xcxeL7a', '5pa4DVyvN2fXpepb', '9Gmnwa5W9PIwaoKq', 'KLa5k73rZvSlv82X', 'LR5vdbQgp3tlMBzB', 'OepoQ9jWQztn5ZqL', 'X4vEl3glp9urv4GN', 'Ze7YenyZvxiB4MYZ', 'eL2w4ZBD7liA85wm', 'eLQ3mNg27GHLkDej', 'ldrknAmwYPcWzp4N', 'mzgVQ4Z5WvHqgNmY', 'nO2pPlZzv3IvOQoP2', 'oNOZxyvRe3Ikx3La', 'roOVZm7kYzS5d4q3', 'rwqzgZjbPaf5dmbL', 'wa3mwLV3ldIqnGnV', 'xPZw23VxroC3N34k', 'ywE435j4gVizvw3R', 'zwKdl7Z2VRudGj2L', '35v28XaVEns4WXOv', 'YbmvamEWQ8faDPx2', 'neaPN7GbBEUex8rV', '9mYN2zmq7aTw4Blo']
speakers = [item for item in speakers if item not in speakers_to_remove]
# speakers = ['Xygv5loxdZtrywr9']

model = models.Model(config)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct = 0, 0, 0, 0
training_idxs = set()
# the following three are used for evaluation
transcript_list = []
phoneme_list = []
intent_list = []
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    print('training for speaker %s' % speakerId)
    tmp = new_train_df[new_train_df['speakerId'] == speakerId]
    transcripts = np.unique(tmp['transcription'])
    # training
    for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
        optim.zero_grad()
        # remove ending punctuation from the transcript
        phoneme_seq = reduce(lambda x, y: x + ['sp'] + y,
                             [d[tk][0] if tk in d else [] for tk in tknz.tokenize(transcript.lower())])
        transcript_list.append(transcript)
        # create (N, T) label
        phoneme_label = torch.tensor(
            [phoneme2id[ph[:-1]] if ph[-1].isdigit() else phoneme2id[ph] for ph in phoneme_seq],
            dtype=torch.long, device=device)
        phoneme_list.append(phoneme_label)
        # print('phoneme_list ', phoneme_list)
        # random choose one file with `transcription`
        rows = tmp[tmp['transcription'] == transcript]
        # sample only one audio file for each distinct transcription
        row = rows.iloc[np.random.randint(len(rows))]
        intent_list.append(row['cache'])
        # add the index to training set, won't use in eval below
        training_idxs.add(row['index'])

        # load the audio file
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
        x_aug = torch.tensor(audio_augment(x), dtype=torch.float, device=device)
        phoneme_label = phoneme_label.repeat(x_aug.shape[0], 1)
        phoneme_pred = model.pretrained_model.compute_phonemes(x_aug)
        pred_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_pred.shape[0], dtype=torch.long)
        label_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_label.shape[-1], dtype=torch.long)

        loss = ctc_loss(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
        # FIXME implement better fix for nan loss
        if torch.isnan(loss).any():
            num_nan_train = num_nan_train + 1
            print('nan training on speaker: %s' % speakerId)
            optim.zero_grad()
        loss.backward()
        optim.step()
    if num_nan_train:
        print('nan in train happens %d times' % num_nan_train)

saved_data = {
    'model': model,
    'transcript_list': transcript_list,
    'phoneme_list': phoneme_list,
    'intent_list': intent_list,
    'training_idxs': training_idxs,
}

filename = f'FSC-phoneme-ctc-all-spk.pkl'
file_path = os.path.join(folder_path, filename)
with open(file_path, 'wb') as f:
    pickle.dump(saved_data, f)
