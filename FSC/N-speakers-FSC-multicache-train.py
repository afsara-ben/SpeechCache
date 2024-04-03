# %%
import data
import os
import time
import soundfile as sf
import torch
import numpy as np
import torch.nn.functional as F
import ast
import csv
from functools import reduce
from collections import defaultdict
import pickle
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from tqdm import tqdm
from torchinfo import summary
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from kmeans_pytorch import kmeans
import models
from utils import audio_augment, get_token_and_weight
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
device = "cpu"
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

# emulate an oracle cloud model
config.phone_rnn_num_hidden = [128, 128]
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models-N-speakers-FSC-multicache')

d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4

spk_group = [['5o9BvRGEGvhaeBwA', 'ro5AaKwypZIqNEp2', 'KqDyvgWm4Wh8ZDM7'], ['73bEEYMKLwtmVwV43', 'R3mXwwoaX9IoRVKe', 'ppzZqYxGkESMdA5Az'], ['2ojo7YRL7Gck83Z3', 'kxgXN97ALmHbaezp', 'zaEBPeMY4NUbDnZy'], ['8e5qRjN7dGuovkRY', 'xRQE5VD7rRHVdyvM', 'n5XllaB4gZFwZXkBz'], ['oRrwPDNPlAieQr8Q', 'W4XOzzNEbrtZz4dW', 'Z7BXPnplxGUjZdmBZ'], ['xEYa2wgAQof3wyEO', 'ObdQbr9wyDfbmW4E', 'EExgNZ9dvgTE3928'], ['gvKeNY2D3Rs2jRdL', 'NWAAAQQZDXC5b9Mk', 'W7LeKXje7QhZlLKe'], ['oXjpaOq4wVUezb3x', 'zZezMeg5XvcbRdg3', 'Gym5dABePPHA8mZK9'], ['Xygv5loxdZtrywr9', 'AY5e3mMgZkIyG3Ox', 'anvKyBjB5OiP5dYZ'], ['G3QxQd7qGRuXAZda', 'WYmlNV2rDkSaALOE', 'RjDBre8jzzhdr4YL'], ['9EWlVBQo9rtqRYdy', 'M4ybygBlWqImBn9oZ', 'jgxq52DoPpsR9ZRx'], ['kNnmb7MdArswxLYw', 'qNY4Qwveojc8jlm4', 'mor8vDGkaOHzLLWBp'], ['5BEzPgPKe8taG9OB', 'AvR9dePW88IynbaE', '52XVOeXMXYuaElyw'], ['2BqVo8kVB2Skwgyb', 'ZebMRl5Z7dhrPKRD', 'R3mexpM2YAtdPbL7'], ['d2waAp3pEjiWgrDEY', 'DWmlpyg93YCXAXgE', '7NEaXjeLX3sg3yDB'], ['xwzgmmv5ZOiVaxXz', 'd3erpmyk9yFlVyrZ', 'BvyakyrDmQfWEABb'], ['Rq9EA8dEeZcEwada2', 'gNYvkbx3gof2Y3V9', 'g2dnA9Wpvzi2WAmZ'], ['xwpvGaaWl5c3G5N3', 'DMApMRmGq5hjkyvX']]

for speakers in spk_group:
    print('training for speakers ', speakers)
    num_nan_train, nan_nan_eval = 0, 0
    train_set, test_set = pd.DataFrame(columns=new_train_df.columns), pd.DataFrame(columns=new_train_df.columns)
    for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
        tmp = new_train_df[new_train_df['speakerId'] == speakerId]
        transcripts = np.unique(tmp['transcription'])
        for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
            rows = tmp[tmp['transcription'] == transcript]
            # sample only one audio file for each distinct transcription
            train_set = train_set.append(rows.iloc[0])
            if len(rows) > 1:
                test_set = test_set.append(rows.iloc[1])

    print('Train and test set created!!')
    transcripts = np.unique(train_set['transcription'])

    model = models.Model(config)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

    training_idxs = set()
    # the following three are used for evaluation
    transcript_list = []
    phoneme_list = []
    intent_list = []
    cluster_ids = []
    cluster_center = None
    cluster_centers = []

    # training
    print('TRAINING START')
    for _, row in train_set.iterrows():
        transcript = row['transcription']
        optim.zero_grad()
        transcript_list.append(transcript)
        intent_list.append(row['cache'])
        # add the index to training set, won't use in eval below
        training_idxs.add(row['index'])

        # load the audio file
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
        # augmentation
        x_aug = torch.tensor(audio_augment(x), dtype=torch.float, device=device)

        # ----------------- kmeans cluster -----------------
        feature = model.pretrained_model.compute_cnn_features(x_aug)
        cluster_id, cluster_center = kmeans(X=feature.reshape(-1, feature.shape[-1]), num_clusters=NUM_CLUSTERS,
                                            distance=dist, tol=tol, device=device)
        # save the cluster center
        intention_label = []
        prev = None
        # collapses the cluster predictions
        for l in cluster_id.view(feature.shape[0], -1)[0]:
            if prev is None or prev != l:
                intention_label.append(l.item())
            prev = l
        cluster_ids.append(torch.tensor(intention_label, dtype=torch.long, device=device))
        cluster_centers.append(cluster_center)

        # ----------------- phoneme ctc -------------------
        # phoneme_seq, weight = get_token_and_weight(transcript.lower())
        phoneme_seq = reduce(lambda x, y: x + ['sp'] + y,
                             [d[tk][0] if tk in d else [] for tk in tknz.tokenize(transcript.lower())])
        phoneme_label = torch.tensor(
            [phoneme2id[ph[:-1]] if ph[-1].isdigit() else phoneme2id[ph] for ph in phoneme_seq],
            dtype=torch.long, device=device)
        phoneme_list.append(phoneme_label)
        phoneme_label = phoneme_label.repeat(x_aug.shape[0], 1)
        phoneme_pred = model.pretrained_model.compute_phonemes(x_aug)
        pred_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_pred.shape[0], dtype=torch.long)
        label_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_label.shape[-1], dtype=torch.long)

        loss = ctc_loss(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
        # FIXME implement better fix for nan loss
        if torch.isnan(loss).any():
            num_nan_train = num_nan_train + 1
            print('nan training on speaker: %s' % speakers)
            optim.zero_grad()
        loss.backward()
        optim.step()
    if num_nan_train:
        print('nan in train happens %d times' % num_nan_train)

    print('TRAINING END')

    saved_data = {
        'model': model,
        'speakers': speakers,
        'train_set': train_set,
        'test_set': test_set,
        'transcript_list': transcript_list,
        'phoneme_list': phoneme_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    filename = f'3-spk-FSC-multicache-{speakers}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)
