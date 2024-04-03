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
import math
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from tqdm import tqdm
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
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
speechcache_config = data.read_config("experiments/speechcache.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
_, _, _ = data.get_SLU_datasets(speechcache_config)  # used to set config.num_phonemes
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

d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

# 70% cached
def train_test_split_70(speakerId):
    train_set, test_set = pd.DataFrame(columns=new_train_df.columns), pd.DataFrame(columns=new_train_df.columns)
    tmp = new_train_df[new_train_df['speakerId'] == speakerId]
    transcripts = np.unique(tmp['transcription'])
    for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
        rows = tmp[tmp['transcription'] == transcript]
        # sample only one audio file for each distinct transcription
        train_set = train_set.append(rows.iloc[0])
        if len(rows) > 1:
            test_set = test_set.append(rows.iloc[1])

    print(len(train_set), len(test_set))
    random_rows = train_set.sample(n=int(0.3 * len(train_set)))  # i.e 70% cached
    test_set = test_set.append(random_rows)
    train_set = train_set.drop(random_rows.index)
    print('Train and test set created!! 70% cached')
    return train_set, test_set

# unseen or 0% cached
def train_test_split_0(speakerId):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    print('speaker %s' % speakerId)
    tmp = new_train_df[new_train_df['speakerId'] == speakerId]
    tmp = tmp.sample(frac=1, random_state=42).reset_index(drop=True)
    transcripts = tmp['transcription']
    train_len = int(0.4 * len(transcripts))
    train_set = tmp.iloc[:train_len]
    test_set = tmp.iloc[train_len - 1:-1]
    train_tscpt = np.unique(train_set['transcription'])
    for _, row in test_set.iterrows():
        if row['transcription'] in train_tscpt:
            train_set = train_set.append(row, ignore_index=True)
            test_set = test_set[test_set['transcription'] != row['transcription']]

    train_pct = math.ceil(len(train_set) / len(transcripts) * 100)
    test_pct = math.floor(len(test_set) / len(transcripts) * 100)
    print('speaker-%s train=%d test=%d ratio: %d:%d' % (speakerId, len(train_set), len(test_set), train_pct, test_pct))
    train_df = pd.concat([train_df, train_set])
    test_df = pd.concat([test_df, test_set])

    if not bool(set(train_set['transcription']).intersection(test_set['transcription'])):
        print('Train and test set created !!')
        return train_df, test_df
    else:
        print('Overlap was found!')
        exit()

pwd = os.getcwd()
speakers = np.unique(new_train_df['speakerId'])
speakers_to_remove = ['4aGjX3AG5xcxeL7a', '5pa4DVyvN2fXpepb', '9Gmnwa5W9PIwaoKq', 'KLa5k73rZvSlv82X',
                      'LR5vdbQgp3tlMBzB', 'OepoQ9jWQztn5ZqL', 'X4vEl3glp9urv4GN', 'Ze7YenyZvxiB4MYZ',
                      'eL2w4ZBD7liA85wm', 'eLQ3mNg27GHLkDej', 'ldrknAmwYPcWzp4N', 'mzgVQ4Z5WvHqgNmY',
                      'nO2pPlZzv3IvOQoP2', 'oNOZxyvRe3Ikx3La', 'roOVZm7kYzS5d4q3', 'rwqzgZjbPaf5dmbL',
                      'wa3mwLV3ldIqnGnV', 'xPZw23VxroC3N34k', 'ywE435j4gVizvw3R', 'zwKdl7Z2VRudGj2L',
                      '35v28XaVEns4WXOv', 'YbmvamEWQ8faDPx2', 'neaPN7GbBEUex8rV', '9mYN2zmq7aTw4Blo']
speakers = [item for item in speakers if item not in speakers_to_remove]

unseen = False
cached = True

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4
cumulative_sample, num_nan_train = 0, 0
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    print('speaker %s' % speakerId)
    if cached:
        train_set, test_set = train_test_split_70(speakerId)
        folder_path = os.path.join(pwd, 'models-FSC-k-means-k%-cached')
    if unseen:
        folder_path = os.path.join(pwd, 'models-FSC-k-means-unseen')
        filename = f'FSC-multicache-unseen-{speakerId}.pkl'
        file_path = folder_path + '/' + filename
        if os.path.exists(file_path):
            print('skipped for ', speakerId)
            continue
        train_set, test_set = train_test_split_0(speakerId)
    model = models.Model(config)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

    training_idxs = set()
    # the following three are used for evaluation
    transcript_list = []
    intent_list = []
    cluster_ids = []
    cluster_center = None
    cluster_centers = []

    # training
    print('TRAINING START')
    for _, row in train_set.iterrows():
        transcript = row['transcription']
        # print(transcript)
        optim.zero_grad()
        transcript_list.append(transcript)
        intent_list.append(row['cache'])
        # add the index to training set, won't use in eval below
        training_idxs.add(row['index'])

        # load the audio file
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
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

    if num_nan_train:
        print('nan in train happens %d times' % num_nan_train)

    saved_data = {
        'model': model,
        'speakerId': speakerId,
        'train_set': train_set,
        'test_set': test_set,
        'transcript_list': transcript_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    if unseen:
        filename = f'FSC-k-means-unseen-{speakerId}.pkl'
    if cached:
        filename = f'FSC-k-means-70%-cached-{speakerId}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)
