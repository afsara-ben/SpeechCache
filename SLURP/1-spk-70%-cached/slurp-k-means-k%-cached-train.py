import ast

import data
import os
import time
import random
import soundfile as sf
import torch
import numpy as np
from functools import reduce
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from tqdm import tqdm
from utils import audio_augment
import models
import pandas as pd
import pickle
from kmeans_pytorch import kmeans
import warnings

warnings.filterwarnings("ignore")

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = 'cpu'

pwd = os.getcwd()
wav_path = os.path.join(pwd, 'slurp_real/')
folder_path = os.path.join(pwd, 'models/SLURP/models-SLURP-k-means-k%-cached')

def train_test_split_70(df):
    train_set, test_set = pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    # get all transcript for speaker X
    transcripts = np.unique(df['sentence'])

    # Randomly select the items and put them into another list
    test_tscpt = np.random.choice(transcripts, size=int(len(transcripts) * 0.3), replace=False)

    # Remove the selected items from the original list
    train_tscpt = [item for item in transcripts if item not in test_tscpt]

    # train test split for each transcript in the 70% list
    for tscpt_idx, transcript in tqdm(enumerate(train_tscpt), total=len(train_tscpt), leave=False):
        rows = df[df['sentence'] == transcript]
        midpoint = len(rows) // 2
        train_set = train_set.append(rows[:midpoint])
        if len(rows) > 1:
            test_set = test_set.append(rows[midpoint:])

    # add the held out 30% transcript to test set, meaning these 30% tscpt never occur in train set
    for tscpt_idx, transcript in tqdm(enumerate(test_tscpt), total=len(test_tscpt), leave=False):
        rows = df[df['sentence'] == transcript]
        test_set = test_set.append(rows[:])

    print('Train and test set created!! 70% cached')
    return train_set, test_set


slurp_df = pd.read_csv(os.path.join(pwd, 'slurp_mini_FE_MO_ME_FO_UNK.csv'))
slurp_df = deepcopy(slurp_df)
speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232']
# speakers = ['MO-433']
NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4
cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct = 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('training for speaker %s' % user_id)
    tmp = slurp_df[slurp_df['user_id'] == user_id]
    train_set, test_set = train_test_split_70(tmp)
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
    for _, row in train_set.iterrows():
        transcript = row['sentence']
        transcript_list.append(transcript)
        intent_list.append(row['intent'])
        training_idxs.add(row[0])

        # load the audio file
        wav = wav_path + row['recording_path']
        x, _ = sf.read(wav)
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

    saved_data = {
        'model': model,
        'speakerId': user_id,
        'train_set':train_set,
        'test_set':test_set,
        'transcript_list': transcript_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    filename = f'slurp-k-means-70%-cached-{user_id}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)

