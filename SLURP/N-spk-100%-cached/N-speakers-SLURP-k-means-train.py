import ast
import data
import os
import time
import soundfile as sf
import torch
import numpy as np
from functools import reduce
from copy import deepcopy
from tqdm import tqdm
from utils import audio_augment, initialize_SLURP_L1, train_test_split, save_model_SLURP_L1
import models
import pandas as pd
import pickle
import random
from kmeans_pytorch import kmeans
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/SLURP/slurp_mini_FE_MO_ME_FO_UNK.csv')
df = deepcopy(df)

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4

config, wav_path, folder_path, spk_group = initialize_SLURP_L1(folder_name='models/SLURP/models-N-speakers-SLURP-k-means')
for _, speakers in tqdm(enumerate(spk_group), total=len(spk_group)):
    print('training for speakers ', speakers)
    filename = f'3-spk-SLURP-k-means-{speakers}.pkl'
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        continue
    train_set, test_set = train_test_split(df, speakers)
    transcripts = np.unique(train_set['sentence'])
    training_idxs = set()
    # the following are used for evaluation
    transcript_list, phoneme_list, intent_list, cluster_ids, cluster_centers = [], [], [], [], []
    # create a new model for this previous speaker
    model = models.Model(config)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location='cpu'))  # load trained model

    # training
    print('TRAINING START')
    for _, row in train_set.iterrows():
        transcript = row['sentence']
        transcript_list.append(transcript)
        intent_list.append(row['intent'])
        training_idxs.add(row[0])

        # load the audio file
        wav = os.path.join(wav_path, row['recording_path'])
        x, _ = sf.read(wav)
        x_aug = torch.tensor(audio_augment(x), dtype=torch.float)

        # ----------------- kmeans cluster -----------------
        feature = model.pretrained_model.compute_cnn_features(x_aug)
        cluster_id, cluster_center = kmeans(X=feature.reshape(-1, feature.shape[-1]), num_clusters=NUM_CLUSTERS,
                                            distance=dist, tol=tol)
        # save the cluster center
        intention_label = []
        prev = None
        # collapses the cluster predictions
        for l in cluster_id.view(feature.shape[0], -1)[0]:
            if prev is None or prev != l:
                intention_label.append(l.item())
            prev = l
        cluster_ids.append(torch.tensor(intention_label, dtype=torch.long))
        cluster_centers.append(cluster_center)

    print('TRAINING END')
    saved_data = {
        'model': model,
        'train_set': train_set,
        'test_set': test_set,
        'speakers': speakers,
        'transcript_list': transcript_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    filename = f'3-spk-SLURP-k-means-{speakers}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)
