# %%
import data
import os
import time
import soundfile as sf
import torch
import numpy as np
import torch.nn.functional as F
import ast
import pickle
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from kmeans_pytorch import kmeans
import models
from utils import audio_augment, FSC_train_test_split, initialize_N_spks_FSC_L1
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
device = "cpu"

d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4

pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models-N-speakers-FSC-k-means')
config, new_train_df, spk_group = initialize_N_spks_FSC_L1()
for speakers in spk_group:
    print('training for speakers ', speakers)
    train_set, test_set = FSC_train_test_split(new_train_df, speakers)

    transcripts = np.unique(train_set['transcription'])

    model = models.Model(config)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

    training_idxs = set()
    # the following three are used for evaluation
    transcript_list, intent_list, cluster_ids, cluster_centers = [], [], [], []

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

    print('TRAINING END')

    saved_data = {
        'model': model,
        'speakers': speakers,
        'train_set': train_set,
        'test_set': test_set,
        'transcript_list': transcript_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    filename = f'3-spk-FSC-k-means-{speakers}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)
