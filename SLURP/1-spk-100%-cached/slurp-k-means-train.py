import ast

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
import pandas as pd
import pickle
from kmeans_pytorch import kmeans


config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = 'cpu'

wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
pwd = os.getcwd()
folder_path = os.path.join(pwd, 'slurp_models_k_means')

slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_FO.csv')
slurp_df = deepcopy(slurp_df)

speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232']
NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4
cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct = 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('training for speaker %s' % user_id)
    tmp = slurp_df[slurp_df['user_id'] == user_id]
    transcripts = np.unique(tmp['sentence'])
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
    for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
        transcript_list.append(transcript)
        # random choose one file with `transcription`
        rows = tmp[tmp['sentence'] == transcript]
        # sample only one audio file for each distinct transcription
        row = rows.iloc[np.random.randint(len(rows))]
        intent_list.append(row['intent'])
        # add the index to training set, won't use in eval below
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
        'transcript_list': transcript_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    filename = f'slurp-k-means-{user_id}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)

