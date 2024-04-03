import data
import os
import time
import soundfile as sf
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import librosa
import models

from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from kmeans_pytorch import kmeans
import pickle
from utils import audio_augment, reduce_dimensions

device = 'cpu'
config = data.read_config("experiments/no_unfreezing.cfg")
speechcache_config = data.read_config("experiments/speechcache.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
_, _, _ = data.get_SLU_datasets(speechcache_config)  # used to set config.num_phonemes
print('config load ok')
dataset_to_use = valid_dataset
dataset_to_use.df.head()
test_dataset.df.head()

pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models-FSC-k-means')

# create model
speechcache_config = data.read_config("experiments/speechcache.cfg")
_, _, _ = data.get_SLU_datasets(speechcache_config)

act_obj_loc = {}
for dataset in [train_dataset, valid_dataset, test_dataset]:
    for _, row in dataset.df.iterrows():
        act, obj, loc = row['action'], row['object'], row['location']
        if (act, obj, loc) not in act_obj_loc:
            act_obj_loc[(act, obj, loc)] = len(act_obj_loc)
len(act_obj_loc)

new_train_df = deepcopy(train_dataset.df)
act_obj_loc_idxs = []
for _, row in new_train_df.iterrows():
    act, obj, loc = row['action'], row['object'], row['location']
    act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
new_train_df['cache'] = act_obj_loc_idxs
new_train_df.head()

d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

speakers = np.unique(new_train_df['speakerId'])
eval_only = False
cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct = 0, 0, 0, 0
ctc_loss_eval = torch.nn.CTCLoss(reduction='none')
THRESHOLD = 500
dist = 'euclidean'
tol = 1e-4
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    print('training for speaker %s' % speakerId)
    # create a new model for this previous speaker
    model = models.Model(config)
    model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

    tmp = new_train_df[new_train_df['speakerId'] == speakerId]
    transcripts = np.unique(tmp['transcription'])
    training_idxs = set()
    # the following three are used for evaluation
    transcript_list = []
    cluster_ids = []
    intent_list = []
    cluster_center = None
    cluster_centers = []
    # training
    for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
        rows = tmp[tmp['transcription'] == transcript]
        # sample only one audio file for each distinct transcription
        row = rows.iloc[np.random.randint(len(rows))]  # selects a random row from rows
        intent_list.append(row['cache'])
        # add the index to training set, won't use in eval below
        training_idxs.add(row['index'])
        # load the audio file
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
        # augmentation
        x_aug = torch.tensor(audio_augment(x), dtype=torch.float, device=device)

        feature = model.pretrained_model.compute_cnn_features(x_aug)
        print(feature.shape)
        cluster_id, cluster_center = kmeans(X=feature.reshape(-1, feature.shape[-1]), num_clusters=70, distance=dist, tol=tol,
                                            device=device)
        # print('cluster_id ', cluster_id.shape)
        # print('cluster_center ', cluster_center.shape)

        # save the cluster center - collapsing
        intention_label = []
        prev = None
        for l in cluster_id.view(feature.shape[0], -1)[0]:  # cluster_id.view(feature.shape[0], -1)[0] takes first row of [16,len,60], reverses the flattened shape
            if prev is None or prev != l:
                intention_label.append(l)
            prev = l

        cluster_ids.append(torch.tensor(intention_label, dtype=torch.long,
                                        device=device))  # appends the cluster labels for each wav file
        cluster_centers.append(cluster_center)
        # print('cluster_ids ', cluster_ids)
        # print('cluster_centers ', cluster_centers)
    saved_data = {
        'model': model,
        'speakerId': speakerId,
        'transcript_list': transcript_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    filename = f'FSC-k-means-{speakerId}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)

