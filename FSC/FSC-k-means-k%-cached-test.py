# %%
import data
import os
import time
import soundfile as sf
import torch
import numpy as np
import torch.nn.functional as F
import ast
from functools import reduce

from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from tqdm import tqdm
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import models
import pickle
from utils import audio_augment, get_token_and_weight, write_to_csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("../experiments/no_unfreezing.cfg")
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

THRESHOLD = 250
d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

model = models.Model(config).eval()
model.load_state_dict(
    torch.load("/models/SLURP/slurp-pretrained.pth", map_location=device))  # load trained model


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

pwd = os.getcwd()
if unseen:
    folder_path = os.path.join(pwd, '../models/FSC/models-FSC-k-means-unseen')
else:
    folder_path = os.path.join(pwd, '../models/FSC/models-FSC-k-means-k%-cached')

cumulative_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, total_train = 0, 0, 0, 0, 0, 0
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    if unseen:
        print('testing for unseen!')
        filename = f'FSC-k-means-unseen-{speakerId}.pkl'
    else:
        print('testing for 70% cached!')
        filename = f'FSC-k-means-70%-cached-{speakerId}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)

    # model = load_data['model']
    train_set = load_data['train_set']
    test_set = load_data['test_set']
    speakerId = load_data['speakerId']
    transcript_list = load_data['transcript_list']
    intent_list = load_data['intent_list']
    training_idxs = load_data['training_idxs']
    cluster_ids = load_data['cluster_ids']
    cluster_centers = load_data['cluster_centers']

    # ----------------- Evaluation -----------------
    # ----------------- prepare for cluster -----------------
    cluster_id_length = torch.tensor(list(map(len, cluster_ids)), dtype=torch.long, device=device)
    cluster_ids = pad_sequence(cluster_ids, batch_first=True, padding_value=0).to(device)
    cluster_centers = torch.stack(cluster_centers).to(device)
    # no reduction, loss on every sequence
    ctc_loss_k_means_eval = torch.nn.CTCLoss(reduction='none')
    # ------------------ variables to record performance --------------------
    tp, total, hits, l1_hits, l2_hits, l1_correct, l2_correct, l2_total = 0, 0, 0, 0, 0, 0, 0, 0
    for _, row in test_set.iterrows():
        if row[0] in training_idxs:
            continue
        # # of total evaluation samples
        total += 1
        wav = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav)
        x = torch.tensor(x, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            tick = time.time()
            # ----------------- l1 -------------------
            x_feature = model.pretrained_model.compute_cnn_features(x)
            dists = torch.cdist(x_feature, cluster_centers)
            # dists = 1 / dists
            # pred = torch.softmax(dists.swapaxes(1, 0), dim=-1)
            dists = dists.max(dim=-1)[0].unsqueeze(-1) - dists
            pred = dists.swapaxes(1, 0)
            pred_lengths = torch.full(size=(cluster_ids.shape[0],), fill_value=pred.shape[0], dtype=torch.long)
            loss = ctc_loss_k_means_eval(pred.log_softmax(dim=-1), cluster_ids, pred_lengths, cluster_id_length)
            pred_intent = loss.argmin().item()
            if loss[pred_intent] < THRESHOLD:
                # go with l1: kmeans
                # print('l1 hit: ', row['transcription'])
                hits += 1
                if row['cache'] == intent_list[pred_intent]:
                    tp += 1

    hit_rate, cache_acc, total_acc = 0, 0, 0
    if total >= 5:  # skip for users with < 5 eval samples
        cumulative_sample += total
        cumulative_correct += tp + (total - hits) * 0.988
        cumulative_hit += hits
        cumulative_hit_correct += tp
        total_train += len(training_idxs)
        hit_rate = round((hits / total), 4)
        total_acc = round((tp + (total - hits) * 0.988) / total, 4)
        print('Train: %d Test: %d' % (len(training_idxs), total))
        print('for speaker ', speakerId)
        if hits:
            cache_acc = round((tp / hits), 4)
            print('hit rate %.4f' % hit_rate)
            print('cache_acc %.3f' % cache_acc)
        print('total_acc: %.3f' % total_acc)
    else:
        print('not enough samples: %s' % speakerId)
    values = [speakerId, THRESHOLD, len(training_idxs), total, hit_rate, cache_acc, total_acc]
    print(values)

print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('Threshold ', THRESHOLD)
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()

