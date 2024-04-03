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
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import models
from utils import audio_augment, get_token_and_weight

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

_, _, _ = data.get_SLU_datasets(speechcache_config)

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

def create_csv_file(filename, headers):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def write_to_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# emulate an oracle cloud model
config.phone_rnn_num_hidden = [128, 128]
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

hit_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
l1_hit_dict = defaultdict(lambda: defaultdict(int))
l2_hit_dict = defaultdict(lambda: defaultdict(int))
l1_correct_dict = defaultdict(lambda: defaultdict(int))
l2_correct_dict = defaultdict(lambda: defaultdict(int))
l1_l2_hit_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
l1_l2_correct_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
correct_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
total_dict = defaultdict(int)
l2_total_dict = defaultdict(int)

d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()
speakers = np.unique(new_train_df['speakerId'])
speakers_to_remove = ['4aGjX3AG5xcxeL7a', '5pa4DVyvN2fXpepb', '9Gmnwa5W9PIwaoKq', 'KLa5k73rZvSlv82X', 'LR5vdbQgp3tlMBzB', 'OepoQ9jWQztn5ZqL', 'X4vEl3glp9urv4GN', 'Ze7YenyZvxiB4MYZ', 'eL2w4ZBD7liA85wm', 'eLQ3mNg27GHLkDej', 'ldrknAmwYPcWzp4N', 'mzgVQ4Z5WvHqgNmY', 'nO2pPlZzv3IvOQoP2', 'oNOZxyvRe3Ikx3La', 'roOVZm7kYzS5d4q3', 'rwqzgZjbPaf5dmbL', 'wa3mwLV3ldIqnGnV', 'xPZw23VxroC3N34k', 'ywE435j4gVizvw3R', 'zwKdl7Z2VRudGj2L', '35v28XaVEns4WXOv', 'YbmvamEWQ8faDPx2', 'neaPN7GbBEUex8rV', '9mYN2zmq7aTw4Blo']
speakers = [item for item in speakers if item not in speakers_to_remove]
# speakers = ['2BqVo8kVB2Skwgyb']
# speakers = ['W7LeKXje7QhZlLKe']
# speakers = ['2ojo7YRL7Gck83Z3'] #test with this to see if phoneme_ctc result is ok or not, onno jaygay > 1 acc. diche
# speakers = ['Xygv5loxdZtrywr9']

L1_THRESHOLDS = [500]
L2_THRESHOLDS = [40]
results_file = f'multi-cache-{L1_THRESHOLDS[0]}-{L2_THRESHOLDS[0]}.csv'
headers = ['SpeakerId', 'samples', 'L1_Threshold', 'l1-hits', 'l1-correct', 'l1-Hit Rate', 'l1-cache_acc',
           'L2_Threshold', 'l2-hits', 'l2-correct', 'l2-Hit Rate',
           'l2-cache_acc', 'Total Acc.']
create_csv_file(results_file, headers)

NUM_CLUSTERS = 80
dist = 'euclidean'
tol = 1e-5
cumulative_sample, num_nan_train = 0, 0
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    tmp = new_train_df[new_train_df['speakerId'] == speakerId]

    pwd = os.getcwd()
    folder_path = os.path.join(pwd, 'model-multicache-80-euclidean-1e-5')
    filename = f'model_multicache_{speakerId}_{NUM_CLUSTERS}_{dist}_{tol}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)
    model = load_data['model']
    transcript_list = load_data['transcript_list']
    phoneme_list = load_data['phoneme_list']
    intent_list = load_data['intent_list']
    training_idxs = load_data['training_idxs']
    cluster_ids = load_data['cluster_ids']
    cluster_centers = load_data['cluster_centers']

    # ----------------- Evaluation -----------------
    # ----------------- prepare for cluster -----------------
    cluster_id_length = torch.tensor(list(map(len, cluster_ids)), dtype=torch.long, device=device)
    cluster_ids = pad_sequence(cluster_ids, batch_first=True, padding_value=0).to(device)
    cluster_centers = torch.stack(cluster_centers).to(device)
    # ----------------- prepare for phoneme -----------------
    # prepare all the potential phoneme sequences
    label_lengths = torch.tensor(list(map(len, phoneme_list)), dtype=torch.long)
    phoneme_label = pad_sequence(phoneme_list, batch_first=True).to(device)
    # no reduction, loss on every sequence
    ctc_loss_k_means_eval = torch.nn.CTCLoss(reduction='none')
    ctc_loss_phoneme_eval = torch.nn.CTCLoss(reduction='none')
    # ------------------ variables to record performance --------------------
    tp, total, l2_total, hits, l1_hits, l2_hits, l1_correct, l2_correct = 0, 0, 0, 0, 0, 0, 0, 0
    for _, row in tmp.iterrows():
        if row['index'] in training_idxs:
            continue
        # # of total evaluation samples
        total += 1
        total_dict[speakerId] += 1
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
        x = torch.tensor(x, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            tick = time.time()
            # ----------------- l1 -------------------
            x_feature = model.pretrained_model.compute_cnn_features(x)
            dists = torch.cdist(x_feature, cluster_centers)
            dists = dists.max(dim=-1)[0].unsqueeze(-1) - dists
            pred = dists.swapaxes(1, 0)
            pred_lengths = torch.full(size=(cluster_ids.shape[0],), fill_value=pred.shape[0], dtype=torch.long)
            k_means_loss = ctc_loss_k_means_eval(pred.log_softmax(dim=-1), cluster_ids, pred_lengths, cluster_id_length)
            pred_intent = k_means_loss.argmin().item()
            # ------------------ l2 -------------------
            phoneme_pred = model.pretrained_model.compute_phonemes(x)
            # repeat it #sentence times to compare with ground truth
            phoneme_pred = phoneme_pred.repeat(1, phoneme_label.shape[0], 1)
            pred_lengths = torch.full(size=(phoneme_label.shape[0],), fill_value=phoneme_pred.shape[0],
                                      dtype=torch.long)
            phoneme_loss = ctc_loss_phoneme_eval(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
            pred_result = phoneme_loss.argmin()

            if torch.isnan(phoneme_loss).any():
                print('nan eval on speaker: %s' % speakerId)

            # branch based on loss value
            for L1_THRESHOLD in L1_THRESHOLDS:  # 300,400,500
                if k_means_loss[pred_intent] < L1_THRESHOLD:
                    # go with l1: kmeans
                    l1_hit_dict[speakerId][L1_THRESHOLD] += 1
                    if intent_list[pred_intent] == row['cache']:
                        l1_correct_dict[speakerId][L1_THRESHOLD] += 1
                    else:
                        print('%s,%s' % (row['transcription'], transcript_list[pred_intent]))
                else:
                    l2_total += 1
                    l2_total_dict[speakerId] += 1
                    for L2_THRESHOLD in L2_THRESHOLDS:
                        if phoneme_loss.min() <= L2_THRESHOLD:
                            l2_hit_dict[speakerId][L2_THRESHOLD] += 1
                            if row['cache'] == intent_list[pred_result]:
                                l2_correct_dict[speakerId][L2_THRESHOLD] += 1
                            else:
                                print('%s,%s' % (row['transcription'], transcript_list[pred_result]))
                        else:
                            # do the calculation
                            # cloud_model.predict_intents(x)
                            print('sent to cloud')

    if total >= 5:  # skip for users with < 5 eval samples
        cumulative_sample += total
        print('for speaker ', speakerId)
        for L1_THRESHOLD in L1_THRESHOLDS:  # 300,400,500
            l1_hit_rate, l1_cache_acc = 0, 0
            # avoid divided by zero
            if l1_hit_dict[speakerId][L1_THRESHOLD]:
                l1_cache_acc = round(l1_correct_dict[speakerId][L1_THRESHOLD] / l1_hit_dict[speakerId][L1_THRESHOLD], 4)
                l1_hit_rate = round(l1_hit_dict[speakerId][L1_THRESHOLD] / total_dict[speakerId], 4)
            else:
                l1_hit_rate = 0
                l1_cache_acc = 0
            for L2_THRESHOLD in L2_THRESHOLDS:  # 30,40
                l2_hit_rate, l2_cache_acc = 0, 0
                # avoid divided by zero
                if l2_hit_dict[speakerId][L2_THRESHOLD]:
                    l2_cache_acc = round(
                        l2_correct_dict[speakerId][L2_THRESHOLD] / l2_hit_dict[speakerId][L2_THRESHOLD], 6)
                    l2_hit_rate = round(l2_hit_dict[speakerId][L2_THRESHOLD] / l2_total, 6)
                else:
                    l2_hit_rate = 0
                    l2_cache_acc = 0

                total_acc = round((total - l1_hit_dict[speakerId][L1_THRESHOLD] - l2_hit_dict[speakerId][L2_THRESHOLD] +
                                   l1_correct_dict[speakerId][L1_THRESHOLD] + l2_correct_dict[speakerId][
                                       L2_THRESHOLD]) / total, 4)
                print('writing to csv %s-%d-%d ' % (speakerId, L1_THRESHOLD, L2_THRESHOLD))
                values = [speakerId, total_dict[speakerId], L1_THRESHOLD, l1_hit_dict[speakerId][L1_THRESHOLD],
                          l1_correct_dict[speakerId][L1_THRESHOLD], l1_hit_rate, l1_cache_acc, L2_THRESHOLD,
                          l2_hit_dict[speakerId][L2_THRESHOLD], l2_correct_dict[speakerId][L2_THRESHOLD], l2_hit_rate,
                          l2_cache_acc, total_acc]
                print(values)
                # write_to_csv(results_file, values)
    else:
        print('not enough samples: %s' % speakerId)

l1_hits = defaultdict(int)
l2_hits = defaultdict(int)
l1_correct = defaultdict(int)
l2_correct = defaultdict(int)
cumulative_acc = defaultdict(lambda: defaultdict(float))

for L1_THRESHOLD in L1_THRESHOLDS:  # 300,400,500
    print('L1_THRESHOLD = %d' % L1_THRESHOLD)
    total = 0
    for k in l1_hit_dict:
        l1_hits[L1_THRESHOLD] += l1_hit_dict[k][L1_THRESHOLD]
        l1_correct[L1_THRESHOLD] += l1_correct_dict[k][L1_THRESHOLD]
        total += total_dict[k]
    print('cumulative_l1_hit_rate-%d %.4f' % (L1_THRESHOLD, (l1_hits[L1_THRESHOLD] / total)))
    if l1_hits[L1_THRESHOLD]:
        print('cumulative l1-hit-acc-%d %.4f' % (L1_THRESHOLD, (l1_correct[L1_THRESHOLD] / l1_hits[L1_THRESHOLD])))
    else:
        print('cumulative l1-hit-acc-%d  = 0' % L1_THRESHOLD)

for L2_THRESHOLD in L2_THRESHOLDS:  # 30,40
    print('L2_THRESHOLD = %d' % L2_THRESHOLD)
    l2_total = 0
    for k in l2_hit_dict:
        l2_hits[L2_THRESHOLD] += l2_hit_dict[k][L2_THRESHOLD]
        l2_correct[L2_THRESHOLD] += l2_correct_dict[k][L2_THRESHOLD]
        l2_total += l2_total_dict[k]
    print('cumulative_l2_hit_rate-%d %.4f' % (L2_THRESHOLD, (l2_hits[L2_THRESHOLD] / l2_total)))
    if l2_hits[L2_THRESHOLD]:
        print('cumulative l2-hit-acc-%d %.4f' % (L2_THRESHOLD, (l2_correct[L2_THRESHOLD] / l2_hits[L2_THRESHOLD])))
    else:
        print('cumulative l2-hit-acc-%d = 0' % L2_THRESHOLD)

for L1_THRESHOLD in L1_THRESHOLDS:
    for L2_THRESHOLD in L2_THRESHOLDS:
        cumulative_acc[L1_THRESHOLD][L2_THRESHOLD] = (cumulative_sample - l1_hits[L1_THRESHOLD] - l2_hits[
            L2_THRESHOLD] + l1_correct[L1_THRESHOLD] + l2_correct[L2_THRESHOLD]) / cumulative_sample
        print('cumulative_total_acc-%d-%d = %.4f' % (
            L1_THRESHOLD, L2_THRESHOLD, cumulative_acc[L1_THRESHOLD][L2_THRESHOLD]))

print()
