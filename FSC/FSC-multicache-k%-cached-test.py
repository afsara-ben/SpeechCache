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

L1_THRESHOLD, L2_THRESHOLD = 300, 25
d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()
# variables to save #hits, #corrects
cumulative_l1_hits, cumulative_l1_corrects = 0, 0
cumulative_l2_hits, cumulative_l2_corrects = 0, 0

speakers = np.unique(new_train_df['speakerId'])
speakers_to_remove = ['4aGjX3AG5xcxeL7a', '5pa4DVyvN2fXpepb', '9Gmnwa5W9PIwaoKq', 'KLa5k73rZvSlv82X',
                      'LR5vdbQgp3tlMBzB', 'OepoQ9jWQztn5ZqL', 'X4vEl3glp9urv4GN', 'Ze7YenyZvxiB4MYZ',
                      'eL2w4ZBD7liA85wm', 'eLQ3mNg27GHLkDej', 'ldrknAmwYPcWzp4N', 'mzgVQ4Z5WvHqgNmY',
                      'nO2pPlZzv3IvOQoP2', 'oNOZxyvRe3Ikx3La', 'roOVZm7kYzS5d4q3', 'rwqzgZjbPaf5dmbL',
                      'wa3mwLV3ldIqnGnV', 'xPZw23VxroC3N34k', 'ywE435j4gVizvw3R', 'zwKdl7Z2VRudGj2L',
                      '35v28XaVEns4WXOv', 'YbmvamEWQ8faDPx2', 'neaPN7GbBEUex8rV', '9mYN2zmq7aTw4Blo']
speakers = [item for item in speakers if item not in speakers_to_remove]

unseen = True
cached = False

pwd = os.getcwd()
if unseen:
    folder_path = os.path.join(pwd, 'models/FSC/models-FSC-multicache-unseen')
else:
    folder_path = os.path.join(pwd, 'models/FSC/models-FSC-multicache-k%-cached')

model = models.Model(config)
model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

cumulative_sample, cumulative_l2_sample, cumulative_correct, cumulative_hits, cumulative_hit_correct = 0, 0, 0, 0, 0
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    if unseen:
        print('unseen')
        filename = f'FSC-multicache-unseen-{speakerId}.pkl'
    else:
        filename = f'FSC-multicache-70%-cached-{speakerId}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)

    # model = load_data['model']
    train_set = load_data['train_set']
    test_set = load_data['test_set']
    speakerId = load_data['speakerId']
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
    tp, total, hits, l1_hits, l2_hits, l1_correct, l2_correct, l2_total = 0, 0, 0, 0, 0, 0, 0, 0
    for _, row in test_set.iterrows():
        if row['index'] in training_idxs:
            continue
        # # of total evaluation samples
        total += 1
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
            loss = ctc_loss_k_means_eval(pred.log_softmax(dim=-1), cluster_ids, pred_lengths, cluster_id_length)
            pred_intent = loss.argmin().item()
            if loss[pred_intent] < L1_THRESHOLD:
                # go with l1: kmeans
                l1_hits += 1
                cumulative_l1_hits += 1
                if row['cache'] == intent_list[pred_intent]:
                    l1_correct += 1
                    cumulative_l1_corrects += 1
            else:
                # ------------------ l2 -------------------
                # phoneme_pred = model.compute_phoneme_from_features(x_feature) #doesnt work RuntimeError: input must have 3 dimensions, got 5
                l2_total += 1
                phoneme_pred = model.pretrained_model.compute_phonemes(x)
                # repeat it #sentence times to compare with ground truth
                phoneme_pred = phoneme_pred.repeat(1, phoneme_label.shape[0], 1)
                pred_lengths = torch.full(size=(phoneme_label.shape[0],), fill_value=phoneme_pred.shape[0],
                                          dtype=torch.long)
                loss = ctc_loss_phoneme_eval(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
                # loss = torch.nan_to_num(loss, nan=float('inf'))  # remove potential nan from loss
                pred_result = loss.argmin()
                if torch.isnan(loss).any():
                    print('nan eval on speaker: %s' % speakerId)
                if loss.min() <= L2_THRESHOLD:
                    # print('l2 hit: ', row['transcription'])
                    l2_hits += 1
                    cumulative_l2_hits += 1
                    if row['cache'] == intent_list[pred_result]:
                        l2_correct += 1
                        cumulative_l2_corrects += 1
                #     else:
                #         print('%s,%s' % (row['transcription'], transcript_list[pred_result]))
                # else:
                #     # do the calculation
                #     # cloud_model.predict_intents(x)
                #     print('cloud')

    l1_hit_rate, l1_cache_acc, l2_hit_rate, l2_cache_acc, total_acc = 0, 0, 0, 0, 0
    if total >= 5:  # skip for users with < 5 eval samples
        cumulative_sample += total
        cumulative_l2_sample += l2_total
        cumulative_hits += l1_hits + l2_hits
        cumulative_hit_correct += l1_correct + l2_correct
        print('speaker ', speakerId)
        if l1_hits:
            l1_hit_rate = round(l1_hits / total, 4)
            l1_cache_acc = round(l1_correct / l1_hits, 4)
            print('l1_hit_rate ', l1_hit_rate)
            print('l1_cache_acc ', l1_cache_acc)
        else:
            print('no hits in l1')
        if l2_hits:
            l2_hit_rate = round(l2_hits / l2_total, 4)
            l2_cache_acc = round(l2_correct / l2_hits, 4)
            print('l2_hit_rate ', l2_hit_rate)
            print('l2_cache_acc ', l2_cache_acc)
        else:
            print('no hits in l2')
        total_acc = (l1_correct + l2_correct + (total - l1_hits - l2_hits) * 0.988) / total
        print('total_acc %.4f' % total_acc)

print('threshold L1: %d L2: %d ' % (L1_THRESHOLD, L2_THRESHOLD))
print('cumulative l1-hit-rate: %.4f' % (cumulative_l1_hits / cumulative_sample))
print('cumulative l2-hit-rate: %.4f' % (cumulative_l2_hits / cumulative_l2_sample))
if cumulative_l1_hits:
    print('cumulative l1-hit-acc: %.4f' % (cumulative_l1_corrects / cumulative_l1_hits))
else:
    print('cumulative l1-hit-acc = 0')
if cumulative_l2_hits:
    print('cumulative l2-hit-acc: %.4f' % (cumulative_l2_corrects / cumulative_l2_hits))
else:
    print('cumulative l2-hit-acc = 0')

print('cumulative hit_rate: %.4f' % (cumulative_hits / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hits))

print('cumulative acc: %.4f' % (((cumulative_sample - cumulative_l1_hits - cumulative_l2_hits) * 0.988 + cumulative_l1_corrects + cumulative_l2_corrects) / cumulative_sample))
