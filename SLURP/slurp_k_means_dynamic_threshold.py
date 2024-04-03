# %%
import sys

import data
import os
import time
import soundfile as sf
import torch
import numpy as np
import torch.nn.functional as F
import ast
from functools import reduce
import pandas as pd

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
import pickle
from utils import audio_augment, get_token_and_weight, write_to_csv, get_audio_duration

device = "cpu"
config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')

# emulate an oracle cloud model
config.phone_rnn_num_hidden = [128, 128]
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

pwd = os.getcwd()
wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
folder_path = os.path.join(pwd, 'slurp_models_k_means')

zero = 0.000000000000001
arguments = sys.argv
THRESHOLDS = []
for arg in arguments[1:]:
    THRESHOLDS.append(int(arg))

slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_FE.csv')
slurp_df = deepcopy(slurp_df)
num_nan_train, nan_nan_eval = 0, 0
speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232', 'ME-144']
# speakers = ['MO-433', 'UNK-326', 'FO-232']
cumulative_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, cumulative_cache_miss,cumulative_hit_incorrect, total_train = 0, 0, 0, 0, 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('SLURP K means EVAL FOR SPEAKER ', user_id)
    tmp = slurp_df[slurp_df['user_id'] == user_id]
    filename = f'slurp-k-means-{user_id}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)
    model = load_data['model']
    user_id = load_data['speakerId']
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
    tp, total, hits = 0, 0, 0
    hit1, hit2, hit3, tp1, tp2, tp3, total1, total2, total3 = zero, zero, zero, 0, 0, 0, zero, zero, zero
    for _, row in tmp.iterrows():
        if row[0] in training_idxs:
            continue
        # # of total evaluation samples
        total += 1
        wav = wav_path + row['recording_path']
        x, _ = sf.read(wav)
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
            # print('loss ', loss[pred_intent])
            if 0 <= (get_audio_duration(wav)) <= 2.7:
                total1 += 1
                THRESHOLD = THRESHOLDS[0]
                if loss[pred_intent] <= THRESHOLD:
                    hit1 += 1
                    if row['intent'] == intent_list[pred_intent]:
                        tp1 += 1
            elif 2.7 < (get_audio_duration(wav)) <= 4:
                total2 += 1
                THRESHOLD = THRESHOLDS[1]
                if loss[pred_intent] <= THRESHOLD:
                    hit2 += 1
                    if row['intent'] == intent_list[pred_intent]:
                        tp2 += 1
            else:
                THRESHOLD = THRESHOLDS[2]
                total3 += 1
                if loss[pred_intent] <= THRESHOLD:
                    hit3 += 1
                    if row['intent'] == intent_list[pred_intent]:
                        tp3 += 1

    print('hit rate %.4f %.4f %.4f' % (hit1 / total1, hit2 / total2, hit3 / total3))
    print('cache acc %.4f %.4f %.4f' % (tp1 / hit1, tp2 / hit2, tp3 / hit3))
    hits = hit1 + hit2 + hit3
    tp = tp1 + tp2 + tp3

    hit_rate, cache_acc, total_acc = 0, 0, 0
    if total >= 5:  # skip for users with < 5 eval samples
        cumulative_sample += total
        cumulative_correct += tp + (total - hits)*0.85
        cumulative_hit += hits
        cumulative_hit_correct += tp
        total_train += len(training_idxs)
        hit_rate = round((hits / total), 4)
        total_acc = round((tp + (total - hits)*0.85) / total, 4)
        print('Train: %d Test: %d' % (len(training_idxs), total))
        print('for speaker ', user_id)
        if hits:
            cache_acc = round((tp / hits), 4)
            print('hit rate %.4f' % hit_rate)
            print('cache_acc %.3f' % cache_acc)
        print('total_acc: %.3f' % total_acc)
    else:
        print('not enough samples: %s' % user_id)
    values = [user_id, THRESHOLDS, len(training_idxs), total, hit_rate, cache_acc, total_acc]
    # write_to_csv(results_file, values)

print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('Threshold ', THRESHOLDS)
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()
