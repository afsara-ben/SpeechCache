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

from utils import create_csv_file, audio_augment, get_token_and_weight, write_to_csv


device = "cpu"
config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
_, _, _ = data.get_SLU_datasets(config)  # used to set config.num_phonemes
print('config load ok')
dataset_to_use = valid_dataset
dataset_to_use.df.head()
test_dataset.df.head()

num_nan_train, nan_nan_eval = 0, 0
pwd = os.getcwd()
wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
folder_path = os.path.join(pwd, 'models/SLURP/models-slurp_k_means_audio_bucket')

model = models.Model(config)
model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

# emulate an oracle cloud model
config.phone_rnn_num_hidden = [128, 128]
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

bucket_results_file = 'SLURP/slurp_k_means_audio_per_bucket_dets.csv'
if not os.path.exists(bucket_results_file):
    headers = ['SpeakerId', 'threshold', 'bucket', 'train', 'test', 'tp', 'hits', 'hit_rate', 'cache_acc']
    create_csv_file(bucket_results_file, headers)

results_file = 'slurp_k_means_audio_bucket.csv'
if not os.path.exists(results_file):
    headers = ['SpeakerId', 'threshold', 'train', 'test', 'hit_rate', 'cache_acc', 'total_acc']
    create_csv_file(results_file, headers)


def test(df, cluster_ids, cluster_centers, transcript_list, training_idxs, intent_list, THRESHOLD):
    # ----------------- prepare for cluster -----------------
    cluster_id_length = torch.tensor(list(map(len, cluster_ids)), dtype=torch.long, device=device)
    cluster_ids = pad_sequence(cluster_ids, batch_first=True, padding_value=0).to(device)
    cluster_centers = torch.stack(cluster_centers).to(device)
    # no reduction, loss on every sequence
    ctc_loss_k_means_eval = torch.nn.CTCLoss(reduction='none')
    tp, total, hits = 0, 0, 0
    for _, row in df.iterrows():
        if row[0] in training_idxs:
            continue
        # # of total evaluation samples
        total += 1
        wav = os.path.join(wav_path, row['recording_path'])
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
            if loss[pred_intent] < THRESHOLD:
                # go with l1: kmeans
                # print('hit')
                hits += 1
                if row['intent'] == intent_list[pred_intent]:
                    # print('tp')
                    tp += 1
            #     else:
            #         print('%s,%s' % (row['sentence'], transcript_list[pred_intent]))
            # else:
            #     print('cloud. loss was %f ' % loss.min())

    return total, hits, tp


buckets = [1, 2, 3]
THRESHOLDS = [400, 700, 1100]
slurp_df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_mini_FE_MO_ME_FO_UNK.csv'))
slurp_df = deepcopy(slurp_df)

speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232', 'ME-144']
# speakers = ['MO-433', 'UNK-326', 'FO-232']

cumulative_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, cumulative_cache_miss, cumulative_hit_incorrect, total_train = 0, 0, 0, 0, 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('EVAL FOR SPEAKER ', user_id)
    bckt_sample, bckt_correct, bckt_hit, bckt_hit_correct, bckt_train = 0, 0, 0, 0, 0

    for bucket in buckets:
        filename = f'slurp_model_k_means_{user_id}_audio_bucket_{bucket}.pkl'
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            load_data = pickle.load(f)

        user_id = load_data['speakerId']
        df = load_data['df']
        transcript_list = load_data['transcript_list']
        intent_list = load_data['intent_list']
        training_idxs = load_data['training_idxs']
        cluster_ids = load_data['cluster_ids']
        cluster_centers = load_data['cluster_centers']

        total, hits, tp = 0, 0, 0
        if cluster_ids and cluster_centers:
            total, hits, tp = test(df, cluster_ids, cluster_centers, transcript_list, training_idxs, intent_list,
                               THRESHOLD=THRESHOLDS[bucket - 1])
        else:
            continue
        hit_rate, cache_acc, total_acc = 0, 0, 0
        if total >= 5:  # skip for users with < 5 eval samples
            print('\n\nEVAL FOR SPEAKER %s: BUCKET %d' % (user_id, bucket))
            bckt_train += len(training_idxs)
            bckt_sample += total
            bckt_correct += tp + (total - hits)*0.9014
            bckt_hit += hits
            bckt_hit_correct += tp
            print('for speaker ', user_id)
            print('Train: %d Test: %d' % (len(training_idxs), total))
            hit_rate = round((hits / total), 4)
            print('hit_rate %.3f ' % hit_rate)
            if hits:
                cache_acc = round((tp / hits), 4)
                print('cache_acc %.3f' % cache_acc)
            total_acc = round((tp + (total - hits)*0.9014) / total, 4)
            print('total_acc: %.3f' % total_acc)
        else:
            print('not enough samples: %s' % user_id)
        values = [user_id, THRESHOLDS[bucket - 1], bucket, len(training_idxs), total, tp, hits, hit_rate, cache_acc]
        write_to_csv(bucket_results_file, values)
        print()

    hit_rate, cache_acc, total_acc = 0, 0, 0
    total_train += bckt_train
    cumulative_sample += bckt_sample
    cumulative_correct += bckt_correct
    cumulative_hit += bckt_hit
    cumulative_hit_correct += bckt_hit_correct
    if bckt_sample:
        hit_rate = round((bckt_hit / bckt_sample), 4)
        total_acc= round((bckt_correct / bckt_sample), 4)
    if bckt_hit:
        cache_acc = round((bckt_hit_correct / bckt_hit), 4)
    print('hit_rate: %.4f' % hit_rate)
    print('cache_acc: %.4f' % cache_acc)
    print('total_acc: %.4f' % total_acc)
    values = [user_id, THRESHOLDS, bckt_train, bckt_sample, hit_rate, cache_acc, total_acc]
    write_to_csv(results_file, values)
    print()

print('Threshold ', THRESHOLDS)
print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()
