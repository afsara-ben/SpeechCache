import data
import os
import time
import soundfile as sf
import torch
from tqdm import tqdm
import pickle
from utils import create_csv_file, write_to_csv, initialize_SLURP_L1, k_means_test
import numpy as np
import pandas as pd
from copy import deepcopy
import sys
import models

config, wav_path, folder_path, spk_group = initialize_SLURP_L1(folder_name='models/SLURP/models-N-speakers-SLURP-k-means')

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4

THRESHOLD = 500
slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_FE_MO_ME_FO_UNK.csv')
slurp_df = deepcopy(slurp_df)

# spk_group = [['UNK-195', 'MO-371', 'UNK-166']]
cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, total_train = 0, 0, 0, 0, 0
for _, speakers in tqdm(enumerate(spk_group), total=len(spk_group)):
    print('SPEAKERS ', speakers)
    filename = f'3-spk-SLURP-k-means-{speakers}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)

    model = load_data['model']
    train_set = load_data['train_set']
    test_set = load_data['test_set']
    speakers = load_data['speakers']
    transcript_list = load_data['transcript_list']
    cluster_ids = load_data['cluster_ids']
    cluster_centers = load_data['cluster_centers']
    if not cluster_ids:
        continue
    intent_list = load_data['intent_list']
    training_idxs = load_data['training_idxs']

    total, hits, tp = 0, 0, 0
    if cluster_ids and cluster_centers:
        total, hits, tp = k_means_test(wav_path, model, train_set, test_set, cluster_ids, cluster_centers, transcript_list, training_idxs, intent_list, THRESHOLD)
    else:
        continue
    hit_rate, cache_acc, total_acc = 0, 0, 0
    if total >= 5:  # skip for users with < 5 eval samples
        cumulative_sample += total
        cumulative_correct += tp + (total - hits) * 0.9014
        cumulative_hit += hits
        cumulative_hit_correct += tp
        total_train += len(training_idxs)
        hit_rate = round((hits / total), 4)
        total_acc = round((tp + (total - hits) * 0.9014) / total, 4)
        print('Train: %d Test: %d' % (len(training_idxs), total))
        if hits:
            cache_acc = round((tp / hits), 4)
            print('hit rate %.4f' % hit_rate)
            print('cache_acc %.3f' % cache_acc)
        print('total_acc: %.3f' % total_acc)
    else:
        print('not enough samples: %s' % speakers)
    values = [speakers, THRESHOLD, hit_rate, cache_acc, total_acc]
    print(values)
    print()

print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('Threshold ', THRESHOLD)
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()
