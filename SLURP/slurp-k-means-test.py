# %%
import os
import data
import experiments
import time
import soundfile as sf
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import models
import pickle

device = "cpu"

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
dataset_to_use = valid_dataset
dataset_to_use.df.head()
test_dataset.df.head()
pwd = os.getcwd()
model = models.Model(config)
model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model


root = '/Users/afsarabenazir/Downloads/speech_projects/speechcache'
wav_path = os.path.join(root, 'SLURP/slurp_real/')
folder_path = os.path.join(root, 'models/SLURP/slurp_models_k_means')

THRESHOLD = 500
slurp_df = pd.read_csv(os.path.join(root, 'SLURP/slurp_mini_FE_MO_ME_FO_UNK.csv'))
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
    # model = load_data['model']
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
            # dists = 1 / dists
            # pred = torch.softmax(dists.swapaxes(1, 0), dim=-1)
            dists = dists.max(dim=-1)[0].unsqueeze(-1) - dists
            pred = dists.swapaxes(1, 0)
            pred_lengths = torch.full(size=(cluster_ids.shape[0],), fill_value=pred.shape[0], dtype=torch.long)
            loss = ctc_loss_k_means_eval(pred.log_softmax(dim=-1), cluster_ids, pred_lengths, cluster_id_length)
            pred_intent = loss.argmin().item()
            if loss[pred_intent] < THRESHOLD:
                # go with l1: kmeans
                # print('l1 hit: ', row['sentence'])
                hits += 1
                if row['intent'] == intent_list[pred_intent]:
                    tp += 1

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
    values = [user_id, THRESHOLD, len(training_idxs), total, hit_rate, cache_acc, total_acc]
    # write_to_csv(results_file, values)

print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('Threshold ', THRESHOLD)
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()
