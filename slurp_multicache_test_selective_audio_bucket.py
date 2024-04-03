# %%
import argparse
import os
import pickle
import time
from copy import deepcopy
from models import MLP
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import csv

import data
import models
from utils import create_csv_file, get_audio_duration
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)

L1_THRESHOLDS = [400, 700, 1100]
L2_THRESHOLDS = [50, 110, 200]

# L2_THRESHOLDS = [30, 85, 170]
# variables to save #hits, #corrects
cumulative_l1_hits, cumulative_l1_corrects, cumulative_l1_hit_correct = 0, 0, 0
cumulative_l2_hits, cumulative_l2_corrects, cumulative_l2_hit_correct = 0, 0, 0
buckets = [1, 2, 3]

pwd = os.getcwd()
wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
folder_path = os.path.join(pwd, 'models/SLURP/models-slurp_multicache_audio_bucket')

parser = argparse.ArgumentParser(description="input cutoff")
parser.add_argument("--dynamic", type=bool, default=False)
parser.add_argument("--in_domain", type=bool, default=False)
args = parser.parse_args()
in_domain = args.in_domain
dynamic = args.dynamic

if in_domain:
    cloud_model = models.Model(config).eval()
    cloud_model.load_state_dict(
        torch.load("/Users/afsarabenazir/Downloads/speech_projects/speechcache/models/SLURP/slurp-pretrained.pth", map_location=device)) # load trained model
else:
    cloud_model = models.Model(config)
    optim = torch.optim.Adam(cloud_model.parameters(), lr=1e-3)
    cloud_model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model


def multicache_test(model, df, cluster_ids, cluster_centers, transcript_list, training_idxs, intent_list, L1_THRESHOLD,
                    L2_THRESHOLD):
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
    tp, total, hits, l1_hits, l2_hits, l1_correct, l2_correct, l2_total, l1_total = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for _, row in df.iterrows():
        if row[0] in training_idxs:
            continue
        # print(row['sentence'])
        # # of total evaluation samples
        total += 1
        # if bucket == 3:
        #     cloud += 1
        #     continue
        wav = os.path.join(wav_path, row['recording_path'])
        x, _ = sf.read(wav)
        x = torch.tensor(x, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            if bucket == 1:
                # print('just do L2')
                l2_total += 1
                phoneme_pred = model.pretrained_model.compute_phonemes(x)
                # repeat it #sentence times to compare with ground truth
                phoneme_pred = phoneme_pred.repeat(1, phoneme_label.shape[0], 1)
                pred_lengths = torch.full(size=(phoneme_label.shape[0],), fill_value=phoneme_pred.shape[0],
                                          dtype=torch.long)
                loss = ctc_loss_phoneme_eval(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
                # print('l2 loss ', loss.min())
                # loss = torch.nan_to_num(loss, nan=float('inf'))  # remove potential nan from loss
                pred_result = loss.argmin()
                if torch.isnan(loss).any():
                    print('nan eval on speaker: %s' % user_id)
                if dynamic:
                    filename = 'utils/slurp-MLP-L2.pkl'
                    with open(filename, 'rb') as f:
                        load_data = pickle.load(f)
                    MLP_model = load_data['model']
                    dur = torch.tensor([[get_audio_duration(wav)]], dtype=torch.float32)
                    L2_THRESHOLD = MLP_model(dur).item()
                if loss.min() <= L2_THRESHOLD:
                    # print('l2 hit: ', row['sentence'])
                    l2_hits += 1
                    if row['intent'] == intent_list[pred_result]:
                        l2_correct += 1
            else:
                # ----------------- l1 -------------------
                x_feature = cloud_model.pretrained_model.compute_cnn_features(x)
                dists = torch.cdist(x_feature, cluster_centers)
                dists = dists.max(dim=-1)[0].unsqueeze(-1) - dists
                pred = dists.swapaxes(1, 0)
                pred_lengths = torch.full(size=(cluster_ids.shape[0],), fill_value=pred.shape[0], dtype=torch.long)
                loss = ctc_loss_k_means_eval(pred.log_softmax(dim=-1), cluster_ids, pred_lengths, cluster_id_length)
                pred_intent = loss.argmin().item()
                # print('l1 loss ', loss[pred_intent])
                l1_total += 1
                if dynamic:
                    filename = 'utils/k_means_MLP.pkl'
                    with open(filename, 'rb') as f:
                        load_data = pickle.load(f)
                    MLP_model_k_means = load_data['model']
                    dur = torch.tensor([[get_audio_duration(wav)]], dtype=torch.float32)
                    L1_THRESHOLD = MLP_model_k_means(dur).item()
                    print(round(dur.item(), 4), L1_THRESHOLD)
                if loss[pred_intent] < L1_THRESHOLD:
                    # go with l1: kmeans
                    # print('l1 hit: ', row['sentence'])
                    l1_hits += 1
                    if row['intent'] == intent_list[pred_intent]:
                        l1_correct += 1
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
                    # print('l2 loss ', loss.min())
                    # loss = torch.nan_to_num(loss, nan=float('inf'))  # remove potential nan from loss
                    pred_result = loss.argmin()
                    if torch.isnan(loss).any():
                        print('nan eval on speaker: %s' % user_id)
                    if dynamic:
                        filename = 'utils/slurp-MLP-L2.pkl'
                        with open(filename, 'rb') as f:
                            load_data = pickle.load(f)
                        MLP_model = load_data['model']
                        dur = torch.tensor([[get_audio_duration(wav)]], dtype=torch.float32)
                        L2_THRESHOLD = MLP_model(dur).item()
                    if loss.min() <= L2_THRESHOLD:
                        # print('l2 hit: ', row['sentence'])
                        l2_hits += 1
                        if row['intent'] == intent_list[pred_result]:
                            l2_correct += 1
                    #     else:
                    #         print('%s,%s' % (row['sentence'], transcript_list[pred_result]))
                    # else:
                    #     # do the calculation
                    #     # cloud_model.predict_intents(x)
                    #     # print('cloud. loss was %f ' % loss.min())
                    #
    return total, l1_total, l2_total, l1_hits, l2_hits, l1_correct, l2_correct


slurp_df = pd.read_csv('SLURP/csv/slurp_mini_FE_MO_ME_FO_UNK.csv')
slurp_df = deepcopy(slurp_df)
num_nan_train, nan_nan_eval = 0, 0
speakers = np.unique(slurp_df['user_id'])
# starts_with = 'UNK'
# speakers = [elem for elem in speakers if not elem.startswith(starts_with)]
# speakers = ['MO-433', 'UNK-326', 'FO-232', 'ME-144']
# speakers = ['MO-433']
cumulative_l1_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_cache_miss, cumulative_hit_incorrect, total_train = 0, 0, 0, 0, 0, 0, 0
cumulative_sample, cumulative_hit_correct, cumulative_hits = 0, 0, 0
bckt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('EVAL FOR SPEAKER ', user_id)
    print('count ', slurp_df[slurp_df['user_id'] == user_id].shape[0])
    bckt_sample, l1_bckt_sample, l2_bckt_sample, l1_bckt_correct, l2_bckt_correct, l1_bckt_hit, l2_bckt_hit, l1_bckt_hit_correct, l2_bckt_hit_correct, bckt_train = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for bucket in buckets:
        filename = f'slurp_model_multicache_{user_id}_audio_bucket_{bucket}.pkl'
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            load_data = pickle.load(f)
        model = load_data['model']
        user_id = load_data['speakerId']
        df = load_data['df']
        transcript_list = load_data['transcript_list']
        phoneme_list = load_data['phoneme_list']
        intent_list = load_data['intent_list']
        training_idxs = load_data['training_idxs']
        cluster_ids = load_data['cluster_ids']
        cluster_centers = load_data['cluster_centers']
        print(f'bucket {bucket} test : {len(df) - len(training_idxs)}')

        total, l1_total, l2_total, l1_hits, l2_hits, l1_correct, l2_correct = 0, 0, 0, 0, 0, 0, 0
        if cluster_ids and cluster_centers and phoneme_list:
            total, l1_total, l2_total, l1_hits, l2_hits, l1_correct, l2_correct = multicache_test(model, df, cluster_ids,
                                                                                        cluster_centers,
                                                                                        transcript_list, training_idxs,
                                                                                        intent_list,
                                                                                        L1_THRESHOLD=L1_THRESHOLDS[
                                                                                            bucket - 1],
                                                                                        L2_THRESHOLD=L2_THRESHOLDS[
                                                                                            bucket - 1])

        bckt[bucket]['l1']['hits'] += l1_hits
        bckt[bucket]['l1']['tp'] += l1_correct
        bckt[bucket]['l2']['hits'] += l2_hits
        bckt[bucket]['l2']['tp'] += l2_correct
        bckt[bucket]['l1']['total'] += l1_total
        bckt[bucket]['l2']['total'] += l2_total

        l1_hit_rate, l1_cache_acc, l2_hit_rate, l2_cache_acc = 0, 0, 0, 0
        bckt_train += len(training_idxs)
        bckt_sample += total
        l1_bckt_sample += l1_total
        l2_bckt_sample += l2_total
        if total >= 5:  # skip for users with < 5 eval samples
            print('EVAL FOR SPEAKER %s: BUCKET %d' % (user_id, bucket))
            # l1_bckt_correct += l1_correct + (total - l1_hits)
            l1_bckt_hit += l1_hits
            l1_bckt_hit_correct += l1_correct

            # l2_bckt_correct += l2_correct + (l2_total - l2_hits)
            l2_bckt_hit += l2_hits
            l2_bckt_hit_correct += l2_correct

            if l1_hits:
                l1_hit_rate = round((l1_hits / l1_total), 4)
                l1_cache_acc = round((l1_correct / l1_hits), 4)
                print('l1_hit_rate ', l1_hit_rate)
                print('l1_cache_acc ', l1_cache_acc)
            else:
                print('no hits in l1')
            if l2_hits:
                l2_hit_rate = round((l2_hits / l2_total), 4)
                l2_cache_acc = round((l2_correct / l2_hits), 4)
                print('l2_hit_rate ', l2_hit_rate)
                print('l2_cache_acc ', l2_cache_acc)
            else:
                print('no hits in l2')
        else:
            print('not enough samples: %s' % user_id)
        # values = [user_id, L1_THRESHOLDS[bucket - 1], L2_THRESHOLDS[bucket - 1], bucket, len(training_idxs), total,
        #           l1_correct, l1_hits, l1_hit_rate, l1_cache_acc, l2_total, l2_correct, l2_hits, l2_hit_rate,
        #           l2_cache_acc]
        # write_to_csv(bucket_results_file, values)
        # print(values)

    total_acc, l1_hit_rate, l2_hit_rate, l1_cache_acc, l2_cache_acc = 0, 0, 0, 0, 0
    total_train += bckt_train

    # following needed for overall evaluation
    cumulative_sample += bckt_sample
    cumulative_l1_sample += l1_bckt_sample
    cumulative_l2_sample += l2_bckt_sample
    cumulative_l1_hits += l1_bckt_hit
    cumulative_l2_hits += l2_bckt_hit

    cumulative_l1_hit_correct += l1_bckt_hit_correct
    cumulative_l2_hit_correct += l2_bckt_hit_correct
    # cumulative_l1_corrects += l1_bckt_correct #tp+total-hits
    # cumulative_l2_corrects += l2_bckt_correct

    print(f"------------------------------------{user_id}------------------------------------")
    if l1_bckt_sample:
        l1_hit_rate = round((l1_bckt_hit / l1_bckt_sample), 4)
        print('l1 hit_rate: %.4f' % l1_hit_rate)
    if l1_bckt_hit:
        l1_cache_acc = round((l1_bckt_hit_correct / l1_bckt_hit), 4)
        print('l1 cache_acc: %.4f' % l1_cache_acc)
    if l2_bckt_sample:
        l2_hit_rate = round((l2_bckt_hit / l2_bckt_sample), 4)
        print('l2 hit_rate: %.4f' % l2_hit_rate)
    if l2_bckt_hit:
        l2_cache_acc = round((l2_bckt_hit_correct / l2_bckt_hit), 4)
        print('l2 cache_acc: %.4f' % l2_cache_acc)

    if bckt_sample:
        total_acc = round((l1_bckt_hit_correct + l2_bckt_hit_correct + (
            bckt_sample - l1_bckt_hit - l2_bckt_hit) * 0.9014) / bckt_sample, 4)
    # tp+total-hit/total
    print('total_acc: %.4f' % total_acc)
    print(f"------------------------------------------------------------------------")
    values = [user_id, L1_THRESHOLDS, L2_THRESHOLDS, bckt_train, bckt_sample, l1_hit_rate, l1_cache_acc, l2_hit_rate,
              l2_cache_acc, total_acc]
    # write_to_csv(results_file, values)
    print(values)
    # print(bckt)

print(L1_THRESHOLDS, L2_THRESHOLDS)
cumulative_hits = cumulative_l1_hits + cumulative_l2_hits
cumulative_hit_correct = cumulative_l1_hit_correct + cumulative_l2_hit_correct

print('cumulative l1-hit-rate: %.4f' % (cumulative_l1_hits / cumulative_l1_sample))
if cumulative_l1_hits:
    print('cumulative l1-hit-acc: %.4f' % (cumulative_l1_hit_correct / cumulative_l1_hits))
else:
    print('cumulative l1-hit-acc = 0')
print('cumulative l2-hit-rate: %.4f' % (cumulative_l2_hits / cumulative_l2_sample))
if cumulative_l2_hits:
    print('cumulative l2-hit-acc: %.4f' % (cumulative_l2_hit_correct / cumulative_l2_hits))
else:
    print('cumulative l2-hit-acc = 0')

# print('cumulative hit_rate: %.4f' % (cumulative_hits / cumulative_sample))
print('cumulative hit_rate: %.4f' % (cumulative_hits / (cumulative_l1_sample+cumulative_l2_sample)))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hits))

cloud = (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits) * float(0.9014)
print('cumulative acc: ',
      round((cumulative_l1_hit_correct + cumulative_l2_hit_correct + cloud) / cumulative_sample, 4))
print('cloud ', (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits))
print('total test ', cumulative_sample)

for i in buckets:
    print('bucket', i)
    if bckt[i]['l1']['total']:
        print(round(bckt[i]['l1']['hits'] / bckt[i]['l1']['total'], 2))
    if bckt[i]['l1']['hits']:
        print(round(bckt[i]['l1']['tp'] / bckt[i]['l1']['hits'], 2))
    if bckt[i]['l2']['total']:
        print(round(bckt[i]['l2']['hits'] / bckt[i]['l2']['total'], 2))
    if bckt[i]['l2']['hits']:
        print(round(bckt[i]['l2']['tp'] / bckt[i]['l2']['hits'], 2))
