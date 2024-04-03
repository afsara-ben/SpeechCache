# %%
import argparse
import gzip

import data
import os
import time
from models import MLP
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
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import models
import pickle
from collections import defaultdict
from utils import write_to_csv, create_csv_file, get_audio_duration


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')

parser = argparse.ArgumentParser(description="input cutoff")
parser.add_argument("--dynamic", type=bool, default=False)
parser.add_argument("--in_domain", type=bool, default=False)
args = parser.parse_args()
in_domain = args.in_domain
dynamic = args.dynamic

cloud_acc = 0.9015
pwd = os.getcwd()
if in_domain:
    path = os.path.join(pwd, "models/SLURP/slurp-pretrained.pth")
    cloud_model = models.Model(config)
    cloud_model.load_state_dict(
        torch.load(path, map_location=device)) # load trained model
else:
    cloud_model = models.Model(config)
    optim = torch.optim.Adam(cloud_model.parameters(), lr=1e-3)
    cloud_model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

L1_THRESHOLDS = [400, 700, 1100]
# L2_THRESHOLDS = [40, 90, 150]
L2_THRESHOLDS = [30, 85, 170]
# variables to save #hits, #corrects
cumulative_l1_hits, cumulative_l1_corrects, cumulative_l1_hit_correct = 0, 0, 0
cumulative_l2_hits, cumulative_l2_corrects, cumulative_l2_hit_correct = 0, 0, 0
buckets = [1, 2, 3]

pwd = os.getcwd()
wav_path = os.path.join(pwd, 'user_study_recordings/')
folder_path = os.path.join(pwd, 'models/SLURP/user-study-3-spk-headset') #aben:user stud

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4

def multicache_test(speakers, model, train_df, test_df, cluster_ids, cluster_centers, transcript_list, training_idxs, intent_list, L1_THRESHOLD, L2_THRESHOLD):
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
    tp, total, hits, l1_hits, l2_hits, l1_correct, l2_correct, l1_total, l2_total = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for _, row in test_df.iterrows():
        if row[0] in training_idxs:
            continue
        total += 1

        wav = os.path.join(wav_path, row['recording_path'])
        x, _ = sf.read(wav)
        x = torch.tensor(x, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            if bucket == 1:
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
                    print('nan eval on speaker: %s' % speakers)
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
                x_feature = model.pretrained_model.compute_cnn_features(x)
                dists = torch.cdist(x_feature, cluster_centers)
                dists = dists.max(dim=-1)[0].unsqueeze(-1) - dists
                pred = dists.swapaxes(1, 0)
                pred_lengths = torch.full(size=(cluster_ids.shape[0],), fill_value=pred.shape[0], dtype=torch.long)
                loss = ctc_loss_k_means_eval(pred.log_softmax(dim=-1), cluster_ids, pred_lengths, cluster_id_length)
                pred_intent = loss.argmin().item()
                l1_total += 1
                # print('l1 loss ', loss[pred_intent])
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
                        print('nan eval on speaker: %s' % speakers)
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
                    #     print('cloud. loss was %f ' % loss.min())
    return total, l1_total, l2_total, l1_hits, l2_hits, l1_correct, l2_correct


slurp_df = pd.read_csv('user_study_recordings/data_headset.csv')
slurp_df = deepcopy(slurp_df)
num_nan_train, nan_nan_eval = 0, 0

cumulative_sample, cumulative_l1_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, cumulative_cache_miss, cumulative_hit_incorrect, total_train = 0, 0, 0, 0, 0, 0, 0, 0, 0
spk_group = [['U1', 'U2', 'U3']]

# curated slurp headset
# spk_group = [['ME-147', 'UNK-323', 'ME-345', 'ME-151'], ['FO-158', 'UNK-335', 'FO-462', 'UNK-322'], ['ME-352', 'UNK-330', 'ME-223'], ['FE-248', 'FO-488', 'MO-164'], ['FE-149', 'FE-235', 'MO-446'], ['FO-444', 'FO-350', 'FE-249'], ['MO-375', 'MO-142', 'MO-431'], ['FO-229', 'UNK-336', 'MO-374'], ['FE-146', 'MO-156', 'MO-355'], ['FO-372', 'ME-143', 'MO-433'], ['ME-414', 'MO-465', 'FO-234'], ['FO-438', 'FO-233', 'FO-413'], ['FO-232', 'UNK-328', 'UNK-329'], ['UNK-334', 'MO-494', 'FO-152'], ['MO-463', 'FO-425', 'ME-373'], ['ME-369', 'ME-144', 'FO-419'], ['UNK-343', 'FO-445', 'UNK-331'], ['MO-422', 'FO-219', 'ME-140'], ['ME-220', 'FO-122', 'FE-141'], ['FO-460', 'FO-493', 'FE-145'], ['FO-231', 'FO-150', 'ME-473'], ['FO-179', 'FO-475', 'MO-222'], ['UNK-326', 'UNK-327', 'FO-171'], ['FO-180', 'FO-461', 'UNK-341']]

# spk_group = [['ME-132', 'ME-352', 'MO-051']]
bckt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for _, speakers in tqdm(enumerate(spk_group), total=len(spk_group)):
    print('SPEAKERS ', speakers)
    bckt_sample, l1_bckt_sample, l2_bckt_sample, l1_bckt_correct, l2_bckt_correct, l1_bckt_hit, l2_bckt_hit, l1_bckt_hit_correct, l2_bckt_hit_correct, bckt_train = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for bucket in buckets:
        filename = f'3-spk-user-study-{speakers}_audio_bucket_{bucket}'
        file_path = os.path.join(folder_path, filename + '.pth')
        model = models.Model(config)
        model.load_state_dict(torch.load(file_path, map_location=device))

        with gzip.open(os.path.join(folder_path, filename + '.pkl.gz'), 'rb') as f:
            metadata = pickle.load(f)

        speakers = metadata['speakers']
        train_set = metadata['train_set']
        test_set = metadata['test_set']
        transcript_list = metadata['transcript_list']
        phoneme_list = metadata['phoneme_list']
        intent_list = metadata['intent_list']
        training_idxs = metadata['training_idxs']
        cluster_ids = metadata['cluster_ids']
        cluster_centers = metadata['cluster_centers']

        # below for slurp - headset
        # filename = f'slurp_curated_headset_base_{speakers}_audio_bucket_{bucket}_3_spk'
        # file_path = os.path.join(folder_path, filename + '.pth')
        #
        # model = models.Model(config)
        # model.load_state_dict(torch.load(file_path, map_location=device))
        #
        # with gzip.open(os.path.join(folder_path, filename + '.pkl.gz'), 'rb') as f:
        #     metadata = pickle.load(f)
        #
        # speakers = metadata['speakers']
        # train_set = metadata['train_set']
        # test_set = metadata['test_set']
        # transcript_list = metadata['transcript_list']
        # phoneme_list = metadata['phoneme_list']
        # intent_list = metadata['intent_list']
        # training_idxs = metadata['training_idxs']
        # cluster_ids = metadata['cluster_ids']
        # cluster_centers = metadata['cluster_centers']
        #
        # print(f'bucket {bucket} train {len(training_idxs)} test: {len(test_set)}')

        total, l1_total, l2_total, l1_hits, l2_hits, l1_correct, l2_correct = 0, 0, 0, 0, 0, 0, 0
        if cluster_ids and cluster_centers and phoneme_list:
            total, l1_total, l2_total, l1_hits, l2_hits, l1_correct, l2_correct = multicache_test(speakers, model, train_set, test_set, cluster_ids, cluster_centers, transcript_list, training_idxs, intent_list, L1_THRESHOLD=L1_THRESHOLDS[bucket - 1], L2_THRESHOLD=L2_THRESHOLDS[bucket - 1])
        # print(f'################################ {l2_hits}################################ ')
        # bckt[bucket]['l1']['hits'] += l1_hits
        # bckt[bucket]['l1']['tp'] += l1_correct
        # bckt[bucket]['l2']['hits'] += l2_hits
        # bckt[bucket]['l2']['tp'] += l2_correct
        # bckt[bucket]['l1']['total'] += l1_total
        # bckt[bucket]['l2']['total'] += l2_total

        l1_hit_rate, l1_cache_acc, l2_hit_rate, l2_cache_acc = 0, 0, 0, 0
        bckt_train += len(training_idxs)
        bckt_sample += total
        l1_bckt_sample += l1_total
        l2_bckt_sample += l2_total
        if total >= 1:  # skip for users with < 5 eval samples
            print('EVAL FOR SPEAKER %s: BUCKET %d' % (speakers, bucket))
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
            print('not enough samples: %s' % speakers)
        values = [speakers, L1_THRESHOLDS[bucket - 1], L2_THRESHOLDS[bucket - 1], bucket, len(training_idxs), total, l1_correct, l1_hits, l1_hit_rate, l1_cache_acc, l2_total, l2_correct, l2_hits, l2_hit_rate, l2_cache_acc]
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
    print(f'-----------------------------------{speakers}----------------------------------------------')
    if bckt_sample:
        if l1_bckt_sample:
            l1_hit_rate = round((l1_bckt_hit / l1_bckt_sample), 4)
        total_acc = round((l1_bckt_hit_correct + l2_bckt_hit_correct + (
                bckt_sample - l1_bckt_hit - l2_bckt_hit) * cloud_acc) / bckt_sample, 4)
        print('l1 hit_rate: %.4f' % l1_hit_rate)
    if l1_bckt_hit:
        l1_cache_acc = round((l1_bckt_hit_correct / l1_bckt_hit), 4)
        print('l1 cache_acc: %.4f' % l1_cache_acc)
    if l2_bckt_sample:
        print(l2_bckt_hit, l2_bckt_sample)
        l2_hit_rate = round((l2_bckt_hit / l2_bckt_sample), 4)
        print('l2 hit_rate: %.4f' % l2_hit_rate)
    if l2_bckt_hit:
        print(l2_bckt_hit_correct, l2_bckt_hit)
        l2_cache_acc = round((l2_bckt_hit_correct / l2_bckt_hit), 4)
        print('l2 cache_acc: %.4f' % l2_cache_acc)

    # tp+total-hit/total
    print('total_acc: %.4f' % total_acc)
    print(f'---------------------------------------------------------------------------------')
    values = [speakers, L1_THRESHOLDS, L2_THRESHOLDS, bckt_train, bckt_sample, l1_hit_rate, l1_cache_acc, l2_hit_rate, l2_cache_acc, total_acc]
    # write_to_csv(results_file, values)
    # print(values)


print(f"------------------------------------------------------------------------")
cumulative_hits = cumulative_l1_hits + cumulative_l2_hits
cumulative_hit_correct = cumulative_l1_hit_correct + cumulative_l2_hit_correct
cumulative_l1_hit_rate, cumulative_l1_hit_acc, cumulative_l2_hit_rate, cumulative_l2_hit_acc = 0, 0, 0, 0
if dynamic:
    print('DYNAMIC')
else:
    print(L1_THRESHOLDS, L2_THRESHOLDS)
if cumulative_l1_sample:
    cumulative_l1_hit_rate = round(cumulative_l1_hits / cumulative_l1_sample, 4)
if cumulative_l1_hits:
    cumulative_l1_hit_acc = round(cumulative_l1_hit_correct / cumulative_l1_hits, 4)
if cumulative_l2_sample:
    cumulative_l2_hit_rate = round(cumulative_l2_hits / cumulative_l2_sample, 4)
if cumulative_l2_hits:
    cumulative_l2_hit_acc = round(cumulative_l2_hit_correct / cumulative_l2_hits, 4)

cumulative_hit_rate = round(cumulative_hits / cumulative_sample, 4)
cumulative_cache_acc = round(cumulative_hit_correct / cumulative_hits, 4)
cloud = (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits) * float(cloud_acc)
cumulative_acc =  round((cumulative_l1_hit_correct + cumulative_l2_hit_correct + cloud) / cumulative_sample, 4)
print(f'total train {total_train} test {cumulative_sample}')
print(f'l1: hit_rate {cumulative_l1_hit_rate} hit_acc {cumulative_l1_hit_acc}')
print(f'l2: hit_rate {cumulative_l2_hit_rate} hit_acc {cumulative_l2_hit_acc}')
print(f'l1+L2: hit_rate {cumulative_hit_rate} hit_acc {cumulative_cache_acc}')
# print(f'acc {cumulative_acc} for cutoff {cutoff}')
print(f'acc {cumulative_acc}')
print(f"------------------------------------------------------------------------")

print('cloud ', (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits))

# OLD code
# if cumulative_l1_sample:
#     print('cumulative l1-hit-rate: %.4f' % (cumulative_l1_hits / cumulative_l1_sample))
# if cumulative_l1_hits:
#     print('cumulative l1-hit-acc: %.4f' % (cumulative_l1_hit_correct / cumulative_l1_hits))
# else:
#     print('cumulative l1-hit-acc = 0')
# if cumulative_l2_sample:
#     print('cumulative l2-hit-rate: %.4f' % (cumulative_l2_hits / cumulative_l2_sample))
# if cumulative_l2_hits:
#     print('cumulative l2-hit-acc: %.4f' % (cumulative_l2_hit_correct / cumulative_l2_hits))
# else:
#     print('cumulative l2-hit-acc = 0')
#
# cumulative_hits = cumulative_l1_hits + cumulative_l2_hits
# cumulative_hit_correct = cumulative_l1_hit_correct + cumulative_l2_hit_correct
# if cumulative_sample:
#     print('cumulative hit_rate: %.4f' % (cumulative_hits / cumulative_sample))
# if cumulative_hits:
#     print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hits))
#
# cloud = (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits) * float(0.9014)
# print('cumulative acc: ',
#       round((cumulative_l1_hit_correct + cumulative_l2_hit_correct + cloud) / cumulative_sample, 4))
# print('cloud ', (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits))
# print('total test ', cumulative_sample)

# for i in buckets:
#     print('bucket', i)
#     if bckt[i]['l1']['total']:
#         print(round(bckt[i]['l1']['hits'] / bckt[i]['l1']['total'], 2))
#     if bckt[i]['l1']['hits']:
#         print(round(bckt[i]['l1']['tp'] / bckt[i]['l1']['hits'], 2))
#     if bckt[i]['l2']['total']:
#         print(round(bckt[i]['l2']['hits'] / bckt[i]['l2']['total'], 2))
#     if bckt[i]['l2']['hits']:
#         print(round(bckt[i]['l2']['tp'] / bckt[i]['l2']['hits'], 2))
