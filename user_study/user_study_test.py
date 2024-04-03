# %%
import argparse
import gzip
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
from collections import defaultdict
import noisereduce as nr

from utils import get_audio_duration

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg")
device = 'cpu'
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)


# L1_THRESHOLDS = [400, 700, 1100]
L1_THRESHOLDS = [700, 1200, 2000]
L2_THRESHOLDS = [30, 85, 170]
# L2_THRESHOLDS = [50, 110, 200]
# L2_THRESHOLDS = [25, 50, 100]
# L2_THRESHOLDS =  [30,60,110]
# L1_THRESHOLDS = [400, 600, 1100]
# L2_THRESHOLDS = [40, 90, 200]

# L2_THRESHOLDS = [30, 85, 170]
# variables to save #hits, #corrects
cumulative_l1_hits, cumulative_l1_corrects, cumulative_l1_hit_correct = 0, 0, 0
cumulative_l2_hits, cumulative_l2_corrects, cumulative_l2_hit_correct = 0, 0, 0
buckets = [1, 2, 3]

parser = argparse.ArgumentParser(description="input cutoff")
parser.add_argument("--cutoff", type=float, default=2.0)
parser.add_argument("--dynamic", type=bool, default=False)
args = parser.parse_args()
cutoff = args.cutoff
dynamic = args.dynamic

# ---------------------------change here ---------------------------
pwd = os.getcwd()
# wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
wav_path = os.path.join(pwd, 'user_study_recordings/')
# wav_path = os.path.join(pwd, 'user_study_recordings/w:o heaset/')  #aben:user study
# folder_path = os.path.join(pwd, f'models/SLURP/slurp_multicache_selective_audio_bucket_k_{cutoff}')
# folder_path = os.path.join(pwd, 'models/SLURP/compressed-curated-slurp-headset')
# folder_path = os.path.join(pwd, 'models/SLURP/user-study-headset')
folder_path = os.path.join(pwd, 'models/SLURP/user-study-wo-headset') #aben:user study
# folder_path = os.path.join(pwd, 'models/SLURP/compressed-curated-slurp-without-headset')
# folder_path = os.path.join(pwd, 'models/SLURP/compressed-curated-slurp-headset-base')


# ---------------------------change here---------------------------
# slurp-pretrained.pth/in-domain
pretrained_file = "slurp-pretrained.pth"
pretrained_path = os.path.join(pwd + "/models/SLURP/", pretrained_file)

# base path
# pretrained_path = "experiments/no_unfreezing/training/model_state.pth"

cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load(pretrained_path, map_location=device))  # load trained model

# ---------------------------------------------------------------------------------

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
        print(row['sentence'])
        # # of total evaluation samples
        total += 1
        wav = os.path.join(wav_path, row['recording_path'])
        x, _ = sf.read(wav)
        x = nr.reduce_noise(y=x, sr=_)
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
                    # filename = 'utils/slurp-MLP-L2_old.pkl'
                    with open(filename, 'rb') as f:
                        load_data = pickle.load(f)
                    MLP_model = load_data['model']
                    dur = torch.tensor([[get_audio_duration(wav)]], dtype=torch.float32)
                    L2_THRESHOLD = MLP_model(dur).item()
                    print('loss ', loss.min())
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
                    # print(round(dur.item(), 4), L1_THRESHOLD)
                print('loss ', loss[pred_intent])
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
                        # filename = 'utils/slurp-MLP-L2_old.pkl'
                        with open(filename, 'rb') as f:
                            load_data = pickle.load(f)
                        MLP_model = load_data['model']
                        dur = torch.tensor([[get_audio_duration(wav)]], dtype=torch.float32)
                        L2_THRESHOLD = MLP_model(dur).item()
                    print('loss ', loss.min())
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


# slurp_df = pd.read_csv('SLURP/csv/slurp_mini_FE_MO_ME_FO_UNK.csv')
# slurp_df = pd.read_csv('SLURP/csv/slurp_mini_FE.csv')
# slurp_df = pd.read_csv('SLURP/csv/slurp_headset.csv')
# slurp_df = pd.read_csv('user_study_recordings/data_headset.csv')
# slurp_df = pd.read_csv('user_study_recordings/data_wo_headset.csv')
slurp_df = pd.read_csv('user_study_recordings/adversarial.csv')
# slurp_df = pd.read_csv('SLURP/csv/slurp_without_headset.csv')
slurp_df = deepcopy(slurp_df)
num_nan_train, nan_nan_eval = 0, 0
speakers = np.unique(slurp_df['user_id'])
# starts_with = 'UNK'
# speakers = [elem for elem in speakers if not elem.startswith(starts_with)]
# speakers = ['MO-433', 'UNK-326', 'FO-232', 'ME-144']
# speakers = ['FO-234', 'FO-462', 'FO-488', 'FO-493', 'ME-144', 'ME-473']
# speakers = ['FO-234', 'FO-462', 'FO-488', 'ME-144', 'ME-369', 'ME-473','MO-030', 'MO-038']
# speakers = ['FO-234']
cumulative_l1_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_cache_miss, cumulative_hit_incorrect, cumulative_hit_rate, cumulative_cache_acc, cumulative_acc, total_train = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
cumulative_sample, cumulative_hit_correct, cumulative_hits = 0, 0, 0
bckt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
#     # print(f'EVAL FOR SPEAKER {user_id} cutoff {cutoff}')
#     print(f'FOLDER: {folder_path}')
#     print(f'EVAL FOR SPEAKER {user_id}')
#     print('count ', slurp_df[slurp_df['user_id'] == user_id].shape[0])
#     bckt_sample, l1_bckt_sample, l2_bckt_sample, l1_bckt_correct, l2_bckt_correct, l1_bckt_hit, l2_bckt_hit, l1_bckt_hit_correct, l2_bckt_hit_correct, bckt_train = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#     for bucket in buckets:
#         # --------------------------- change here ---------------------------
#         # filename = f'slurp_model_multicache_{user_id}_audio_bucket_{bucket}_cutoff_{cutoff}'
#         # filename = f'slurp_curated_multicache_{user_id}_audio_bucket_{bucket}'
#         # filename = f'slurp_curated_wo_headset_multicache_{user_id}_audio_bucket_{bucket}'
#         filename = f'slurp_curated_headset_base_multicache_{user_id}_audio_bucket_{bucket}'
#         file_path = os.path.join(folder_path, filename + '.pth')
#
#         model = models.Model(config)
#         model.load_state_dict(torch.load(file_path, map_location=device))
#
#         del model.pretrained_model.word_layers
#         del model.pretrained_model.word_linear
#         del model.intent_layers
#         new_file_path = os.path.join(new_folder_path, filename + '.pth')
#         torch.save(model, new_file_path)

for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    # print(f'EVAL FOR SPEAKER {user_id} cutoff {cutoff}')
    print(f'EVAL FOR SPEAKER {user_id}')
    print('count ', slurp_df[slurp_df['user_id'] == user_id].shape[0])
    bckt_sample, l1_bckt_sample, l2_bckt_sample, l1_bckt_correct, l2_bckt_correct, l1_bckt_hit, l2_bckt_hit, l1_bckt_hit_correct, l2_bckt_hit_correct, bckt_train = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for bucket in buckets:
        # --------------------------- change here ---------------------------
        # filename = f'slurp_model_multicache_{user_id}_audio_bucket_{bucket}_cutoff_{cutoff}'
        # filename = f'slurp_curated_multicache_{user_id}_audio_bucket_{bucket}'
        # filename = f'user_study_headset_base_multicache_{user_id}_audio_bucket_{bucket}'
        filename = f'user_study_wo_headset_base_multicache_{user_id}_audio_bucket_{bucket}'
        # filename = f'slurp_curated_wo_headset_multicache_{user_id}_audio_bucket_{bucket}'
        # filename = f'slurp_curated_headset_base_multicache_{user_id}_audio_bucket_{bucket}'
        file_path = os.path.join(folder_path, filename + '.pth')
        model = torch.load(file_path)

        with gzip.open(os.path.join(folder_path, filename + '.pkl.gz'), 'rb') as f:
            metadata = pickle.load(f)

        user_id = metadata['speakerId']
        df = metadata['df']
        transcript_list = metadata['transcript_list']
        phoneme_list = metadata['phoneme_list']
        intent_list = metadata['intent_list']
        training_idxs = metadata['training_idxs']
        cluster_ids = metadata['cluster_ids']
        cluster_centers = metadata['cluster_centers']
        print(f'bucket {bucket} train {len(training_idxs)} test: {len(df)-len(training_idxs)}')

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

        if total >= 0:  # skip for users with < 5 eval samples
            # print('EVAL FOR SPEAKER %s: BUCKET %d' % (user_id, bucket))
            # l1_bckt_correct += l1_correct + (total - l1_hits)
            bckt_train += len(training_idxs)
            bckt_sample += total
            l1_bckt_sample += l1_total
            l2_bckt_sample += l2_total
            l1_bckt_hit += l1_hits
            l1_bckt_hit_correct += l1_correct

            # l2_bckt_correct += l2_correct + (l2_total - l2_hits)
            l2_bckt_hit += l2_hits
            l2_bckt_hit_correct += l2_correct

            if l1_hits:
                l1_hit_rate = round((l1_hits / l1_total), 4)
                l1_cache_acc = round((l1_correct / l1_hits), 4)
                # print('l1_hit_rate ', l1_hit_rate)
                # print('l1_cache_acc ', l1_cache_acc)
            else:
                print('no hits in l1')
            if l2_hits:
                l2_hit_rate = round((l2_hits / l2_total), 4)
                l2_cache_acc = round((l2_correct / l2_hits), 4)
                # print('l2_hit_rate ', l2_hit_rate)
                # print('l2_cache_acc ', l2_cache_acc)
            else:
                print('no hits in l2')
            print(f'Bucket {bucket} [l1: hit_rate {l1_hit_rate} cache_acc {l1_cache_acc}], [l2: hit_rate {l2_hit_rate} cache_acc {l2_cache_acc}]')
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
    print(f'total train {bckt_train} test {bckt_sample}')
    if l1_bckt_sample:
        l1_hit_rate = round((l1_bckt_hit / l1_bckt_sample), 4)
        # print('l1 hit_rate: %.4f' % l1_hit_rate)
    if l1_bckt_hit:
        l1_cache_acc = round((l1_bckt_hit_correct / l1_bckt_hit), 4)
        # print('l1 cache_acc: %.4f' % l1_cache_acc)
    if l2_bckt_sample:
        l2_hit_rate = round((l2_bckt_hit / l2_bckt_sample), 4)
        # print('l2 hit_rate: %.4f' % l2_hit_rate)
    if l2_bckt_hit:
        l2_cache_acc = round((l2_bckt_hit_correct / l2_bckt_hit), 4)
        # print('l2 cache_acc: %.4f' % l2_cache_acc)
    print(f'{user_id} \nl1: hit_rate {l1_hit_rate} cache_acc {l1_cache_acc}, l2: hit_rate {l2_hit_rate} cache_acc {l2_cache_acc}')
    if bckt_sample:
        total_acc = round((l1_bckt_hit_correct + l2_bckt_hit_correct + (
            bckt_sample - l1_bckt_hit - l2_bckt_hit) * 0.9014) / bckt_sample, 4)
    # tp+total-hit/total
    print('total_acc: %.4f' % total_acc)
    print(f"------------------------------------------------------------------------")
    values = [user_id, L1_THRESHOLDS, L2_THRESHOLDS, bckt_train, bckt_sample, l1_hit_rate, l1_cache_acc, l2_hit_rate,
              l2_cache_acc, total_acc]
    # write_to_csv(results_file, values)
    # print(values)
    # print(bckt)


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
cloud = (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits) * float(0.9014)
cumulative_acc =  round((cumulative_l1_hit_correct + cumulative_l2_hit_correct + cloud) / cumulative_sample, 4)
print(f'total train {total_train} test {cumulative_sample}')
print(f'l1: hit_rate {cumulative_l1_hit_rate} hit_acc {cumulative_l1_hit_acc}')
print(f'l2: hit_rate {cumulative_l2_hit_rate} hit_acc {cumulative_l2_hit_acc}')
print(f'l1+L2: hit_rate {cumulative_hit_rate} hit_acc {cumulative_cache_acc}')
# print(f'acc {cumulative_acc} for cutoff {cutoff}')
print(f'acc {cumulative_acc}')
print(f"------------------------------------------------------------------------")

print('cloud ', (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits))
print()
#
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
