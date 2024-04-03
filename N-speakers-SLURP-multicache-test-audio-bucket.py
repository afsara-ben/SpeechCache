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
import models
import pickle

from utils import write_to_csv, create_csv_file

device = "cpu"
config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')

# emulate an oracle cloud model
config.phone_rnn_num_hidden = [128, 128]
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

L1_THRESHOLDS = [400, 700, 1100]
# L2_THRESHOLDS = [40, 90, 150]
L2_THRESHOLDS = [30, 85, 170]
# variables to save #hits, #corrects
cumulative_l1_hits, cumulative_l1_corrects, cumulative_l1_hit_correct = 0, 0, 0
cumulative_l2_hits, cumulative_l2_corrects, cumulative_l2_hit_correct = 0, 0, 0
buckets = [1, 2, 3]

bucket_results_file = 'slurp_multicache_audio_per_bucket_dets.csv'
if not os.path.exists(bucket_results_file):
    headers = ['SpeakerId', 'l1_threshold', 'l2_threshold', 'bucket', 'train', 'test', 'l1 tp', 'l1 hits',
               'l1_hit_rate', 'l1_cache_acc', 'l2_sample', 'l2_tp', 'l2_hits', 'l2_hit_rate', 'l2_cache_acc']
    create_csv_file(bucket_results_file, headers)

results_file = 'slurp_multicache_audio_bucket.csv'
if not os.path.exists(results_file):
    headers = ['SpeakerId', 'l1_threshold', 'l2_threshold', 'train', 'test', 'l1_hit_rate', 'l1_cache_acc',
               'l2_hit_rate', 'l2_cache_acc', 'total_acc']
    create_csv_file(results_file, headers)

pwd = os.getcwd()
wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
folder_path = os.path.join(pwd, 'models/SLURP/models-N-speakers-SLURP-multicache-audio-bucket')

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4


def multicache_test(speakers, model, train_df, test_df, cluster_ids, cluster_centers, transcript_list, training_idxs,
                    intent_list, L1_THRESHOLD, L2_THRESHOLD):
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
    for _, row in test_df.iterrows():
        if row[0] in training_idxs:
            continue
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
            # print('l1 loss ', loss[pred_intent])
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
    return total, l2_total, l1_hits, l2_hits, l1_correct, l2_correct


slurp_df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_mini_FE_MO_ME_FO_UNK.csv'))
slurp_df = deepcopy(slurp_df)
num_nan_train, nan_nan_eval = 0, 0

cumulative_l1_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, cumulative_cache_miss, cumulative_hit_incorrect, total_train = 0, 0, 0, 0, 0, 0, 0, 0
spk_group = [['UNK-195', 'MO-371', 'UNK-166'], ['ME-132', 'ME-352', 'MO-051'], ['FO-152', 'MO-355', 'ME-354'],
             ['MO-222', 'FO-230', 'MO-465'], ['UNK-340', 'UNK-432', 'UNK-187'], ['ME-376', 'MO-424', 'MO-127'],
             ['UNK-162', 'ME-140', 'UNK-160'], ['UNK-182', 'MO-422', 'FE-178'], ['FO-445', 'FO-126', 'FE-248'],
             ['UNK-177', 'MO-055', 'FO-372'], ['MO-423', 'UNK-320', 'UNK-199'], ['UNK-200', 'UNK-385', 'ME-223'],
             ['UNK-342', 'MO-026', 'ME-220'], ['FO-444', 'UNK-458', 'FO-133'], ['UNK-331', 'FE-145', 'FO-228'],
             ['MO-019', 'UNK-322', 'FO-474'], ['UNK-165', 'MO-185', 'ME-373'], ['FO-233', 'UNK-198', 'ME-028'],
             ['UNK-168', 'FO-419', 'MO-433'], ['MO-036', 'UNK-196', 'FO-150'], ['MO-038', 'FE-149', 'FO-179'],
             ['ME-218', 'ME-151', 'ME-345'], ['UNK-325', 'MO-116', 'FO-124'], ['FO-425', 'FO-123', 'MO-030'],
             ['UNK-197', 'UNK-242', 'FE-146'], ['ME-369', 'FO-461', 'UNK-163'], ['FO-158', 'ME-434', 'UNK-341'],
             ['MO-142', 'UNK-240', 'FO-413'], ['ME-143', 'UNK-335', 'UNK-343'], ['MO-156', 'UNK-201', 'UNK-334'],
             ['FO-493', 'UNK-326', 'FO-488'], ['MO-442', 'FO-231', 'UNK-324'], ['ME-138', 'FE-235', 'MO-374'],
             ['FO-234', 'FE-249', 'MO-431'], ['FO-129', 'FE-141', 'UNK-323'], ['FO-438', 'FO-122', 'FO-171'],
             ['FO-420', 'ME-492', 'FO-184'], ['FO-229', 'MO-190', 'ME-414'], ['UNK-386', 'FO-219', 'FO-356'],
             ['MO-446', 'UNK-336', 'UNK-329'], ['UNK-159', 'FO-125', 'MO-463'], ['FO-405', 'MO-189', 'ME-144'],
             ['FO-139', 'UNK-241', 'UNK-328'], ['MO-191', 'FO-462', 'FO-180'], ['UNK-226', 'MO-044', 'UNK-436'],
             ['ME-473', 'FO-475', 'MO-418'], ['ME-147', 'FO-350', 'UNK-167'], ['MO-494', 'MO-164', 'UNK-245'],
             ['UNK-244', 'MO-467', 'UNK-208'], ['MO-375', 'UNK-327', 'FO-232'], ['MO-029', 'MO-050', 'UNK-225'],
             ['UNK-188', 'UNK-330', 'FO-460']]

# spk_group = [['ME-132', 'ME-352', 'MO-051']]
for _, speakers in tqdm(enumerate(spk_group), total=len(spk_group)):
    print('SPEAKERS ', speakers)
    bckt_sample, l2_bckt_sample, l1_bckt_correct, l2_bckt_correct, l1_bckt_hit, l2_bckt_hit, l1_bckt_hit_correct, l2_bckt_hit_correct, bckt_train = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for bucket in buckets:
        filename = f'3-spk-SLURP-multicache-{speakers}_audio_bucket_{bucket}.pkl'
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            load_data = pickle.load(f)
        model = load_data['model']
        speakers = load_data['speakers']
        train_set = load_data['train_set']
        test_set = load_data['test_set']
        transcript_list = load_data['transcript_list']
        phoneme_list = load_data['phoneme_list']
        intent_list = load_data['intent_list']
        training_idxs = load_data['training_idxs']
        cluster_ids = load_data['cluster_ids']
        cluster_centers = load_data['cluster_centers']

        total, l2_total, l1_hits, l2_hits, l1_correct, l2_correct = 0, 0, 0, 0, 0, 0
        if cluster_ids and cluster_centers and phoneme_list:
            total, l2_total, l1_hits, l2_hits, l1_correct, l2_correct = multicache_test(speakers, model, train_set,
                                                                                        test_set, cluster_ids,
                                                                                        cluster_centers,
                                                                                        transcript_list, training_idxs,
                                                                                        intent_list,
                                                                                        L1_THRESHOLD=L1_THRESHOLDS[
                                                                                            bucket - 1],
                                                                                        L2_THRESHOLD=L2_THRESHOLDS[
                                                                                            bucket - 1])

        print(f'################################ {l2_hits}################################ ')
        l1_hit_rate, l1_cache_acc, l2_hit_rate, l2_cache_acc = 0, 0, 0, 0
        bckt_train += len(training_idxs)
        bckt_sample += total
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
                l1_hit_rate = round((l1_hits / total), 4)
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
        values = [speakers, L1_THRESHOLDS[bucket - 1], L2_THRESHOLDS[bucket - 1], bucket, len(training_idxs), total,
                  l1_correct, l1_hits, l1_hit_rate, l1_cache_acc, l2_total, l2_correct, l2_hits, l2_hit_rate,
                  l2_cache_acc]
        # write_to_csv(bucket_results_file, values)
        print(values)

    total_acc, l1_hit_rate, l2_hit_rate, l1_cache_acc, l2_cache_acc = 0, 0, 0, 0, 0
    total_train += bckt_train

    # following needed for overall evaluation
    cumulative_l1_sample += bckt_sample
    cumulative_l2_sample += l2_bckt_sample
    cumulative_l1_hits += l1_bckt_hit
    cumulative_l2_hits += l2_bckt_hit
    cumulative_l1_hit_correct += l1_bckt_hit_correct
    cumulative_l2_hit_correct += l2_bckt_hit_correct

    # cumulative_l1_corrects += l1_bckt_correct #tp+total-hits
    # cumulative_l2_corrects += l2_bckt_correct

    print(f'---------------------------------{speakers}-----------------------------------------------')
    if bckt_sample:
        l1_hit_rate = round((l1_bckt_hit / bckt_sample), 4)
        total_acc = round((l1_bckt_hit_correct + l2_bckt_hit_correct + (
                bckt_sample - l1_bckt_hit - l2_bckt_hit) * 0.9014) / bckt_sample, 4)
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

    values = [speakers, L1_THRESHOLDS, L2_THRESHOLDS, bckt_train, bckt_sample, l1_hit_rate, l1_cache_acc, l2_hit_rate,
              l2_cache_acc, total_acc]
    # write_to_csv(results_file, values)
    # print(values)

print('\n\n', L1_THRESHOLDS, L2_THRESHOLDS)
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

cumulative_hits = cumulative_l1_hits + cumulative_l2_hits
cumulative_hit_correct = cumulative_l1_hit_correct + cumulative_l2_hit_correct
print('cumulative hit_rate: %.4f' % (cumulative_hits / cumulative_l1_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hits))

cloud = (cumulative_l1_sample - cumulative_l1_hits - cumulative_l2_hits) * float(0.9014)
print('cumulative acc: ',
      round((cumulative_l1_hit_correct + cumulative_l2_hit_correct + cloud) / cumulative_l1_sample, 4))
print()
