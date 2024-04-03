import data
import os
import time
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle
from utils import create_csv_file, write_to_csv
import numpy as np
import pandas as pd
from copy import deepcopy
import sys
import models

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = 'cpu'

bucket_results_file = 'slurp_audio_per_bucket_dets.csv'
if not os.path.exists(bucket_results_file):
    headers = ['SpeakerId', 'threshold', 'bucket', 'train', 'test', 'tp', 'hits', 'hit_rate', 'cache_acc']
    create_csv_file(bucket_results_file, headers)

results_file = 'slurp_audio_bucket.csv'
if not os.path.exists(results_file):
    headers = ['SpeakerId', 'threshold', 'train', 'test', 'hit_rate', 'cache_acc', 'total_acc']
    create_csv_file(results_file, headers)

wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models-N-speakers-SLURP-k-means-audio-bucket')

model = models.Model(config)
model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4


def test(train_set, test_set, cluster_ids, cluster_centers, transcript_list, training_idxs, intent_list, THRESHOLD):
    # ----------------- prepare for cluster -----------------
    cluster_id_length = torch.tensor(list(map(len, cluster_ids)), dtype=torch.long, device=device)
    cluster_ids = pad_sequence(cluster_ids, batch_first=True, padding_value=0).to(device)
    cluster_centers = torch.stack(cluster_centers).to(device)
    # no reduction, loss on every sequence
    ctc_loss_k_means_eval = torch.nn.CTCLoss(reduction='none')
    tp, total, hits = 0, 0, 0
    for _, row in test_set.iterrows():
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
            if loss[pred_intent] < THRESHOLD:
                # go with l1: kmeans
                # print('hit')
                hits += 1
                if row['intent'] == intent_list[pred_intent]:
                    # print('tp')
                    tp += 1
                else:
                    print('%s,%s' % (row['sentence'], transcript_list[pred_intent]))
            # else:
            #     print('cloud. loss was %f ' % loss.min())

    return total, hits, tp


buckets = [1, 2, 3]
THRESHOLDS = [400, 700, 1100]
# arguments = sys.argv
# for arg in arguments[1:]:
#     THRESHOLDS.append(int(arg))

slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_FE_MO_ME_FO_UNK.csv')
slurp_df = deepcopy(slurp_df)

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

cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, total_train = 0, 0, 0, 0, 0
for _, speakers in tqdm(enumerate(spk_group), total=len(spk_group)):
    print('SPEAKERS ', speakers)
    bckt_sample, bckt_correct, bckt_hit, bckt_hit_correct, bckt_train = 0, 0, 0, 0, 0
    for bucket in buckets:
        print('\nloading bucket %d model.....' % bucket)
        filename = f'3-spk-SLURP-k-means-{speakers}_audio_bucket_{bucket}.pkl'
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            load_data = pickle.load(f)

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
            total, hits, tp = test(train_set, test_set, cluster_ids, cluster_centers, transcript_list, training_idxs,
                                   intent_list,
                                   THRESHOLD=THRESHOLDS[bucket - 1])
        else:
            continue
        hit_rate, cache_acc, total_acc = 0, 0, 0
        if total >= 5:  # skip for users with < 5 eval samples
            print('EVAL FOR SPEAKER %s: BUCKET %d' % (speakers, bucket))
            bckt_train += len(training_idxs)
            bckt_sample += total
            bckt_correct += tp + (total - hits) * 0.9014
            bckt_hit += hits
            bckt_hit_correct += tp
            print('Train: %d Test: %d' % (len(training_idxs), total))
            hit_rate = round((hits / total), 4)
            print('hit_rate %.3f ' % hit_rate)
            if hits:
                cache_acc = round((tp / hits), 4)
                print('cache_acc %.3f' % cache_acc)
            total_acc = round((tp + (total - hits) * 0.9014) / total, 4)
            print('total_acc: %.3f' % total_acc)
        else:
            print('not enough samples: %s' % speakers)
        values = [speakers, THRESHOLDS[bucket - 1], bucket, len(training_idxs), total, tp, hits, hit_rate, cache_acc]
        # write_to_csv(bucket_results_file, values)
        print(values)
        print()

    hit_rate, cache_acc, total_acc = 0, 0, 0
    total_train += bckt_train
    cumulative_sample += bckt_sample
    cumulative_correct += bckt_correct
    cumulative_hit += bckt_hit
    cumulative_hit_correct += bckt_hit_correct
    if bckt_sample:
        hit_rate = round((bckt_hit / bckt_sample), 4)
        total_acc = round((bckt_correct / bckt_sample), 4)
    if bckt_hit:
        cache_acc = round((bckt_hit_correct / bckt_hit), 4)
    print('hit_rate: %.4f' % hit_rate)
    print('cache_acc: %.4f' % cache_acc)
    print('total_acc: %.4f' % total_acc)
    values = [speakers, THRESHOLDS, bckt_train, bckt_sample, hit_rate, cache_acc, total_acc]
    print(values)
    print()

print('Threshold ', THRESHOLDS)
print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()
