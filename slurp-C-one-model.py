import ast
import gzip

import data
import os
import soundfile as sf
import torch
import numpy as np
from functools import reduce
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from tqdm import tqdm
from utils import audio_augment
import models
import pandas as pd
import pickle
from torch.nn.utils.rnn import pad_sequence
from kmeans_pytorch import kmeans

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4
L1_THRESHOLD = 500
L2_THRESHOLD = 50

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = 'cpu'

with open('phoneme_list.txt', 'r') as file:
    id2phoneme = ast.literal_eval(file.read())
phoneme2id = {v: k for k, v in id2phoneme.items()}

d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

pwd = os.getcwd()
wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
# folder_path = os.path.join(pwd, 'models/SLURP/curated-slurp-with-headset-multicache')
folder_path = os.path.join(pwd, 'models/SLURP/compressed-curated-slurp-headset-base-multicache')

# in domain
# pretrained_file = "slurp-pretrained.pth"
# pretrained_path = os.path.join(pwd + "/models/SLURP/", pretrained_file)
# base pretrained
pretrained_path = "experiments/no_unfreezing/training/model_state.pth"

# original
# folder_path = os.path.join(pwd, 'models/SLURP/models-slurp-multicache')
# df = pd.read_csv(os.path.join(pwd, 'SLURP/slurp_mini_FE_MO_ME_FO_UNK.csv'))
# df = deepcopy(df)
# num_nan_train, nan_nan_eval = 0, 0
# speakers = np.unique(df['user_id'])
# speakers = ['FO-232']

cloud_model = models.Model(config).eval()
optim = torch.optim.Adam(cloud_model.parameters(), lr=1e-3)
cloud_model.load_state_dict(
        torch.load(pretrained_path, map_location=device))  # load trained model

# trainign code start
"""
num_nan_train, nan_nan_eval = 0, 0
slurp_df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_headset.csv'))
slurp_df = deepcopy(slurp_df)
speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232']
# slurp_df = slurp_df[slurp_df['user_id'] == 'FE-141']
# speakers = ['FE-141']

transcripts = np.unique(slurp_df['sentence'])
model = models.Model(config).eval()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
model.load_state_dict(
        torch.load(pretrained_path, map_location=device))  # load trained model

# the following three are used for evaluation
transcript_list = []
phoneme_list = []
intent_list = []
cluster_ids = []
cluster_centers = []
training_idxs = set()

for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
    optim.zero_grad()

    # training
    # remove ending punctuation from the transcript
    phoneme_seq = reduce(lambda x, y: x + ['sp'] + y,
                         [d[tk][0] if tk in d else [] for tk in tknz.tokenize(transcript.lower())])
    transcript_list.append(transcript)
    # random choose one file with `transcription`
    rows = slurp_df[slurp_df['sentence'] == transcript]
    # sample only one audio file for each distinct transcription
    row = rows.iloc[np.random.randint(len(rows))]
    intent_list.append(row['intent'])
    # add the index to training set, won't use in eval below
    training_idxs.add(row[0])

    # load the audio file
    wav = wav_path + row['recording_path']
    x, _ = sf.read(wav)
    x_aug = torch.tensor(audio_augment(x), dtype=torch.float, device=device)

    # ----------------- kmeans cluster -----------------
    feature = model.pretrained_model.compute_cnn_features(x_aug)
    cluster_id, cluster_center = kmeans(X=feature.reshape(-1, feature.shape[-1]), num_clusters=NUM_CLUSTERS,
                                            distance=dist, tol=tol, device=device)
    # save the cluster center
    intention_label = []
    prev = None
    # collapses the cluster predictions
    for l in cluster_id.view(feature.shape[0], -1)[0]:
        if prev is None or prev != l:
            intention_label.append(l.item())
        prev = l
    cluster_ids.append(torch.tensor(intention_label, dtype=torch.long, device=device))
    cluster_centers.append(cluster_center)

    # ----------------- phoneme ctc -------------------
    phoneme_seq = reduce(lambda x, y: x + ['sp'] + y,
                         [d[tk][0] if tk in d else [] for tk in tknz.tokenize(transcript.lower())])
    phoneme_label = torch.tensor(
        [phoneme2id[ph[:-1]] if ph[-1].isdigit() else phoneme2id[ph] for ph in phoneme_seq],
            dtype=torch.long, device=device)
    phoneme_list.append(phoneme_label)
    phoneme_label = phoneme_label.repeat(x_aug.shape[0], 1)
    phoneme_pred = model.pretrained_model.compute_phonemes(x_aug)
    pred_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_pred.shape[0], dtype=torch.long)
    label_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_label.shape[-1], dtype=torch.long)

    loss = ctc_loss(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
    # FIXME implement better fix for nan loss
    if torch.isnan(loss).any():
        num_nan_train = num_nan_train + 1
        optim.zero_grad()

    loss.backward()
    optim.step()

# filename = f'slurp_curated_headset_multicache_{user_id}'
filename = f'slurp_C-all-spk'
file_path = os.path.join(folder_path, filename + '.pth')

# del model.pretrained_model.word_linear
# del model.pretrained_model.word_layers
# del model.intent_layers
torch.save(model, file_path)

metadata = {
    'df': slurp_df,
    'transcript_list': transcript_list,
    'phoneme_list': phoneme_list,
    'intent_list': intent_list,
    'training_idxs': training_idxs,
    'cluster_ids': cluster_ids,
    'cluster_centers': cluster_centers,
}
with gzip.open(os.path.join(folder_path, filename + '.pkl.gz'), 'wb') as f:
    pickle.dump(metadata, f)

print()
"""

# testing
# variables to save #hits, #corrects
cumulative_l1_hits, cumulative_l1_corrects = 0, 0
cumulative_l2_hits, cumulative_l2_corrects = 0, 0
cumulative_sample, cumulative_l1_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, cumulative_cache_miss,cumulative_hit_incorrect, total_train, cumulative_l1_hit_correct, cumulative_l2_hit_correct = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

filename = f'slurp_C-all-spk'
file_path = os.path.join(folder_path, filename + '.pth')
model = torch.load(file_path)
with gzip.open(os.path.join(folder_path, filename + '.pkl.gz'), 'rb') as f:
    metadata = pickle.load(f)

# df = metadata['df']
transcript_list = metadata['transcript_list']
phoneme_list = metadata['phoneme_list']
intent_list = metadata['intent_list']
training_idxs = metadata['training_idxs']
cluster_ids = metadata['cluster_ids']
cluster_centers = metadata['cluster_centers']

slurp_df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_headset.csv'))
slurp_df = deepcopy(slurp_df)
speakers = np.unique(slurp_df['user_id'])
# speakers = ['FO-234']

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
# tp, total, hits, l1_hits, l2_hits, l1_correct, l2_correct, l2_total = 0, 0, 0, 0, 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('SLURP-C EVAL ', user_id)
    df = slurp_df[slurp_df['user_id'] == user_id]
    tp, total, hits, l1_hits, l2_hits, l1_correct, l2_correct, l2_total = 0, 0, 0, 0, 0, 0, 0, 0
    for _, row in df.iterrows():
        if row[0] in training_idxs:
            continue
        print(row['sentence'])
        # # of total evaluation samples
        total += 1
        wav = os.path.join(wav_path, row['recording_path'])
        x, _ = sf.read(wav)
        x = torch.tensor(x, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            # ----------------- l1 -------------------
            x_feature = cloud_model.pretrained_model.compute_cnn_features(x)
            dists = torch.cdist(x_feature, cluster_centers)
            dists = dists.max(dim=-1)[0].unsqueeze(-1) - dists
            pred = dists.swapaxes(1, 0)
            pred_lengths = torch.full(size=(cluster_ids.shape[0],), fill_value=pred.shape[0], dtype=torch.long)
            loss = ctc_loss_k_means_eval(pred.log_softmax(dim=-1), cluster_ids, pred_lengths, cluster_id_length)
            pred_intent = loss.argmin().item()

            if loss[pred_intent] < L1_THRESHOLD:
                # go with l1: kmeans
                # print('l1 hit: ', row['sentence'])
                l1_hits += 1
                # cumulative_l1_hits += 1
                if row['intent'] == intent_list[pred_intent]:
                    l1_correct += 1
                    # cumulative_l1_corrects += 1
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
                if loss.min() <= L2_THRESHOLD:
                    # print('l2 hit: ', row['sentence'])
                    l2_hits += 1
                    # cumulative_l2_hits += 1
                    if row['intent'] == intent_list[pred_result]:
                        l2_correct += 1
                        # cumulative_l2_corrects += 1
                    # else:
                    #     print('%s,%s' % (row['sentence'], transcript_list[pred_result]))
                else:
                    # do the calculation
                    # cloud_model.predict_intents(x)
                    print('cloud. loss was %f ' % loss.min())

    if total >= 5:  # skip for users with < 5 eval samples
        total_train += len(training_idxs)
        cumulative_sample += total
        cumulative_l1_sample += total
        cumulative_l2_sample += l2_total
        cumulative_l1_hits += l1_hits
        cumulative_l2_hits += l2_hits
        cumulative_l1_hit_correct += l1_correct
        cumulative_l2_hit_correct += l2_correct
        total_acc = round((l1_correct + l2_correct + (total - l1_hits - l2_hits) * 0.9014) / total, 4)
        print('----------- total acc ', total_acc)
        if l1_hits:
            print('l1_hit_rate ', l1_hits/total)
            print('l1_cache_acc ', l1_correct / l1_hits)
        else:
            print('no hits in l1')
        if l2_hits:
            print('l2_hit_rate ', l2_hits / l2_total)
            print('l2_cache_acc ', l2_correct / l2_hits)
        else:
            print('no hits in l2')

print(f"------------------------------------------------------------------------")
cumulative_hits = cumulative_l1_hits + cumulative_l2_hits
cumulative_hit_correct = cumulative_l1_hit_correct + cumulative_l2_hit_correct
cumulative_l1_hit_rate, cumulative_l1_hit_acc, cumulative_l2_hit_rate, cumulative_l2_hit_acc = 0, 0, 0, 0


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
cumulative_acc = round((cumulative_l1_hit_correct + cumulative_l2_hit_correct + cloud) / cumulative_sample, 4)
print(f'total train {total_train} test {cumulative_sample}')
print(f'l1: hit_rate {cumulative_l1_hit_rate} hit_acc {cumulative_l1_hit_acc}')
print(f'l2: hit_rate {cumulative_l2_hit_rate} hit_acc {cumulative_l2_hit_acc}')
print(f'l1+L2: hit_rate {cumulative_hit_rate} hit_acc {cumulative_cache_acc}')
# print(f'acc {cumulative_acc} for cutoff {cutoff}')
print(f'acc {cumulative_acc}')
print(f"------------------------------------------------------------------------")

print('cloud ', (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits))

