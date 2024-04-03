# %%
import argparse

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
from models import MLP
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from tqdm import tqdm
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from kmeans_pytorch import kmeans
import gzip
import models
import pickle
from utils import audio_augment, get_token_and_weight, get_audio_duration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
dataset_to_use = valid_dataset
dataset_to_use.df.head()
test_dataset.df.head()

with open('phoneme_list.txt', 'r') as file:
    id2phoneme = ast.literal_eval(file.read())
phoneme2id = {v: k for k, v in id2phoneme.items()}

# emulate an oracle cloud model
config.phone_rnn_num_hidden = [128, 128]
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

# local, personalized cache model
# threshold for ctc_loss, if less than it, use cache
# else, use the full model
L1_THRESHOLD, L2_THRESHOLD = 500, 50
NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4
# variables to save #hits, #corrects
cumulative_l1_hits, cumulative_l1_corrects = 0, 0
cumulative_l2_hits, cumulative_l2_corrects = 0, 0


parser = argparse.ArgumentParser(description="in-domain & dynamic")
parser.add_argument("--dynamic", type=bool, default=False)
parser.add_argument("--in_domain", type=bool, default=False)
args = parser.parse_args()
in_domain = args.in_domain
dynamic = args.dynamic

pwd = os.getcwd()
if in_domain:
    folder_path = os.path.join(pwd, 'models/SLURP/slurp-pretrain-multicache')
    cloud_model = models.Model(config).eval()
    cloud_model.load_state_dict(
        torch.load("/Users/afsarabenazir/Downloads/speech_projects/speechcache/models/SLURP/slurp-pretrained.pth",
                   map_location=device))  # load trained model
else:
    folder_path = os.path.join(pwd, 'models/SLURP/models-slurp-multicache')
    cloud_model = models.Model(config)
    optim = torch.optim.Adam(cloud_model.parameters(), lr=1e-3)
    cloud_model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
wav_path = os.path.join(pwd, 'SLURP/slurp_real/')

df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_mini_FE_MO_ME_FO_UNK.csv'))
df = deepcopy(df)
num_nan_train, nan_nan_eval = 0, 0
speakers = np.unique(df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232', 'ME-144']

cumulative_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, cumulative_cache_miss,cumulative_hit_incorrect, total_train = 0, 0, 0, 0, 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('SLURP MULTI CACHE EVAL FOR SPEAKER ', user_id)
    tmp = df[df['user_id'] == user_id]
    filename = f'slurp-pretrain-multicache-{user_id}.pth'
    file_path = os.path.join(folder_path, filename)

    model = models.Model(config)
    model.load_state_dict(torch.load(file_path))

    with gzip.open(os.path.join(folder_path, f'slurp-pretrain-multicache-{user_id}.pkl.gz'), 'rb') as f:
        metadata = pickle.load(f)

    user_id = metadata['speakerId']
    transcript_list = metadata['transcript_list']
    phoneme_list = metadata['phoneme_list']
    intent_list = metadata['intent_list']
    training_idxs = metadata['training_idxs']
    cluster_ids = metadata['cluster_ids']
    cluster_centers = metadata['cluster_centers']
    if not cluster_ids:
        continue
    if not phoneme_list:
        continue

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
    tp, total, hits, l1_hits, l2_hits, l1_correct, l2_correct, l2_total = 0, 0, 0, 0, 0, 0, 0, 0
    for _, row in tmp.iterrows():
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
            x_feature = cloud_model.pretrained_model.compute_cnn_features(x)
            dists = torch.cdist(x_feature, cluster_centers)
            dists = dists.max(dim=-1)[0].unsqueeze(-1) - dists
            pred = dists.swapaxes(1, 0)
            pred_lengths = torch.full(size=(cluster_ids.shape[0],), fill_value=pred.shape[0], dtype=torch.long)
            loss = ctc_loss_k_means_eval(pred.log_softmax(dim=-1), cluster_ids, pred_lengths, cluster_id_length)
            pred_intent = loss.argmin().item()
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
                cumulative_l1_hits += 1
                if row['intent'] == intent_list[pred_intent]:
                    l1_correct += 1
                    cumulative_l1_corrects += 1
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
                    cumulative_l2_hits += 1
                    if row['intent'] == intent_list[pred_result]:
                        l2_correct += 1
                        cumulative_l2_corrects += 1
                    # else:
                    #     print('%s,%s' % (row['sentence'], transcript_list[pred_result]))
                # else:
                #     # do the calculation
                #     # cloud_model.predict_intents(x)
                #     print('cloud. loss was %f ' % loss.min())

    if total >= 5:  # skip for users with < 5 eval samples
        cumulative_sample += total
        cumulative_l2_sample += l2_total
        total_acc = round((l1_correct + l2_correct + (total - l1_hits - l2_hits) * 0.9014) / total, 4)
        print('total acc ', total_acc)
        if l1_hits:
            print('l1_hit_rate %.2f' % (l1_hits/total))
            print('l1_cache_acc %.2f' % (l1_correct / l1_hits))
        else:
            print('no hits in l1')
        if l2_hits:
            print('l2_hit_rate %.2f' % (l2_hits / l2_total))
            print('l2_cache_acc %.2f' % (l2_correct / l2_hits))
        else:
            print('no hits in l2')

print(f"SLURP MULTICACHE - regular - in-domain {in_domain}")
print('threshold L1: %d L2: %d ' % (L1_THRESHOLD, L2_THRESHOLD))
if cumulative_sample:
    print('cumulative l1-hit-rate: %.4f' % (cumulative_l1_hits / cumulative_sample))
if cumulative_l1_hits:
    print('cumulative l1-hit-acc: %.4f' % (cumulative_l1_corrects / cumulative_l1_hits))
else:
    print('cumulative l1-hit-acc = 0')
if cumulative_l2_sample:
    print('cumulative l2-hit-rate: %.4f' % (cumulative_l2_hits / cumulative_l2_sample))
if cumulative_l2_hits:
    print('cumulative l2-hit-acc: %.4f' % (cumulative_l2_corrects / cumulative_l2_hits))
else:
    print('cumulative l2-hit-acc = 0')
if cumulative_sample:
    print('cumulative acc:', round((cumulative_l1_corrects + cumulative_l2_corrects + (cumulative_sample - cumulative_l1_hits - cumulative_l2_hits) * 0.9014) / cumulative_sample, 4))
