# %%
from charset_normalizer import models
from collections import defaultdict
import data
import os
import time
import soundfile as sf
import torch
import numpy as np
from functools import reduce
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils import audio_augment
import pandas as pd
import models
import random
import csv
import math
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
dataset_to_use = valid_dataset
dataset_to_use.df.head()
test_dataset.df.head()

act_obj_loc = {}
for dataset in [train_dataset, valid_dataset, test_dataset]:
    for _, row in dataset.df.iterrows():
        act, obj, loc = row['action'], row['object'], row['location']
        if (act, obj, loc) not in act_obj_loc:
            act_obj_loc[(act, obj, loc)] = len(act_obj_loc)
len(act_obj_loc)

device = torch.device('cpu')

# process data
new_train_df = deepcopy(train_dataset.df)
act_obj_loc_idxs = []
for _, row in new_train_df.iterrows():
    act, obj, loc = row['action'], row['object'], row['location']
    act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
new_train_df['cache'] = act_obj_loc_idxs
new_train_df.head()
# new_train_df.to_csv('train_data_cache.csv', index=None)

id2phoneme = {
    0: 'sil',
    1: 'S',
    2: 'IH',
    3: 'N',
    4: 'Y',
    5: 'UW',
    6: 'AA',
    7: 'R',
    8: 'AH',
    9: 'F',
    10: 'EH',
    11: 'D',
    12: 'V',
    13: 'M',
    14: 'AY',
    15: 'K',
    16: 'Z',
    17: 'HH',
    18: 'P',
    19: 'IY',
    20: 'B',
    21: 'sp',
    22: 'SH',
    23: 'UH',
    24: 'AE',
    25: 'ER',
    26: 'T',
    27: 'OW',
    28: 'DH',
    29: 'CH',
    30: 'L',
    31: 'EY',
    32: 'JH',
    33: 'AO',
    34: 'W',
    35: 'G',
    36: 'AW',
    37: 'TH',
    38: 'NG',
    39: 'OY',
    40: 'ZH',
    41: 'spn',
}
phoneme2id = {v: k for k, v in id2phoneme.items()}


def create_csv_file(filename, headers):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def write_to_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

config_file = 'train_test_per_speaker.csv'
headers = ['SpeakerId', 'Train(%)', 'Test(%)', 'Train_Dup', 'Test_Dup', 'Total']
create_csv_file(config_file, headers)

results_file = 'per_speaker_unseen_test.csv'
headers = ['SpeakerId', 'Threshold', 'Hit Rate', 'cache_acc', 'total_acc']
create_csv_file(results_file, headers)

# emulate an oracle cloud model
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

# local, personalized cache model
# threshold for ctc_loss, if less than it, use cache
# else, use the full model
d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

num_nan_train, nan_nan_eval = 0, 0
speakers = np.unique(new_train_df['speakerId'])

hit_dict = defaultdict(lambda: defaultdict(int))
tp_dict = defaultdict(lambda: defaultdict(int))
total_dict = defaultdict(int)
THRESHOLDS = [30, 40, 50, 60, 70, 80, 90, 100]

for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    print('training for speaker %s' % speakerId)

    # create train and test set
    tmp = new_train_df[new_train_df['speakerId'] == speakerId]
    tmp = tmp.sample(frac=1, random_state=42).reset_index(drop=True)
    transcripts = tmp['transcription']
    train_len = int(0.5*len(transcripts))
    train_set = tmp.iloc[:train_len]
    test_set = tmp.iloc[train_len-1:-1]
    train_tscpt = np.unique(train_set['transcription'])
    for _, row in test_set.iterrows():
        if row['transcription'] in train_tscpt:
            train_set = train_set.append(row, ignore_index=True)
            test_set = test_set[test_set['transcription'] != row['transcription']]

    train_dup = train_set[train_set.duplicated('transcription', keep=False)]
    test_dup = test_set[test_set.duplicated('transcription', keep=False)]
    train_pct = math.ceil(len(train_set)/len(transcripts)*100)
    test_pct = math.floor(len(test_set)/len(transcripts)*100)
    print('train: test: ratio ', len(train_set), len(test_set), train_pct, test_pct)
    print('train_dup: test_dup  ', len(train_dup), len(test_dup))

    values = [speakerId, train_pct, test_pct, len(train_dup), len(test_dup), len(transcripts)]
    write_to_csv(config_file, values)

    training_idxs = set()

    # the following three are used for evaluation
    transcript_list = []
    phoneme_list = []
    intent_list = []
    # create a new model for this previous speaker
    model = models.Model(config)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

    # training
    for _,row in train_set.iterrows():
        optim.zero_grad()
        transcript = row['transcription']
        print(transcript)
        # remove ending punctuation from the transcript
        phoneme_seq = reduce(lambda x, y: x + ['sp'] + y,
                             [d[tk][0] if tk in d else [] for tk in tknz.tokenize(transcript.lower())])
        transcript_list.append(transcript)
        # create (N, T) label
        phoneme_label = torch.tensor(
            [phoneme2id[ph[:-1]] if ph[-1].isdigit() else phoneme2id[ph] for ph in phoneme_seq],
            dtype=torch.long, device=device)
        phoneme_list.append(phoneme_label)
        intent_list.append(row['cache'])
        # add the index to training set, won't use in eval below
        training_idxs.add(row['index'])
        # load the audio file
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
        x_aug = torch.tensor(audio_augment(x), dtype=torch.float, device=device)
        phoneme_label = phoneme_label.repeat(x_aug.shape[0], 1)
        phoneme_pred = model.pretrained_model.compute_phonemes(x_aug)
        pred_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_pred.shape[0], dtype=torch.long)
        label_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_label.shape[-1], dtype=torch.long)

        loss = ctc_loss(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
        # FIXME implement better fix for nan loss
        if torch.isnan(loss).any():
            num_nan_train = num_nan_train + 1
            print('nan training on speaker: %s' % speakerId)
            optim.zero_grad()
        loss.backward()
        optim.step()
    if num_nan_train:
        print('nan in train happens %d times' % num_nan_train)

    # testing
    # prepare all the potential phoneme sequences
    print('\nTESTING')
    label_lengths = torch.tensor(list(map(len, phoneme_list)), dtype=torch.long)
    phoneme_label = pad_sequence(phoneme_list, batch_first=True).to(device)
    # no reduction, loss on every sequence
    ctc_loss_eval = torch.nn.CTCLoss(reduction='none')
    total = 0
    total_time_spk = 0
    for _, row in tmp.iterrows():
        if row['index'] in training_idxs:
            continue
        total += 1
        total_dict[speakerId] += 1
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
        x = torch.tensor(x, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            tick = time.time()
            phoneme_pred = model.pretrained_model.compute_phonemes(x)
            # repeat it #sentence times to compare with ground truth
            phoneme_pred = phoneme_pred.repeat(1, phoneme_label.shape[0], 1)
            pred_lengths = torch.full(size=(phoneme_label.shape[0],), fill_value=phoneme_pred.shape[0],
                                      dtype=torch.long)
            loss = ctc_loss_eval(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
            pred_result = loss.argmin()
            total_time_spk += (time.time() - tick)
            if torch.isnan(loss).any():
                print('nan eval on speaker: %s' % speakerId)

            # branch based on loss value
            for THRESHOLD in THRESHOLDS:
                if loss.min() <= THRESHOLD:
                    hit_dict[speakerId][THRESHOLD] += 1
                    if row['cache'] == intent_list[pred_result]:
                        tp_dict[speakerId][THRESHOLD] += 1
                    else:
                        print('%s,%s' % (row['transcription'], transcript_list[pred_result]))

    if total >= 5:  # skip for users with < 5 eval samples
        for threshold in THRESHOLDS:
            hit_rate, cache_acc, total_acc= 0,0,0
            # avoid divided by zero
            if hit_dict[speakerId][threshold] != 0:
                cache_acc = round(tp_dict[speakerId][threshold] / hit_dict[speakerId][threshold], 4)
            hit_rate = round(hit_dict[speakerId][threshold] / total_dict[speakerId], 4)
            total_acc = round((tp_dict[speakerId][threshold] + (total_dict[speakerId] - hit_dict[speakerId][threshold])) / total_dict[speakerId], 4)
            print('writing to csv %d' % threshold)
            values = [speakerId, threshold, hit_rate, cache_acc, total_acc]
            print(values)
            write_to_csv(results_file, values)
    else:
        print('not enough samples: %s' % speakerId)

for threshold in THRESHOLDS:
    print('THRESHOLD = %d' % threshold)
    tp, hits, total, correct = 0, 0, 0, 0
    for k in hit_dict:
        hits += hit_dict[k][threshold]
        tp += tp_dict[k][threshold]
        total += total_dict[k]
        correct += tp_dict[k][threshold] + (total_dict[k] - hit_dict[k][threshold])  # cumulative_correct += tp + (total - hits)
    print('hit_rate %.4f' % (hits / total))
    print('cache acc %.4f' % (tp / hits))
    print('total_acc %.4f' % (correct / total))

