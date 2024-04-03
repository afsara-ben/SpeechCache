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

filename = 'model_unseen_train_few_speakers.pkl'
with open(filename, 'rb') as f:
    load_data = pickle.load(f)
model = load_data['model']
transcript_list = load_data['transcript_list']
phoneme_list = load_data['phoneme_list']
intent_list = load_data['intent_list']
training_idxs = load_data['training_idxs']
test_df = load_data['test_df']

hit_dict = defaultdict(int)
tp_dict = defaultdict(int)
total_dict = defaultdict(int)
THRESHOLDS = [30, 40, 50, 60, 70, 80, 90, 100]

# testing
# prepare all the potential phoneme sequences
print('\nTESTING')
label_lengths = torch.tensor(list(map(len, phoneme_list)), dtype=torch.long)
phoneme_label = pad_sequence(phoneme_list, batch_first=True).to(device)
# no reduction, loss on every sequence
ctc_loss_eval = torch.nn.CTCLoss(reduction='none')
total = 0
total_time_spk = 0

for _, row in test_df.iterrows():
    total += 1
    wav_path = os.path.join(config.slu_path, row['path'])
    x, _ = sf.read(wav_path)
    x = torch.tensor(x, dtype=torch.float, device=device).unsqueeze(0)
    with torch.no_grad():
        tick = time.time()
        phoneme_pred = model.pretrained_model.compute_phonemes(x)
        # repeat it #sentence times to compare with ground truth
        phoneme_pred = phoneme_pred.repeat(1, phoneme_label.shape[0], 1)
        pred_lengths = torch.full(size=(phoneme_label.shape[0],), fill_value=phoneme_pred.shape[0], dtype=torch.long)
        loss = ctc_loss_eval(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
        pred_result = loss.argmin()
        total_time_spk += (time.time() - tick)
        if torch.isnan(loss).any():
            print('nan eval')

        # branch based on loss value
        for THRESHOLD in THRESHOLDS:
            if loss.min() <= THRESHOLD:
                hit_dict[THRESHOLD] += 1
                if row['cache'] == intent_list[pred_result]:
                    tp_dict[THRESHOLD] += 1
                else:
                    print('%s,%s' % (row['transcription'], transcript_list[pred_result]))

if total >= 5:  # skip for users with < 5 eval samples
    for threshold in THRESHOLDS:
        print('THRESHOLD = %d' % threshold)
        hit_rate, cache_acc, total_acc= 0,0,0
        # avoid divided by zero
        if hit_dict[threshold] != 0:
            cache_acc = round(tp_dict[threshold] / hit_dict[threshold], 4)
        hit_rate = round(hit_dict[threshold] / total, 4)
        total_acc = round((tp_dict[threshold] + (total - hit_dict[threshold])) / total, 4)
        print('hit_rate %.4f' % hit_rate)
        print('cache acc %.4f' % cache_acc)
        print('total_acc %.4f' % total_acc)
