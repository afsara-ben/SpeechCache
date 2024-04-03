import ast

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
from tqdm import tqdm
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils import audio_augment, create_csv_file, write_to_csv, get_audio_duration
import models
import pandas as pd
import pickle
import sys

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = 'cpu'

pwd = os.getcwd()
wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
folder_path = os.path.join(pwd, 'slurp_models_phoneme_ctc')
slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_UNK.csv')

results_file = 'slurp_phoneme_ctc_dynamic_threshold.csv'
if not os.path.exists(results_file):
    headers = ['SpeakerId', 'threshold', 'train', 'test', 'hit_rate', 'cache_acc', 'total_acc']
    create_csv_file(results_file, headers)

zero = 0.000000000000001
def test(model, df, transcript_list, phoneme_list, intent_list, training_idxs):
    hit1,hit2,hit3, tp1, tp2, tp3, total, total1, total2, total3 = zero, zero, zero, 0, 0, 0, zero, zero, zero, zero,
    # prepare all the potential phoneme sequences
    label_lengths = torch.tensor(list(map(len, phoneme_list)), dtype=torch.long)
    phoneme_label = pad_sequence(phoneme_list, batch_first=True).to(device)
    # no reduction, loss on every sequence
    ctc_loss_eval = torch.nn.CTCLoss(reduction='none')
    total_time_spk = 0
    for _, row in df.iterrows():
        if row[0] in training_idxs:
            continue
        total += 1
        wav = os.path.join(wav_path, row['recording_path'])
        x, _ = sf.read(wav)
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

            if 0 < (get_audio_duration(wav)) <= 2.7:
                total1 += 1
                THRESHOLD = THRESHOLDS[1]
                if loss.min() <= THRESHOLD:
                    hit1 += 1
                    if row['intent'] == intent_list[pred_result]:
                        tp1 += 1
            elif 2.7 < (get_audio_duration(wav)) <= 4:
                total2 += 1
                THRESHOLD = THRESHOLDS[2]
                if loss.min() <= THRESHOLD:
                    hit2 += 1
                    if row['intent'] == intent_list[pred_result]:
                        tp2 += 1
            else:
                THRESHOLD = THRESHOLDS[3]
                total3 += 1
                if loss.min() <= THRESHOLD:
                    hit3 += 1
                    if row['intent'] == intent_list[pred_result]:
                        tp3 += 1
            #     else:
            #         print('%s,%s' % (row['sentence'], transcript_list[pred_result]))
            #     total_time_spk += (time.time() - tick)
            # else:
            #     # do the calculation
            #     tick = time.time()
            #     print('cloud. loss was %f ' % loss.min())
            #     # cloud_model.predict_intents(x)
            #     total_time_spk += (time.time() - tick)

    print('hit rate %.4f %.4f %.4f' % (hit1/total1, hit2/total2, hit3/total3))
    print('cache acc %.4f %.4f %.4f' % (tp1/hit1, tp2/hit2, tp3/hit3))
    hits = hit1 + hit2 + hit3
    tp = tp1 + tp2 + tp3
    return total, hits, tp


slurp_df = deepcopy(slurp_df)
arguments = sys.argv

THRESHOLDS = []
for arg in arguments[1:]:
    THRESHOLDS.append(int(arg))

num_nan_train, nan_nan_eval = 0, 0
speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232', 'ME-144']
cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, cumulative_cache_miss,cumulative_hit_incorrect, total_train = 0, 0, 0, 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('EVAL FOR SPEAKER ', user_id)
    df = slurp_df[slurp_df['user_id'] == user_id]
    filename = f'slurp_model_phoneme_ctc_{user_id}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)
    model = load_data['model']
    transcript_list = load_data['transcript_list']
    phoneme_list = load_data['phoneme_list']
    intent_list = load_data['intent_list']
    training_idxs = load_data['training_idxs']
    total, hits, tp = test(model, df, transcript_list, phoneme_list, intent_list, training_idxs)

    hit_rate, cache_acc, total_acc = 0, 0, 0
    if total >= 5:  # skip for users with < 5 eval samples
        print('\n\nEVAL FOR SPEAKER ', user_id)
        cumulative_sample += total
        cumulative_correct += tp + (total - hits)*0.85
        cumulative_hit += hits
        cumulative_hit_correct += tp
        total_train += len(training_idxs)
        hit_rate = round((hits / total), 4)
        total_acc = round((tp + (total - hits)*0.85) / total, 4)
        print('Train: %d Test: %d' % (len(training_idxs), total))
        print('hit_rate %.3f ' % hit_rate)
        if hits:
            cache_acc = round((tp/hits), 4)
            print('cache_acc %.3f' % cache_acc)
        print('total_acc: %.3f' % total_acc)
    else:
        print('not enough samples: %s' % user_id)
    values = [user_id, THRESHOLDS, len(training_idxs), total, hit_rate, cache_acc, total_acc]
    write_to_csv(results_file, values)

print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('Threshold ', THRESHOLDS)
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()