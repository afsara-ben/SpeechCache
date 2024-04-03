# %%
from charset_normalizer import models

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
from utils import audio_augment
import models
import pickle
import ast

device = torch.device('cpu')
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

# process data
new_train_df = deepcopy(train_dataset.df)
act_obj_loc_idxs = []
for _, row in new_train_df.iterrows():
    act, obj, loc = row['action'], row['object'], row['location']
    act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
new_train_df['cache'] = act_obj_loc_idxs
new_train_df.head()
# new_train_df.to_csv('train_data_cache.csv', index=None)

with open('phoneme_list.txt', 'r') as file:
    id2phoneme = ast.literal_eval(file.read())
phoneme2id = {v: k for k, v in id2phoneme.items()}

# emulate an oracle cloud model
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

# local, personalized cache model
# threshold for ctc_loss, if less than it, use cache
# else, use the full model
THRESHOLD = 25
d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

unseen = True
k_cached = False
pwd = os.getcwd()
if unseen:
    folder_path = os.path.join(pwd, 'models/FSC/models-FSC-phoneme-ctc-unseen')
else:
    folder_path = os.path.join(pwd, 'models/FSC/models-FSC-phoneme-ctc-k%-cached')

num_nan_train, nan_nan_eval = 0, 0
speakers = np.unique(new_train_df['speakerId'])
speakers_to_remove = ['4aGjX3AG5xcxeL7a', '5pa4DVyvN2fXpepb', '9Gmnwa5W9PIwaoKq', 'KLa5k73rZvSlv82X', 'LR5vdbQgp3tlMBzB', 'OepoQ9jWQztn5ZqL', 'X4vEl3glp9urv4GN', 'Ze7YenyZvxiB4MYZ', 'eL2w4ZBD7liA85wm', 'eLQ3mNg27GHLkDej', 'ldrknAmwYPcWzp4N', 'mzgVQ4Z5WvHqgNmY', 'nO2pPlZzv3IvOQoP2', 'oNOZxyvRe3Ikx3La', 'roOVZm7kYzS5d4q3', 'rwqzgZjbPaf5dmbL', 'wa3mwLV3ldIqnGnV', 'xPZw23VxroC3N34k', 'ywE435j4gVizvw3R', 'zwKdl7Z2VRudGj2L', '35v28XaVEns4WXOv', 'YbmvamEWQ8faDPx2', 'neaPN7GbBEUex8rV', '9mYN2zmq7aTw4Blo']
speakers = [item for item in speakers if item not in speakers_to_remove]

cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, total_train = 0, 0, 0, 0, 0
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    print('EVAL FOR SPEAKER ', speakerId)
    if unseen:
        print('unseen')
        filename = f'FSC-phoneme-ctc-unseen-{speakerId}.pkl'
    else:
        filename = f'FSC-phoneme-ctc-70%-cached-{speakerId}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)
    model = load_data['model']
    speakerId = load_data['speakerId']
    train_set = load_data['train_set']
    test_set = load_data['test_set']
    transcript_list = load_data['transcript_list']
    phoneme_list = load_data['phoneme_list']
    intent_list = load_data['intent_list']
    training_idxs = load_data['training_idxs']

    # prepare all the potential phoneme sequences
    label_lengths = torch.tensor(list(map(len, phoneme_list)), dtype=torch.long)
    phoneme_label = pad_sequence(phoneme_list, batch_first=True).to(device)
    # no reduction, loss on every sequence
    ctc_loss_eval = torch.nn.CTCLoss(reduction='none')
    tp, total, hits = 0, 0, 0
    total_time_spk = 0
    for _, row in test_set.iterrows():
        if row['index'] in training_idxs:
            continue
        total += 1
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
            if loss.min() <= THRESHOLD:
                tick = time.time()
                hits += 1
                if row['cache'] == intent_list[pred_result]:
                    tp += 1
                else:
                    print('%s,%s' % (row['transcription'], transcript_list[pred_result]))

            #     total_time_spk += (time.time() - tick)
            # else:
            #     # do the calculation
            #     tick = time.time()
            #     cloud_model.predict_intents(x)
            #     total_time_spk += (time.time() - tick)

    hit_rate, cache_acc, total_acc = 0, 0, 0
    if total >= 5:  # skip for users with < 5 eval samples
        print('\n\nEVAL FOR SPEAKER ', speakerId)
        cumulative_sample += total
        cumulative_correct += tp + (total - hits) * 0.988
        cumulative_hit += hits
        cumulative_hit_correct += tp
        total_train += len(training_idxs)
        hit_rate = round((hits / total), 4)
        total_acc = round((tp + (total - hits) * 0.988) / total, 4)
        print('Train: %d Test: %d' % (len(training_idxs), total))
        print('hit_rate %.3f ' % hit_rate)
        if hits:
            cache_acc = round((tp / hits), 4)
            print('cache_acc %.3f' % cache_acc)
        print('total_acc: %.3f' % total_acc)
    else:
        print('not enough samples: %s' % speakerId)
    values = [speakerId, THRESHOLD, len(training_idxs), total, hit_rate, cache_acc, total_acc]
    print(values)

print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('Threshold ', THRESHOLD)
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()
