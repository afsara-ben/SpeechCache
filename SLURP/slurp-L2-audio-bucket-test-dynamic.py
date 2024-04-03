import data
import os
import time
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pickle
from utils import create_csv_file, write_to_csv, get_audio_duration
import numpy as np
import pandas as pd
from copy import deepcopy
import sys
from models import MLP

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = 'cpu'

pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models/SLURP/models-slurp_phoneme_ctc_audio_bucket')


def test(model, df, transcript_list, phoneme_list, intent_list, training_idxs):
    # prepare all the potential phoneme sequences
    label_lengths = torch.tensor(list(map(len, phoneme_list)), dtype=torch.long)
    phoneme_label = pad_sequence(phoneme_list, batch_first=True).to(device)
    # no reduction, loss on every sequence
    ctc_loss_eval = torch.nn.CTCLoss(reduction='none')
    tp, fp, total, hits = 0, 0, 0, 0
    total_time_spk = 0
    test_df = []
    for _, row in df.iterrows():
        if row[0] in training_idxs:
            continue
        else:
            test_df.append(row)
    for _, row in df.iterrows():
        if row[0] in training_idxs:
            continue
        wav_path = os.path.join(pwd, 'SLURP/slurp_real/' + row['recording_path'])
        total += 1
        # print(row['sentence'])
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
                print('nan eval on speaker: %s' % user_id)
            filename = 'utils/MLP.pkl'
            with open(filename, 'rb') as f:
                load_data = pickle.load(f)
            MLP_model = load_data['model']

            dur = torch.tensor([[get_audio_duration(wav_path)]], dtype=torch.float32)
            THRESHOLD = MLP_model(dur).item()
            # print(round(dur.item(), 4), THRESHOLD)
            if loss.min() <= THRESHOLD:
                # print('hit')
                tick = time.time()
                hits += 1
                if row['intent'] == intent_list[pred_result]:
                    # print('tp')
                    tp += 1
            #     else:
            #         print('loss %.4f %s,%s' % (loss.min(), row['sentence'], transcript_list[pred_result]))
            #     total_time_spk += (time.time() - tick)
            # else:
            #     # do the calculation
            #     # tick = time.time()
            #     print('cloud. loss was %f ' % loss.min())
            #     # cloud_model.predict_intents(x)
            #     # total_time_spk += (time.time() - tick)

    return total, hits, tp


buckets = [1, 2, 3]

slurp_df = pd.read_csv(os.path.join(pwd, 'slurp_mini_FE_MO_ME_FO_UNK.csv'))
slurp_df = deepcopy(slurp_df)
speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232', 'ME-144']
# speakers = ['MO-433']
cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, cumulative_cache_miss, cumulative_hit_incorrect, total_train = 0, 0, 0, 0, 0, 0, 0

for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('EVAL FOR SPEAKER ', user_id)
    print('count ', slurp_df[slurp_df['user_id'] == user_id].shape[0])
    bckt_sample, bckt_correct, bckt_hit, bckt_hit_correct, bckt_train = 0, 0, 0, 0, 0
    for bucket in buckets:
        filename = f'slurp_model_phoneme_ctc_{user_id}_audio_bucket_{bucket}.pkl'
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'rb') as f:
            load_data = pickle.load(f)

        model = load_data['model']
        df = load_data['df']
        print(len(df))
        transcript_list = load_data['transcript_list']
        phoneme_list = load_data['phoneme_list']
        if not phoneme_list:
            continue
        intent_list = load_data['intent_list']
        training_idxs = load_data['training_idxs']
        total, hits, tp = test(model, df, transcript_list, phoneme_list, intent_list, training_idxs)

        hit_rate, cache_acc = 0, 0
        bckt_train += len(training_idxs)
        if total >= 5:  # skip for users with < 5 eval samples
            bckt_sample += total
            print('BUCKET %d' % bucket)
            bckt_correct += tp + (total - hits) * 0.9014
            bckt_hit += hits
            bckt_hit_correct += tp
            print('Train: %d Test: %d' % (len(training_idxs), total))
            hit_rate = round((hits / total), 4)
            print('hit_rate %.4f ' % hit_rate)
            if hits:
                cache_acc = round((tp / hits), 4)
                print('hit_acc %.4f' % cache_acc)
            else:
                print('cache_acc', 0)
            print('total_acc: %.4f' % ((tp + (total - hits) * 0.9014) / total))
        else:
            print('not enough samples: %s' % user_id)
        values = [user_id, bucket, len(training_idxs), total, tp, hits, hit_rate, cache_acc]

    total_acc, hit_rate, cache_acc = 0, 0, 0
    total_train += bckt_train
    cumulative_sample += bckt_sample
    cumulative_correct += bckt_correct
    cumulative_hit += bckt_hit
    cumulative_hit_correct += bckt_hit_correct
    print('-------------- SPEAKER %s -------------', user_id)
    if bckt_sample:
        hit_rate = round((bckt_hit / bckt_sample), 4)
        print('hit_rate: %.4f' % hit_rate)
        total_acc = round((bckt_correct / bckt_sample), 4)
    if bckt_hit:
        cache_acc = round((bckt_hit_correct / bckt_hit), 4)
        print('cache_acc: %.4f' % cache_acc)
    print('total_acc: %.4f' % total_acc)
    values = [user_id, bckt_train, bckt_sample, hit_rate, cache_acc, total_acc]


print('Dynamic L2 Threshold ')
print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()
