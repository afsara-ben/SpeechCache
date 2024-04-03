# %%
import data
import os
import time
import soundfile as sf
import torch
import ast
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import models
import pickle

device = "cpu"
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

pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models/FSC/models-N-speakers-FSC-multicache')

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
config.phone_rnn_num_hidden = [128, 128]
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

L1_THRESHOLD, L2_THRESHOLD = 300, 25
d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()
# variables to save #hits, #corrects
cumulative_l1_hits, cumulative_l1_corrects = 0, 0
cumulative_l2_hits, cumulative_l2_corrects = 0, 0

spk_group = [['5o9BvRGEGvhaeBwA', 'ro5AaKwypZIqNEp2', 'KqDyvgWm4Wh8ZDM7'], ['73bEEYMKLwtmVwV43', 'R3mXwwoaX9IoRVKe', 'ppzZqYxGkESMdA5Az'], ['2ojo7YRL7Gck83Z3', 'kxgXN97ALmHbaezp', 'zaEBPeMY4NUbDnZy'], ['8e5qRjN7dGuovkRY', 'xRQE5VD7rRHVdyvM', 'n5XllaB4gZFwZXkBz'], ['oRrwPDNPlAieQr8Q', 'W4XOzzNEbrtZz4dW', 'Z7BXPnplxGUjZdmBZ'], ['xEYa2wgAQof3wyEO', 'ObdQbr9wyDfbmW4E', 'EExgNZ9dvgTE3928'], ['gvKeNY2D3Rs2jRdL', 'NWAAAQQZDXC5b9Mk', 'W7LeKXje7QhZlLKe'], ['oXjpaOq4wVUezb3x', 'zZezMeg5XvcbRdg3', 'Gym5dABePPHA8mZK9'], ['Xygv5loxdZtrywr9', 'AY5e3mMgZkIyG3Ox', 'anvKyBjB5OiP5dYZ'], ['G3QxQd7qGRuXAZda', 'WYmlNV2rDkSaALOE', 'RjDBre8jzzhdr4YL'], ['9EWlVBQo9rtqRYdy', 'M4ybygBlWqImBn9oZ', 'jgxq52DoPpsR9ZRx'], ['kNnmb7MdArswxLYw', 'qNY4Qwveojc8jlm4', 'mor8vDGkaOHzLLWBp'], ['5BEzPgPKe8taG9OB', 'AvR9dePW88IynbaE', '52XVOeXMXYuaElyw'], ['2BqVo8kVB2Skwgyb', 'ZebMRl5Z7dhrPKRD', 'R3mexpM2YAtdPbL7'], ['d2waAp3pEjiWgrDEY', 'DWmlpyg93YCXAXgE', '7NEaXjeLX3sg3yDB'], ['xwzgmmv5ZOiVaxXz', 'd3erpmyk9yFlVyrZ', 'BvyakyrDmQfWEABb'], ['Rq9EA8dEeZcEwada2', 'gNYvkbx3gof2Y3V9', 'g2dnA9Wpvzi2WAmZ'], ['xwpvGaaWl5c3G5N3', 'DMApMRmGq5hjkyvX']]
cumulative_sample, cumulative_l2_sample, cumulative_correct, cumulative_hits, cumulative_hit_correct = 0, 0, 0, 0, 0
for _, speakers in tqdm(enumerate(spk_group), total=len(spk_group)):
    print('loading model.....')
    filename = f'3-spk-FSC-multicache-{speakers}.pkl'
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

    print('eval for speakers ', speakers)

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
    for _, row in test_set.iterrows():
        if row['index'] in training_idxs:
            continue
        # # of total evaluation samples
        total += 1
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
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
            if loss[pred_intent] < L1_THRESHOLD:
                # go with l1: kmeans
                l1_hits += 1
                cumulative_l1_hits += 1
                if row['cache'] == intent_list[pred_intent]:
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
                    print('nan eval on speaker: %s' % speakers)
                if loss.min() <= L2_THRESHOLD:
                    # print('l2 hit: ', row['transcription'])
                    l2_hits += 1
                    cumulative_l2_hits += 1
                    if row['cache'] == intent_list[pred_result]:
                        l2_correct += 1
                        cumulative_l2_corrects += 1
                #     else:
                #         print('%s,%s' % (row['transcription'], transcript_list[pred_result]))
                # else:
                #     # do the calculation
                #     # cloud_model.predict_intents(x)
                #     print('cloud')

    l1_hit_rate, l1_cache_acc, l2_hit_rate, l2_cache_acc, total_acc = 0, 0, 0, 0, 0
    if total >= 5:  # skip for users with < 5 eval samples
        cumulative_sample += total
        cumulative_l2_sample += l2_total
        cumulative_hits += l1_hits + l2_hits
        cumulative_hit_correct += l1_correct + l2_correct
        print(speakers)
        if l1_hits:
            l1_hit_rate = round(l1_hits / total, 4)
            l1_cache_acc = round(l1_correct / l1_hits, 4)
            print('l1_hit_rate ', l1_hit_rate)
            print('l1_cache_acc ', l1_cache_acc)
        else:
            print('no hits in l1')
        if l2_hits:
            l2_hit_rate = round(l2_hits / l2_total, 4)
            l2_cache_acc = round(l2_correct / l2_hits, 4)
            print('l2_hit_rate ', l2_hit_rate)
            print('l2_cache_acc ', l2_cache_acc)
        else:
            print('no hits in l2')
        total_acc = (l1_correct + l2_correct + (total - l1_hits - l2_hits) * 0.988) / total
        print('total_acc %.4f' % total_acc)

print('total >= 5')
print('threshold L1: %d L2: %d ' % (L1_THRESHOLD, L2_THRESHOLD))
print('cumulative l1-hit-rate: %.4f' % (cumulative_l1_hits / cumulative_sample))
print('cumulative l2-hit-rate: %.4f' % (cumulative_l2_hits / cumulative_l2_sample))
if cumulative_l1_hits:
    print('cumulative l1-hit-acc: %.4f' % (cumulative_l1_corrects / cumulative_l1_hits))
else:
    print('cumulative l1-hit-acc = 0')
if cumulative_l2_hits:
    print('cumulative l2-hit-acc: %.4f' % (cumulative_l2_corrects / cumulative_l2_hits))
else:
    print('cumulative l2-hit-acc = 0')

print('cumulative hit_rate: %.4f' % (cumulative_hits / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hits))

print('cumulative acc: %.4f' % (((cumulative_sample - cumulative_l1_hits - cumulative_l2_hits) * 0.988 + cumulative_l1_corrects + cumulative_l2_corrects) / cumulative_sample))
