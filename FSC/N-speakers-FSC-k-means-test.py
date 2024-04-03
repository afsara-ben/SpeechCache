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
speechcache_config = data.read_config("experiments/speechcache.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
_, _, _ = data.get_SLU_datasets(speechcache_config)  # used to set config.num_phonemes
print('config load ok')
dataset_to_use = valid_dataset
dataset_to_use.df.head()
test_dataset.df.head()


pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models/FSC/models-N-speakers-FSC-k-means')

THRESHOLD = 300

# variables to save #hits, #corrects
cumulative_l1_hits, cumulative_l1_corrects = 0, 0
cumulative_l2_hits, cumulative_l2_corrects = 0, 0

model = models.Model(config)
model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

spk_group = [['5o9BvRGEGvhaeBwA', 'ro5AaKwypZIqNEp2', 'KqDyvgWm4Wh8ZDM7'], ['73bEEYMKLwtmVwV43', 'R3mXwwoaX9IoRVKe', 'ppzZqYxGkESMdA5Az'], ['2ojo7YRL7Gck83Z3', 'kxgXN97ALmHbaezp', 'zaEBPeMY4NUbDnZy'], ['8e5qRjN7dGuovkRY', 'xRQE5VD7rRHVdyvM', 'n5XllaB4gZFwZXkBz'], ['oRrwPDNPlAieQr8Q', 'W4XOzzNEbrtZz4dW', 'Z7BXPnplxGUjZdmBZ'], ['xEYa2wgAQof3wyEO', 'ObdQbr9wyDfbmW4E', 'EExgNZ9dvgTE3928'], ['gvKeNY2D3Rs2jRdL', 'NWAAAQQZDXC5b9Mk', 'W7LeKXje7QhZlLKe'], ['oXjpaOq4wVUezb3x', 'zZezMeg5XvcbRdg3', 'Gym5dABePPHA8mZK9'], ['Xygv5loxdZtrywr9', 'AY5e3mMgZkIyG3Ox', 'anvKyBjB5OiP5dYZ'], ['G3QxQd7qGRuXAZda', 'WYmlNV2rDkSaALOE', 'RjDBre8jzzhdr4YL'], ['9EWlVBQo9rtqRYdy', 'M4ybygBlWqImBn9oZ', 'jgxq52DoPpsR9ZRx'], ['kNnmb7MdArswxLYw', 'qNY4Qwveojc8jlm4', 'mor8vDGkaOHzLLWBp'], ['5BEzPgPKe8taG9OB', 'AvR9dePW88IynbaE', '52XVOeXMXYuaElyw'], ['2BqVo8kVB2Skwgyb', 'ZebMRl5Z7dhrPKRD', 'R3mexpM2YAtdPbL7'], ['d2waAp3pEjiWgrDEY', 'DWmlpyg93YCXAXgE', '7NEaXjeLX3sg3yDB'], ['xwzgmmv5ZOiVaxXz', 'd3erpmyk9yFlVyrZ', 'BvyakyrDmQfWEABb'], ['Rq9EA8dEeZcEwada2', 'gNYvkbx3gof2Y3V9', 'g2dnA9Wpvzi2WAmZ'], ['xwpvGaaWl5c3G5N3', 'DMApMRmGq5hjkyvX']]
cumulative_sample, cumulative_l2_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct, cumulative_cache_miss,cumulative_hit_incorrect, total_train = 0, 0, 0, 0, 0, 0, 0, 0

for _, speakers in tqdm(enumerate(spk_group), total=len(spk_group)):
    print('loading model.....')
    filename = f'3-spk-FSC-k-means-{speakers}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)
    # model = load_data['model']
    speakers = load_data['speakers']
    train_set = load_data['train_set']
    test_set = load_data['test_set']
    transcript_list = load_data['transcript_list']
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
    # no reduction, loss on every sequence
    ctc_loss_k_means_eval = torch.nn.CTCLoss(reduction='none')
    # ------------------ variables to record performance --------------------
    tp, total, hits, l1_hits, l2_hits, l1_correct, l2_correct, l2_total = 0, 0, 0, 0, 0, 0, 0, 0
    for _, row in test_set.iterrows():
        if row[0] in training_idxs:
            continue
        # # of total evaluation samples
        total += 1
        wav = os.path.join(config.slu_path, row['path'])
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
            # print('loss ', loss[pred_intent])
            if loss[pred_intent] < THRESHOLD:
                # go with l1: kmeans
                # print('l1 hit: ', row['transcription'])
                hits += 1
                if row['cache'] == intent_list[pred_intent]:
                    tp += 1

    hit_rate, cache_acc, total_acc = 0, 0, 0
    if total >= 5:  # skip for users with < 5 eval samples
        cumulative_sample += total
        cumulative_correct += tp + (total - hits) * 0.988
        cumulative_hit += hits
        cumulative_hit_correct += tp
        total_train += len(training_idxs)
        hit_rate = round((hits / total), 4)
        total_acc = round((tp + (total - hits) * 0.988) / total, 4)
        print('Train: %d Test: %d' % (len(training_idxs), total))

        if hits:
            cache_acc = round((tp / hits), 4)
            print('hit rate %.4f' % hit_rate)
            print('cache_acc %.3f' % cache_acc)
        print('total_acc: %.3f' % total_acc)
    else:
        print('not enough samples: %s' % speakers)
    values = [speakers, THRESHOLD, len(training_idxs), total, hit_rate, cache_acc, total_acc]
    print(values)
    # write_to_csv(results_file, values)

print('Train: %d Test: %d' % (total_train, cumulative_sample))
print('Threshold ', THRESHOLD)
print('cumulative hit_rate: %.4f' % (cumulative_hit / cumulative_sample))
print('cumulative cache_acc: %.4f' % (cumulative_hit_correct / cumulative_hit))
print('cumulative total_acc: %.4f' % (cumulative_correct / cumulative_sample))
print()

