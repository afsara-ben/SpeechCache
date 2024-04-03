import ast
import gzip

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
import pandas as pd
import pickle
from kmeans_pytorch import kmeans

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4

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

num_nan_train, nan_nan_eval = 0, 0
slurp_df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_headset.csv'))
slurp_df = deepcopy(slurp_df)
speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232']
# speakers = ['FE-141']

cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct = 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('training for speaker %s' % user_id)
    df = slurp_df[slurp_df['user_id'] == user_id]
    transcripts = np.unique(df['sentence'])

    model = models.Model(config).eval()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device))  # load trained model

    training_idxs = set()
    # the following three are used for evaluation
    transcript_list = []
    phoneme_list = []
    intent_list = []
    cluster_ids = []
    cluster_center = None
    cluster_centers = []
    # training
    for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
        optim.zero_grad()
        # remove ending punctuation from the transcript
        phoneme_seq = reduce(lambda x, y: x + ['sp'] + y,
                             [d[tk][0] if tk in d else [] for tk in tknz.tokenize(transcript.lower())])
        transcript_list.append(transcript)
        # random choose one file with `transcription`
        rows = df[df['sentence'] == transcript]
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
        # phoneme_seq, weight = get_token_and_weight(transcript.lower())
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
            print('nan training on speaker: %s' % user_id)
            optim.zero_grad()
        loss.backward()
        optim.step()
    if num_nan_train:
        print('nan in train happens %d times' % num_nan_train)

    # filename = f'slurp_curated_headset_multicache_{user_id}'
    filename = f'slurp_curated_headset_base_multicache_{user_id}'
    file_path = os.path.join(folder_path, filename + '.pth')

    del model.pretrained_model.word_linear
    del model.pretrained_model.word_layers
    del model.intent_layers
    torch.save(model, file_path)

    metadata = {
        'df': df,
        'speakerId': user_id,
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

