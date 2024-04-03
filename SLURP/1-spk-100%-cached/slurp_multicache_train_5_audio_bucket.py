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
from utils import audio_augment, get_audio_duration
import models
import pandas as pd
import pickle
from kmeans_pytorch import kmeans
import warnings

# Filter and suppress all warnings
warnings.filterwarnings("ignore")

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = 'cpu'

pwd = os.getcwd()
wav_path = os.path.join(pwd, 'slurp_real/')
folder_path = os.path.join(pwd, 'models/SLURP/models-slurp_multicache_five_audio_bucket')

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4

with open('phoneme_list.txt', 'r') as file:
    id2phoneme = ast.literal_eval(file.read())
phoneme2id = {v: k for k, v in id2phoneme.items()}

d = cmudict.dict()
tknz = NLTKWordTokenizer()


def train(model, optim, df, ctc_loss, bucket):
    num_nan_train = 0
    transcripts = np.unique(df['sentence'])
    training_idxs = set()
    # the following three are used for evaluation
    transcript_list = []
    phoneme_list = []
    intent_list = []
    cluster_ids = []
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

    saved_data = {
        'model': model,
        'df': df,
        'bucket': bucket,
        'speakerId': user_id,
        'transcript_list': transcript_list,
        'phoneme_list': phoneme_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    filename = f'slurp_model_multicache_{user_id}_five_audio_bucket_{bucket}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)


slurp_df = pd.read_csv(os.path.join(pwd, 'slurp_mini_FE_MO_ME_FO_UNK.csv'))
slurp_df = deepcopy(slurp_df)

speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433']

cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct = 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    filename = f'slurp_model_multicache_{user_id}_five_audio_bucket_5.pkl'
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        continue
    print('training for speaker %s' % user_id)
    df = slurp_df[slurp_df['user_id'] == user_id]
    df1 = pd.DataFrame(columns=slurp_df.columns)
    df2 = pd.DataFrame(columns=slurp_df.columns)
    df3 = pd.DataFrame(columns=slurp_df.columns)
    df4 = pd.DataFrame(columns=slurp_df.columns)
    df5 = pd.DataFrame(columns=slurp_df.columns)
    for _, row in df.iterrows():
        wav = wav_path + row['recording_path']
        if 0 <= (get_audio_duration(wav)) <= 2.7:
            df1 = df1.append(row)
        elif 2.7 < (get_audio_duration(wav)) <= 4:
            df2 = df2.append(row)
        elif 4 < (get_audio_duration(wav)) <= 5.5:
            df3 = df3.append(row)
        elif 5.5 < (get_audio_duration(wav)) <= 7:
            df4 = df4.append(row)
        else:
            df5 = df5.append(row)
    print('%d' % (len(df1) + len(df2) + len(df3) + len(df4) + len(df5)))

    model1 = models.Model(config)
    optim1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    model1.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
    ctc_loss1 = torch.nn.CTCLoss()
    train(model1, optim1, df1, ctc_loss1, bucket=1)

    model2 = models.Model(config)
    optim2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    model2.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
    ctc_loss2 = torch.nn.CTCLoss()
    train(model2, optim2, df2, ctc_loss2, bucket=2)

    model3 = models.Model(config)
    optim3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
    model3.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
    ctc_loss3 = torch.nn.CTCLoss()
    train(model3, optim3, df3, ctc_loss3, bucket=3)

    model4 = models.Model(config)
    optim4 = torch.optim.Adam(model4.parameters(), lr=1e-3)
    model4.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
    ctc_loss4 = torch.nn.CTCLoss()
    train(model4, optim4, df4, ctc_loss4, bucket=4)

    model5 = models.Model(config)
    optim5 = torch.optim.Adam(model5.parameters(), lr=1e-3)
    model5.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
    ctc_loss5 = torch.nn.CTCLoss()
    train(model5, optim5, df5, ctc_loss5, bucket=5)
