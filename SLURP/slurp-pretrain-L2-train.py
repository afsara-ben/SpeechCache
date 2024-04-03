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
from utils import audio_augment
import models
import pandas as pd
import pickle

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = 'cpu'

with open('phoneme_list.txt', 'r') as file:
    id2phoneme = ast.literal_eval(file.read())
phoneme2id = {v: k for k, v in id2phoneme.items()}


pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models/SLURP/slurp-pretrain-phoneme-ctc')
wav_path = os.path.join(pwd, 'SLURP/slurp_real')

d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

df = pd.read_csv(os.path.join(pwd, 'slurp_mini_FE_MO_ME_FO_UNK.csv'))
df = deepcopy(df)
num_nan_train, nan_nan_eval = 0, 0
# speakers = np.unique(new_train_df['speakerId'])
speakers = np.unique(df['user_id'])
cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct = 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('training for speaker %s' % user_id)
    tmp = df[df['user_id'] == user_id]
    transcripts = np.unique(tmp['sentence'])
    training_idxs = set()
    # the following three are used for evaluation
    transcript_list = []
    phoneme_list = []
    intent_list = []
    file_path = os.path.join(pwd, 'models/SLURP/slurp_pretrained.pkl')
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)
    model = load_data['model']
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # training
    for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
        optim.zero_grad()
        # remove ending punctuation from the transcript
        phoneme_seq = reduce(lambda x, y: x + ['sp'] + y,
                             [d[tk][0] if tk in d else [] for tk in tknz.tokenize(transcript.lower())])
        transcript_list.append(transcript)
        # create (N, T) label
        phoneme_label = torch.tensor(
            [phoneme2id[ph[:-1]] if ph[-1].isdigit() else phoneme2id[ph] for ph in phoneme_seq],
            dtype=torch.long, device=device)
        phoneme_list.append(phoneme_label)
        # print('phoneme_list ', phoneme_list)
        # random choose one file with `transcription`
        rows = tmp[tmp['sentence'] == transcript]
        # sample only one audio file for each distinct transcription
        row = rows.iloc[np.random.randint(len(rows))]
        intent_list.append(row['intent'])
        # add the index to training set, won't use in eval below
        training_idxs.add(row[0])

        # load the audio file
        wav = os.path.join(wav_path, row['recording_path'])
        x, _ = sf.read(wav)
        x_aug = torch.tensor(audio_augment(x), dtype=torch.float, device=device)
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
        'speakerId': user_id,
        'transcript_list': transcript_list,
        'phoneme_list': phoneme_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
    }

    filename = f'slurp-pretrain-L2-{user_id}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)

