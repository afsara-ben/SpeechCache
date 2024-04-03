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
from utils import audio_augment
import models
import pandas as pd
import pickle

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('phoneme_list.txt', 'r') as file:
    id2phoneme = ast.literal_eval(file.read())
phoneme2id = {v: k for k, v in id2phoneme.items()}


pwd = os.getcwd()
wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
folder_path = os.path.join(pwd, 'models/SLURP/')

d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_10_pc.csv'))
df = deepcopy(df)
df = df.sort_values(by='sentence', key=lambda x: x.str.len())
num_nan_train, nan_nan_eval = 0, 0

model = models.Model(config)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

count = 0
for _, row in df.iterrows():
    count += 1
    optim.zero_grad()
    transcript = row['sentence']
    print(count, transcript)
    # remove ending punctuation from the transcript
    phoneme_seq = reduce(lambda x, y: x + ['sp'] + y,
                         [d[tk][0] if tk in d else [] for tk in tknz.tokenize(transcript.lower())])
    # create (N, T) label
    phoneme_label = torch.tensor(
        [phoneme2id[ph[:-1]] if ph[-1].isdigit() else phoneme2id[ph] for ph in phoneme_seq],
        dtype=torch.long, device=device)

    # load the audio file
    wav = os.path.join(wav_path, row['recording_path'])
    x, _ = sf.read(wav)
    # x_aug = torch.tensor(x, dtype=torch.float, device=device)
    x_aug = torch.tensor(audio_augment(x), dtype=torch.float, device=device)
    phoneme_label = phoneme_label.repeat(x_aug.shape[0], 1)
    phoneme_pred = model.pretrained_model.compute_phonemes(x_aug)
    pred_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_pred.shape[0], dtype=torch.long)
    label_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_label.shape[-1], dtype=torch.long)

    if len(label_lengths) > len(pred_lengths):
        continue
    loss = ctc_loss(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
    print(loss.item())
    if torch.isinf(loss).any():
        model.requires_grad_(True)
        loss = torch.tensor(0.0000001, requires_grad=True)
    # FIXME implement better fix for nan loss
    if torch.isnan(loss).any() or torch.isinf(loss).any:
        num_nan_train = num_nan_train + 1
        optim.zero_grad()
    loss.backward()
    optim.step()
if num_nan_train:
    print('nan in train happens %d times' % num_nan_train)

saved_data = {
    'model': model,
}

torch.save(model.state_dict(), 'slurp-pretrained.pth')

filename = f'slurp_pretrained.pkl'
file_path = os.path.join(folder_path, filename)
with open(file_path, 'wb') as f:
    pickle.dump(saved_data, f)
