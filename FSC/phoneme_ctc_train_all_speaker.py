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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# %%
device = torch.device('cpu')


# create model
def create_model(config, device, path="experiments/no_unfreezing/training/model_state.pth"):
    model = models.SpeechCacheModel(config).to(device)
    saved_params = torch.load(path, map_location=device)
    # for k, v in saved_params.items():
    #     print(k)
    # a workaround to load pretrained weight to speechcache model
    for name, param in model.named_parameters():
        tmp = 'pretrained_model.' + name
        if tmp in saved_params:
            param.weight = saved_params[tmp]
            param.requires_grad = False
    return model


model = create_model(config, device)

# process data
new_train_df = deepcopy(train_dataset.df)
act_obj_loc_idxs = []
for _, row in new_train_df.iterrows():
    act, obj, loc = row['action'], row['object'], row['location']
    act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
new_train_df['cache'] = act_obj_loc_idxs
new_train_df.head()
# new_train_df.to_csv('train_data_cache.csv', index=None)

id2phoneme = {
    0: 'sil',
    1: 'S',
    2: 'IH',
    3: 'N',
    4: 'Y',
    5: 'UW',
    6: 'AA',
    7: 'R',
    8: 'AH',
    9: 'F',
    10: 'EH',
    11: 'D',
    12: 'V',
    13: 'M',
    14: 'AY',
    15: 'K',
    16: 'Z',
    17: 'HH',
    18: 'P',
    19: 'IY',
    20: 'B',
    21: 'sp',
    22: 'SH',
    23: 'UH',
    24: 'AE',
    25: 'ER',
    26: 'T',
    27: 'OW',
    28: 'DH',
    29: 'CH',
    30: 'L',
    31: 'EY',
    32: 'JH',
    33: 'AO',
    34: 'W',
    35: 'G',
    36: 'AW',
    37: 'TH',
    38: 'NG',
    39: 'OY',
    40: 'ZH',
    41: 'spn',
}
phoneme2id = {v: k for k, v in id2phoneme.items()}

# emulate an oracle cloud model
cloud_model = models.Model(config).eval()
cloud_model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

# local, personalized cache model
# threshold for ctc_loss, if less than it, use cache
# else, use the full model
d = cmudict.dict()
tknz = NLTKWordTokenizer()
ctc_loss = torch.nn.CTCLoss()

num_nan_train, nan_nan_eval = 0, 0
# speakers = ['2ojo7YRL7Gck83Z3', '8e5qRjN7dGuovkRY', '9EWlVBQo9rtqRYdy']
# speakers = ['2ojo7YRL7Gck83Z3'] #248 train
# speakers = ['KqDyvgWm4Wh8ZDM7'] #133 train
speakers = np.unique(new_train_df['speakerId'])
eval_only = False
model = models.Model(config)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
# the following three are used for evaluation
transcript_list = []
phoneme_list = []
intent_list = []
training_idxs = set()
total_train, total_test = 0,0

for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    print('training for speaker %s' % speakerId)
    tmp = new_train_df[new_train_df['speakerId'] == speakerId]
    transcripts = np.unique(tmp['transcription'])

    # train for train_set<100
    if len(transcripts) > 100:
        continue

    total_train += len(transcripts)
    total_test += len(tmp['transcription']) - len(transcripts)
    print('train: ', len(transcripts) )
    print('test: ', len(tmp['transcription']) - len(transcripts))

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
        # random choose one file with `transcription`
        rows = tmp[tmp['transcription'] == transcript]
        # sample only one audio file for each distinct transcription
        row = rows.iloc[np.random.randint(len(rows))]
        intent_list.append(row['cache'])
        # add the index to training set, won't use in eval below
        training_idxs.add(row['index'])
        if eval_only:
            continue
        # load the audio file
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
        x_aug = torch.tensor(audio_augment(x), dtype=torch.float, device=device)
        phoneme_label = phoneme_label.repeat(x_aug.shape[0], 1)
        phoneme_pred = model.pretrained_model.compute_phonemes(x_aug)
        pred_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_pred.shape[0], dtype=torch.long)
        label_lengths = torch.full(size=(x_aug.shape[0],), fill_value=phoneme_label.shape[-1], dtype=torch.long)

        loss = ctc_loss(phoneme_pred, phoneme_label, pred_lengths, label_lengths)
        # FIXME implement better fix for nan loss
        if torch.isnan(loss).any():
            num_nan_train = num_nan_train + 1
            print('nan training on speaker: %s' % speakerId)
            optim.zero_grad()
        loss.backward()
        optim.step()
    print('train loss: ', loss.min())
    if num_nan_train:
        print('nan in train happens %d times' % num_nan_train)


saved_data = {
    'model': model,
    'transcript_list': transcript_list,
    'phoneme_list': phoneme_list,
    'intent_list': intent_list,
    'training_idxs': training_idxs,
}

filename = r'models/model_data_all_speakers'
with open(filename, 'wb') as f:
    pickle.dump(saved_data, f)

print('train:test', total_train, total_test)
print('here')
