import ast
import gzip

import data
import os
import time
import random
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pwd = os.getcwd()
wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
folder_path = os.path.join(pwd, 'models/SLURP/models-N-speakers-SLURP-multicache-audio-bucket')
# folder_path = os.path.join(pwd, 'models/SLURP/curated-slurp-headset-base-audio-bucket-N-speakers')

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4

with open('phoneme_list.txt', 'r') as file:
    id2phoneme = ast.literal_eval(file.read())
phoneme2id = {v: k for k, v in id2phoneme.items()}

d = cmudict.dict()
tknz = NLTKWordTokenizer()

spk_group = [['UNK-195', 'MO-371', 'UNK-166'], ['ME-132', 'ME-352', 'MO-051'], ['FO-152', 'MO-355', 'ME-354'],
             ['MO-222', 'FO-230', 'MO-465'], ['UNK-340', 'UNK-432', 'UNK-187'], ['ME-376', 'MO-424', 'MO-127'],
             ['UNK-162', 'ME-140', 'UNK-160'], ['UNK-182', 'MO-422', 'FE-178'], ['FO-445', 'FO-126', 'FE-248'],
             ['UNK-177', 'MO-055', 'FO-372'], ['MO-423', 'UNK-320', 'UNK-199'], ['UNK-200', 'UNK-385', 'ME-223'],
             ['UNK-342', 'MO-026', 'ME-220'], ['FO-444', 'UNK-458', 'FO-133'], ['UNK-331', 'FE-145', 'FO-228'],
             ['MO-019', 'UNK-322', 'FO-474'], ['UNK-165', 'MO-185', 'ME-373'], ['FO-233', 'UNK-198', 'ME-028'],
             ['UNK-168', 'FO-419', 'MO-433'], ['MO-036', 'UNK-196', 'FO-150'], ['MO-038', 'FE-149', 'FO-179'],
             ['ME-218', 'ME-151', 'ME-345'], ['UNK-325', 'MO-116', 'FO-124'], ['FO-425', 'FO-123', 'MO-030'],
             ['UNK-197', 'UNK-242', 'FE-146'], ['ME-369', 'FO-461', 'UNK-163'], ['FO-158', 'ME-434', 'UNK-341'],
             ['MO-142', 'UNK-240', 'FO-413'], ['ME-143', 'UNK-335', 'UNK-343'], ['MO-156', 'UNK-201', 'UNK-334'],
             ['FO-493', 'UNK-326', 'FO-488'], ['MO-442', 'FO-231', 'UNK-324'], ['ME-138', 'FE-235', 'MO-374'],
             ['FO-234', 'FE-249', 'MO-431'], ['FO-129', 'FE-141', 'UNK-323'], ['FO-438', 'FO-122', 'FO-171'],
             ['FO-420', 'ME-492', 'FO-184'], ['FO-229', 'MO-190', 'ME-414'], ['UNK-386', 'FO-219', 'FO-356'],
             ['MO-446', 'UNK-336', 'UNK-329'], ['UNK-159', 'FO-125', 'MO-463'], ['FO-405', 'MO-189', 'ME-144'],
             ['FO-139', 'UNK-241', 'UNK-328'], ['MO-191', 'FO-462', 'FO-180'], ['UNK-226', 'MO-044', 'UNK-436'],
             ['ME-473', 'FO-475', 'MO-418'], ['ME-147', 'FO-350', 'UNK-167'], ['MO-494', 'MO-164', 'UNK-245'],
             ['UNK-244', 'MO-467', 'UNK-208'], ['MO-375', 'UNK-327', 'FO-232'], ['MO-029', 'MO-050', 'UNK-225'],
             ['UNK-188', 'UNK-330', 'FO-460']]

def train_test_split(df, speakers):
    train_set, test_set = pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)
    tmp = pd.DataFrame(columns=df.columns)
    for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
        tmp = tmp._append(df[df['user_id'] == speakerId])
    transcripts = np.unique(tmp['sentence'])
    for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
        rows = tmp[tmp['sentence'] == transcript]
        train_set = train_set._append(rows.iloc[0])
        if len(rows) > 1:
            test_set = test_set._append(rows.iloc[1:])

    print('train %d test %d' % (len(train_set), len(test_set)))
    print('Train and test set created!!')
    return train_set, test_set

def train(speakers, model, optim, train_df, test_df, ctc_loss, bucket):
    num_nan_train = 0
    training_idxs = set()
    # the following three are used for evaluation
    transcript_list = []
    phoneme_list = []
    intent_list = []
    cluster_ids = []
    cluster_centers = []

    # training
    print('TRAINING START')
    for _, row in train_df.iterrows():
        transcript = row['sentence']
        optim.zero_grad()
        # remove ending punctuation from the transcript
        phoneme_seq = reduce(lambda x, y: x + ['sp'] + y,
                             [d[tk][0] if tk in d else [] for tk in tknz.tokenize(transcript.lower())])
        transcript_list.append(transcript)
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
            print('nan training on speaker: %s' % speakers)
            optim.zero_grad()
        loss.backward()
        optim.step()
    if num_nan_train:
        print('nan in train happens %d times' % num_nan_train)

    print('train %d test %d' % (len(training_idxs), len(test_df)))

    filename = f'3-spk-SLURP-multicache-{speakers}_audio_bucket_{bucket}'
    # filename = f'slurp_70%_multicache_{user_id}_audio_bucket_{bucket}'
    # filename = f'slurp_curated_headset_base_{speakers}_audio_bucket_{bucket}_3_spk'
    file_path = os.path.join(folder_path, filename + '.pth')
    torch.save(model.state_dict(), file_path)

    metadata = {
        'train_set': train_df,
        'test_set': test_df,
        'bucket': bucket,
        'speakers': speakers,
        'transcript_list': transcript_list,
        'phoneme_list': phoneme_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    with gzip.open(os.path.join(folder_path, filename + '.pkl.gz'), 'wb') as f:
        pickle.dump(metadata, f)


slurp_df = pd.read_csv(os.path.join(pwd, 'slurp_mini_FE_MO_ME_FO_UNK.csv'))
# slurp_df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_headset.csv'))
slurp_df = deepcopy(slurp_df)

# curated slurp headset
# spk_group = [['ME-147', 'UNK-323', 'ME-345', 'ME-151'], ['FO-158', 'UNK-335', 'FO-462', 'UNK-322'], ['ME-352', 'UNK-330', 'ME-223'], ['FE-248', 'FO-488', 'MO-164'], ['FE-149', 'FE-235', 'MO-446'], ['FO-444', 'FO-350', 'FE-249'], ['MO-375', 'MO-142', 'MO-431'], ['FO-229', 'UNK-336', 'MO-374'], ['FE-146', 'MO-156', 'MO-355'], ['FO-372', 'ME-143', 'MO-433'], ['ME-414', 'MO-465', 'FO-234'], ['FO-438', 'FO-233', 'FO-413'], ['FO-232', 'UNK-328', 'UNK-329'], ['UNK-334', 'MO-494', 'FO-152'], ['MO-463', 'FO-425', 'ME-373'], ['ME-369', 'ME-144', 'FO-419'], ['UNK-343', 'FO-445', 'UNK-331'], ['MO-422', 'FO-219', 'ME-140'], ['ME-220', 'FO-122', 'FE-141'], ['FO-460', 'FO-493', 'FE-145'], ['FO-231', 'FO-150', 'ME-473'], ['FO-179', 'FO-475', 'MO-222'], ['UNK-326', 'UNK-327', 'FO-171'], ['FO-180', 'FO-461', 'UNK-341']]

for _, speakers in tqdm(enumerate(spk_group), total=len(spk_group)):
    print('training for speakers ', speakers)
    filename = f'3-spk-SLURP-multicache-{speakers}_audio_bucket_3'
    if os.path.exists(os.path.join(folder_path, filename)):
        continue

    train_set, test_set = train_test_split(slurp_df, speakers)

    df1, df2, df3 = pd.DataFrame(columns=slurp_df.columns), pd.DataFrame(columns=slurp_df.columns), pd.DataFrame(
        columns=slurp_df.columns)
    df1_test, df2_test, df3_test = pd.DataFrame(columns=slurp_df.columns), pd.DataFrame(
        columns=slurp_df.columns), pd.DataFrame(columns=slurp_df.columns)

    for _, row in train_set.iterrows():
        wav = wav_path + row['recording_path']
        if 0 <= (get_audio_duration(wav)) <= 2.7:
            df1 = df1.append(row)
        elif 2.7 < (get_audio_duration(wav)) <= 4:
            df2 = df2.append(row)
        else:
            df3 = df3.append(row)
    print('%d' % (len(df1) + len(df2) + len(df3)))

    for _, row in test_set.iterrows():
        wav = wav_path + row['recording_path']
        if 0 <= (get_audio_duration(wav)) <= 2.7:
            df1_test = df1_test.append(row)
        elif 2.7 < (get_audio_duration(wav)) <= 4:
            df2_test = df2_test.append(row)
        else:
            df3_test = df3_test.append(row)

    model1 = models.Model(config)
    optim1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    model1.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
    ctc_loss1 = torch.nn.CTCLoss()
    train(speakers, model1, optim1, df1, df1_test, ctc_loss1, bucket=1)

    model2 = models.Model(config)
    optim2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    model2.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
    ctc_loss2 = torch.nn.CTCLoss()
    train(speakers, model2, optim2, df2, df2_test, ctc_loss2, bucket=2)

    model3 = models.Model(config)
    optim3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
    model3.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model
    ctc_loss3 = torch.nn.CTCLoss()
    train(speakers, model3, optim3, df3, df3_test, ctc_loss3, bucket=3)
