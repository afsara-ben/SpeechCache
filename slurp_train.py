import ast
import data
import os
import soundfile as sf
import torch
import numpy as np
from functools import reduce
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
from copy import deepcopy
from tqdm import tqdm
from utils import audio_augment, get_audio_duration
import models
import pandas as pd
import pickle
from kmeans_pytorch import kmeans
import warnings
import gzip
import argparse

# Filter and suppress all warnings
warnings.filterwarnings("ignore")

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config) #this line is needed otherwise to correctly get config
print('config load ok')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', dest='model_dir', type=str, help='Add the path where you want to save your trained model')
parser.add_argument('--wav_path', dest='wav_path', type=str, help='Add the path to your wav files')
args = parser.parse_args()

pwd = os.getcwd()
# wav_path = os.path.join(pwd, 'SLURP/slurp-wav/')
wav_path = args.wav_path
# model_dir = os.path.join(pwd, 'SLURP/models/slurp-C')
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# hyperparameters
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

    print('train %d test %d' % (len(training_idxs), len(df) - len(training_idxs)))

    del model.pretrained_model.word_layers
    del model.pretrained_model.word_linear
    del model.intent_layers

    filename = f'slurp_C_{user_id}_audio_bucket_{bucket}'
    user_model_dir = os.path.join(model_dir, user_id)
    if not os.path.exists(user_model_dir):
        os.makedirs(user_model_dir)
    new_file_path = os.path.join(user_model_dir, filename + '.pth')
    torch.save(model, new_file_path)

    metadata = {
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

    with gzip.open(os.path.join(user_model_dir, filename + '.pkl.gz'), 'wb') as f:
        pickle.dump(metadata, f)


slurp_df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_headset.csv'))
slurp_df = deepcopy(slurp_df)

speakers = np.unique(slurp_df['user_id'])
# speakers = ['MO-433', 'UNK-326', 'FO-232']
speakers = ['FE-146']
cumulative_sample, cumulative_correct, cumulative_hit, cumulative_hit_correct = 0, 0, 0, 0
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    print('training for speaker %s' % user_id)
    df = slurp_df[slurp_df['user_id'] == user_id]

    df1 = pd.DataFrame(columns=slurp_df.columns)
    df2 = pd.DataFrame(columns=slurp_df.columns)
    df3 = pd.DataFrame(columns=slurp_df.columns)
    for _, row in df.iterrows():
        wav = wav_path + row['recording_path']
        if 0 <= (get_audio_duration(wav)) <= 2.7:
            df1 = df1._append(row)
        elif 2.7 < (get_audio_duration(wav)) <= 4:
            df2 = df2._append(row)
        else:
            df3 = df3._append(row)
    print('%d' % (len(df1) + len(df2) + len(df3)))

    # for indomain training uncomment the following
    # pretrained_file = "slurp-pretrained.pth"
    # pretrained_path = os.path.join(pwd + "/models/SLURP/", pretrained_file)
    # for using the base pretrained model
    pretrained_path = "experiments/no_unfreezing/training/model_state.pth"

    model1 = models.Model(config).eval()
    optim1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    model1.load_state_dict(
        torch.load(pretrained_path, map_location=device))  # load trained model
    ctc_loss1 = torch.nn.CTCLoss()
    train(model1, optim1, df1, ctc_loss1, bucket=1)

    model2 = models.Model(config).eval()
    optim2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    model2.load_state_dict(
        torch.load(pretrained_path, map_location=device))  # load trained model
    ctc_loss2 = torch.nn.CTCLoss()
    train(model2, optim2, df2, ctc_loss2, bucket=2)

    model3 = models.Model(config).eval()
    optim3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
    model3.load_state_dict(
        torch.load(pretrained_path, map_location=device))  # load trained model
    ctc_loss3 = torch.nn.CTCLoss()
    train(model3, optim3, df3, ctc_loss3, bucket=3)
