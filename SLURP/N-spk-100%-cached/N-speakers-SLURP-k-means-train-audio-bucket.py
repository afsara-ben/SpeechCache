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
from utils import audio_augment
import models
import pandas as pd
import pickle
from utils import get_audio_duration
import warnings
from kmeans_pytorch import kmeans

# Filter and suppress all warnings
warnings.filterwarnings("ignore")

config = data.read_config("experiments/no_unfreezing.cfg")
train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
print('config load ok')
device = 'cpu'

with open('phoneme_list.txt', 'r') as file:
    id2phoneme = ast.literal_eval(file.read())
phoneme2id = {v: k for k, v in id2phoneme.items()}

d = cmudict.dict()
tknz = NLTKWordTokenizer()
slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_FE_MO_ME_FO_UNK.csv')
slurp_df = deepcopy(slurp_df)

wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
pwd = os.getcwd()
folder_path = os.path.join(pwd, 'models-N-speakers-SLURP-k-means-audio-bucket')

model = models.Model(config)
model.load_state_dict(
    torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device))  # load trained model

NUM_CLUSTERS = 70
dist = 'euclidean'
tol = 1e-4

def train(speakers, train_df, test_df, bucket):
    training_idxs = set()
    # the following three are used for evaluation
    transcript_list = []
    intent_list = []
    cluster_ids = []
    cluster_centers = []

    for _, row in train_df.iterrows():
        transcript = row['sentence']
        transcript_list.append(transcript)
        intent_list.append(row['intent'])
        training_idxs.add(row[0])

        # load the audio file
        wav = os.path.join(wav_path, row['recording_path'])
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

    saved_data = {
        'train_set': train_df,
        'test_set': test_df,
        'speakers': speakers,
        'bucket': bucket,
        'transcript_list': transcript_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    filename = f'3-spk-SLURP-k-means-{speakers}_audio_bucket_{bucket}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)


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

spk_group = [['FO-152', 'MO-355', 'ME-354']]
def train_test_split(df, speakers):
    train_set, test_set = pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)
    for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
        tmp = df[df['user_id'] == speakerId]
        transcripts = np.unique(tmp['sentence'])
        for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
            rows = tmp[tmp['sentence'] == transcript]
            midpoint = len(rows) // 2
            train_set = train_set.append(rows[:midpoint])
            if len(rows) > 1:
                test_set = test_set.append(rows[midpoint:])

    print('train %d test %d' % (len(train_set), len(test_set)))
    print('Train and test set created!!')
    return train_set, test_set


for _, speakers in tqdm(enumerate(spk_group), total=len(spk_group)):
    filename = f'3-spk-SLURP-k-means-{speakers}_audio_bucket_1.pkl'
    # if os.path.isfile(os.path.join(folder_path, filename)):
    #     print('continued')
    #     continue
    print('training for speakers ', speakers)

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

    train(speakers, df1, df1_test, bucket=1)
    train(speakers, df2, df2_test, bucket=2)
    train(speakers, df3, df3_test, bucket=3)
    print()
