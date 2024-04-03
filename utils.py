import os
import data
import numpy as np
import librosa
import random
import soundfile as sf
import torch
import io
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import cmudict
from nltk.tokenize import NLTKWordTokenizer
import torchaudio
import csv
import pandas as pd
from tqdm import tqdm
import pickle
from torch.nn.utils.rnn import pad_sequence
import models
import ast
from copy import deepcopy
from pydub import AudioSegment
import sys
import configparser
from subprocess import call
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

nltk_tknz = NLTKWordTokenizer()
cmud = cmudict.dict()

# class Config:
# 	def __init__(self):
# 		self.use_sincnet = True

# def read_config(config_file):
#     config = Config()
#     parser = configparser.ConfigParser()
#     parser.read(config_file)
#
#     # [experiment]
#     config.seed = 1234
#     config.folder = '/Users/afsarabenazir/Downloads/speech_projects/speechcache/experiments/no_unfreezing'
#     # config.seed = int(parser.get("experiment", "seed"))
#     # config.folder = parser.get("experiment", "folder")
#
#     # Make a folder containing experiment information
#     if not os.path.isdir(config.folder):
#         os.mkdir(config.folder)
#         os.mkdir(os.path.join(config.folder, "pretraining"))
#         os.mkdir(os.path.join(config.folder, "training"))
#     call("cp " + config_file + " " + os.path.join(config.folder, "experiment.cfg"), shell=True)
#
#     # [phoneme_module]
#     config.use_sincnet = True
#     config.fs = 16000
#
#     config.cnn_N_filt = [80, 60, 60]
#     config.cnn_len_filt = [401, 5, 5]
#     config.cnn_stride = [80, 1, 1]
#     config.cnn_max_pool_len = [2, 1, 1]
#     config.cnn_act = ['leaky_relu', 'leaky_relu', 'leaky_relu']
#     config.cnn_drop = [0.0, 0.0, 0.0]
#
#     config.phone_rnn_num_hidden = [128, 128]
#     config.phone_downsample_len = [2, 2]
#     config.phone_downsample_type = ['avg', 'avg']
#     config.phone_rnn_drop = [0.5, 0.5]
#     config.phone_rnn_bidirectional = True
#
#     # [word_module]
#     config.word_rnn_num_hidden = [128, 128]
#     config.word_downsample_len = [2, 2]
#     config.word_downsample_type = ['avg', 'avg']
#     config.word_rnn_drop = [0.5, 0.5]
#     config.word_rnn_bidirectional = True
#     config.vocabulary_size = 10000
#
#     # [intent_module]
#     config.intent_rnn_num_hidden = [128]
#     config.intent_downsample_len = [1]
#     config.intent_downsample_type = ['none']
#     config.intent_rnn_drop = [0.5]
#     config.intent_rnn_bidirectional = True
#     try:
#         config.intent_encoder_dim = int(parser.get("intent_module", "intent_encoder_dim"))
#         print("xzl:intent_encoder_dim")
#         config.num_intent_encoder_layers = int(parser.get("intent_module", "num_intent_encoder_layers"))
#         print("xzl:num_intent_encoder_layers")
#         config.intent_decoder_dim = int(parser.get("intent_module", "intent_decoder_dim"))
#         print("xzl:intent_decoder_dim")
#         config.num_intent_decoder_layers = int(parser.get("intent_module", "num_intent_decoder_layers"))
#         print("xzl:num_intent_decoder_layers")
#         config.intent_decoder_key_dim = int(parser.get("intent_module", "intent_decoder_key_dim"))
#         print("xzl:intent_decoder_key_dim")
#         config.intent_decoder_value_dim = int(parser.get("intent_module", "intent_decoder_value_dim"))
#         print("xzl:intent_decoder_value_dim")
#     except:
#         print("no seq2seq hyperparameters")  # xzl: if none, so what??
#
#
#     config.asr_path = '/scratch/lugosch/librispeech'
#     config.pretraining_type = 2  # 0 - no pre-training, 1 - phoneme pre-training, 2 - phoneme + word pre-training, 3 - word pre-training
#     if config.pretraining_type == 0: config.starting_unfreezing_index = 1 + len(config.word_rnn_num_hidden) + len(
#         config.phone_rnn_num_hidden) + len(config.cnn_N_filt)
#     if config.pretraining_type == 1: config.starting_unfreezing_index = 1 + len(config.word_rnn_num_hidden)
#     if config.pretraining_type == 2: config.starting_unfreezing_index = 1
#     if config.pretraining_type == 3: config.starting_unfreezing_index = 1
#     config.pretraining_lr = 0.001
#     config.pretraining_batch_size = 64
#     config.pretraining_num_epochs = 10
#     config.pretraining_length_mean = 2.25
#     config.pretraining_length_var = 1
#
#     # [training]
#     config.slu_path = 'fluent_speech_commands_dataset'
#     config.unfreezing_type = 0
#     config.training_lr = 0.001
#     config.training_batch_size = 64
#     config.training_num_epochs = 20
#     config.real_dataset_subset_percentage = 1.0
#     config.synthetic_dataset_subset_percentage = 1.0
#     config.real_speaker_subset_percentage = 1.0
#     config.synthetic_speaker_subset_percentage = 0.0
#     config.train_wording_path = 'None'
#     if config.train_wording_path == "None": config.train_wording_path = None
#     config.test_wording_path = 'None'
#     if config.test_wording_path == "None": config.test_wording_path = None
#     try:
#         config.augment = (parser.get("training", "augment") == "True")
#     except:
#         # old config file with no augmentation
#         config.augment = False
#
#     # print("xzl: ", parser.get("training", "seq2seq"))
#
#     try:
#         config.seq2seq = (parser.get("training", "seq2seq") == "True")
#     except:
#         # old config file with no seq2seq
#         config.seq2seq = False
#
#     try:
#         config.dataset_upsample_factor = int(parser.get("training", "dataset_upsample_factor"))
#     except:
#         # old config file
#         config.dataset_upsample_factor = 1
#
#     # [speechcache]
#     config.num_intent = 31
#     # compute downsample factor (divide T by this number)
#     # xzl input raw is reduced by this factor??
#     config.phone_downsample_factor = 1
#     for factor in config.cnn_stride + config.cnn_max_pool_len + config.phone_downsample_len:  # xzl concatenate. not addition
#         config.phone_downsample_factor *= factor
#     print("xzl phone_downsample_factor=", config.phone_downsample_factor)
#     print("xzl config.cnn_stride + config.cnn_max_pool_len + config.phone_downsample_len",
#           config.cnn_stride + config.cnn_max_pool_len + config.phone_downsample_len)
#
#     config.word_downsample_factor = 1
#     for factor in config.cnn_stride + config.cnn_max_pool_len + config.phone_downsample_len + config.word_downsample_len:
#         config.word_downsample_factor *= factor
#
#     return config
#
#
# # xzl: also invoked for inference, populating sy_intent etc
# def get_SLU_datasets(config):
#     """
#     config: Config object (contains info about model and training)
#     """
#     print('here')
#     base_path = config.slu_path
#
#     # Split				xzl: why use different datasets for seq2seq or not??
#     if not config.seq2seq:
#         synthetic_train_df = pd.read_csv(os.path.join(base_path, "data", "synthetic_data.csv"))
#         real_train_df = pd.read_csv(os.path.join(base_path, "data", "train_data.csv"))
#         if "\"Unnamed: 0\"" in list(real_train_df): real_train_df = real_train_df.drop(columns="Unnamed: 0")
#     else:
#         synthetic_train_df = pd.read_csv(os.path.join(base_path, "data", "synthetic_data_seq2seq.csv"))
#         real_train_df = pd.read_csv(os.path.join(base_path, "data", "train_data_seq2seq.csv"))
#         if "\"Unnamed: 0\"" in list(real_train_df): real_train_df = real_train_df.drop(columns="Unnamed: 0")
#
#     # Select random subset of speakers
#     # First, check if "speakerId" is in the df columns
#     if "speakerId" in list(real_train_df) and "speakerId" in list(synthetic_train_df):
#         if config.real_speaker_subset_percentage < 1:
#             speakers = np.array(list(Counter(real_train_df.speakerId)))
#             np.random.shuffle(speakers)
#             selected_speaker_count = round(config.real_speaker_subset_percentage * len(speakers))
#             selected_speakers = speakers[:selected_speaker_count]
#             real_train_df = real_train_df[real_train_df["speakerId"].isin(selected_speakers)]
#         if config.synthetic_speaker_subset_percentage < 1:
#             speakers = np.array(list(Counter(synthetic_train_df.speakerId)))
#             np.random.shuffle(speakers)
#             selected_speaker_count = round(config.synthetic_speaker_subset_percentage * len(speakers))
#             selected_speakers = speakers[:selected_speaker_count]
#             synthetic_train_df = synthetic_train_df[synthetic_train_df["speakerId"].isin(selected_speakers)]
#     else:
#         if "speakerId" in list(real_train_df): real_train_df = real_train_df.drop(columns="speakerId")
#         if "speakerId" in list(synthetic_train_df): synthetic_train_df = synthetic_train_df.drop(columns="speakerId")
#         if config.real_speaker_subset_percentage < 1:
#             print("no speaker id listed in dataset .csv; ignoring speaker subset selection")
#         if config.synthetic_speaker_subset_percentage < 1:
#             print("no speaker id listed in dataset .csv; ignoring speaker subset selection")
#
#     # Select random subset of training data
#     if config.real_dataset_subset_percentage < 1:
#         subset_size = round(config.real_dataset_subset_percentage * len(real_train_df))
#         real_train_df = real_train_df.loc[np.random.choice(len(real_train_df), subset_size, replace=False)]
#     # real_train_df = real_train_df.set_index(np.arange(len(real_train_df)))
#     if config.synthetic_dataset_subset_percentage < 1:
#         subset_size = round(config.synthetic_dataset_subset_percentage * len(synthetic_train_df))
#         synthetic_train_df = synthetic_train_df.loc[
#             np.random.choice(len(synthetic_train_df), subset_size, replace=False)]
#     # synthetic_train_df = synthetic_train_df.set_index(np.arange(len(synthetic_train_df)))
#
#     train_df = pd.concat([synthetic_train_df, real_train_df]).reset_index()
#     if not config.seq2seq:
#         valid_df = pd.read_csv(os.path.join(base_path, "data", "valid_data.csv"))
#         test_df = pd.read_csv(os.path.join(base_path, "data", "test_data.csv"))
#     else:
#         valid_df = pd.read_csv(os.path.join(base_path, "data", "valid_data_seq2seq.csv"))
#         test_df = pd.read_csv(os.path.join(base_path, "data", "test_data_seq2seq.csv"))
#
#     print("xzl: config.seq2seq", config.seq2seq)
#
#     if not config.seq2seq:
#         # Get list of slots
#         Sy_intent = {"action": {}, "object": {}, "location": {}}
#
#         values_per_slot = []
#         for slot in ["action", "object", "location"]:
#             slot_values = Counter(train_df[slot])
#             for idx, value in enumerate(slot_values):
#                 Sy_intent[slot][
#                     value] = idx  # xzl: Sy_intent like a hashtable, from slot/value strings to idx. later reverse lookup will be used
#             values_per_slot.append(len(slot_values))
#         config.values_per_slot = values_per_slot  # xzl: a list of integers...
#         print("xzl values_per_slot", values_per_slot)
#         config.Sy_intent = Sy_intent
#         print("xzl Sy_intent", Sy_intent)
#     else:  # seq2seq
#         import string  # xzl: inc possible values & printable chars (for asr target?)
#         all_chars = "".join(train_df.loc[i]["semantics"] for i in
#                             range(len(train_df))) + string.printable  # all printable chars; TODO: unicode?
#         all_chars = list(set(all_chars))
#         Sy_intent = ["<sos>"]
#         Sy_intent += all_chars
#         Sy_intent.append("<eos>")
#         config.Sy_intent = Sy_intent
#         print("xzl: seq2seq, Sy_intent=", Sy_intent)
#
#     # If certain phrases are specified, only use those phrases
#     if config.train_wording_path is not None:
#         with open(config.train_wording_path, "r") as f:
#             train_wordings = [line.strip() for line in f.readlines()]
#         train_df = train_df.loc[train_df.transcription.isin(train_wordings)]
#         train_df = train_df.set_index(np.arange(len(train_df)))
#
#     if config.test_wording_path is not None:
#         with open(config.test_wording_path, "r") as f:
#             test_wordings = [line.strip() for line in f.readlines()]
#         valid_df = valid_df.loc[valid_df.transcription.isin(test_wordings)]
#         valid_df = valid_df.set_index(np.arange(len(valid_df)))
#         test_df = test_df.loc[test_df.transcription.isin(test_wordings)]
#         test_df = test_df.set_index(np.arange(len(test_df)))
#
#     # Get number of phonemes
#     if os.path.isfile(os.path.join(config.folder, "pretraining", "phonemes.txt")):
#         Sy_phoneme = []
#         with open(os.path.join(config.folder, "pretraining", "phonemes.txt"), "r") as f:
#             for line in f.readlines():
#                 if line.rstrip("\n") != "": Sy_phoneme.append(line.rstrip("\n"))
#         config.num_phonemes = len(Sy_phoneme)
#     else:
#         print("No phoneme file found.")
#
#     # Create dataset objects
#     train_dataset = data.SLUDataset(train_df, base_path, Sy_intent, config, upsample_factor=config.dataset_upsample_factor)
#     valid_dataset = data.SLUDataset(valid_df, base_path, Sy_intent, config)
#     test_dataset = data.SLUDataset(test_df, base_path, Sy_intent, config)
#
#     return train_dataset, valid_dataset, test_dataset


# config = data.read_config("experiments/no_unfreezing.cfg")

def add_random_noise(data, scale=.005):
    '''
    data: np.array | torch.Tensor
    '''
    length = data.shape[0]
    noise = np.random.randn(length)
    data = data + scale * noise
    return data


def temporal_shift(data, max_ratio=.05):
    '''
    shift the audio to left, overflow goes to the left,
    e.g., [1,2,3,4] -> [4,1,2,3]
    '''
    length = len(data)
    ratio = random.uniform(0, max_ratio)
    num_roll = round(ratio * length)
    data = np.roll(data, num_roll, axis=-1)
    return data


def alter_freq(data, rate=1.):
    '''
    alter the frequency of the audio via stretching its length
    scale < 1: lower freq; scale > 1: higher freq
    '''
    length = data.shape[0]
    data = librosa.effects.time_stretch(data, rate=rate)
    if data.shape[0] > length:
        return data[:length]
    else:
        return np.pad(data, (0, length - data.shape[0]), mode='constant')


def random_alter_freq(data, low=.9, high=1.1):
    '''
    randomly alter the frequency of the audio
    a, b: lower and upper bound for scale
    '''
    rate = random.uniform(a=low, b=high)
    return alter_freq(data, rate)


def audio_augment(x, num=5):
    '''
    apply each augmentation `num` times
    '''
    augs = [add_random_noise, random_alter_freq, temporal_shift] * num
    random.shuffle(augs)
    ret = [x]
    for aug in augs:
        ret.append(aug(x))
    return np.stack(ret)


def generate_samples(x, df, speakerId, config):
    length = x.shape[0]
    x_aug = audio_augment(x)
    n = len(x_aug)
    rows = df.sample(n=n)
    x_neg = []
    for _, row in rows.iterrows():
        wav_path = os.path.join(config.slu_path, row['path'])
        x, _ = sf.read(wav_path)
        x_neg.append(np.pad(x, (0, length - x.shape[0]), mode='constant'))
    x_neg = np.stack(x_neg)
    x_in = np.concatenate([x_aug, x_neg], axis=0)


def weighted_ctc(pred, label, weight, pred_lengths, label_lengths, blank=0):
    log_probs = F.log_softmax(pred, dim=1)
    ctc_loss = F.ctc_loss(log_probs.permute(2, 0, 1), label, pred_lengths, label_lengths,
                          blank=blank, reduction='sum')
    ctc_grad, _ = torch.autograd.grad(ctc_loss, (pred,), retain_graph=True)
    temporal_mask = (torch.arange(pred.shape[-1], device=pred_lengths.device,
                                  dtype=pred_lengths.dtype).unsqueeze(0) < pred_lengths.unsqueeze(1))[:, None, :]
    alignment_pred = (log_probs.exp() * temporal_mask * weight - ctc_grad).detach()
    ctc_weighted_loss = (-alignment_pred * log_probs).sum()
    return ctc_weighted_loss


def reduce_dimensions(cluster_centers):
    # Concatenate the list of tensors into a 2D numpy array
    cluster_centers = np.concatenate(cluster_centers, axis=0)
    # Perform PCA to reduce the dimension from 60 to 2
    pca = PCA(n_components=2)
    cluster_centers_pca = pca.fit_transform(cluster_centers)
    return cluster_centers_pca


def plot_cluster_centers(cluster_centers):
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot each iteration with a different color
    for i, centers in enumerate(cluster_centers):
        reduced_centers = reduce_dimensions(centers)
        ax.scatter(reduced_centers[:, 0], reduced_centers[:, 1], label=f"Iteration {i + 1}")

    # Add title, x-label, y-label, and legend
    ax.set_title("PCA of Cluster Centers")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    # Show the plot
    plt.show()


def get_token_and_weight(transcript):
    # compute weight
    # tks = roberta_tknz(transcript, return_tensors='pt')
    # output = roberta_model(**tks)
    # shape is (1, <bos>+T+<eos>, 768)
    # last_hidden_state = output.last_hidden_state
    # weight = last_hidden_state.sum(dim=-1).squeeze(0)[1:-1]

    # compute phoneme
    phoneme_seq = []
    weight = []
    for tk in nltk_tknz.tokenize(transcript):
        if tk in cmud:
            phonemes = cmud[tk][0]
            phoneme_seq.extend(phonemes)
            weight.extend([1.] * len(phonemes))
            phoneme_seq.append('sp')
            weight.append(0.)
    if phoneme_seq[-1] == 'sp':
        phoneme_seq.pop()
        weight.pop()
    assert (len(phoneme_seq) == len(weight))
    return phoneme_seq, weight


def get_audio_duration(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.size(1) / sample_rate
    return duration


def create_csv_file(filename, headers):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def write_to_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


def initialize_SLURP_L1(folder_name):
    config = data.read_config("experiments/no_unfreezing.cfg")
    train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
    print('config load ok')
    device = 'cpu'
    wav_path = os.path.join(pwd, 'SLURP/slurp_real/')
    pwd = os.getcwd()
    folder_path = os.path.join(pwd, folder_name)

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
    return config, wav_path, folder_path, spk_group


def train_test_split(df, speakers):
    train_set, test_set = pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)
    for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
        # select rows of speaker X
        tmp = df[df['user_id'] == speakerId]
        # get all transcript for speaker X
        transcripts = np.unique(tmp['sentence'])
        # train test split for each transcript
        for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
            rows = tmp[tmp['sentence'] == transcript]
            midpoint = len(rows) // 2
            train_set = train_set.append(rows[:midpoint])
            if len(rows) > 1:
                test_set = test_set.append(rows[midpoint:])

    print('Train and test set created!!')
    return train_set, test_set


def initialize_N_spks_FSC_L1():
    config = data.read_config("experiments/no_unfreezing.cfg")
    speechcache_config = data.read_config("experiments/speechcache.cfg")
    train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
    _, _, _ = data.get_SLU_datasets(speechcache_config)  # used to set config.num_phonemes
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

    # process data
    new_train_df = deepcopy(train_dataset.df)
    act_obj_loc_idxs = []
    for _, row in new_train_df.iterrows():
        act, obj, loc = row['action'], row['object'], row['location']
        act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
    new_train_df['cache'] = act_obj_loc_idxs
    new_train_df.head()
    # new_train_df.to_csv('train_data_cache.csv', index=None)

    # emulate an oracle cloud model
    config.phone_rnn_num_hidden = [128, 128]
    cloud_model = models.Model(config).eval()
    cloud_model.load_state_dict(
        torch.load("experiments/no_unfreezing/training/model_state.pth", map_location='cpu'))  # load trained model

    spk_group = [['5o9BvRGEGvhaeBwA', 'ro5AaKwypZIqNEp2', 'KqDyvgWm4Wh8ZDM7'],
                 ['73bEEYMKLwtmVwV43', 'R3mXwwoaX9IoRVKe', 'ppzZqYxGkESMdA5Az'],
                 ['2ojo7YRL7Gck83Z3', 'kxgXN97ALmHbaezp', 'zaEBPeMY4NUbDnZy'],
                 ['8e5qRjN7dGuovkRY', 'xRQE5VD7rRHVdyvM', 'n5XllaB4gZFwZXkBz'],
                 ['oRrwPDNPlAieQr8Q', 'W4XOzzNEbrtZz4dW', 'Z7BXPnplxGUjZdmBZ'],
                 ['xEYa2wgAQof3wyEO', 'ObdQbr9wyDfbmW4E', 'EExgNZ9dvgTE3928'],
                 ['gvKeNY2D3Rs2jRdL', 'NWAAAQQZDXC5b9Mk', 'W7LeKXje7QhZlLKe'],
                 ['oXjpaOq4wVUezb3x', 'zZezMeg5XvcbRdg3', 'Gym5dABePPHA8mZK9'],
                 ['Xygv5loxdZtrywr9', 'AY5e3mMgZkIyG3Ox', 'anvKyBjB5OiP5dYZ'],
                 ['G3QxQd7qGRuXAZda', 'WYmlNV2rDkSaALOE', 'RjDBre8jzzhdr4YL'],
                 ['9EWlVBQo9rtqRYdy', 'M4ybygBlWqImBn9oZ', 'jgxq52DoPpsR9ZRx'],
                 ['kNnmb7MdArswxLYw', 'qNY4Qwveojc8jlm4', 'mor8vDGkaOHzLLWBp'],
                 ['5BEzPgPKe8taG9OB', 'AvR9dePW88IynbaE', '52XVOeXMXYuaElyw'],
                 ['2BqVo8kVB2Skwgyb', 'ZebMRl5Z7dhrPKRD', 'R3mexpM2YAtdPbL7'],
                 ['d2waAp3pEjiWgrDEY', 'DWmlpyg93YCXAXgE', '7NEaXjeLX3sg3yDB'],
                 ['xwzgmmv5ZOiVaxXz', 'd3erpmyk9yFlVyrZ', 'BvyakyrDmQfWEABb'],
                 ['Rq9EA8dEeZcEwada2', 'gNYvkbx3gof2Y3V9', 'g2dnA9Wpvzi2WAmZ'],
                 ['xwpvGaaWl5c3G5N3', 'DMApMRmGq5hjkyvX']]

    return config, new_train_df, spk_group


def FSC_train_test_split(new_train_df, speakers):
    train_set, test_set = pd.DataFrame(columns=new_train_df.columns), pd.DataFrame(columns=new_train_df.columns)
    for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
        tmp = new_train_df[new_train_df['speakerId'] == speakerId]
        transcripts = np.unique(tmp['transcription'])
        for tscpt_idx, transcript in tqdm(enumerate(transcripts), total=len(transcripts), leave=False):
            rows = tmp[tmp['transcription'] == transcript]
            # sample only one audio file for each distinct transcription
            train_set = train_set.append(rows.iloc[0])
            if len(rows) > 1:
                test_set = test_set.append(rows.iloc[1])

    print('Train and test set created!!')
    return train_set, test_set


def save_model_SLURP_L1(model, speakers, train_set, test_set, transcript_list, training_idxs, cluster_ids,
                        cluster_centers, intent_list, folder_path, filename):
    saved_data = {
        'train_set': train_set,
        'test_set': test_set,
        'speakers': speakers,
        'transcript_list': transcript_list,
        'intent_list': intent_list,
        'training_idxs': training_idxs,
        'cluster_ids': cluster_ids,
        'cluster_centers': cluster_centers,
    }

    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(saved_data, f)


def k_means_test(wav_path, model, train_set, test_set, cluster_ids, cluster_centers, transcript_list, training_idxs,
                 intent_list, THRESHOLD):
    # ----------------- prepare for cluster -----------------
    cluster_id_length = torch.tensor(list(map(len, cluster_ids)), dtype=torch.long, device='cpu')
    cluster_ids = pad_sequence(cluster_ids, batch_first=True, padding_value=0).to('cpu')
    cluster_centers = torch.stack(cluster_centers).to('cpu')
    # no reduction, loss on every sequence
    ctc_loss_k_means_eval = torch.nn.CTCLoss(reduction='none')
    tp, total, hits = 0, 0, 0
    for _, row in test_set.iterrows():
        if row[0] in training_idxs:
            continue
        # # of total evaluation samples
        total += 1
        wav = os.path.join(wav_path, row['recording_path'])
        x, _ = sf.read(wav)
        x = torch.tensor(x, dtype=torch.float, device='cpu').unsqueeze(0)
        with torch.no_grad():
            # ----------------- l1 -------------------
            x_feature = model.pretrained_model.compute_cnn_features(x)
            dists = torch.cdist(x_feature, cluster_centers)
            dists = dists.max(dim=-1)[0].unsqueeze(-1) - dists
            pred = dists.swapaxes(1, 0)
            pred_lengths = torch.full(size=(cluster_ids.shape[0],), fill_value=pred.shape[0], dtype=torch.long)
            loss = ctc_loss_k_means_eval(pred.log_softmax(dim=-1), cluster_ids, pred_lengths, cluster_id_length)
            pred_intent = loss.argmin().item()
            if loss[pred_intent] < THRESHOLD:
                # go with l1: kmeans
                # print('hit')
                hits += 1
                if row['intent'] == intent_list[pred_intent]:
                    # print('tp')
                    tp += 1
                # else:
                #     print('%s,%s' % (row['sentence'], transcript_list[pred_intent]))
            # else:
            #     print('cloud. loss was %f ' % loss.min())

    return total, hits, tp

def initialize_FSC():
    device = torch.device('cpu')
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

    # process data
    new_train_df = deepcopy(train_dataset.df)
    act_obj_loc_idxs = []
    for _, row in new_train_df.iterrows():
        act, obj, loc = row['action'], row['object'], row['location']
        act_obj_loc_idxs.append(act_obj_loc[(act, obj, loc)])
    new_train_df['cache'] = act_obj_loc_idxs
    new_train_df.head()
    return new_train_df, config


def flac_to_wav(input_file):
    sys.path.append('/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')
    # Load the FLAC audio file
    audio = AudioSegment.from_file(input_file, format="flac")

    # Export the audio to WAV format
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")

    return wav_buffer
