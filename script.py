# import pandas as pd
# import shutil
# import os
# from copy import deepcopy
#
# pwd = os.getcwd()
# slurp_df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_headset.csv'))
# slurp_df = deepcopy(slurp_df)
# # Example DataFrame
# data = slurp_df['recording_path']
# df = pd.DataFrame(data)
#
# source_folder = '/Users/afsarabenazir/Downloads/speech_datasets/slurp-wav/slurp-headset'
# destination_folder = '/Users/afsarabenazir/Downloads/speech_datasets/slurp-wav/slurp-headset2'
#
# # Create the destination folder if it doesn't exist
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)
#
# # Iterate over the filenames in the DataFrame
# for filename in slurp_df['recording_path']:
#     src_path = os.path.join(source_folder, filename.replace(".flac", ".wav"))
#     dest_path = os.path.join(destination_folder, filename.replace(".flac", ".wav"))
#
#     # Check if the file exists in the source folder
#     if os.path.exists(src_path):
#         shutil.copy(src_path, dest_path)
# print()

def calculate(L1_hit_rate, L2_hit_rate):
    # First expression calculation
    # result1 = L1_hit_rate * 42.2 + (1 - L1_hit_rate) * L2_hit_rate * 120.64 + (1 - L1_hit_rate) * (1 - L2_hit_rate) * 200 -- old
    result1 = L1_hit_rate * 95.89 + (1 - L1_hit_rate) * L2_hit_rate * 185 + (1 - L1_hit_rate) * (1 - L2_hit_rate) * audio_dur * 290

    # Second expression calculation
    # result2 = L1_hit_rate * 1.08 + (1 - L1_hit_rate) * L2_hit_rate * 3.1 + (1 - L1_hit_rate) * (1 - L2_hit_rate) * 200

    return result1

# Given hit rates
L1_hit_rate = 0.19
L2_hit_rate = 0.27
audio_dur = 3

# Calculate results
result1 = calculate(L1_hit_rate, L2_hit_rate)

print(f"Result for m7: {round(result1,2)}")
# print(f"Result for rp4: {round(result2,2)}")


# SC:
# Result for m7: 131.57
# Result for rp4: 92.86
# SC (with pretrained slurp)
# Result for m7: 118.28
# Result for rp4: 75.37
# SC 3 bucket
# Result for m7: 143.84
# Result for rp4: 104.91
# SC-dynamic+3 bucket
# Result for m7: 154.49
# Result for rp4: 131.59
# SC pretrained+3 bucket+dynamic
# Result for m7: 149.47
# Result for rp4: 118.61

# 131.57, 118.28, 143.84, 0, 154.49,0

# import pickle
# import os
# import torch
# import io
# import models
# import data
# import gzip
#
# config = data.read_config("experiments/no_unfreezing.cfg")
# train_dataset, valid_dataset, test_dataset = data.get_SLU_datasets(config)
# print('config load ok')
#
# class CustomUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)
#
# folder = os.path.join(os.getcwd(), 'models/SLURP/slurp-pretrain-multicache')
# file_names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
# file_names = [os.path.splitext(f)[0] for f in file_names]
# file_names = [f for f in file_names if f != '.DS_Store']
#
# for filename in file_names:
#     with open(os.path.join(folder, filename+'.pkl'), 'rb') as f:
#         model = CustomUnpickler(f).load()
#
#     model_state_dict = model['model'].state_dict()
#     metadata = {
#         'speakerId': model['speakerId'],
#         'transcript_list': model['transcript_list'],
#         'phoneme_list': model['phoneme_list'],
#         'intent_list': model['intent_list'],
#         'training_idxs': model['training_idxs'],
#         'cluster_ids': model['cluster_ids'],
#         'cluster_centers': model['cluster_centers']
#     }
#
#     with gzip.open(os.path.join(folder, filename+'.pkl.gz'), 'wb') as f:
#         pickle.dump(metadata, f)
#
#     # combined = {
#     #     'model_state_dict': model_state_dict,
#     #     'metadata': metadata
#     # }
#
#     # torch.save(combined, os.path.join(folder, 'slurp-pretrain-multicache-FE-249.pth'))
#     my_model_instance = models.Model(config)
#     my_model_instance.load_state_dict(model_state_dict)  # load trained model
#     torch.save(my_model_instance.state_dict(), os.path.join(folder, filename+'.pth'))
#     #
#     # model = torch.load(os.path.join(folder, 'slurp-pretrain-multicache-FE-249.pkl'), map_location=lambda storage, loc: storage)
#     # print('here')
#     # model = torch.load(os.path.join(folder, 'slurp-pretrain-multicache-FE-249.pkl'), map_location=torch.device('cpu'))

# import fairseq
# facebook/hubert-base-ls960
# ckpt_path = "./hubert_large_ll60k.pt"
# models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path], strict=False)