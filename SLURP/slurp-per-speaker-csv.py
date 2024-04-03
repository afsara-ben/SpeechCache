import pandas as pd
import json
import os
import data
from tqdm import tqdm
from copy import  deepcopy
import numpy as np
from utils import get_audio_duration

pwd = os.getcwd()
# Load JSON data from file
with open('/Users/afsarabenazir/Downloads/speech_datasets/slurp/dataset/slurp/metadata.json', 'r') as file:
    json_data = json.load(file)

# # Convert JSON data to DataFrame
df = pd.DataFrame(json_data)
df = df.transpose()
#
# slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/SLURP/slurp_df.csv')
slurp_df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/SLURP/slurp_test_df.csv')
df_rows = []
for slurp_id, row in df.iterrows():
    slurp_id = int(slurp_id)
    if not slurp_df['slurp_id'].isin([slurp_id]).any():
        continue
    intent = slurp_df.loc[slurp_df['slurp_id'] == slurp_id, 'intent'].values[0]
    action = slurp_df.loc[slurp_df['slurp_id'] == slurp_id, 'action'].values[0]
    scenario = slurp_df.loc[slurp_df['slurp_id'] == slurp_id, 'scenario'].values[0]
    nlub_id = row["nlub_id"]
    recordings = row["recordings"]
    sentence = row['sentence_normalised']
    for recording, details in recordings.items():
        rec_path = recording
        usrid = details["usrid"]
        df_rows.append([usrid, slurp_id, nlub_id, sentence, intent, action, scenario, rec_path])

df = pd.DataFrame(df_rows, columns=["user_id", "slurp_id", "nlub_id", "sentence", "intent", "action", "scenario", "recording_path"])
df = df.sort_values('user_id')
df.to_csv('slurp_test_per_speaker.csv', index=True)

print()
# config = data.read_config("experiments/no_unfreezing.cfg")
# df = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/SLURP/slurp_per_speaker.csv')
# spk = pd.DataFrame(columns=df.columns)

# starts_with = 'MO'
# speakers = np.unique(df['user_id'])
# speakers = [elem for elem in speakers if elem.startswith(starts_with)]
# # speakers = ['MO-433', 'UNK-326', 'FO-232', 'ME-144']
#
# for speaker in speakers:
#     count = df[df['user_id'] == speaker].shape[0]
#     print(count)
#     # if count > 1000:
#     #     continue
#     spk = spk.append(df[df['user_id'] == speaker])
# spk.to_csv(f'slurp_mini_{starts_with}.csv', index=True)
print()
df1 = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_FE.csv')
df2 = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_MO.csv')
df3 = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_ME.csv')
df4 = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_FO.csv')
df5 = pd.read_csv('/Users/afsarabenazir/Downloads/speech_projects/speechcache/slurp_mini_UNK.csv')

dfs = [df1, df2, df3, df4, df5]
# Concatenate the two dataframes vertically
combined_df = pd.concat(dfs, axis=0)

# Write the combined dataframe to a new CSV file
combined_df.to_csv(f'slurp_mini_FE_MO_ME_FO_UNK.csv', index=True)