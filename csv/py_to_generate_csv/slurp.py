import pandas as pd
import json
import numpy as np


# Load JSON data from file
with open('/Users/afsarabenazir/Downloads/speech_datasets/slurp-master/dataset/slurp/metadata.json', 'r') as file:
    json_data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame(json_data)
df = df.transpose()
sen_df = df['sentence_normalised']
tmp = dict()
for index, row in df.iterrows():
    recording = pd.DataFrame(row['recordings']).transpose()
    tmp[row['nlub_id']] = dict(recording['usrid'].value_counts())

unique_tscpt = np.unique(df['sentence_normalised']) #15910
new_df = df.drop_duplicates(subset=['nlub_id']) #15930

spk_count = [[value] for value in tmp.values()]
# spk_count = pd.Series(spk_count)
new_df['spk_count'] = spk_count
columns_to_drop = ['recordings', 'status', 'userid_amt']
new_df = new_df.drop(columns=columns_to_drop)
new_df.to_csv('slurp.csv', index=False)

# sen = np.unique(new_df['sentence_normalised'], return_counts=True)
# value, count = np.unique(df['sentence_normalised'], return_counts=True)

print('here')