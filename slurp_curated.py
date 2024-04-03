import pandas as pd
from copy import deepcopy
import numpy as np
from tqdm import tqdm

slurp_df = pd.read_csv('SLURP/csv/slurp_mini_FE_MO_ME_FO_UNK.csv')
# slurp_df = pd.read_csv('SLURP/csv/slurp_per_speaker.csv')
slurp_df = deepcopy(slurp_df)

main_df = pd.DataFrame(columns=slurp_df.columns)
count_df = slurp_df.groupby(['user_id', 'sentence']).size().reset_index(name='utterance_count')
filtered_df = count_df.loc[count_df['utterance_count'] > 2]

filtered_slurp_df= slurp_df[slurp_df.set_index(['user_id', 'sentence']).index.isin(filtered_df.set_index(['user_id', 'sentence']).index)]

speakers = np.unique(filtered_slurp_df['user_id'])
# speakers = ['FO-488']
for _, user_id in tqdm(enumerate(speakers), total=len(speakers)):
    tmp = filtered_slurp_df[filtered_slurp_df['user_id'] == user_id]
    with_headset = tmp[tmp['recording_path'].str.contains('headset')]
    without_headset = tmp[~tmp['recording_path'].str.contains('headset')]
    main_df = pd.concat([main_df, with_headset], ignore_index=False)
    print()
# main_df.drop([main_df.columns[0], main_df.columns[1]], axis=1, inplace=True)
# main_df.to_csv('slurp_with_headset.csv')
print()

