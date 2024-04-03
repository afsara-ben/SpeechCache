import pandas as pd
import os
from copy import deepcopy
import numpy as np
import csv

pwd = os.getcwd()
slurp_df = pd.read_csv(os.path.join(pwd, 'SLURP/csv/slurp_mini_FE_MO_ME_FO_UNK.csv'))
slurp_df = deepcopy(slurp_df)

slurp_df = slurp_df.sample(frac=1).reset_index(drop=True)
slurp_df = slurp_df.drop_duplicates(subset='sentence', keep='first')
slurp_df = slurp_df[0:5293] #10% held out set
slurp_df.to_csv('slurp_10_pc.csv')
# slurp_df = slurp_df[~slurp_df['recording_path'].str.contains('headset')]
# slurp_df.to_csv('slurp_no_headset.csv')


print()

