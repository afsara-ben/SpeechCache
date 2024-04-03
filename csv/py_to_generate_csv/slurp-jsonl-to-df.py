import pandas as pd
import json
import numpy as np

# import jsonlines
#
# def combine_jsonl(file1, file2, output_file):
#     combined_data = []
#
#     # Read the first JSONL file and append its contents to the combined list
#     with jsonlines.open(file1, 'r') as reader:
#         for obj in reader:
#             combined_data.append(obj)
#
#     # Read the second JSONL file and append its contents to the combined list
#     with jsonlines.open(file2, 'r') as reader:
#         for obj in reader:
#             combined_data.append(obj)
#
#     # Write the combined list to the output JSONL file
#     with jsonlines.open(output_file, 'w') as writer:
#         writer.write_all(combined_data)
#
#     print("Combined JSONL files successfully.")
#
# # Path to the JSONL file
# train_jsonl = '/Users/afsarabenazir/Downloads/speech_datasets/slurp/dataset/slurp/train.jsonl'
# test_jsonl = '/Users/afsarabenazir/Downloads/speech_datasets/slurp/dataset/slurp/test.jsonl'
# combine_jsonl(train_jsonl, test_jsonl, 'combined.jsonl')

# Initialize an empty list to store the parsed JSON objects
data = []
file_path = '/Users/afsarabenazir/Downloads/speech_datasets/slurp/dataset/slurp/test.jsonl'
# file_path = '/Users/afsarabenazir/Downloads/speech_datasets/slurp/dataset/slurp/train.jsonl'
# Read the JSONL file line by line and parse each JSON object
with open(file_path, 'r') as file:
    for line in file:
        # Parse the JSON object
        json_object = json.loads(line)
        data.append(json_object)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)
# df.to_csv('slurp_df.csv', index=True)
# df.to_csv('slurp_test_df.csv', index=True)
df.to_csv('slurp_train_df.csv', index=True)
intents = np.unique(df['intent'])
print()

# train.jsonl = 11514
# test.jsonl = 2974
# combined.jsonl = 14488