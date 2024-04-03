import os
import numpy as np
import pickle
from tqdm import tqdm
from utils import initialize_FSC

new_train_df, config = initialize_FSC()
pwd = os.getcwd()
# folder_path = os.path.join(pwd, 'models-FSC-k-means')
folder_path = os.path.join(pwd, 'models-FSC-phoneme-ctc')

speakers = np.unique(new_train_df['speakerId'])
speakers_to_remove = ['4aGjX3AG5xcxeL7a', '5pa4DVyvN2fXpepb', '9Gmnwa5W9PIwaoKq', 'KLa5k73rZvSlv82X',
                      'LR5vdbQgp3tlMBzB', 'OepoQ9jWQztn5ZqL', 'X4vEl3glp9urv4GN', 'Ze7YenyZvxiB4MYZ',
                      'eL2w4ZBD7liA85wm', 'eLQ3mNg27GHLkDej', 'ldrknAmwYPcWzp4N', 'mzgVQ4Z5WvHqgNmY',
                      'nO2pPlZzv3IvOQoP2', 'oNOZxyvRe3Ikx3La', 'roOVZm7kYzS5d4q3', 'rwqzgZjbPaf5dmbL',
                      'wa3mwLV3ldIqnGnV', 'xPZw23VxroC3N34k', 'ywE435j4gVizvw3R', 'zwKdl7Z2VRudGj2L',
                      '35v28XaVEns4WXOv', 'YbmvamEWQ8faDPx2', 'neaPN7GbBEUex8rV', '9mYN2zmq7aTw4Blo']
speakers = [item for item in speakers if item not in speakers_to_remove]
cache = 0
for spk_idx, speakerId in tqdm(enumerate(speakers), total=len(speakers)):
    # filename = f'FSC-k-means-{speakerId}.pkl'
    # file_path = os.path.join(folder_path, filename)
    # with open(file_path, 'rb') as f:
    #     load_data = pickle.load(f)
    # model = load_data['model']
    # transcript_list = load_data['transcript_list']
    # intent_list = load_data['intent_list']
    # training_idxs = load_data['training_idxs']
    # cluster_ids = load_data['cluster_ids']
    # cluster_centers = load_data['cluster_centers']
    # cluster_ids_bytes = sum(tensor.element_size() * tensor.numel() for tensor in cluster_ids)
    # cluster_centers_bytes = sum(tensor.element_size() * tensor.numel() for tensor in cluster_centers)
    # intent_list_bytes = sum(array.nbytes for array in intent_list)
    #
    # total_size_bytes = cluster_ids_bytes + cluster_centers_bytes + intent_list_bytes
    # total_size_kb = total_size_bytes / (1024)  # 1 KB = 1024 bytes
    # total_size_mb = total_size_bytes / (1024 * 1024)  # 1 MB = 1024 KB
    #
    # print(f"Total size in bytes: {total_size_bytes} bytes")
    # print(f"Total size in kilobytes (KB): {total_size_kb:.2f} KB")
    # print(f"Total size in megabytes (MB): {total_size_mb:.2f} MB")
    #
    # print(f"each cache entry: {total_size_kb / len(cluster_ids):.2f} KB")
    # cache += total_size_kb / len(cluster_ids)

    # load model
    filename = f'FSC-phoneme-ctc-{speakerId}.pkl'
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        load_data = pickle.load(f)
    model = load_data['model']
    transcript_list = load_data['transcript_list']
    phoneme_list = load_data['phoneme_list']
    intent_list = load_data['intent_list']
    training_idxs = load_data['training_idxs']

    phoneme_list_bytes = sum(tensor.element_size() * tensor.numel() for tensor in phoneme_list)
    intent_list_bytes = sum(array.nbytes for array in intent_list)

    total_size_bytes = phoneme_list_bytes + intent_list_bytes
    total_size_kb = total_size_bytes / (1024)  # 1 KB = 1024 bytes
    total_size_mb = total_size_bytes / (1024 * 1024)  # 1 MB = 1024 KB

    print(f"Total size in bytes: {total_size_bytes} bytes")
    print(f"Total size in kilobytes (KB): {total_size_kb:.2f} KB")
    print(f"Total size in megabytes (MB): {total_size_mb:.2f} MB")

    print(f"each cache entry: {total_size_kb/len(phoneme_list):.2f} KB")
    cache += total_size_kb/len(phoneme_list)
print()