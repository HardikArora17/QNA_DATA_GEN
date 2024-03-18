from tqdm.autonotebook import tqdm
import pandas as pd


def create_data(dataset):
    instruct_data = []

    # for type_of_split in dataset:
    #   print(type_of_split)
    for row in tqdm(dataset['train']):
      instruct_data.append(row['text'])

    print("TOTAL TRAINING SAMPLES GATHERED: ", len(instruct_data))

    return instruct_data
