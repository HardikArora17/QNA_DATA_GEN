import os
from tqdm import tqdm
import pandas as pd
from datasets import Dataset

def create_data(input_data_path, data_output_path):
  data_folder  = input_data_path
  list_of_files = os.listdir(data_folder)
  example = []

  for file_name in tqdm(list_of_files):
    with open (os.path.join(data_folder, file_name), 'r') as file:
      lines = file.readlines()
      lines = " ".join(lines)
      example.append({'text': lines})

  df_data = pd.DataFrame(example)
  dataset = Dataset.from_pandas(df_data)
  dataset.save_to_disk(data_output_path)


input_data_path = '/content/train_data'
output_data_path = 'astro_full_text_data'
create_data(input_data_path, output_data_path)
