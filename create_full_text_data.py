import os
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, concatenate_datasets

# Assuming your data is a large list of dictionaries like:
# data = [{"text": "example text 1", "label": 1}, ..., {"text": "example text 300k", "label": 0}]

def chunk_data(data, chunk_size):
    """Generator to yield chunks of data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def process_data_chunks(data, chunk_size=10000):
    """Process data in chunks and return a list of datasets."""
    datasets = []
    for chunk in chunk_data(data, chunk_size):
        chunk_dataset = Dataset.from_list(chunk)
        datasets.append(chunk_dataset)
    return datasets

def combine_datasets(datasets):
    """Combine a list of datasets into a single dataset."""
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset
  
def create_data(input_data_path, input_data_path_2, data_output_path):
  data_folder_1  = input_data_path_1
  data_folder_2  = input_data_path_2
  
  list_of_files_1 = os.listdir(data_folder_1)
  list_of_files_2 = os.listdir(data_folder_2)
  
  example = []

  for file_name in tqdm(list_of_files_1):
    try:
      with open (os.path.join(data_folder_1, file_name), 'r') as file:
        lines = file.readlines()
        lines = " ".join(lines)
        example.append({'text': lines})
    except:
      print(file_name)

 for file_name in tqdm(list_of_files_2):
    try:
      with open (os.path.join(data_folder_2, file_name), 'r') as file:
        lines = file.readlines()
        lines = " ".join(lines)
        example.append({'text': lines})
    except:
      print(file_name)
        
  data_chunks_datasets = process_data_chunks(data, chunk_size=50000) 
  print("chunks created")
  combined_dataset = combine_datasets(data_chunks_datasets)
  print("chunks_combined")
  combined_dataset.save_to_disk(output_data_path)
  print("data_saved")

input_data_path_1 = 'data/astro-full-text/arxiv_markdowns'
input_data_path_2 = 'data/astro-full-text/astro-ph_markdowns' 
output_data_path = 'astro_full_text'
create_data(input_data_path_1, input_data_path_2, output_data_path)
