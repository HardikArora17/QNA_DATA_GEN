import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import SFTTrainer
import pandas as pd
from global_variables_phi3 import *
import os
from itertools import chain
from datasets import load_dataset
import json

def group_texts(examples, block_size):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

def run_finetuning(raw_dataset, model_name, new_model_name, output_path):
    
    tokenizer_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.add_eos_token = True
    print("Tokenizer Loaded")
    

    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True)
    
    tokenized_datasets = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Running tokenizer on every text in dataset",
        )

    processed_dataset = tokenized_datasets.map(
                group_texts,
                fn_kwargs={'max_seq_length': block_size},
                batched=True,
                desc=f"Grouping texts in chunks of {block_size}",
            )
    
    print("DATA_LOADED")
    print(processed_dataset)
    
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', trust_remote_code =True)
   
    print("MODEL INITIALIZED")
    print("Model's parameters device:", next(model.parameters()).device)
    print("Model's device:", next(model.parameters()).device)

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optim=optim,
        bf16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {'use_reentrant':False},
        lr_scheduler_type=lr_scheduler_type)

    print("LOADED TRAINING ARGUMENTS")
    
    # Set supervised fine-tuning parameters
    trainer = Trainer(
                model=model,
                train_dataset=processed_dataset,
                tokenizer=tokenizer,
                args=training_arguments)
              

    trainer.train()
    trainer.model.save_pretrained( f'{output_path}/{new_model_name}_pre')
    trainer.save_model(f'{output_path}/{new_model_name}')
    
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, f'{output_path}/{new_model_name}_state.pt')

    print("Model state dictionary saved.")


if __name__ == '__main__':
    model_name = 'microsoft/Phi-3-mini-4k-instruct'
    new_model_name = 'astrophi-3-mini-4k-instruct'

    dataset_name = 'KnightHardik/temp-astro-full-text'
    dataset = load_dataset(dataset_name)['train']
    
    output_file_path = 'stored_output_model_cpt'
    
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok = True)
    
    print("directories_made")
    run_finetuning(dataset, model_name, new_model_name, output_file_path)
