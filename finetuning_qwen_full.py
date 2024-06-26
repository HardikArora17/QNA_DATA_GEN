import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
from global_variables_phi2 import *
import os
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
import deepspeed
import json

def run_finetuning(dataset, model_name, new_model_name, output_path):
    
    #device_string = PartialState().process_index
    #device_map = {'': device_string}
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    tokenizer_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.add_eos_token = True
    
    print("tokenizer done")
    
    #batch_size = 256
    def batched_tokenize_function(batch):
        #tokenized_examples = []
        #for i in trange(0, len(examples), batch_size):
           # batch = examples[i:i+batch_size]
        tokenized_examples  = tokenizer(batch['text'], truncation=True, max_length=512)
        return {"input_ids" : tokenized_examples["input_ids"] }
    
    chunk_size = 2
    train_dataset = dataset.shuffle().map(batched_tokenize_function, batched=True, batch_size = chunk_size)
    print("DATA_LOADED")
    print(train_dataset)
    
    with open('ds_config_zero.json' ,'r') as out:
        ds_config = json.load(out)

    #acc= Accelerator()
    model_path = model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code =True)
    #model, _, _ ,_  = deepspeed.initialize(model=model, model_parameters = model.parameters(), config =ds_config)

    #model = acc.prepare(model)    # model = model.to('cuda')
    print("MODEL INITIALIZED")
    print("Model's parameters device:", next(model.parameters()).device)

    # Alternatively, you can print the device of the model itself
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
        deepspeed = 'ds_config_zero.json',
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
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                args=training_arguments)
    
    # Train model
    print("training_started")

    trainer.train()
    # Save trained model
    trainer.model.save_pretrained( f'{output_path}/{new_model_name}')

    model_state_dict = model.state_dict()
    torch.save(model_state_dict, f'{output_path}/{new_model_name}_state.pt')

    print("Model state dictionary saved.")


if __name__ == '__main__':
    model_name = 'Qwen/Qwen1.5-1.8B'

    dataset_name = 'KnightHardik/temp-astro-full-text'
    dataset_path = dataset_name
    dataset = load_dataset(dataset_path)['train'].select(range(100))
    # dataset = load_from_disk(dataset_path)
    new_model_name = 'astroqwen-1.8B-full'
    output_file_path = 'stored_output_model_cpt'
    
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok = True)
    
    print("directories_made")
    run_finetuning(dataset, model_name, new_model_name, output_file_path)
