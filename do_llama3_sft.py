import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
from global_variables_phi2 import *
import os
from datasets import load_dataset, load_from_disk
from upload_to_hub import upload_to_hub
from accelerate import PartialState

def run_finetuning(dataset, model_name, new_model_name, output_path):
    device_string = PartialState().process_index
    device_map = {'': device_string}
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    tokenizer_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.add_eos_token = True

    print("tokenizer done")
    
    train_dataset = dataset #.shuffle().map(tokenize_function, batched=True)
    print(dataset)
    print("DATA_LOADED")

    model_path = model_name
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)
    
    print("MODEL INITIALIZED")
    print("Model's parameters device:", next(model.parameters()).device)
    print("Model's device:", next(model.parameters()).device)
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj"
        ])

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
        deepspeed = 'ds_config_zero.json',
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type)

    print("LOADED TRAINING ARGUMENTS")
    def formatting_prompts_func(examples):
        output_texts = []
        for i in range(len(examples)):
            text = f"### Instruction: Generate a hypothesis about the following problem: {examples['Problem'][i]}\n ### Hypothesis\n: {examples['Problem'][i]}\n {examples['Solution'][i]}\n {examples['Methodology'][i]}\n {examples['Evaluation'][i]}\n {examples['Results'][i]}"
            output_texts.append(text)
        return output_texts

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length= 32,
        dataset_text_field = 'Abstract',
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing)

    print("training_started")
    trainer.train()
    trainer.model.save_pretrained( f'{output_path}/{new_model_name}')
    print("Model saved.")


if __name__ == '__main__':
    base_model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
    #Supervised finetuning
    model_name = base_model_name
    new_sft_model_name = 'llama-3-70B-Instruct-sft'
    
    sft_dataset_name = 'charlieoneill/hypogen_shared_task'
    sft_dataset_path = sft_dataset_name
    sft_dataset = load_dataset(sft_dataset_path)['train']
    
    sft_output_file_path = 'stored_output_model_sft'
    
    if not os.path.exists(sft_output_file_path):
        os.makedirs(sft_output_file_path, exist_ok = True)
      
    run_finetuning(sft_dataset, model_name, new_sft_model_name, sft_output_file_path)
    #upload_to_hub(adapter_model_name, os.path.join(sft_output_file_path, new_sft_model_name), 'KnightHardik/temp_sft_phi2_aic')
    
    
