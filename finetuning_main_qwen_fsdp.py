import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
from data_create import create_data
from global_variables_phi2 import *
import os
from mpi4py import MPI
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from datetime import timedelta

def run_finetuning(data_filename, model_name, new_model_name, output_path):

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    master_addr = os.environ["MASTER_ADDR"]
    print("Master address from main:", master_addr)

    master_port = "29500"
    #default_pg_timeout = timedelta(minutes=1)
    
    def setup_distributed_env(init_method=None, rank = 0, world_size=16): 
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        world_rank = rank = comm.Get_rank()
        #world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        #world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        backend = None
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(world_rank)
        os.environ['LOCAL_RANK'] = "0"#str(world_rank % 8)
        print("initialization parameters:", init_method, backend, rank, world_size)
        torch.distributed.init_process_group(backend,
                                            #timeout=default_pg_timeout,
                                            init_method=init_method,
                                            rank=rank,
                                            world_size=world_size)
        using_mpi = torch.distributed.get_backend() == 'mpi'
        print("using_mpi=", using_mpi)
    
    setup_distributed_env()


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config)

    model = FSDP(model)
    print("MODEL INITIALIZED")
    print("Model's parameters device:", next(model.parameters()).device)

    # Alternatively, you can print the device of the model itself
    print("Model's device:", next(model.parameters()).device)

    instruct_data = create_data(data_filename)

    data = pd.DataFrame({'text': instruct_data})
    print("DATA_LOADED")

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(data)

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.add_eos_token = True

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
        )

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
        fp16=fp16,
        optim=optim,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type)

    print("LOADED TRAINING ARGUMENTS")
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=200,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing
    )

    # Train model
    print("training_started")

    trainer.train()
    # Save trained model
    trainer.model.save_pretrained(f'{output_path}/{new_model_name}')

    model_state_dict = model.state_dict()
    torch.save(model_state_dict, f'{output_path}/{new_model_name}_state.pt')

    print("Model state dictionary saved.")



if __name__ == '__main__':
    model_name = 'Qwen/Qwen1.5-1.8B'
    dataset = load_dataset("universeTBD/arxiv-astro-abstracts-all")
    new_model_name = 'finetuned_qwen_astro_data'
    output_file_path = 'output_finetuned_model'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("directories_made")
    run_finetuning(dataset, model_name, new_model_name, output_file_path)
