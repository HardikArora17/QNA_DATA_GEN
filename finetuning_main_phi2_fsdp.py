import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import pandas as pd
from global_variables_llama import *
import os
from datasets import load_from_disk
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from datetime import timedelta
from mpi4py import MPI


def run_finetuning(dataset, model_name, new_model_name, output_path):
    
     # Load LLaMA tokenizer
    tokenizer_path = os.path.join('/lustre/orion/proj-shared/stf218/tirthankar/astro_finetuning/QNA_DATA_GEN/tokenizers/', model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.add_eos_token = True

    print("tokenizer done")
    
    def tokenize_function(examples):
      return tokenizer(examples["text"], truncation=True)

    train_dataset = dataset.shuffle().map(tokenize_function, batched=True)
    print("DATA_LOADED")

    master_addr = os.environ["MASTER_ADDR"]
    print("Master address from main:", master_addr)

    master_port = "29500"

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

    model_path = os.path.join('/lustre/orion/proj-shared/stf218/tirthankar/astro_finetuning/QNA_DATA_GEN/model/', model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = FSDP(model)
    # model = model.to('cuda:0')

    print("MODEL INITIALIZED")
    print("Model's parameters device:", next(model.parameters()).device)

    # Alternatively, you can print the device of the model itself
    print("Model's device:", next(model.parameters()).device)

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    training_arguments = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optim='adamw_hf',
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard")

    print("LOADED TRAINING ARGUMENTS")
    
    collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator = collator,
        args=training_arguments,
        )

    # Train model
    print("training_started")

    trainer.train()
    # Save trained model
    trainer.model.save_pretrained( f'{output_path}/{new_model_name}')

    model_state_dict = model.state_dict()
    torch.save(model_state_dict, f'{output_path}/{new_model_name}_state.pt')

    print("Model state dictionary saved.")

if __name__ == '__main__':
    model_name = 'microsoft-phi2'
    dataset_name = 'astro-abstracts'
    dataset_path = os.path.join('/lustre/orion/proj-shared/stf218/tirthankar/astro_finetuning/QNA_DATA_GEN/dataset', dataset_name)
    dataset = load_from_disk(dataset_path)['train'].select(range(100))
    
    new_model_name = 'finetuned_astr_phi2'
    
    output_file_path = '/lustre/orion/proj-shared/stf218/tirthankar/astro_finetuning/QNA_DATA_GEN/stored_output_model'
    
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path, exist_ok = True)
    
    print("directories_made")
    run_finetuning(dataset, model_name, new_model_name, output_file_path)

