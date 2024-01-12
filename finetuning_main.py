import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
from data_create import create_data
from global_variables_llama import *
import os
from mpi4py import MPI


def run_finetuning(data_filename, model_name, new_model_name, output_path):
    instruct_data = create_data(data_filename)

    data = pd.DataFrame({'text': instruct_data})

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(data)

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

    # default_pg_timeout = timedelta(minutes=1)

    def setup_distributed_env(init_method=None, rank=0, world_size=16):
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        world_rank = rank = comm.Get_rank()
        # world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        backend = None
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(world_rank)
        os.environ['LOCAL_RANK'] = "0"  # str(world_rank % 8)
        print("initialization parameters:", init_method, backend, rank, world_size)
        torch.distributed.init_process_group(backend,
                                             # timeout = default_pg_timeout,
                                             init_method=init_method,
                                             rank=rank,
                                             world_size=world_size)

        using_mpi = torch.distributed.get_backend() == 'mpi'
        print("using_mpi=", using_mpi)

    setup_distributed_env()

    # Check GPU compatibility with float16

    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map)

    model = FSDP(model)

    # If the GPU MEMORY IS FILLED
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     load_in_4bit=True,
    #     quantization_config=bnb_config,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"])

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
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb")

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model_name)

    model_state_dict = model.state_dict()
    torch.save(model_state_dict, f'{output_path}/{new_model_name}_state.pt')

    print("Model state dictionary saved.")


if __name__ == '__main__':
    model_name = 'mistralai/Mistral-7B-v0.1'
    data_file_path = 'train_data.csv'
    new_model_name = 'finetuned_mistral_science_qna'
    output_file_path = 'stored_output_model'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_finetuning(model_name, data_file_path, new_model_name, output_file_path)

#<Tirthankar-Ghosal>
