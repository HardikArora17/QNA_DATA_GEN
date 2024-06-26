from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def upload_to_hub(base_model_name, adapter_model_name, output_model_name):
  base_model  = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='auto')  
  tokenizer = AutoTokenizer.from_pretrained(base_model_name)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"
  
  peft_model = PeftModel.from_pretrained(base_model, adapter_model_name)
  peft_model = peft_model.merge_and_unload()
  
  peft_model.push_to_hub(output_model_name)
  tokenizer.push_to_hub(output_model_name)

if __name__ == '__main__':
  base_model_name = 'microsoft/phi-2'
  adapter_model_name = 'stored_output_model_sft/sft_astrophi-full'
  output_model_name = 'KnightHardik/astrophi-full'

  upload_to_hub(base_model_name, adapter_model_name, output_model_name)
   
