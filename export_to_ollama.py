"""
🚀 Export Fine-Tuned FunctionGemma to Ollama
Merges LoRA adapters and converts to GGUF format.

Usage:
    python export_to_ollama.py
"""

import os
import torch
import shutil

CONFIG = {
    "base_model": "google/functiongemma-270m-it",
    "lora_path": "./functiongemma-lora",
    "merged_output": "./functiongemma-merged",
    "ollama_model_name": "functiongemma-ada",
}


def main():
    print("\n" + "=" * 60)
    print("🚀 Export FunctionGemma to Ollama")
    print("=" * 60 + "\n")
    
    # Check LoRA exists
    if not os.path.exists(CONFIG["lora_path"]):
        print(f"❌ LoRA adapters not found at {CONFIG['lora_path']}")
        return
    print(f"✅ Found LoRA adapters at {CONFIG['lora_path']}")
    
    # Import here to avoid slow startup
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print("\n🔄 Step 1: Loading base model (this may take a minute)...")
    
    # Load base model WITHOUT quantization for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    print("✅ Base model loaded")
    
    print("\n🔄 Step 2: Loading and merging LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, CONFIG["lora_path"])
    model = model.merge_and_unload()
    print("✅ LoRA adapters merged")
    
    print(f"\n🔄 Step 3: Saving merged model to {CONFIG['merged_output']}...")
    if os.path.exists(CONFIG["merged_output"]):
        shutil.rmtree(CONFIG["merged_output"])
    os.makedirs(CONFIG["merged_output"], exist_ok=True)
    
    model.save_pretrained(CONFIG["merged_output"])
    tokenizer.save_pretrained(CONFIG["merged_output"])
    print("✅ Merged model saved!")
    
    # Create Modelfile
    modelfile = """FROM ./functiongemma-finetuned.gguf
PARAMETER temperature 0.0
PARAMETER top_p 0.9
PARAMETER num_predict 100
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<end_function_call>"
PARAMETER stop "<start_function_response>"
"""
    with open("Modelfile", "w") as f:
        f.write(modelfile)
    print("✅ Created Modelfile")
    
    # Print next steps
    print("\n" + "=" * 60)
    print("📦 NEXT STEPS")
    print("=" * 60)
    print("""
1. Clone llama.cpp and convert to GGUF:

   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   pip install -r requirements.txt
   python convert_hf_to_gguf.py ../functiongemma-merged --outfile ../functiongemma-finetuned.gguf --outtype q4_k_m
   cd ..

2. Import to Ollama:

   ollama create functiongemma-ada -f Modelfile

3. Test it:

   ollama run functiongemma-ada "Turn on the lights"
""")
    print("=" * 60)
    print(f"🎯 Model will be named: {CONFIG['ollama_model_name']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
