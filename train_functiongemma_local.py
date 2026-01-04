"""
🚀 FunctionGemma 270M Local Fine-Tuning Script
Matches the Colab notebook EXACTLY for consistent results.
Optimized for RTX 3060 Ti (8GB VRAM)

Usage:
    python train_functiongemma_local.py

Prerequisites:
    1. Accept the license at: https://huggingface.co/google/functiongemma-270m-it
    2. Login with: huggingface-cli login
    3. Install dependencies: pip install torch transformers datasets peft accelerate bitsandbytes trl
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime

CONFIG = {
    # Model settings
    "model_name": "google/functiongemma-270m-it",
    
    # Training data - supports both JSONL and JSON array formats
    "training_file": "training_data/functiongemma_training.jsonl",
    "training_file_json": "training_data/functiongemma_training_readable.json",
    
    # Output directories
    "output_dir": "./functiongemma-finetuned",
    "lora_output_dir": "./functiongemma-lora",
    
    # Training hyperparameters - EXACTLY matching Colab
    "num_epochs": 1,
    "batch_size": 2,              
    "gradient_accumulation": 4,   
    "learning_rate": 2e-4,        
    "max_seq_length": 1024,       
    "warmup_ratio": 0.03,
    
    # LoRA settings - Same as Colab
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Logging
    "logging_steps": 10,
    "save_steps": 100,
    
    # Memory optimization for 8GB VRAM
    "use_4bit": True,
    "use_gradient_checkpointing": True,
}


def print_banner():
    """Print a nice banner."""
    print("\n" + "=" * 60)
    print("🚀 FunctionGemma 270M Local Fine-Tuning")
    print("   Matching Colab Configuration EXACTLY")
    print("   Optimized for RTX 3060 Ti (8GB VRAM)")
    print("=" * 60 + "\n")


def check_gpu():
    """Check GPU availability and memory."""
    print("📊 Checking GPU...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available! Please install CUDA.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_memory:.1f} GB")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    
    # Clear any existing GPU memory
    torch.cuda.empty_cache()
    
    return True


def load_training_data(config: dict):
    """
    Load training data - supports both JSONL and JSON array formats.
    Matches the Colab notebook data loading exactly.
    """
    print("\n📂 Loading training data...")
    
    data = []
    
    # Try JSONL first (one JSON object per line)
    jsonl_path = config["training_file"]
    json_path = config["training_file_json"]
    
    if os.path.exists(jsonl_path):
        print(f"   Loading from JSONL: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    # EXACTLY like Colab: combine prompt + completion
                    text = entry["prompt"] + entry["completion"]
                    data.append({"text": text})
    
    elif os.path.exists(json_path):
        print(f"   Loading from JSON array: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        for entry in entries:
            # EXACTLY like Colab: combine prompt + completion
            text = entry["prompt"] + entry["completion"]
            data.append({"text": text})
    
    else:
        raise FileNotFoundError(f"No training data found at {jsonl_path} or {json_path}")
    
    print(f"✅ Loaded {len(data)} training examples")
    
    # Show a sample
    if data:
        sample = data[0]["text"]
        print(f"   Sample length: {len(sample)} chars")
        print(f"   Sample preview: {sample[:100]}...")
    
    return data


def load_model_and_tokenizer(config: dict):
    """
    Load the model with 4-bit quantization for memory efficiency.
    Matches Colab but uses 4-bit for 8GB VRAM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"\n🔄 Loading {config['model_name']}...")
    print("   (This may take a few minutes on first run)")
    
    # Configure 4-bit quantization (needed for 8GB VRAM)
    if config["use_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("   Using 4-bit quantization for memory efficiency")
    else:
        bnb_config = None
    
    # Load tokenizer - EXACTLY like Colab
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded! ({param_count:,} parameters)")
    
    return model, tokenizer


def apply_lora(model, config: dict):
    """
    Apply LoRA adapters to the model.
    EXACTLY like Colab notebook.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    print("\n🔧 Applying LoRA adapters...")
    
    # Prepare model for k-bit training - EXACTLY like Colab
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA - EXACTLY like Colab
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA applied! Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    
    return model


def prepare_dataset(data: list, tokenizer, config: dict):
    """
    Prepare the dataset for training.
    EXACTLY like Colab notebook.
    """
    from datasets import Dataset
    
    print("\n📊 Preparing dataset...")
    
    dataset = Dataset.from_list(data)
    
    # Tokenize - EXACTLY like Colab (max_length=1024)
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=config["max_seq_length"],  # 1024 like Colab
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])
    print(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
    
    return tokenized_dataset


def train(model, tokenizer, dataset, config: dict):
    """
    Run the training loop.
    EXACTLY like Colab notebook.
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    print("\n🚀 Starting training...")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Gradient accumulation: {config['gradient_accumulation']}")
    print(f"   Effective batch size: {config['batch_size'] * config['gradient_accumulation']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Max sequence length: {config['max_seq_length']}")
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Training arguments - EXACTLY like Colab
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        # Memory optimizations for 8GB VRAM
        gradient_checkpointing=config["use_gradient_checkpointing"],
        dataloader_pin_memory=False,
    )
    
    # Create trainer - EXACTLY like Colab
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train!
    start_time = datetime.now()
    print(f"\n⏱️  Training started at {start_time.strftime('%H:%M:%S')}")
    print("=" * 60)
    
    trainer.train()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print("=" * 60)
    print(f"✅ Training complete! Duration: {duration}")
    
    return trainer


def save_model(model, tokenizer, config: dict):
    """Save the LoRA adapters."""
    print(f"\n💾 Saving LoRA adapters to {config['lora_output_dir']}...")
    
    os.makedirs(config["lora_output_dir"], exist_ok=True)
    model.save_pretrained(config["lora_output_dir"])
    tokenizer.save_pretrained(config["lora_output_dir"])
    
    print("✅ LoRA adapters saved!")


def test_model(model, tokenizer, config: dict):
    """
    Test the fine-tuned model with sample prompts.
    Uses EXACT prompt format from training data.
    """
    print("\n🧪 Testing the fine-tuned model...")
    
    # Load a sample from training data to get the EXACT prompt structure
    jsonl_path = config["training_file"]
    json_path = config["training_file_json"]
    
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline())
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
            sample = entries[0]
    
    # Extract the function declarations (everything before user message)
    base_prompt = sample["prompt"].rsplit("<start_of_turn>user", 1)[0]
    
    test_prompts = [
        "Turn on the bedroom lights",
        "Set a timer for 5 minutes",
        "Hello",
        "Search for weather",
    ]
    
    model.eval()
    
    print("-" * 60)
    for user_input in test_prompts:
        # Build prompt EXACTLY like training data
        full_prompt = base_prompt + f"<start_of_turn>user {user_input}<end_of_turn>\n<start_of_turn>model"
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Get only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        # Extract function call
        if "<start_function_call>" in response:
            start = response.find("<start_function_call>")
            end = response.find("<end_function_call>")
            if end > start:
                func = response[start + len("<start_function_call>"):end]
                print(f"� Input: {user_input}")
                print(f"🤖 Output: {func}")
            else:
                print(f"📝 Input: {user_input}")
                print(f"🤖 Raw: {response[:100]}")
        else:
            print(f"� Input: {user_input}")
            print(f"🤖 Raw: {response[:100]}")
        print("-" * 60)


def main():
    """Main training function."""
    print_banner()
    
    # Check GPU
    if not check_gpu():
        return
    
    try:
        # Load training data
        data = load_training_data(CONFIG)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(CONFIG)
        
        # Apply LoRA
        model = apply_lora(model, CONFIG)
        
        # Prepare dataset
        dataset = prepare_dataset(data, tokenizer, CONFIG)
        
        # Train
        trainer = train(model, tokenizer, dataset, CONFIG)
        
        # Save LoRA adapters
        save_model(model, tokenizer, CONFIG)
        
        # Test the model
        test_model(model, tokenizer, CONFIG)
        
        print("\n" + "=" * 60)
        print("🎉 Training complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Run: python export_to_ollama.py")
        print(f"  2. Follow the GGUF conversion steps")
        print(f"  3. Import into Ollama")
        print("=" * 60)
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU Out of Memory!")
        print("   Try reducing batch_size to 2 in the CONFIG")
        print("   Or reduce max_seq_length to 768")
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted. Checkpoints saved in output directory.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
