import torch
from transformers import AutoTokenizer

def check_setup():
    print("🚀 Checking Function Gemma Setup...")
    
    # 1. Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("❌ CUDA not available")

    # 2. Check Tokenizer & Template
    model_name = "google/functiongemma-270m-it"
    print(f"\n🔍 Inspecting Tokenizer for: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")
        
        print("\n🔑 Special Tokens:")
        for k, v in tokenizer.special_tokens_map.items():
            print(f"   {k}: {v}")
            
        print("\n📝 Chat Template:")
        if tokenizer.chat_template:
            print(tokenizer.chat_template)
        else:
            print("   No chat template found in tokenizer config")
            
        # Test a sample formatting if possible
        print("\n🧪 Testing Template Formatting:")
        messages = [
            {"role": "user", "content": "What is the weather in London?"}
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print("Formatted User Prompt:")
            print(formatted)
        except Exception as e:
            print(f"   Could not apply chat template: {e}")

    except Exception as e:
        print(f"\n❌ Error loading tokenizer: {e}")
        print("   (You might need to login with `huggingface-cli login` first)")

if __name__ == "__main__":
    check_setup()
