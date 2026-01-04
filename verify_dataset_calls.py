import json

def verify_single_call_dataset():
    input_file = "training_data/functiongemma_training_readable.json"
    print(f"🔍 Verifying {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    multi_call_count = 0
    zero_call_count = 0
    total_items = len(data)
    
    for i, item in enumerate(data):
        completion = item.get("completion", "")
        call_count = completion.count("<start_function_call>")
        
        if call_count > 1:
            multi_call_count += 1
            print(f"❌ Item {i} has {call_count} calls: {completion}")
        elif call_count == 0:
            zero_call_count += 1
            # Passthrough might have 0 calls if it's just text, but let's check format
            # Actually passthrough usually has <start_function_call>call:passthrough...
            # if it uses the strict tool format.
            
    print("-" * 30)
    print(f"Total items: {total_items}")
    if multi_call_count == 0:
        print("✅ CONFIRMED: No examples have more than 1 function call.")
    else:
        print(f"⚠️ FOUND {multi_call_count} examples with multiple calls.")
        
    if zero_call_count > 0:
        print(f"ℹ️ Found {zero_call_count} examples with 0 calls (likely pure text responses if allowed).")
    else:
        print("ℹ All examples have at least 1 function call.")

if __name__ == "__main__":
    verify_single_call_dataset()
