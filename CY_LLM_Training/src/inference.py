import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="deepseek-ai/deepseek-llm-7b-chat", help="Base model ID")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--character", type=str, default="派蒙", help="Character name to roleplay")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model} in 4-bit...")
    # 1. 4-bit 量化配置 (与训练时保持一致)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. 加载 LoRA 适配器
    print(f"Loading LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    print("\n" + "="*50)
    print("模型加载完成！开始对话 (输入 'quit' 退出)")
    print("="*50)

    # 5. 交互循环
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        # 构建 Prompt (与训练时格式保持一致)
        # 训练格式: ### Instruction:\n你现在是{character}...\n\n### Input:\n{input}\n\n### Response:\n
        
        prompt = f"### Instruction:\n你现在是{args.character}。请模仿{args.character}的语气、口癖和性格进行对话。和你对话的是旅行者。回答必须符合人物设定和对话背景。\n\n### Input:\n{user_input}\n\n### Response:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # 增加停止词，防止模型生成完回复后继续胡言乱语
                stop_strings=["<|endoftext|>", "### Instruction:", "### Input:"],
                tokenizer=tokenizer
            )
        
        # 解码并去除 Prompt 部分
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 截断可能多余生成的内容
        if "<|endoftext|>" in generated_text:
            generated_text = generated_text.split("<|endoftext|>")[0]
        
        response = generated_text.split("### Response:\n")[-1].strip()
        
        print(f"{args.character}: {response}")

if __name__ == "__main__":
    main()
