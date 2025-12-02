import json
import os
import argparse
from typing import List, Dict

def load_raw_data(data_dir: str) -> List[Dict]:
    """加载目录下所有的原始 JSON 数据"""
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return all_data

def convert_to_conversation(raw_data: List[Dict], target_character: str, window_size: int = 3) -> List[Dict]:
    """
    将对话数据转换为指令微调格式。
    
    针对 `speaker/角色名` 这种只有单人台词的数据源：
    由于缺乏上下文（对方说了什么），我们将构建 "无上下文" 的风格模仿训练数据。
    
    Args:
        raw_data: 原始数据列表
        target_character: 目标训练角色名字
    
    Returns:
        List[Dict]: 格式化后的训练数据
    """
    training_samples = []
    
    # 检查数据是否包含上下文信息 (简单的启发式检查)
    # 如果数据中只有目标角色的发言，且没有 reply_to 或 context 字段，则视为无上下文模式
    # 这里假设 raw_data 已经是过滤后的单人数据（基于用户提供的路径 speaker/派蒙）
    
    # 定义属于背景描述的 input 键值
    background_keys = {
        "更多描述", "角色详细", "角色故事1", "角色故事2", "角色故事3", 
        "角色故事4", "角色故事5", "孤心沙龙", "神之眼"
    }

    for line in raw_data:
        # 优先使用原始数据中的 input 字段（如果存在且不为空）
        # 这样可以支持 settings_modified_furina.json 这种包含 "input": "角色故事1" 的数据
        if 'input' in line and line['input']:
            input_text = line['input']
            
            if input_text in background_keys:
                # 针对背景描述类数据（第三人称描述）
                instruction = f"以下是关于{target_character}的背景故事和设定描述。"
            else:
                # 针对角色台词类数据（第一人称）
                instruction = f"你现在是{target_character}。请根据你的背景设定回答问题。"
        else:
            # 否则保持为空（针对纯对话数据）
            input_text = ""
            # 针对纯对话数据的 Instruction：侧重于语气模仿
            instruction = f"你现在是{target_character}。请模仿{target_character}的语气、口癖和性格进行对话。"

        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": line['text']
        }
        training_samples.append(sample)
                    
    return training_samples

def main():
    parser = argparse.ArgumentParser(description="Convert raw dialogue to LLM training format")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to speaker json directory (e.g. .../speaker/派蒙)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output jsonl file")
    parser.add_argument("--character", type=str, default=None, help="Target character name (optional, defaults to dir name)")
    
    args = parser.parse_args()
    
    # 自动推断角色名
    if args.character is None:
        args.character = os.path.basename(os.path.normpath(args.raw_dir))
        print(f"Inferred character name: {args.character}")
    
    print(f"Loading data from {args.raw_dir}...")
    raw_data = load_raw_data(args.raw_dir)
    print(f"Loaded {len(raw_data)} raw lines.")
    
    print(f"Extracting conversations for character: {args.character}...")
    dataset = convert_to_conversation(raw_data, args.character)
    print(f"Generated {len(dataset)} training samples.")
    
    # Save as JSONL
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
