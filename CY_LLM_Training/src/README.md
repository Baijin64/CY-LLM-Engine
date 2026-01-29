# CY_LLM_Training 指南

本目录包含用于微调 DeepSeek（或其他兼容 LLM）以模仿特定角色语气与记忆/设定的工具链。

**要点**：本仓库区分两类训练数据——“语气模仿（dialogue）”与“人物设定（settings）”。训练时建议同时使用两者，并对 `settings` 提高权重以确保模型记住设定。

---

## 环境准备

建议创建一个新的 Conda/virtualenv 环境并安装依赖：

```bash
conda create -n cy_llm_train python=3.10 -y
conda activate cy_llm_train
pip install -r requirements.txt
```

---

## 数据预处理

1. 生成语气模仿数据（例如 `furina_train.jsonl`）：

```bash
python CY_LLM_Training/src/dataset_converter.py \
  --raw_dir ./CY_LLM_Training/preparation/dialogue_gi/extraction \
  --output_file ./CY_LLM_Training/data/furina_train.jsonl \
  --character "芙宁娜"
```

2. 生成设定/背景数据（例如 `settings_modified_furina.json` -> `settings_train.jsonl`）：

```bash
python CY_LLM_Training/src/dataset_converter.py \
  --raw_dir CY_LLM_Training/data \
  --output_file ./CY_LLM_Training/data/settings_train.jsonl \
  --character "芙宁娜"
```

说明：`dataset_converter.py` 会根据每条记录的 `input` 字段自动区分三类 Instruction（背景描述 / 设定问答 / 纯对话模仿），无需手动更改 JSON 内容。

---

## 训练（LoRA 微调）

`src/train_lora.py` 支持同时加载两份数据：主对话数据（`--dataset_path`）和设定数据（`--settings_path`）。当提供 `--settings_path` 时，脚本会对 `settings` 做过采样（默认 x5）以提高权重，随后把两者合并并打乱训练。

基本训练命令（联网可用，使用 HF ID）：

```bash
python CY_LLM_Training/src/train_lora.py \
  --model_name "facebook/opt-2.7b" \
  --dataset_path ./CY_LLM_Training/data/furina_train.jsonl \
  --settings_path ./CY_LLM_Training/data/settings_train.jsonl \
  --output_dir ./CY_LLM_Training/checkpoints/furina_lora_v3 \
  --batch_size 2 \
  --grad_accum 8 \
  --epochs 3
```

离线 / 无网络的替代方法：

- 如果你已经在本地有模型目录（例如 `/home/username/models/deepseek-llm-7b-chat`），直接将 `--model_name` 指向本地路径：

```bash
python CY_LLM_Training/src/train_lora.py \
  --model_name /home/username/models/deepseek-llm-7b-chat \
  --dataset_path ./CY_LLM_Training/data/furina_train.jsonl \
  --settings_path ./CY_LLM_Training/data/settings_train.jsonl \
  --output_dir ./CY_LLM_Training/checkpoints/furina_lora_v2
```

- 如果模型已缓存在 Hugging Face 本地缓存（`~/.cache/huggingface`），可以启用离线模式：

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python CY_LLM_Training/src/train_lora.py \
  --dataset_path ./CY_LLM_Training/data/furina_train.jsonl \
  --settings_path ./CY_LLM_Training/data/settings_train.jsonl \
  --output_dir ./CY_LLM_Training/checkpoints/furina_lora_v2
```

注意：离线模式要求基座模型已经完整下载到本机缓存或你提供了本地模型路径。

调整 `settings` 权重：当前实现是通过过采样（复制数据）来加权，默认倍数为 5。如需改变，请在 `train_lora.py` 中修改 `settings_weight` 变量。

训练输出：LoRA 权重会保存在 `--output_dir` 指定目录（例如 `checkpoints/furina_lora_v2`）。

---

## 推理（使用 LoRA）

训练完成后可用 `src/inference.py` 做交互式测试：

```bash
python CY_LLM_Training/src/inference.py \
  --base_model /home/username/models/deepseek-llm-7b-chat \
  --lora_path CY_LLM_Training/checkpoints/furina_lora_v2 \
  --character 芙宁娜
```

说明：`--base_model` 可以是 HF ID（需联网）或本地路径；LoRA 适配器需要可用的基座模型来加载并推理。

---

## 常见问题与故障排查

- 报错无法访问 `huggingface.co`（如 `Network is unreachable`）：说明程序尝试从 Hugging Face 下载模型但网络不可用。解决办法：使用本地已下载模型或在另一台可联网机器上下载后传回，或者使用国内镜像HF_ENDPOINT=https://hf-mirror.com
- 查找本地缓存目录：

```bash
ls -la ~/.cache/huggingface/hub
ls -la ~/.cache/huggingface/transformers
```

- 如果模型不在本地缓存：在有网络的机器上运行下载脚本，然后用 `scp`/`rsync` 复制到训练服务器：

```bash
# 在有网的机器上
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-llm-7b-chat', trust_remote_code=True); AutoTokenizer.from_pretrained('deepseek-ai/deepseek-llm-7b-chat', trust_remote_code=True)"

# 然后将下载目录传回服务器
scp -r /path/to/downloaded/deepseek-llm-7b-chat user@server:/home/username/models/
```

---

## 训练质量提示

- 如果训练时 Loss 很低但模型“答非所问”，通常是因为训练数据中 Instruction 固定或缺少 `input`，导致模型只学会了“在看到指令就生成台词”的模式。混合并加权设定数据可以显著改善这种问题。
- 更好的评估方式是只对 `Response` 计算 Loss（避免 Instruction 的重复信息拉低 Loss）。可选改进：使用 TRL/transformers 的专用 data collator，让 Loss 仅覆盖输出区域；需要时我可以帮你把 `train_lora.py` 调整为只对 Response 计算 Loss。

---
