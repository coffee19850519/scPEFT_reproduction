# 灵活推理代码使用说明

我已经为您创建了一个改进的推理代码 `flexible_inference.py`，它解决了原始示例代码只支持 hidden=128 的限制，**特别针对您的模型文件命名方式（如 `model_hidden64.pt`, `model_hidden8.pt`）进行了优化**。

## 主要改进

1. **从文件名自动提取hidden size**：自动从文件名如 `model_hidden64.pt` 中提取 hidden=64
2. **自动配置检测**：自动从训练好的模型中检测所有配置参数（embsize, d_hid, nlayers, nhead等）
3. **支持所有PEFT类型**：LORA, ENCODER, TOKEN, PREFIX, HYBRID 等
4. **智能模型文件选择**：当有多个模型文件时自动选择包含hidden信息的文件
5. **灵活参数配置**：不再固定任何参数，完全动态适配

## 针对您的模型使用方法

### 使用您的 ENCODER 模型目录：
```bash
python flexible_inference.py \
    --model_path /mnt/c/Users/wxy/Desktop/esmc/ENCODER \
    --data_path /path/to/test_data.h5ad \
    --output_dir ./results
```

### 指定特定的模型文件：
```bash
# 使用 hidden64 模型
python flexible_inference.py \
    --model_path /mnt/c/Users/wxy/Desktop/esmc/ENCODER \
    --model_file model_hidden64.pt \
    --data_path /path/to/test_data.h5ad \
    --output_dir ./results_hidden64

# 使用 hidden8 模型  
python flexible_inference.py \
    --model_path /mnt/c/Users/wxy/Desktop/esmc/ENCODER \
    --model_file model_hidden8.pt \
    --data_path /path/to/test_data.h5ad \
    --output_dir ./results_hidden8
```

### 批量处理多个模型：
我还创建了 `run_inference_example.py` 脚本来演示如何批量使用不同hidden size的模型：
```bash
python run_inference_example.py /path/to/test_data.h5ad
```

## 参数说明

### 必需参数：
- `--model_path`: 训练好的模型目录路径（包含model.pt和vocab.json）
- `--data_path`: 测试数据文件路径（.h5ad格式）

### 可选参数：
- `--output_dir`: 输出目录（默认：./inference_output）
- `--batch_size`: 批处理大小（默认：32）
- `--seed`: 随机种子（默认：0）

### 高级参数（通常自动检测）：
- `--peft_type`: PEFT类型 [HYBRID, ENCODER, TOKEN, PREFIX, LORA]
- `--peft_r`: LoRA rank参数
- `--embsize`: 嵌入维度
- `--d_hid`: 隐藏层维度
- `--nlayers`: Transformer层数
- `--nhead`: 注意力头数

## 输出文件

程序会在输出目录中生成以下文件：

1. `inference_results.pkl` - 完整的结果数据（包含预测、嵌入、配置等）
2. `predictions.csv` - 预测结果表格
3. `embeddings.npy` - 细胞嵌入向量

## 与原始代码的区别

| 特性 | 原始示例代码 | 新的灵活推理代码 |
|------|------------|----------------|
| hidden size | 固定 128 | 自动检测，支持任意值 |
| PEFT支持 | 有限 | 完整支持所有类型 |
| 配置方式 | 硬编码 | 自动检测 + 命令行参数 |
| 兼容性 | 特定模型 | 适配训练代码的所有配置 |
| 错误处理 | 基础 | 完善的验证和错误提示 |

## 使用示例

假设您有一个使用LORA训练的模型，embsize=256, r=64：

```bash
python flexible_inference.py \
    --model_path ./save/peft128LORA_COVID-19-Dec23-10-30 \
    --data_path ./data/COVID-19_test.h5ad \
    --output_dir ./inference_results
```

程序会自动：
- 检测到这是LORA模型
- 读取embsize=256
- 检测r=64
- 使用相应的配置进行推理

## 注意事项

1. 确保模型目录包含 `vocab.json` 和至少一个 `.pt` 文件
2. 测试数据应为 `.h5ad` 格式
3. 程序会自动匹配词汇表中的基因
4. 支持CPU和GPU推理（自动检测）

这个改进的推理代码完全兼容您的训练代码，支持所有配置参数，不再有任何硬编码的限制！