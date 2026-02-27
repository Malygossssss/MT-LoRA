# MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning

## Introduction

This is the official implementation of the paper: **MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning** developed at [Brown University SCALE lab](https://scale-lab.github.io).

This repository provides a Python-based implementation of MTLoRA including [`MTLoRALinear`](models/lora.py) (the main module) and MTL architectures.

The repository is built on top of [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and uses some modules from [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch).

---

## 环境准备与数据准备

### 1) 克隆仓库

```bash
git clone https://github.com/scale-lab/MTLoRA.git
cd MTLoRA
```

### 2) 安装依赖

- Python >= 3.8（推荐 3.8/3.9）
- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- CUDA >= 11.6

```bash
pip install -r requirements.txt
```

### 3) 数据集与权重

- 数据集路径可通过配置文件或命令行参数指定：
  - `--pascal PASCAL_MT`：PASCAL 多任务设置
  - `--nyud NYUV2`：NYUDv2 多任务设置
- Swin backbone 预训练权重可在 [Swin Transformer 官方仓库](https://github.com/microsoft/Swin-Transformer) 下载，常见路径：
  - `./backbone/swin_tiny_patch4_window7_224.pth`
  - `./backbone/swin_base_patch4_window7_224.pth`

---

## 训练/评估启动方式（详细）

> 本仓库默认采用分布式入口：`python -m torch.distributed.launch ... main.py`。
> 建议为每组实验使用不同 `--master_port`，避免端口冲突。

### 命令结构模板

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> \
python -m torch.distributed.launch \
  --nproc_per_node <gpu_num> \
  --master_port <port> \
  main.py \
  --cfg <config.yaml> \
  --tasks <task1,task2,...> \
  --batch-size <bs> \
  [--epochs <num>] \
  [--ckpt-freq <num>] \
  [--eval-freq <num>] \
  [--resume-backbone <backbone.pth>] \
  [--resume <checkpoint.pth>] \
  [--eval]
```

### 常用参数说明

- `--cfg`：实验配置文件（模型结构、优化器、数据增强等主配置）。
- `--tasks`：任务列表，逗号分隔，如：`semseg,normals,sal,human_parts` 或 `semseg,normals,depth`。
- `--batch-size`：单 GPU batch size。
- `--epochs`：训练轮数。
- `--ckpt-freq`：每 N 个 epoch 存一次 checkpoint。
- `--eval-freq`：每 N 个 epoch 做一次验证评估。
- `--resume-backbone`：加载 backbone 预训练权重（通常用于开始训练）。
- `--resume`：恢复某个完整 checkpoint（可用于继续训练或配合 `--eval` 仅评估）。
- `--eval`：仅评估，不进行训练。
- `--freeze-backbone`：冻结 backbone，仅训练头部/适配层（取决于配置）。

### 输出目录

默认输出在 `output/` 下，通常结构类似：

```text
output/<exp_name>/<tag>/
  ├── log_rank0.txt
  ├── ckpt_epoch_xxx.pth
  └── ...
```

---

## 你常用命令的归纳整理（可直接复用）

下面将你给出的命令按用途分组，建议直接复制后按需改路径/端口/GPU 编号。

### A. PASCAL-MT：从 checkpoint 做纯评估

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node 2 --master_port 29502 main.py --cfg configs/mtlora/tiny_448/mtlora_tiny_448_r64_scale4_pertask_stage2.yaml --pascal PASCAL_MT --tasks semseg,normals,sal,human_parts --batch-size 32 --resume /home/soft/users/wyf/MTLoRA-main/output/mtlora_tiny_448_r64_scale4_pertask-maml/default/ckpt_epoch_299.pth --eval
```

适用场景：已有训练好 checkpoint，只想复现实验指标。

### B. PASCAL-MT：从 backbone 启动训练（Tiny / ProMLoRA）

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node 2 --master_port 29501 main.py --cfg configs/mtlora/tiny_448/promlora_tiny_448_r32_prom50_scale4_pertask.yaml --pascal PASCAL_MT --tasks semseg,normals,sal,human_parts --batch-size 12 --ckpt-freq=20 --eval-freq=5 --epochs=300 --resume-backbone ./backbone/swin_tiny_patch4_window7_224.pth
```

适用场景：新实验从 ImageNet 预训练 backbone 开始训。

### C. PASCAL-MT：从已有 checkpoint 继续训练/微调（Base）

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29501 main.py --cfg configs/mtlora/base_448/pascal/promlora_base_448_r64_prom60_scale4_pertask.yaml --pascal PASCAL_MT --tasks semseg,normals,sal,human_parts --batch-size 8 --ckpt-freq=20 --eval-freq=5 --epochs=100 --resume output/promlora_base_448_r64_prom60_scale4_pertask/default/ckpt_epoch_200.pth
```

适用场景：在中间轮次 checkpoint 基础上继续训练或短程微调。

### D. NYUDv2：冻结 backbone 训练 Decoder/任务头

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29501 main.py --cfg configs/swin/tiny_448/nyud/swin_tiny_patch4_window7_448_nyud_alltask_decoder.yaml --freeze-backbone --nyud NYUV2 --tasks semseg,normals,depth --batch-size 32 --ckpt-freq=20 --eval-freq=5 --epochs=300 --resume-backbone ./backbone/swin_tiny_patch4_window7_224.pth
```

适用场景：先固定特征提取器，验证解码器或任务头设计。

### E. PASCAL-MT：Base 模型从 backbone 开始训练

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29501 main.py --cfg configs/mtlora/base_448/pascal/promlora_base_448_r64_prom60_scale4_pertask.yaml --pascal PASCAL_MT --tasks semseg,normals,sal,human_parts --batch-size 16 --ckpt-freq=20 --eval-freq=5 --epochs=300 --resume-backbone backbone/swin_base_patch4_window7_224.pth
```

适用场景：Base 规模模型完整训练。

---

## 各脚本文件使用方式

### 1) `main.py`：统一训练/评估入口

常见模式：

- **训练（从 backbone 启动）**：`--resume-backbone` + 不加 `--eval`
- **继续训练（从 checkpoint 恢复）**：`--resume <ckpt>` + 不加 `--eval`
- **仅评估**：`--resume <ckpt> --eval`

最小训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29511 main.py --cfg configs/mtlora/tiny_448/mtlora_tiny_448_r64_scale4_pertask_maml.yaml --pascal PASCAL_MT --tasks semseg,normals,sal,human_parts --batch-size 8 --epochs 300 --resume-backbone ./backbone/swin_tiny_patch4_window7_224.pth
```

### 2) `compute_delta_m.py`：从日志计算 Δm（推荐）

#### PASCAL 四任务示例

```bash
python compute_delta_m.py \
--tasks semseg,human,saliency,normals \
--log-file output/swin_base_patch4_window7_448_PASCAL_alltask/default/log_rank0.txt \
--semseg-st 67.21 \
--human-st 61.93 \
--saliency-st 62.35 \
--normals-st 17.97 \
--csv-out csv/swin_base_patch4_window7_448_PASCAL_alltask.csv
```

#### NYUD 三任务示例

```bash
python compute_delta_m.py \
--tasks semseg,normals,depth \
--log-file output/promlora_tiny_448_r64_prom60_scale4_pertask_nyud/default/log_rank0.txt \
--semseg-st 40.65 \
--normals-st 22.97 \
--depth-st 0.64 \
--csv-out csv/promlora_tiny_448_r64_prom60_scale4_pertask_nyud.csv
```

说明：
- `--tasks` 与提供的 `*-st` 单任务基线要一致。
- `--csv-out` 可导出每次 eval 的指标和 Δm，便于画曲线。

### 3) `calcu_stats.py`：快速手动统计（简单脚本）

```bash
python calcu_stats.py
```

该脚本目前通过修改文件中固定变量 `a, b, c, d` 来计算均值和方差，适合临时快速验证。

### 4) `scripts/prepare_nyudv2.py`：NYUDv2 数据预处理

```bash
python scripts/prepare_nyudv2.py
```

用于 NYUDv2 数据准备流程，建议先阅读脚本内注释并确认输入/输出目录。

### 5) `scripts/inspect_mat.py`：检查 `.mat` 文件内容

```bash
python scripts/inspect_mat.py
```

适合排查 NYUD 或其他 mat 数据文件结构。

### 6) `scripts/repro_diagnose.py`：复现实验诊断辅助

```bash
python scripts/repro_diagnose.py
```

用于复现性排查（如随机种子、日志对比等）。

### 7) 你的标签统计命令（若本地有 `scripts/label_stats.py`）

```bash
python scripts/label_stats.py \
  --cfg configs/mtlora/tiny_448/promlora_tiny_448_r64_prom60_scale4_pertask.yaml \
  --nyud NYUV2 \
  --tasks semseg,depth,normals \
  --max-batches 50 \
  --split train
```

> 说明：当前仓库默认文件列表中未包含 `scripts/label_stats.py`。如果这是你的本地自定义脚本，可保留该命令用于统计标签分布。

---

## 预训练模型评估

你可以下载模型权重：[Google Drive](https://drive.google.com/file/d/1AzzOgX6X0VFKyXUBXhwlgmba5NbPUq3m/view?usp=drive_link)

假设权重在 `./mtlora.pth`，评估方式：

```bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29502 main.py --cfg configs/mtlora/tiny_448/mtlora_tiny_448_r64_scale4_pertask.yaml --pascal PASCAL_MT --tasks semseg,normals,sal,human_parts --batch-size 32 --resume ./mtlora.pth --eval
```

---

## Authorship

Since the release commit is squashed, the GitHub contributors tab doesn't reflect the authors' contributions. The following authors contributed equally to this codebase:

- [Ahmed Agiza](https://github.com/ahmed-agiza)
- [Marina Neseem](https://github.com/marina-neseem)

## Citation

If you find MTLoRA helpful in your research, please cite our paper:

```bibtex
@inproceedings{agiza2024mtlora,
  title={MTLoRA: Low-Rank Adaptation Approach for Efficient Multi-Task Learning},
  author={Agiza, Ahmed and Neseem, Marina and Reda, Sherief},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16196--16205},
  year={2024}
}
```

## License

MIT License. See [LICENSE](LICENSE) file.
