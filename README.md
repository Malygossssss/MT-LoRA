# MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning

## Introduction

This is the official implementation of the paper: **MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning** developed at [Brown University SCALE lab](https://scale-lab.github.io).

This repository provides a Python-based implementation of MTLoRA including [`MTLoRALinear`](models/lora.py) (the main module) and MTL architectures. 

The repository is built on top of [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and uses some modules from [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch).


## How to Run

Running MTLoRA code, is very simmilar to Swin's codebase:

1. **Clone the repository**
    ```bash
    git clone https://github.com/scale-lab/MTLoRA.git
    cd MTLoRA
    ```

2. **Install the prerequisites**
    - Install `PyTorch>=1.12.0` and `torchvision>=0.13.0` with `CUDA>=11.6`
    - Install dependencies: `pip install -r requirements.txt`

3. **Run the code**
    ```python
    CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node 2 --master_port 29501 main.py --cfg configs/mtlora/tiny_448/mtlora_tiny_448_r64_scale4_pertask_maml.yaml --pascal PASCAL_MT --tasks semseg,normals,sal,human_parts --batch-size 32 --ckpt-freq=20 --eval-freq=10 --epoch=300 --resume-backbone ./backbone/swin_tiny_patch4_window7_224.pth
    ```
    Swin variants and their weights can be found at the official [Swin Transformer repository](https://github.com/microsoft/Swin-Transformer). 
  
    The outputs will be saved in `output/` folder unless overridden by the argument `--output`.

4. **Using the pre-trained model**

    You can download the model weights from the following [link](https://drive.google.com/file/d/1AzzOgX6X0VFKyXUBXhwlgmba5NbPUq3m/view?usp=drive_link).

    To run and evaluate the pre-trained model (assuming the model weight file is at `./mtlora.pth`), use `--eval` and `--resume <checkpoint>` as follows:
    ```python
    CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29502 main.py --cfg configs/mtlora/tiny_448/mtlora_tiny_448_r64_scale4_pertask.yaml --pascal PASCAL_MT --tasks semseg,normals,sal,human_parts --batch-size 32 --resume ./mtlora.pth --eval
    ```
   
5. **通过脚本计算Δm**
   ```python
   python compute_delta_m.py \
   --log-file /sde/wyf/project/MTLoRA-main/output/mtlora_tiny_448_r64_scale4_pertask_cotodrop/default/log_rank0.txt \
   --semseg-st 67.21 \
   --human-st 61.93 \
   --saliency-st 62.35 \
   --normals-st 17.97 \
   --csv-out delta_m_history.csv
   ```

  
## Authorship
Since the release commit is squashed, the GitHub contributors tab doesn't reflect the authors' contributions. The following authors contributed equally to this codebase:
- [Ahmed Agiza](https://github.com/ahmed-agiza)
- [Marina Neseem](https://github.com/marina-neseem)

## Citation
If you find MTLoRA helpful in your research, please cite our paper:
```
@inproceedings{agiza2024mtlora,
  title={MTLoRA: Low-Rank Adaptation Approach for Efficient Multi-Task Learning},
  author={Agiza, Ahmed and Neseem, Marina and Reda, Sherief},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16196--16205},
  year={2024}
}
```

## License
MIT License. See [LICENSE](LICENSE) file


