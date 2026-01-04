import argparse
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from data.mtl_ds import PASCALContext, collate_mil, get_tasks_config, get_transformations
from models.swin_transformer import SwinTransformer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Hook:
	def __init__(self, module, backward=False):
		if backward is False:
			self.hook = module.register_forward_hook(self.hook_fn)
		else:
			self.hook = module.register_backward_hook(self.hook_fn)

	def hook_fn(self, module, input, output):
		self.output = input[0]

	def close(self):
		self.hook.remove()


def _reduce_activation_stats(activation):
	reduce_dims = tuple(range(activation.ndim - 1))
	if reduce_dims:
		beta = activation.mean(dim=reduce_dims)
		gamma = activation.std(dim=reduce_dims, unbiased=False)
	else:
		beta = activation
		gamma = torch.zeros_like(beta)
	return beta, gamma


def _compute_importance(activation):
	beta, gamma = _reduce_activation_stats(activation)
	normal_part = 0.5 * (1 + torch.erf((0 - beta) / (gamma * np.sqrt(2))))
	zero_gamma_condition = gamma == 0
	cdf_current = torch.where(zero_gamma_condition, (0 >= beta).float(), normal_part)
	entropy_current = -torch.mean(
		cdf_current * torch.log2(torch.clamp(cdf_current, min=1e-5))
		+ (1 - cdf_current) * torch.log2(torch.clamp(1 - cdf_current, min=1e-5))
	)
	mean_abs_beta = beta.abs().mean()
	return mean_abs_beta.item(), entropy_current.item()


def load_backbone(checkpoint_path):
	model = SwinTransformer()
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	if isinstance(checkpoint, dict) and 'model' in checkpoint:
		state_dict = checkpoint['model']
	elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
		state_dict = checkpoint['state_dict']
	else:
		state_dict = checkpoint
	missing, unexpected = model.load_state_dict(state_dict, strict=False)
	if missing:
		print(f"Missing keys when loading checkpoint: {missing}")
	if unexpected:
		print(f"Unexpected keys when loading checkpoint: {unexpected}")
	return model


def _task_flags(task_name):
	flags = {
		'do_semseg': task_name == 'semseg',
		'do_human_parts': task_name == 'human_parts',
		'do_sal': task_name == 'sal',
		'do_normals': task_name == 'normals',
	}
	return flags


def build_task_loader(data_root, task_name, batch_size, num_workers, img_size, split):
	task_cfg, _ = get_tasks_config('PASCALContext', [task_name], img_size)
	_, transforms_ts = get_transformations('PASCALContext', task_cfg)
	flags = _task_flags(task_name)
	dataset = PASCALContext(
		root=data_root,
		split=[split],
		transform=transforms_ts,
		retname=False,
		**flags,
	)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		collate_fn=collate_mil,
		pin_memory=torch.cuda.is_available(),
	)


def compute_linear_importance(model, data_loader):
	model.eval()
	hooks = {}
	for name, module in model.named_modules():
		if isinstance(module, nn.Linear):
			hooks[name] = Hook(module)
	with torch.no_grad():
		for batch in tqdm(data_loader, desc="Collecting activations"):
			inputs = batch['image'].to(device)
			model(inputs)
			break
	results = {}
	for name, hook in hooks.items():
		if not hasattr(hook, 'output'):
			continue
		mean_abs_beta, entropy = _compute_importance(hook.output)
		results[name] = {
			'mean_abs_beta': mean_abs_beta,
			'entropy': entropy,
		}
	for hook in hooks.values():
		hook.close()
	return results


def main():
	parser = argparse.ArgumentParser(description='Swin-T Linear Importance (PASCAL_MT)')
	parser.add_argument('--data-root', default='PASCAL_MT', help='PASCAL_MT dataset root')
	parser.add_argument('--backbone', default='backbone/swin_tiny_patch4_window7_224.pth',
						help='Path to Swin-Tiny checkpoint')
	parser.add_argument('--tasks', default='semseg,human_parts,saliency,normals',
						help='Comma-separated tasks: semseg,human_parts,saliency,normals')
	parser.add_argument('--batch-size', type=int, default=4, help='Batch size for importance estimation')
	parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers')
	parser.add_argument('--img-size', type=int, default=224, help='Input image size')
	parser.add_argument('--split', default='val', help='Dataset split (train/val)')
	args = parser.parse_args()

	task_map = {
		'semseg': 'semseg',
		'human_parts': 'human_parts',
		'saliency': 'sal',
		'sal': 'sal',
		'normal': 'normals',
		'normals': 'normals',
	}
	task_list = []
	for task in args.tasks.split(','):
		task_key = task.strip().lower()
		if task_key not in task_map:
			raise ValueError(f"Unknown task: {task}")
		task_list.append(task_map[task_key])

	model = load_backbone(args.backbone).to(device)
	for task in task_list:
		print(f"\n=== Task: {task} ===")
		loader = build_task_loader(
			args.data_root,
			task,
			args.batch_size,
			args.num_workers,
			args.img_size,
			args.split,
		)
		importance = compute_linear_importance(model, loader)
		sorted_layers = sorted(
			importance.items(),
			key=lambda item: item[1]['mean_abs_beta'],
			reverse=True,
		)
		print("Layer importance (sorted by mean |beta|):")
		for name, stats in sorted_layers:
			print(f"{name}\tmean_abs_beta={stats['mean_abs_beta']:.6f}\tentropy={stats['entropy']:.6f}")


if __name__ == '__main__':
	main()
