#!/usr/bin/env python3
"""
swin transformer with prompt
"""
import math
import logging
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn import Conv2d, Dropout

from timm.models.layers import to_2tuple

from .swin_transformer_mtlora import (
    SwinTransformerMTLoRA,
    SwinTransformerBlock,
    PatchMerging,
    window_partition,
    window_reverse,
    WindowAttention,
)

logger = logging.getLogger("visual_prompt")


class PromptedSwinTransformer(SwinTransformerMTLoRA):
    def __init__(
            self,
            prompt_config,
            tasks,
            mtlora,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            **kwargs,
    ):
        if prompt_config.LOCATION == "pad":
            img_size += 2 * prompt_config.NUM_TOKENS

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            fused_window_process=kwargs.get("fused_window_process", False),
            tasks=tasks,
            mtlora=mtlora,
        )

        # rebuild layers with prompt-aware blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PromptedBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PromptedPatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                num_prompts=prompt_config.NUM_TOKENS,
                prompt_location=prompt_config.LOCATION,
                deep_prompt=prompt_config.DEEP,
                tasks=tasks,
                mtlora=mtlora,
                layer_idx=i_layer,
            )
            self.layers.append(layer)

        self.prompt_config = prompt_config
        self.tasks = tasks
        self.mtlora = mtlora
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        if self.prompt_config.LOCATION == "add":
            num_tokens = self.embeddings.position_embeddings.shape[1]
        elif self.prompt_config.LOCATION == "add-1":
            num_tokens = 1
        else:
            num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)
        if self.prompt_config.PROJECT > -1:
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, embed_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
        else:
            self.prompt_proj = nn.Identity()

        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + embed_dim))

            if self.prompt_config.LOCATION == "below":
                self.patch_embed.proj = Conv2d(
                    in_channels=num_tokens + 3,
                    out_channels=embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
                nn.init.zeros_(self.patch_embed.proj.bias)

                self.prompt_embeddings = nn.ParameterDict()
                for task in tasks:
                    emb = torch.zeros(1, num_tokens, img_size[0], img_size[1])
                    nn.init.uniform_(emb, -val, val)
                    self.prompt_embeddings[task] = nn.Parameter(emb)

            elif self.prompt_config.LOCATION == "pad":
                self.prompt_embeddings_tb = nn.ParameterDict()
                self.prompt_embeddings_lr = nn.ParameterDict()
                for task in tasks:
                    emb_tb = torch.zeros(1, 3, 2 * num_tokens, img_size[0])
                    emb_lr = torch.zeros(1, 3, img_size[0] - 2 * num_tokens, 2 * num_tokens)
                    nn.init.uniform_(emb_tb, 0.0, 1.0)
                    nn.init.uniform_(emb_lr, 0.0, 1.0)
                    self.prompt_embeddings_tb[task] = nn.Parameter(emb_tb)
                    self.prompt_embeddings_lr[task] = nn.Parameter(emb_lr)

                self.prompt_norm = tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )

            else:
                self.prompt_embeddings = nn.ParameterDict()
                for task in tasks:
                    emb = torch.zeros(1, num_tokens, embed_dim)
                    nn.init.uniform_(emb, -val, val)
                    self.prompt_embeddings[task] = nn.Parameter(emb)

                if self.prompt_config.DEEP:
                    self.deep_prompt_embeddings_0 = nn.ParameterDict()
                    self.deep_prompt_embeddings_1 = nn.ParameterDict()
                    self.deep_prompt_embeddings_2 = nn.ParameterDict()
                    self.deep_prompt_embeddings_3 = nn.ParameterDict()
                    for task in tasks:
                        emb0 = torch.zeros(depths[0] - 1, num_tokens, embed_dim)
                        emb1 = torch.zeros(depths[1], num_tokens, embed_dim * 2)
                        emb2 = torch.zeros(depths[2], num_tokens, embed_dim * 4)
                        emb3 = torch.zeros(depths[3], num_tokens, embed_dim * 8)
                        nn.init.uniform_(emb0, -val, val)
                        nn.init.uniform_(emb1, -val, val)
                        nn.init.uniform_(emb2, -val, val)
                        nn.init.uniform_(emb3, -val, val)
                        self.deep_prompt_embeddings_0[task] = nn.Parameter(emb0)
                        self.deep_prompt_embeddings_1[task] = nn.Parameter(emb1)
                        self.deep_prompt_embeddings_2[task] = nn.Parameter(emb2)
                        self.deep_prompt_embeddings_3[task] = nn.Parameter(emb3)
        else:
            raise ValueError("Other initiation scheme is not supported")

    def forward(self, x, task=None, return_stages=False, flatten_ft=False):
        x = self.forward_features(x, task, return_stages)
        if return_stages:
            return x
        x = self.head(x)
        if flatten_ft:
            return x
        return x

    def incorporate_prompt(self, x, task):
        """Combine prompt embeddings with image-patch embeddings for a task."""
        B = x.shape[0]

        if self.prompt_config.LOCATION == "prepend":
            x = self.get_patch_embeddings(x)
            prompt_embd = self.prompt_dropout(
                self.prompt_embeddings[task].expand(B, -1, -1)
            )
            x = torch.cat((prompt_embd, x), dim=1)

        elif self.prompt_config.LOCATION == "add":
            x = self.get_patch_embeddings(x)
            x = x + self.prompt_dropout(
                self.prompt_embeddings[task].expand(B, -1, -1)
            )

        elif self.prompt_config.LOCATION == "add-1":
            x = self.get_patch_embeddings(x)
            L = x.shape[1]
            prompt_emb = self.prompt_dropout(
                self.prompt_embeddings[task].expand(B, -1, -1)
            )
            x = x + prompt_emb.expand(-1, L, -1)

        elif self.prompt_config.LOCATION == "pad":
            prompt_emb_lr = self.prompt_norm(
                self.prompt_embeddings_lr[task]
            ).expand(B, -1, -1, -1)
            prompt_emb_tb = self.prompt_norm(
                self.prompt_embeddings_tb[task]
            ).expand(B, -1, -1, -1)
            x = torch.cat(
                (
                    prompt_emb_lr[:, :, :, : self.num_tokens],
                    x,
                    prompt_emb_lr[:, :, :, self.num_tokens :],
                ),
                dim=-1,
            )
            x = torch.cat(
                (
                    prompt_emb_tb[:, :, : self.num_tokens, :],
                    x,
                    prompt_emb_tb[:, :, self.num_tokens :, :],
                ),
                dim=-2,
            )
            x = self.get_patch_embeddings(x)

        elif self.prompt_config.LOCATION == "below":
            x = torch.cat(
                (
                    x,
                    self.prompt_norm(self.prompt_embeddings[task]).expand(
                        B, -1, -1, -1
                    ),
                ),
                dim=1,
            )
            x = self.get_patch_embeddings(x)
        else:
            raise ValueError("Other prompt locations are not supported")

        return x

    def get_patch_embeddings(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def train(self, mode=True):
        if mode and not getattr(self.mtlora, "ENABLED", False):
            for module in self.children():
                module.train(False)
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x, task, return_stages=False):
        x = self.incorporate_prompt(x, task)

        feats = []
        if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:
            for layer, deep_prompt_embd_dict in zip(
                    self.layers,
                    [
                        self.deep_prompt_embeddings_0,
                        self.deep_prompt_embeddings_1,
                        self.deep_prompt_embeddings_2,
                        self.deep_prompt_embeddings_3,
                    ],
            ):
                deep_prompt_embd = self.prompt_dropout(deep_prompt_embd_dict[task])
                x = layer(x, task, deep_prompt_embd)
                if return_stages:
                    feats.append(x)
        else:
            for layer in self.layers:
                x = layer(x, task)
                if return_stages:
                    feats.append(x)

        x = self.norm(x)
        if return_stages:
            return feats
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def load_state_dict(self, state_dict, strict):
        if self.prompt_config.LOCATION == "below":
            # modify state_dict first   [768, 4, 16, 16]
            conv_weight = state_dict["patch_embed.proj.weight"]
            conv_weight = torch.cat(
                (conv_weight, self.patch_embed.proj.weight[:, 3:, :, :]),
                dim=1
            )
            state_dict["patch_embed.proj.weight"] = conv_weight

        super(PromptedSwinTransformer, self).load_state_dict(state_dict, strict)


class PromptedBasicLayer(nn.Module):
    def __init__(
            self,
            dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            num_prompts=None,
            prompt_location=None,
            deep_prompt=None,
            tasks=None,
            mtlora=None,
            layer_idx=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.tasks = tasks
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        self.deep_prompt = deep_prompt

        self.blocks = nn.ModuleList([
            PromptedSwinTransformerBlock(
                num_prompts,
                prompt_location,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                tasks=tasks,
                mtlora=mtlora,
                layer_idx=layer_idx,
                lora=(i == depth - 1),
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(
                num_prompts,
                prompt_location,
                deep_prompt,
                input_resolution,
                dim=dim,
                norm_layer=norm_layer,
                layer_idx=layer_idx,
                mtlora=mtlora,
            )
        else:
            self.downsample = None

    def forward(self, x, task, deep_prompt_embd=None):
        if self.deep_prompt and deep_prompt_embd is None and self.prompt_location == "prepend":
            raise ValueError("need deep_prompt embeddings")

        if not self.deep_prompt:
            for blk in self.blocks:
                x, tasks_lora = blk(x)
                if tasks_lora is not None:
                    x = tasks_lora[task]
        else:
            B = x.shape[0]
            num_blocks = len(self.blocks)
            if deep_prompt_embd.shape[0] != num_blocks:
                for i in range(num_blocks):
                    if i == 0:
                        x, tasks_lora = self.blocks[i](x)
                        if tasks_lora is not None:
                            x = tasks_lora[task]
                    else:
                        prompt_emb = deep_prompt_embd[i - 1].expand(B, -1, -1)
                        x = torch.cat((prompt_emb, x[:, self.num_prompts:, :]), dim=1)
                        x, tasks_lora = self.blocks[i](x)
                        if tasks_lora is not None:
                            x = tasks_lora[task]
            else:
                for i in range(num_blocks):
                    prompt_emb = deep_prompt_embd[i].expand(B, -1, -1)
                    x = torch.cat((prompt_emb, x[:, self.num_prompts:, :]), dim=1)
                    x, tasks_lora = self.blocks[i](x)
                    if tasks_lora is not None:
                        x = tasks_lora[task]

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PromptedPatchMerging(PatchMerging):
    r""" Patch Merging Layer."""

    def __init__(
            self,
            num_prompts,
            prompt_location,
            deep_prompt,
            input_resolution,
            dim,
            norm_layer=nn.LayerNorm,
            layer_idx=0,
            mtlora=None,
    ):
        super().__init__(input_resolution, dim, norm_layer, layer_idx=layer_idx, mtlora=mtlora)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if prompt_location == "prepend":
            if not deep_prompt:
                self.prompt_upsampling = None
            else:
                self.prompt_upsampling = None

    def upsample_prompt(self, prompt_emb):
        if self.prompt_upsampling is not None:
            prompt_emb = self.prompt_upsampling(prompt_emb)
        else:
            prompt_emb = torch.cat(
                (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        return prompt_emb

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]
            L = L - self.num_prompts
            prompt_emb = self.upsample_prompt(prompt_emb)

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H*W, L)
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # add the prompt back:
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.norm(x)
        x, _ = self.reduction(x)

        return x


class PromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
            self,
            num_prompts,
            prompt_location,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            tasks=None,
            mtlora=None,
            layer_idx=0,
            lora=False,
    ):
        super().__init__(
            dim,
            input_resolution,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
            fused_window_process=False,
            lora=lora,
            tasks=tasks,
            mtlora=mtlora,
            layer_idx=layer_idx,
        )
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if self.prompt_location == "prepend":
            self.attn = PromptedWindowAttention(
                num_prompts,
                prompt_location,
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                tasks=tasks,
                mtlora=mtlora,
                layer_idx=layer_idx,
                lora=lora,
            )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.prompt_location == "prepend":
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]
            L = L - self.num_prompts

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H * W, L)

        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        num_windows = int(x_windows.shape[0] / B)
        if self.prompt_location == "prepend":
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
            x_windows = torch.cat((prompt_emb, x_windows), dim=1)

        attn_windows, attn_windows_lora_tasks = self.attn(x_windows, mask=self.attn_mask)

        if self.prompt_location == "prepend":
            prompt_emb = attn_windows[:, :self.num_prompts, :]
            attn_windows = attn_windows[:, self.num_prompts:, :]
            prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
            prompt_emb = prompt_emb.mean(0)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)
        x = shortcut + self.drop_path(x)
        mlp_result, mlp_lora_tasks = self.mlp(self.norm2(x), attn_windows_lora_tasks)
        x = x + self.drop_path(mlp_result)
        return x, mlp_lora_tasks


class PromptedWindowAttention(WindowAttention):
    def __init__(
            self,
            num_prompts,
            prompt_location,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            tasks=None,
            mtlora=None,
            layer_idx=0,
            lora=False,
    ):
        super().__init__(
            dim,
            window_size,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            lora=lora,
            tasks=tasks,
            mtlora=mtlora,
            layer_idx=layer_idx,
        )
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv, _ = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        if self.prompt_location == "prepend":
            _C, _H, _W = relative_position_bias.shape
            relative_position_bias = torch.cat(
                (torch.zeros(_C, self.num_prompts, _W, device=attn.device), relative_position_bias), dim=1
            )
            relative_position_bias = torch.cat(
                (torch.zeros(_C, _H + self.num_prompts, self.num_prompts, device=attn.device), relative_position_bias), dim=-1
            )

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            if self.prompt_location == "prepend":
                mask = torch.cat(
                    (torch.zeros(nW, self.num_prompts, _W, device=attn.device), mask), dim=1
                )
            if self.prompt_location == "prepend":
                mask = torch.cat(
                    (
                        torch.zeros(
                            nW, _H + self.num_prompts, self.num_prompts, device=attn.device
                        ),
                        mask,
                    ),
                    dim=-1,
                )
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x, x_proj_lora_tasks = self.proj(x)
        x = self.proj_drop(x)
        if x_proj_lora_tasks is not None:
            for task in self.tasks:
                x_proj_lora_tasks[task] = self.proj_drop(x_proj_lora_tasks[task])
        return x, x_proj_lora_tasks
