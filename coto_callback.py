from transformers import TrainerCallback
import torch, math


class CoToSchedulerCallback(TrainerCallback):
    def __init__(self, adapter_modules, initial_p=0.0, final_p=1.0, stage1_ratio=0.75):
        self.loras = adapter_modules
        self.initial_p = initial_p
        self.final_p = final_p
        self.stage1_ratio = stage1_ratio
        self.total_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        self.update_dropout_rate(self.initial_p)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        end_step = math.ceil(self.total_steps * self.stage1_ratio)
        rate = self.initial_p + (self.final_p - self.initial_p) * (step / end_step)
        self.update_dropout_rate(min(rate, self.final_p))

    def update_dropout_rate(self, rate):
        ta_loras = [l for l in self.loras if getattr(l, "lora_type", "TA") == "TA"]
        ts_loras = [l for l in self.loras if getattr(l, "lora_type", "TA") == "TS"]

        # TS-LoRA 永远开
        for lora in ts_loras:
            lora.cotodrop = False

        # TA-LoRA 按概率
        active_flags = []
        for lora in ta_loras:
            drop = torch.rand(1).item() > rate
            lora.cotodrop = drop
            active_flags.append(not drop)

        # 保证至少有一个 TA-LoRA 激活
        if ta_loras and not any(active_flags):
            idx = torch.randint(len(ta_loras), (1,)).item()
            ta_loras[idx].cotodrop = False