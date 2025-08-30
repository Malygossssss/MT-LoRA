from transformers import Trainer
from coto_callback import CoToSchedulerCallback


def get_all_loras(model):
    """收集模型中所有 LoRA 层"""
    return [module for _, module in model.named_modules() if hasattr(module, "lora_type")]


def build_trainer(model, training_args, train_data, eval_data):
    """构建带有 CoToSchedulerCallback 的 Trainer"""
    loras = get_all_loras(model)
    callbacks = [CoToSchedulerCallback(loras, initial_p=0.0, final_p=1.0, stage1_ratio=0.75)]
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=callbacks,
    )