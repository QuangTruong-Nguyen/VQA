from transformers import TrainingArguments, Trainer
from safetensors.torch import save_file
import torch
import os
from typing import List, Optional, Union

def setup_training():
    return TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=10,
        dataloader_pin_memory=True,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=20,
        output_dir="./",
        evaluation_strategy="steps",
        eval_steps=20,
        # push_to_hub=True,
        # hub_model_id='VQA2',
        # hub_token='YOUR_HUB_TOKEN',
        remove_unused_columns=False,
        report_to="wandb"
    )

# class CustomTrainer(Trainer):
#     def save_model(self, output_dir=None, _internal_call=False):
#         if output_dir is None:
#             output_dir = self.args.output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         state_dict = self.model.state_dict()
#         new_state_dict = {name: tensor.clone() if tensor.is_shared() else tensor for name, tensor in state_dict.items()}
#         save_file(new_state_dict, f"{output_dir}/model.safetensors")
#         if hasattr(self.model, 'config'):
#             self.model.config.save_pretrained(output_dir)
#         if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'save_pretrained'):
#             self.tokenizer.save_pretrained(output_dir)


class CustomTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy the model state dict
        state_dict = self.model.state_dict()
        new_state_dict = {}
        
        for name, tensor in state_dict.items():
            # Clone the tensor if it shares memory
            if tensor.is_shared():
                new_state_dict[name] = tensor.clone()
            else:
                new_state_dict[name] = tensor
        
        # Save the model state dict using safetensors
        save_file(new_state_dict, f"{output_dir}/model.safetensors")
        
        # If the model has a config, save it
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(output_dir)
        
        # If the tokenizer exists and has the save_pretrained method, save it
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(output_dir)