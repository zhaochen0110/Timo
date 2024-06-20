# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import pathlib
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother


from trl import DPOTrainer
from datasets import load_dataset
from functools import partial
import pdb

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

Instruction = '''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
'''


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attn: bool = False


@dataclass
class DataArguments:
    data_id: str = field(
        default = None, metadata = {"help": "Dataset id name of the training data."}
    )
    
    data_split: str = field(
        default = None, metadata = {"help": "Chosen split of the training data."}
    )
    
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    
    cache_path: str = field(
        default=None, metadata={"help": "Path to cache the training data."}
    )
    
    num_proc: int = field(
        default=32
    )
    
    
    json_path: str = field(
        default = None, metadata = {"help": "Path to the json file containing the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    beta: float = field(default = 0.1, metadata = {
        "help": "Control the deviation from the reference model."
    })
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    min_lr: float = field(
        default = None
    )
    mask_user: bool = field(
        default = True    
    )
    
    save_global_steps: bool = field(
        default = True
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def preprocess(
    sample
) -> Dict:
    
    prompt = Instruction.format(sample["prompt"])
    
    # Apply prompt templates
    chosen_conversations = sample["chosen"]

    rejected_conversations = sample["rejected"]
    
    return dict(
        prompt=prompt,
        chosen=chosen_conversations,
        rejected=rejected_conversations,
    )


def make_dpo_dataset(
    data_args: DataArguments,
    sanity_check: bool = False
) -> Dataset:
    """
    Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
        "Question: " + <prompt> + "\n\nAnswer: "
    """
    
    data_id: str = data_args.data_id
    data_split: str = data_args.data_split
    data_dir: str = data_args.data_path
    cache_dir: str = data_args.cache_path
    num_proc: int = data_args.num_proc
    
    json_path: str = data_args.json_path
    
    if not json_path:
        dataset = load_dataset(
            data_id,
            split=data_split,
            cache_dir=cache_dir,
            data_dir=data_dir,
        )
    else:
        dataset = load_dataset(
            "json",
            data_files = {"train": json_path}
        )
        


    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    
    return dataset.map(
        preprocess,
        num_proc=num_proc
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    """DPO Training"""
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.do_eval = False
    local_rank = training_args.local_rank
    
    # print("Load Model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2 = True
    )
    model.config.use_cache = False
    
    # print("Load Refer Model")
    model_refer = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2 = True
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    tokenizer.pad_token = tokenizer.unk_token
    train_dataset = make_dpo_dataset(data_args=data_args)
    
    trainer = DPOTrainer(
        model, model_refer, tokenizer = tokenizer, beta = training_args.beta, args=training_args, train_dataset = train_dataset['train'],
        max_prompt_length = 512, max_length = 2048
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Checkpoint found, resuming training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    trainer.save_state()
    trainer.save_model()
    

if __name__ == "__main__":
    train()