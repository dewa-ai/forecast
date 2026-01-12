# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Optional, Dict, Any
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# @dataclass
# class LLMConfig:
#     hf_id: str
#     use_4bit: bool = True
#     max_new_tokens: int = 256
#     temperature: float = 0.0
#     top_p: float = 1.0
#     repetition_penalty: float = 1.05

# class LLMForecaster:
#     def __init__(self, cfg: LLMConfig):
#         self.cfg = cfg
#         self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, use_fast=True)

#         quant = None
#         if cfg.use_4bit:
#             quant = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#             )

#         self.model = AutoModelForCausalLM.from_pretrained(
#             cfg.hf_id,
#             device_map="auto",
#             torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#             quantization_config=quant,
#         )
#         self.model.eval()

#     @torch.inference_mode()
#     def generate(self, prompt: str) -> str:
#         # Many instruct models work better with chat template if available
#         if hasattr(self.tokenizer, "apply_chat_template"):
#             messages = [{"role": "user", "content": prompt}]
#             input_ids = self.tokenizer.apply_chat_template(
#                 messages, add_generation_prompt=True, return_tensors="pt"
#             )
#             input_ids = input_ids.to(self.model.device)
#         else:
#             input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

#         gen = self.model.generate(
#             input_ids=input_ids,
#             max_new_tokens=self.cfg.max_new_tokens,
#             do_sample=(self.cfg.temperature > 0),
#             temperature=self.cfg.temperature if self.cfg.temperature > 0 else None,
#             top_p=self.cfg.top_p,
#             repetition_penalty=self.cfg.repetition_penalty,
#             eos_token_id=self.tokenizer.eos_token_id,
#         )
#         out = self.tokenizer.decode(gen[0], skip_special_tokens=True)

#         # Try to remove prompt echo
#         if out.startswith(prompt):
#             out = out[len(prompt):].strip()
#         return out.strip()


#-------------------------------------------------------------------------------
# llm.py - revised
#-------------------------------------------------------------------------------

# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# _NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# class HFLLM:
#     def __init__(self, model_name: str):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             device_map="auto",
#         )
#         self.model.eval()

#     @torch.inference_mode()
#     def generate_number(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 64) -> float:
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

#         do_sample = temperature is not None and temperature > 1e-6
#         out = self.model.generate(
#             **inputs,
#             do_sample=do_sample,
#             temperature=float(temperature) if do_sample else None,
#             max_new_tokens=int(max_new_tokens),
#             pad_token_id=self.tokenizer.eos_token_id,
#         )
#         text = self.tokenizer.decode(out[0], skip_special_tokens=True)

#         # Try to parse the last number in the output
#         nums = _NUM_RE.findall(text)
#         if not nums:
#             raise ValueError(f"Cannot parse numeric output from model. Output was: {text[-200:]}")
#         return float(nums[-1])



#-------------------------------------------------------------------------------
# llm.py - final
#-------------------------------------------------------------------------------

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

class LLM:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, prompt, temperature=0.0, max_new_tokens=64):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        nums = NUM_RE.findall(text)
        if not nums:
            raise ValueError("No numeric output from LLM")
        return float(nums[-1])
