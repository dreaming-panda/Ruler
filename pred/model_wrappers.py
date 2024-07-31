# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional
CHUNK_SIZE = 4096

class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from transformers import LlamaForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        
        self.pipeline = None
        self.model = LlamaForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, **model_kwargs)
        # self.model.config.K = 12
        # self.model.config.L = 300
        # self.model.config.window = 16
        # self.model.config.cache_mode = "anns"
        # self.model.eval()
        # self.model.set_sparse_attn(sparse=0.01, window_size=16, kernel_size=5, random_sparse=0.1, vsparse=1.0)
    
        # from snapkv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral
        # replace_llama()
            
        # self.model = LlamaForCausalLM.from_pretrained(
        #     name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16,
        #     use_flash_attention_2=True
        # )
        # self.model.config.max_capacity_prompt = 1024
        #self.model.config.ratio = sparse
        #self.model.config.window_size = window_size
        #self.model.config.pooling = "maxpool"
        #self.model.eval()
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            seq_len = inputs["input_ids"].shape[1]
            num_chunk = (seq_len // CHUNK_SIZE - 1) if (seq_len % CHUNK_SIZE == 0) else (seq_len // CHUNK_SIZE)
            past_key_values = None
            with torch.inference_mode():
                for chunk_id in range(num_chunk):
                    outputs = self.model(input_ids=inputs["input_ids"][:,chunk_id * CHUNK_SIZE : (chunk_id + 1) * CHUNK_SIZE],
                                        past_key_values=past_key_values,
                                        use_cache=True)
                    past_key_values = outputs.past_key_values
            input_ids = inputs["input_ids"]

            output = self.model.generate(
                input_ids=input_ids,
                past_key_values=past_key_values,
                **self.generation_kwargs
            )
            generated_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompt, **self.generation_kwargs,)
            assert len(output) == 1
            generated_text = output[0]["generated_text"]
            
        # remove the input form the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :]
                
        if self.stop is not None:
            for s in self.stop:
                generated_text = generated_text.split(s)[0]
        return {'text': [generated_text]}


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1] :])]}