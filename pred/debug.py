import torch
from llama_sim_new import LlamaForCausalLM

inputs = torch.load("../inputs.pt")
generation = torch.load("../generation.pt")
CHUNK_SIZE = 1024
model:LlamaForCausalLM = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True,  torch_dtype=torch.bfloat16, _attn_implementation="eager").cuda(0)
model.config.K = 11
model.config.L = 120
model.config.window = 16
model.config.cache_mode = "anns"
model.eval()
model.set_sparse_attn(sparse=0.05, window_size=16, kernel_size=5, random_sparse=0.1, vsparse=1.0)


seq_len = inputs["input_ids"].shape[1]
num_chunk = (seq_len // CHUNK_SIZE - 1) if (seq_len % CHUNK_SIZE == 0) else (seq_len // CHUNK_SIZE)
past_key_values = None
model.select_kv(False)
with torch.inference_mode():
    for chunk_id in range(1):
        outputs = model(input_ids=inputs["input_ids"][:,chunk_id * CHUNK_SIZE : (chunk_id + 1) * CHUNK_SIZE],
                                        attention_mask = inputs["attention_mask"][:,chunk_id * CHUNK_SIZE : (chunk_id + 1) * CHUNK_SIZE],
                                        past_key_values=past_key_values,
                                        use_cache=True)
        print(outputs)
