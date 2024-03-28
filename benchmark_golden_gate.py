from transformers import GemmaForCausalLM, GemmaTokenizerFast
import time
import torch

# MODEL_ID = "golden-gate-dummy"
MODEL_ID = "gg-hf/gemma-2b"
MAX_NEW_TOKENS = 128
DO_SAMPLE = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained model
model = GemmaForCausalLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
tokenizer = GemmaTokenizerFast.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"

# Define inputs (bsz=4)
# input_text = ["The capital of France is", "Hi", "Recipe for pasta:", "Hottest countries in the world:"]
input_text = ["The capital of France is"]
input_ids = tokenizer(input_text, return_tensors="pt", padding=True)

model.to(device)
input_ids.to(device)

# Benchmark non-jit generation
start = time.time()
generated_ids = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE)
runtime = time.time() - start

print(f"Runtime: {runtime}")
pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(pred_text[0])