from transformers import FlaxGemmaForCausalLM, GemmaTokenizerFast
import time
import jax
from flax import jax_utils
from flax.training.common_utils import shard
import jax.numpy as jnp

# MODEL_ID = "golden-gate-dummy"
MODEL_ID = "gg-hf/gemma-2b"
MAX_NEW_TOKENS = 128
DO_SAMPLE = True

# Load pre-trained model
model, params = FlaxGemmaForCausalLM.from_pretrained(MODEL_ID, revision="flax", _do_init=False, dtype=jnp.bfloat16)
params = model.to_bf16(params)

tokenizer = GemmaTokenizerFast.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left"

# Define inputs (bsz=4)
# input_text = ["The capital of France is", "Hi", "Recipe for pasta:", "Hottest countries in the world:"]
input_text = 4 * ["The capital of France is"]
input_ids = tokenizer(input_text, return_tensors="np", padding=True).input_ids

# Benchmark non-jit generation
start = time.time()
generated_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, params=params, do_sample=DO_SAMPLE).sequences
runtime = time.time() - start

print(f"Runtime without pmap: {runtime}")
pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(pred_text[0])

# shard the parameters and data for DP across TPU cores
params = jax_utils.replicate(params)
input_ids = shard(input_ids)

# pmap generate and run compilation step
def generate(input_ids, params, max_new_tokens, do_sample):
    generated_ids = model.generate(input_ids, params=params, max_new_tokens=max_new_tokens, do_sample=do_sample)
    return generated_ids.sequences

p_generate = jax.pmap(
    generate, "input_ids", in_axes=(0, 0, None, None), out_axes=0, static_broadcasted_argnums=(2,3,)
)
_ = p_generate(input_ids, params, MAX_NEW_TOKENS, DO_SAMPLE)

# Benchmark pjit generation
start = time.time()
generated_ids = p_generate(input_ids, params, MAX_NEW_TOKENS, DO_SAMPLE)
runtime = time.time() - start

# post-process generated ids
generated_ids = jax.device_get(generated_ids.reshape(-1, generated_ids.shape[-1]))
print(f"Runtime with pmap: {runtime}")
pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(pred_text[0])
