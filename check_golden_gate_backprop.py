from transformers import GoldenGateForCausalLM, FlaxGoldenGateForCausalLM, GoldenGateTokenizerFast
import numpy as np
import torch
import jax
import optax
from flax.training.common_utils import onehot
import tempfile
from flax.traverse_util import flatten_dict

MODEL_ID = "golden-gate-2b"

fx_model, params = FlaxGoldenGateForCausalLM.from_pretrained(MODEL_ID, _do_init=False)
tokenizer = GoldenGateTokenizerFast.from_pretrained(MODEL_ID)

try:
    pt_model = GoldenGateForCausalLM.from_pretrained(MODEL_ID)
except OSError or EnvironmentError:
    pt_model = GoldenGateForCausalLM.from_pretrained(MODEL_ID, from_flax=True)

input_text = ["Hello my name is"] #"Hi", "Recipe for pasta:", "Hottest countries in the world:"]
inputs = tokenizer(input_text, return_tensors="np", padding=True)

# cut bos token here as it's append later anyways
input_ids = inputs.input_ids#[:, :-1]
labels = inputs.input_ids#[:, 1:]
attention_mask = inputs.attention_mask#[:, 1:]

# replace padding with -100 to ignore correctly when computing the loss
labels = np.ma.array(labels, mask=np.not_equal(attention_mask, 1))
labels = labels.filled(fill_value=-100)

# Flax cross entropy loss
def loss_fn(logits, labels):
    vocab_size = logits.shape[-1]
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    loss = optax.softmax_cross_entropy(shift_logits, onehot(shift_labels, vocab_size))
    # ignore padded tokens from loss, i.e. where labels are not set to -100
    padding = shift_labels >= 0
    loss = loss * padding
    loss = loss.sum() / padding.sum()
    return loss


# Flax training step (single device)
def fx_train_step(fx_model, fx_batch, params):
    def compute_loss(params):
        labels = fx_batch.pop("labels")
        outputs = fx_model(**fx_batch, params=params)
        loss = loss_fn(outputs.logits, labels)
        return loss, outputs

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, outputs), grad = grad_fn(params)

    return loss, outputs, grad

fx_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
pt_batch = {key: torch.LongTensor(value) for key, value in fx_batch.items()}

fx_loss, fx_outputs, fx_grad = fx_train_step(fx_model, fx_batch, params)

# PyTorch training step (single device)
pt_outputs = pt_model(**pt_batch)
pt_logits = pt_outputs.logits
pt_loss = pt_outputs.loss
pt_loss.backward()


def assert_almost_equals(a: np.ndarray, b: np.ndarray, tol: float = 3e-3):
    """Assert whether the maximum absolute difference between two NumPy arrays a and b is within a given tolerance tol.
    Due to the pad_to_multiple_of nature of the FlaxDataCollator, the length of the Flax array a will always be greater than
    or equal to the length of the PyTorch array b. If a and b are of different lengths, array a (Flax, padded) will be
    reshaped to the shape of b (PyTorch)."""
    if a.shape != b.shape:
        a = a[:, :b.shape[1]]

    diff = np.abs((a - b))
    if diff.max() < tol:
        print(f"✅ Difference between Flax and PyTorch is {diff.max()} (< {tol}), avg is {diff.mean()}")
    else:
        print(f"❌ Difference between Flax and PyTorch is {diff.max()} (>= {tol}), avg is {diff.mean()}")

print("--------------------------Checking logits match--------------------------")
print(f"Flax logits shape: {fx_outputs.logits.shape}, PyTorch logits shape: {pt_logits.shape}")
assert_almost_equals(fx_outputs.logits, pt_logits.detach().numpy())

print("--------------------------Checking losses match--------------------------")
print(f"Flax loss: {fx_loss}, PyTorch loss: {pt_loss}")
assert_almost_equals(fx_loss, pt_loss.detach().numpy())

def assert_dict_equal(a: dict, b: dict, tol: float = 3e-3):
    if a.keys() != b.keys():
        print("❌ Dictionary keys for PyTorch and Flax do not match")
    results_fail = []
    results_correct = []
    result_diffs = []

    results_fail_rel = []
    results_correct_rel = []
    result_diffs_rel = []
    for k in a:
        ak_norm = np.linalg.norm(a[k])
        bk_norm = np.linalg.norm(b[k])
        diff = np.abs(ak_norm - bk_norm)
        diff_rel = np.abs(ak_norm - bk_norm) / np.abs(ak_norm)
        if diff < tol:
            results_correct.append(f"✅ Layer {k} diff is {diff} < {tol}).")
        else:
            results_fail.append(f"❌ Layer {k} has PT grad norm {bk_norm} and flax grad norm {ak_norm}.")
        result_diffs.append(diff)
        if diff_rel < tol:
            results_correct_rel.append(f"✅ Layer {k} rel diff is {diff} < {tol}).")
        else:
            results_fail_rel.append(f"❌ Layer {k} has PT grad norm {bk_norm} and flax grad norm {ak_norm}.")
        result_diffs_rel.append(diff_rel)
    return results_fail_rel, results_correct_rel, results_fail, results_correct, result_diffs, result_diffs_rel

# Convert PyTorch gradients to Flax
pt_grad_dict = {k: v.grad if v.grad is not None else torch.zeros(v.shape) for k, v in pt_model.named_parameters()}
missing_grads = [k for k in pt_model.state_dict().keys() if k not in pt_grad_dict]

missing_keys, unexpected_keys = pt_model.load_state_dict(pt_grad_dict, strict=False)

assert missing_grads == missing_keys, f"Error with either grads {missing_keys} or keys {unexpected_keys}"

with tempfile.TemporaryDirectory() as tmpdirname:
    pt_model.save_pretrained(tmpdirname, safe_serialization=False)
    pt_grad_model_to_fx = FlaxGoldenGateForCausalLM.from_pretrained(tmpdirname, from_pt=True)

pt_grad_to_fx = pt_grad_model_to_fx.params
fx_grad = flatten_dict(fx_grad)
pt_grad_to_fx = flatten_dict(pt_grad_to_fx)

results_fail_rel, results_correct_rel, results_fail, results_correct, result_diffs, result_diffs_rel = assert_dict_equal(fx_grad, pt_grad_to_fx)

print("--------------------------Checking gradients match--------------------------")
if len(results_fail) == 0:
    print(f"✅ Difference between Flax and PyTorch is {np.max(result_diffs)} (< 0.01), average is {np.mean(result_diffs)}")
else:
    print("\n".join(results_fail))

print("--------------------------Checking rel gradients match----------------------")

if len(results_fail_rel) == 0:
    print(f"✅ Difference between Flax and PyTorch is {np.max(result_diffs_rel)} (< 0.01), average is {np.mean(result_diffs_rel)}")
else:
    print("\n".join(results_fail_rel))
