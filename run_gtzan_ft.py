from datasets import load_dataset

gtzan = load_dataset("marsyas/gtzan")
print(gtzan)

gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
print(gtzan)

print(gtzan["train"][0])

id2label_fn = gtzan["train"].features["genre"].int2str
print(id2label_fn(gtzan["train"][0]["genre"]))

from transformers import AutoFeatureExtractor

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True, return_attention_mask=True)

sampling_rate = feature_extractor.sampling_rate

from datasets import Audio

gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))

print(gtzan["train"][0])

import numpy as np

sample = gtzan["train"][0]["audio"]

print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")

inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])

print(f"inputs keys: {list(inputs.keys())}")

print(f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}")

max_duration = 30.0


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs

import psutil

num_proc = psutil.cpu_count()
gtzan_encoded = gtzan.map(
    preprocess_function, remove_columns=["audio", "file"], batched=True, num_proc=num_proc
)
print(gtzan_encoded)

gtzan_encoded = gtzan_encoded.rename_column("genre", "label")

id2label = {
            str(i): id2label_fn(i) for i in range(len(gtzan_encoded["train"].features["label"].names))
            }
label2id = {v: k for k, v in id2label.items()}

id2label["7"]

from transformers import AutoModelForAudioClassification

num_labels = len(id2label)

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

from transformers import TrainingArguments

model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 10

training_args = TrainingArguments(
    f"{model_name}-finetuned-gtzan",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=True,
)

import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=gtzan_encoded["train"],
    eval_dataset=gtzan_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

kwargs = {
    "dataset_tags": "marsyas/gtzan",
    "dataset": "GTZAN",
    "model_name": f"{model_name}-finetuned-gtzan",
    "finetuned_from": model_id,
    "tasks": "audio-classification",
}

trainer.push_to_hub(**kwargs)

