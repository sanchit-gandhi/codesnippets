---
language:
- hi
license: apache-2.0
tags:
- hf-asr-leaderboard
- generated_from_trainer
datasets:
- mozilla-foundation/common_voice_11_0
metrics:
- wer
model-index:
- name: 'Whisper Small Hi - Sanchit Gandhi'
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice 11.0
      type: mozilla-foundation/common_voice_11_0
      config: null
      split: None
    metrics:
    - name: Wer
      type: wer
      value: 32.09599593667993
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the Common Voice 11.0 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4519
- Wer: 32.01

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 5000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 0.1011        | 2.44  | 1000 | 0.3075          | 34.63 |
| 0.0264        | 4.89  | 2000 | 0.3558          | 33.13 |
| 0.0025        | 7.33  | 3000 | 0.4214          | 32.59 |
| 0.0006        | 9.78  | 4000 | 0.4519          | 32.01 |
| 0.0002        | 12.22 | 5000 | 0.4679          | 32.10 |

### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.12.1
- Datasets 2.5.3.dev0
- Tokenizers 0.12.1
