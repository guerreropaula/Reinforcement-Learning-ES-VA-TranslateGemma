# Paper
---
**Paula Guerrero Castelló**  

This repository contains the paper *“Enhancing LLM Translation Performance for Spanish–Valencian through Supervised Fine-Tuning and Reinforcement Learning”*.

## Abstract

Valencian, the Western Catalan variety spoken in the Valencian Community, lacks a dedicated language code in most multilingual machine translation (MT) systems and is systematically translated into standard Eastern Catalan. We address this limitation by adapting TranslateGemma-4B-IT, a 4-billion-parameter instruction-tuned (IT) large language model (LLM) specialized for translation, through three post-training strategies using only publicly available corpora and Quantized Low-Rank Adaptation (QLoRA): (i) supervised fine-tuning (SFT); (ii) Group Relative Policy Optimization (GRPO), a reinforcement learning (RL) technique, with chrF and a naturalness reward (GRPOv1); and (iii) GRPO with a composite automatic-metric reward (GRPOv2). Our results suggest that reward-function alignment with the target dialect is a key determinant of RL success in low-resource dialectal MT.

## Keywords

Valencian, dialectal MT, reinforcement learning, GRPO, supervised fine-tuning


© 2026 The authors. No derivative works permitted.