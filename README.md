# PrefKD To-Do List

- [x] calculate parent tokens for train/valid datasets
- [ ] calculate weights for each parent tokens
    - [ ] train contrastive teacher LLM
    - [x] weight function
- ~~[ ] gen COT samples~~
- [x] bash file - eval script for baseline of teacher and student
    - should include harmless and helpfulness
    - use a reward model or cost model or guard model?
    - llm as a judge: prometheus
    - open llm leaderboard: ifeval, bbh
- [ ] implement baseline models: use DSKD script
- [ ] testing 
- [x] thay ref model thành DPO teacher model

## Need adapt
- [x] max length should be along token or parent token? -> fix from begining
- each batch should contains:
    - input_ids
    - attention_mask
    - weights
    - parent_token_ids
- [x] concanated inputs should concat the parent token list also
- tính weight có cho prompt không? không
- max len có ảnh hưởng tới tính parent token không? -> có, nên có 1 hàm set max len trong tokenize trước khi tính parent token
    - nên theo batch hay universal?

## 1. Loss Implementation
- [x] Implement parent token mechanism
- [x] Add weight calculation functionality
- [x] Develop token level logits handling
- [x] DSKD


## 2. Data Processing
- [x] Format the dataset for training (chat template, train on completion, etc)


## 3. Evaluation
**Models**
- Teacher: Mistral 7B v3 (32k vocab)
- Student: (phi 3/3.5 3.8B (32k vocab) /) qwen 2.5 1.5B

**Benchmark**
<!-- - [ ] MTBench
- [ ] AlpacaEval -->
- [ ] OpenLLM Leaderboard
- [ ] Anthropic HH


## 4. Baseline
- [ ] Teacher base
- [ ] Student base
**Preference Alignment Baseline**
- [ ] DPO
- [ ] TIS-DPO

**Cross-tokenizer KD Baseline**
- [ ] DSKD
- [ ] ULD
- [ ] CDM

## 5. Ablation Study
- [ ] Discard DSKD loss
- [ ] impact of weights?

# Notes
- **SET ALL SEEDS TO 1 FOR REPRODUCIBLE!!**

## Data for training
**Based on the method to decide**
- Preference data: [link](https://huggingface.co/datasets/tonyshelby/ultra-feedback_v7)
<!-- - SFT data: Deita 10k V0 -->

## Training Procedures
- [ ] Initialize 2 model with SFT
- [ ] Train 2 contrastive model to calculate weights
- [ ] Train the students model
- train on completion only: still tokenize full prompt + answer to calculate logits, but only use the answer part to calculate loss

## Considerations for method tuning
- Should include SeqKL type in the loss or not? Maybe not, the sequence level is include by DSKD distill loss
- calculate with teacher and student? dont understand
- should I include weights into COT2Align framework? (the empirical distribution?)
- should I use base or instruct model? I need to ensure similar data distribution along different phases. -> base
- [ ] check parent token meaning -> find original motivation in the paper for weight meaning, the if weight for parent token has any meaning.
- derive the paper carefully
    - weight calculation formula, why is that form, what goal trying to achieve
    - why we calculate parent token probs like that? what meaning, what goals? perplexity? why we only take local from element of sub token? what meaning?
- [ ] connect 2 loss naturally -> token level and sequence level
- [ ] include COT? -> not sure, but it is a good idea -> step level control
- [ ] review computational graph -> what to detach
- [ ] tune the threshold of weights, consider weight transform.
    note: trong code tis dpo không có clamp như trong công thức, ngược lại có weight transform
- [ ] try different optimizer
- [ ] try differrent KL function
