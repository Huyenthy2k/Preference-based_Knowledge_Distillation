# Preference-based Knowledge Distillation for Language Models

Preference-based Knowledge Distillation is a framework for distilling knowledge from large language models (LLMs) to smaller models using preference-based learning and knowledge distillation techniques. The framework combines the benefits of preference alignment (DPO) with knowledge distillation to create more efficient and aligned student models.

## Features

- **Parent Token Mechanism**: Implements a novel approach to token-level knowledge distillation using parent tokens
- **Weight Calculation**: Dynamic weight calculation for parent tokens to guide the distillation process
- **Multiple Loss Functions**: Supports various loss functions including:
  - DPO (Direct Preference Optimization)
  - TIS-DPO (Token-level Importance Sampling DPO)
  - DSKD (Distillation with Soft Knowledge)
  - Custom loss combinations

## Project Structure

```
PrefKD/
├── src/
│   └── prefkd/
│       ├── config/         # Configuration files
│       ├── loss/          # Loss function implementations
│       ├── utils/         # Utility functions
│       ├── weight/        # Weight calculation modules
│       ├── train.py       # Main training script
│       └── trainers.py    # Trainer implementations
├── script/               # Training and evaluation scripts
├── run_train/           # Training run configurations
├── run_eval/            # Evaluation run configurations
└── config/              # Hydra configuration files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Huyenthy2k/DPO_distillation
cd PrefKD
```

2. Create and activate a conda environment:
```bash
conda env create -f train_env.yaml
conda activate prefkd
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

1. Create a `.env` file in the project root with your API tokens:
```
HUGGINGFACE_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_token_here
```

2. Make sure the `.env` file is in `.gitignore` to keep your tokens secure.

## Usage

### Training

1. Configure your training parameters in the config files under `config/`
2. Run training:
```bash
python src/prefkd/train.py
```

### Evaluation

Run evaluation scripts from the `run_eval/` directory:
```bash
bash run_eval/eval_script.sh
```

## Models

### Teacher Models
- Mistral 7B v3 (32k vocab)

### Student Models
- Qwen 2.5 1.5B
- Phi 3/3.5 3.8B (optional)

## Benchmarks

The framework supports evaluation on:
- OpenLLM Leaderboard
- Anthropic HH (Helpful & Harmless)
- Custom benchmarks

## Development Status

### Completed
- [x] Parent token mechanism implementation
- [x] Weight calculation functionality
- [x] Token-level logits handling
- [x] DSKD implementation
- [x] Dataset formatting and processing
- [x] Basic evaluation scripts

### In Progress
- [ ] Comprehensive evaluation suite
- [ ] Additional baseline implementations
- [ ] Ablation studies
- [ ] Performance optimizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:
```
[Add citation information]
```
