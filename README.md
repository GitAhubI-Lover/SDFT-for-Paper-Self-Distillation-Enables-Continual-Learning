# Self-Distillation Fine-Tuning (SDFT) Implementation

This repository contains a PyTorch implementation of the Self-Distillation Fine-Tuning (SDFT) method described in the paper "Self-Distillation Enables Continual Learning" by Shenfeld et al.

## Paper Overview

The paper introduces Self-Distillation Fine-Tuning (SDFT), a method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills.

### Key Concepts

1. **On-Policy vs Off-Policy Learning**: SDFT addresses the issue of catastrophic forgetting in off-policy learning methods like Supervised Fine-Tuning (SFT) by using on-policy updates.

2. **Self-Distillation Mechanism**: The same model serves as both student and teacher:
   - **Student**: Conditioned only on the query `x`
   - **Teacher**: Same model, but additionally conditioned on expert demonstration `c`

3. **Demonstration-Conditioned Context**: The teacher is created using the prompt template:
   ```
   <Question>
   [QUERY]
   This is an example for a response to the question:
   <Demonstration>
   [DEMONSTRATION]
   Now answer with a response of your own, including the thinking process:
   ```

4. **Reverse KL Divergence**: The training objective minimizes the reverse KL divergence:
   ```
   L(θ) = D_KL(π_θ(·|x) ∥ π(·|x, c))
   ```

5. **Exponential Moving Average (EMA)**: The teacher model parameters are updated as an EMA of student parameters to ensure stability.

## Implementation Structure

```
├── complete_sdft_implementation.py    # Core SDFT implementation
├── sdft_implementation.py            # Basic implementation
└── README.md                         # This file
```

## Core Components

### 1. SDFTModel
- Contains both student and teacher encoders
- Implements EMA update for teacher parameters
- Handles context creation for both models

### 2. SDFTLoss
- Implements reverse KL divergence loss function
- Computes `D_KL(π_θ(·|x) || π(·|x, c))`

### 3. SDFTTrainer
- Main training loop with epoch management
- Handles batching and tokenization
- Performs validation when available

## How to Use

### Installation
```bash
pip install torch transformers datasets
```

### Basic Usage
```python
from complete_sdft_implementation import SDFTTrainer, create_sample_data

# Create training data
train_data, val_data = create_sample_data()

# Initialize trainer
trainer = SDFTTrainer(
    model_name="gpt2",  # Replace with your preferred model
    train_data=train_data,
    val_data=val_data,
    learning_rate=5e-6,
    ema_alpha=0.02,
    batch_size=2
)

# Train the model
trainer.train(num_epochs=3)

# Generate response to a query
response = trainer.generate("What is machine learning?")
print(response)
```

## Paper Reproduction Notes

This implementation follows the methodology described in the paper:

- Uses the specified prompt template for teacher creation
- Implements on-policy distillation via reverse KL divergence
- Applies EMA for teacher model updates
- Follows the hyperparameter recommendations from the paper:
  - Learning rate: 5e-6 (in paper's range: {5e-6, 1e-5, 5e-5})
  - EMA α: 0.02 (in paper's range: {0.01, 0.02, 0.05})
  - Epochs: 2-4 (depending on task)

## Expected Benefits

According to the paper, SDFT should provide:

1. **Reduced Catastrophic Forgetting**: Better preservation of prior capabilities
2. **Improved New Task Performance**: Higher accuracy on new tasks compared to SFT
3. **Better Generalization**: Both in-distribution and out-of-distribution
4. **Continual Learning**: Ability to accumulate multiple skills over time

## Limitations & Considerations

- The effectiveness depends on strong in-context learning capabilities
- Computationally more expensive than SFT (~2.5x FLOPs)
- Requires careful hyperparameter tuning
- May inherit spurious linguistic patterns from teacher

## Program Requirements

### 1. Runtime Environment
- Python: 3.8 or higher
- PyTorch: 1.12 or higher
- Transformers: 4.20 or higher
- CUDA: Recommended for GPU acceleration (optional but highly recommended)

### 2. Hardware Requirements
#### (1) Minimum Requirements
- CPU: Modern multi-core processor (Intel i5 / AMD Ryzen 5 or equivalent)
- RAM: 16 GB minimum
- GPU: Not required but can use CPU (very slow training)
#### (2) Recommended Requirements
- CPU: Multi-core processor (Intel i7 / AMD Ryzen 7 or better)
- RAM: 32 GB or more
- GPU: NVIDIA GPU with at least 8GB VRAM (RTX 3070/4070 or better recommended)
- For larger models (7B+ parameters): 16GB+ VRAM (RTX 4090, A6000, or similar)
#### （3）For Production Use:
- GPU: 24GB+ VRAM (A100, H100, RTX 6000 Ada, or similar) for large models like Qwen2.5-7B
- VRAM Requirements: Scale with model size - for 7B models, expect 12-16GB VRAM usage

Note: Training large language models using SDFT requires significant computational resources, similar to fine-tuning requirements mentioned in the paper (~2.5x computational cost compared to SFT).

### 3. SDFTTrainer
- Main training loop with epoch management
- Handles batching and tokenization
- Performs validation when available

## Comparison with GitHub Repositories

The implementation addresses the same core concepts as the referenced repositories:
- `https://github.com/uthmandevsec/Self-Distillation`: Implements on-policy self-distillation for continual learning
- `https://github.com/erickfm/distill`: Focuses on distillation techniques for continual learning

However, this implementation follows the specific methodology outlined in the "Self-Distillation Enables Continual Learning" paper with the precise algorithmic details.