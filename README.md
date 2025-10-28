# RL-HF-with-Pytorch-for-LLM-Allignment
### check pdf for detailed notes
LLM Allignment usinng techniques like PPO and DPO

## Key Concepts

### RLHF (Reinforcement Learning from Human Feedback)

- Three-stage process: Supervised Fine-Tuning (SFT) → Reward Model Training → RL Optimization
- Aligns language models with human values and preferences
- Enables models to generate more helpful, harmless, and honest outputs

### PPO (Proximal Policy Optimization)

- **Policy Network**: The language model being optimized (actor)
- **Value Network**: Estimates expected reward for states (critic)
- **Reference Model**: Frozen copy of initial policy to prevent drift
- **Clipped Objective**: Limits policy updates to prevent catastrophic forgetting
- **KL Penalty**: Penalizes divergence from reference model
- **Advantage Estimation**: Uses GAE (Generalized Advantage Estimation) for variance reduction

**PPO Algorithm Flow**:

1. Generate responses from current policy
1. Score responses with reward model
1. Compute advantages using value network
1. Update policy with clipped surrogate objective
1. Update value network to predict rewards

### DPO (Direct Preference Optimization)

- Directly optimizes policy from preference data without reward model
- **Simpler Pipeline**: Skips reward model training stage
- **Bradley-Terry Model**: Models preference probability
- **Implicit Reward**: Derives reward from preference likelihood ratio
- **Single-Stage Training**: Jointly optimizes for reward and KL constraint
- More stable and computationally efficient than PPO

**DPO Loss**:

```
L = -log(σ(β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x))))
```

where y_w = preferred, y_l = rejected, β = temperature parameter




## key metrics during training:

- **Reward**: Average reward from reward model
- **KL Divergence**: Distance from reference policy
- **Policy Loss**: PPO objective value
- **Value Loss**: Critic prediction error (PPO only)
- **Entropy**: Policy exploration measure

## Comparison: PPO vs DPO

|Aspect         |PPO            |DPO         |
|---------------|---------------|------------|
|Stages         |3 (SFT, RM, RL)|2 (SFT, DPO)|
|Reward Model   |Required       |Not required|
|Stability      |Less stable    |More stable |
|Compute        |Higher         |Lower       |
|Hyperparameters|More complex   |Simpler     |

<img width="593" height="820" alt="image" src="https://github.com/user-attachments/assets/c77f663e-0956-44ed-ad91-b66fd73e7cf7" />
