# How StochasticGoose Works

## Overview

**StochasticGoose** is a reinforcement learning agent for the ARC-AGI-3 Agent Preview Competition. It uses a CNN (Convolutional Neural Network) to learn which actions cause game state changes, enabling more efficient exploration than random action selection.

## Where Does the Code Run?

The agent runs **on your machine**, but communicates with a remote competition server:

1. **Local execution**: The CNN and all training happen on your computer (using your GPU if available, otherwise CPU)
2. **Remote connection**: Your agent connects to the ARC-AGI-3 competition server via HTTP API using your API key
3. **Game loop**: The server sends you game frames, you process them locally, then send back actions

## What is a CNN?

A **CNN** (Convolutional Neural Network) is a deep learning model specifically designed to work with images and spatial data. Think of it as a pattern-recognition machine that learns to detect features.

### The CNN in This Project

The CNN is defined in `custom_agents/action.py` in the `ActionModel` class. It has:

**Convolutional Backbone** (4 layers):
- Layer 1: 16 input channels → 32 channels
- Layer 2: 32 → 64 channels
- Layer 3: 64 → 128 channels
- Layer 4: 128 → 256 channels

Each convolutional layer scans the image with filters (3x3 pixel patterns) and extracts spatial features.

**Two Output Heads**:
1. **Action Head**: Predicts probabilities for 5 simple actions (ACTION1-ACTION5)
2. **Coordinate Head**: Predicts where to click on a 64×64 grid (ACTION6)

Total action space: 5 + 4,096 = **4,101 possible actions**

## How the Agent Decides on an Action - Step by Step

### 1. Receive Game Frame

The server sends a 64×64 grid where each cell contains a color value (0-15).

```python
current_frame = self._frame_to_tensor(latest_frame)
# Converts to shape: (16, 64, 64) - 16 one-hot encoded color channels
```

### 2. Run CNN Forward Pass

The frame flows through the CNN's convolutional layers:

```python
combined_logits = self.action_model(current_frame.unsqueeze(0))
# Input: (1, 16, 64, 64) - one frame with 16 color channels
# Output: (1, 4101) - raw prediction scores for each action
```

The CNN outputs **logits** (raw prediction scores) for all 4,101 possible actions.

### 3. Convert to Probabilities

Raw logits are converted to probabilities using **sigmoid**:

```python
action_probs = torch.sigmoid(action_logits)      # 5 values (0-1)
coord_probs = torch.sigmoid(coord_logits)        # 4,096 values (0-1)
```

Each probability represents: *"How confident am I that this action will change the frame?"*

### 4. Apply Action Masking

If certain actions aren't available, they're masked out:

```python
if available_actions is not None:
    action_mask = torch.full_like(action_logits, float('-inf'))
    # Only unmask allowed actions
    action_logits = action_logits + action_mask
```

### 5. Stochastically Sample an Action

Instead of always picking the highest probability action, the agent **randomly samples** weighted by probabilities:

```python
all_probs_sampling = torch.cat([action_probs, coord_probs_scaled])
all_probs_sampling = all_probs_sampling / all_probs_sampling.sum()
selected_idx = np.random.choice(len(all_probs_sampling), p=all_probs_sampling)
```

This introduces **exploration**: even low-probability actions get tried sometimes, helping the agent discover new strategies.

### 6. Return the Action

```python
if selected_idx < 5:
    # Selected ACTION1-ACTION5
    return self.action_list[selected_idx]
else:
    # Selected a coordinate for ACTION6
    coord_idx = selected_idx - 5
    y_idx = coord_idx // 64
    x_idx = coord_idx % 64
    return GameAction.ACTION6 with x, y coordinates
```

## What Happens with That Information?

### On the Next Frame

The agent receives the new frame and compares it to the previous one:

```python
frame_changed = not np.array_equal(self.prev_frame, current_frame)

experience = {
    'state': self.prev_frame,
    'action_idx': self.prev_action_idx,
    'reward': 1.0 if frame_changed else 0.0  # 1 if action caused change
}

self.experience_buffer.append(experience)
```

This creates **training data**: *"When I saw this frame and took this action, the frame changed (or didn't)."*

### Training Loop

Every 5 actions, the CNN is trained on random batches from the experience buffer:

```python
def _train_action_model(self):
    # Sample batch of 64 random experiences
    batch = random_sample(experience_buffer, batch_size=64)

    # Forward pass: predict if action will change frame
    predicted_logits = self.action_model(batch['states'])

    # Compute loss: how wrong was the prediction?
    loss = F.binary_cross_entropy_with_logits(predicted_logits, batch['rewards'])

    # Update weights to predict better next time
    loss.backward()
    self.optimizer.step()
```

The CNN gradually learns patterns: *"When I see patterns like this, ACTION1 usually changes the frame, but ACTION3 rarely does."*

### Entropy Regularization

To encourage exploration, the agent adds a small bonus for having diverse action probabilities:

```python
action_entropy = action_probs.mean()  # Average confidence
coord_entropy = coord_probs.mean()

total_loss = main_loss - 0.0001 * action_entropy - 0.00001 * coord_entropy
```

Higher entropy = more diverse actions tried = better exploration.

## The Complete Loop

```
1. Server sends frame
   ↓
2. CNN processes frame locally
   ↓
3. CNN outputs action probabilities
   ↓
4. Stochastically sample an action
   ↓
5. Send action to server
   ↓
6. Server executes action and returns new frame
   ↓
7. Compare frames → create training example
   ↓
8. Train CNN every 5 actions
   ↓
9. Return to step 1
```

## Key Features

### Experience Buffer
- Stores up to 200,000 unique state-action pairs
- Uses MD5 hashing to avoid storing duplicates
- Provides training data for supervised learning

### Action Masking
- Agent respects which actions are available
- Prevents wasting actions on invalid moves

### Adaptive Training
- When score increases (new game level):
  - Clears experience buffer
  - Resets CNN to start fresh
  - Resets optimizer with fresh learning rate

### Time Management
- Agent has 8-hour time limit per run (with 5-minute safety buffer)
- Checks elapsed time and exits gracefully when time is up

### Monitoring
- Logs metrics to TensorBoard: training loss, accuracy, entropy, scores
- Optional action visualizations: heatmaps showing click probabilities
- Git commit hash and diff saved for reproducibility

## Architecture Summary

```
Game Frame (64×64, 16 colors)
        ↓
One-Hot Encode (16, 64, 64)
        ↓
Convolutional Backbone (4 layers)
        ↓
    ┌───────────────────────┐
    ↓                       ↓
Action Head            Coordinate Head
(5 logits)             (4096 logits)
    ↓                       ↓
Action Probs           Coordinate Probs
    ↓                       ↓
    └───────────────────────┘
            ↓
    Stochastic Sampling
            ↓
    Selected Action
```

## No LLM Required

This is a pure **computer vision + reinforcement learning** project. There is **no language model involved**. The CNN:
- Works entirely with visual game frames
- Uses PyTorch for training
- Requires no text generation or natural language processing
- Only requires an ARC API key for server authentication

## Getting Started

```bash
# Setup environment
make install

# Add your API key
cd ARC-AGI-3-Agents
cp .env-example .env
# Edit .env and add ARC_API_KEY from https://three.arcprize.org/user

# Run the agent
make action

# Monitor training
make tensorboard
# Opens on http://localhost:6006
```
