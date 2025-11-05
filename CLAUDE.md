# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**StochasticGoose** - An action-learning reinforcement learning agent for the ARC-AGI-3 Agent Preview Competition. This agent uses a CNN-based model to predict which actions (ACTION1-ACTION6) will cause frame state changes, enabling more efficient exploration than random action selection.

The project is built on top of the ARC-AGI-3-Agents framework (included as a git submodule) and implements a custom action-learning agent that learns from experience to predict state transitions.

## Development Commands

### Setup and Installation
```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:DriesSmit/ARC3-solution.git

# Install dependencies (creates venv and installs packages)
make install

# Environment setup
cd ARC-AGI-3-Agents
cp .env-example .env
# Edit .env and add ARC_API_KEY from https://three.arcprize.org/user
```

### Running the Agent
```bash
# Run the action-learning agent
make action

# Run with specific game ID
uv run ARC-AGI-3-Agents/main.py --agent=action --game=vc33

# View training metrics in TensorBoard
make tensorboard
# Opens on http://localhost:6006
```

### Cleanup
```bash
# Remove generated run logs and TensorBoard data
make clean
```

### Testing
```bash
# Run tests (from ARC-AGI-3-Agents submodule)
pytest
```

## Architecture

### Key Components

**ActionModel** (CNN-based predictor in `custom_agents/action.py`):
- Input: 16-channel one-hot encoded frames (64x64 grid, colors 0-15)
- Shared convolutional backbone: 4 layers (32→64→128→256 channels)
- Two-headed output:
  - **Action head**: Predicts probabilities for ACTION1-ACTION5 (simple actions)
  - **Coordinate head**: Predicts 64x64 click position probabilities for ACTION6 using convolutional layers to maintain 2D spatial bias
- Total action space: 5 + 4096 = 4101 possible actions

**Action Agent** (`custom_agents/action.py:Action`):
- Inherits from base `Agent` class in ARC-AGI-3-Agents framework
- Implements `choose_action()` and `is_done()` methods required by framework
- Uses experience replay buffer (200K max unique state-action pairs)
- Hash-based deduplication to avoid storing duplicate experiences
- Dynamic model reset when reaching new game levels (when score increases)

**Experience Buffer**:
- Stores (state, action_idx, reward) tuples where reward=1.0 if frame changed, 0.0 otherwise
- Supervised learning: predicts binary "will this action change the frame?"
- Uniqueness ensured via MD5 hashing of frame+action combinations

**Exploration Strategy**:
- Hierarchical sampling: First sample action type (ACTION1-5 or coordinates), then if coordinates, sample specific position
- Stochastic selection biased by sigmoid probabilities from model
- Entropy regularization to encourage exploration (light weight: 0.0001 for actions, 0.00001 for coordinates)

### Integration with ARC-AGI-3-Agents Framework

The custom agent is registered by modifying the submodule:

1. Add to `ARC-AGI-3-Agents/agents/__init__.py` (before `load_dotenv()`):
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_agent import *
```

2. Add field to `FrameData` class in `ARC-AGI-3-Agents/agents/structs.py`:
```python
available_actions: list[GameAction] = Field(default_factory=list)
```

### File Structure

```
ARC3-solution/
├── custom_agents/
│   ├── action.py          # Main ActionModel and Action agent implementation
│   └── view_utils.py      # Visualization utilities (grid images, heatmaps, charts)
├── custom_agent.py        # Agent registration (imports Action)
├── utils.py               # Experiment logging, git tracking, directory setup
├── Makefile               # Build/run commands
├── requirements.txt       # numpy, tensorboard, torch
└── ARC-AGI-3-Agents/      # Git submodule - competition framework
    ├── agents/
    │   ├── agent.py       # Base Agent class
    │   ├── structs.py     # GameAction, FrameData, GameState enums
    │   └── ...
    └── main.py            # Entry point
```

### Key Data Structures

**GameAction** enum (from framework):
- `RESET` (0): Reset the game
- `ACTION1-5` (1-5): Simple actions (no coordinates)
- `ACTION6` (6): Complex action requiring x,y coordinates

**FrameData** (from framework, modified):
- Contains current game frame (64x64 grid), score, state, and `available_actions` list
- `available_actions` field added to support action masking

**State Representation**:
- Frames are one-hot encoded: shape (16, 64, 64) for 16 possible colors
- Stored as boolean numpy arrays in experience buffer for memory efficiency

### Training Process

1. Agent receives FrameData from framework
2. Converts frame to tensor and runs forward pass through ActionModel
3. Samples action hierarchically from combined action space
4. Executes action via framework
5. On next frame, creates experience tuple (previous_state, action_idx, frame_changed)
6. Adds to experience buffer (if unique hash)
7. Periodically trains on random batch (default: every 5 actions)
8. Uses BCE loss with light entropy regularization
9. When score increases (new level): clears buffer and resets model

### Logging and Monitoring

- Experiment outputs saved to `runs/{timestamp}/` directory
- Per-game logs in `runs/{timestamp}/{game_id}/`
- TensorBoard metrics: training loss, accuracy, entropy, action counts, scores
- Optional action visualizations (disabled by default): heatmaps showing action probabilities overlaid on game frames
- Git commit hash and diff saved on each run for reproducibility

## Development Notes

### GPU Usage
- Model automatically uses CUDA if available, falls back to CPU
- GPU memory is cleared periodically during training (`torch.cuda.empty_cache()`)

### Time Limits
- Agent has 8-hour time limit per run (with 5-minute safety buffer)
- Check via `_has_time_elapsed()` method

### Randomization
- Seeds are based on current time + game_id hash for reproducibility per game
- Different games get different seeds even in same run

### Action Masking
- Agent respects `available_actions` from FrameData
- Masks unavailable actions by setting logits to -inf before sampling

### Visualization Control
- Set `save_action_visualizations = True` in `Action.__init__()` to enable image generation
- Configure frequency with `vis_save_frequency` and `vis_samples_per_save`

## Dependencies

- **uv**: Package manager for Python projects
- **PyTorch**: Deep learning framework (2.8.0)
- **NumPy**: Array operations (2.3.2)
- **TensorBoard**: Metric visualization
- **PIL**: Image generation for visualizations
- **Pydantic**: Data validation in framework
- **LangGraph**: Part of framework (optional extras in submodule)

## Important Configuration Notes

- API key must be set in `ARC-AGI-3-Agents/.env` before running
- Submodule modifications (to `__init__.py` and `structs.py`) are required for agent to work
- Model hyperparameters are hardcoded in `action.py` (learning rate: 0.0001, batch size: 64, buffer: 200K)
