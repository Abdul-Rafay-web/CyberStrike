# CyberStrike: Network Infiltration Game

An adversarial turn-based strategy AI system on an enterprise network topology.

An **Attacker AI** infiltrates through an entry point attempting to breach the Crown Jewel server. A **Defender AI / Player** deploys security controls (patches, firewalls, and honeypots) relying on noisy traffic alerts from Bayesian sensors under partial observability.

---

### Core Tech Stack

- **Python 3** (Core rules engine, headless simulation, and AI agents)
- **Pygame** (Real-time cyber battlefield GUI, overlays, and interactive visualizer)
- **NumPy** (Reproducible PRNG streams, vector math, and Bayesian belief calculations)
- **Matplotlib** (Experiment plotting, ROC curves, calibration curves, and performance charts)

---

### Project Architecture & Rules

- **Headless First**: Pure Python rules engine (`core/`) never imports Pygame. AI simulations and tournaments run headless at thousands of games per minute.
- **Strict Reproducibility**: All randomness is drawn through seeded NumPy streams (`utils/rng.py`).
- **Structured Logging**: Events are logged as structured JSON lines (`utils/logging_setup.py`) for replayability and SQLite indexing.
- **Config Driven**: All game, search, seed, and visual constants are centralized in `config.py`.

---

### Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Run test suite
pytest

# Run the game window
python main.py play
```

