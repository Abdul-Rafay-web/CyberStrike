from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from agents.base import AttackerAgent, DefenderAgent
from agents.belief_csp_defender import BeliefCSPDefender
from agents.belief_defender import load_sensor_rates
from agents.coach import CoachReport, analyze_defender_game
from agents.greedy_attacker import GreedyAttacker
from agents.minimax_attacker import MinimaxAttacker
from agents.random_agents import RandomAttacker, RandomDefender
from agents.simple_defenders import StaticDefender
from config import CONFIG, SensorConfig
from core.actions import Action
from core.network import Network
from core.sensors import RateSensor, Sensor
from core.state import GameState, Side
from evo.localsearch import load_tuned_weights
from ml.pipeline import load_sensor, train_sensor
from ml.sensor import ClassifierSensor
from search.evaluation import DEFAULT_WEIGHTS, EvalWeights
from utils.paths import DATA_DIR

DifficultyTier = Literal["easy", "standard", "hard"]

_CACHED_SENSOR: ClassifierSensor | None = None
_CACHED_RATES: SensorConfig | None = None


def _get_classifier_sensor_and_rates() -> tuple[ClassifierSensor, SensorConfig]:
    """Load Phase 5's trained Naive Bayes sensor once per process for zero UI lag."""
    global _CACHED_SENSOR, _CACHED_RATES
    if _CACHED_SENSOR is not None and _CACHED_RATES is not None:
        return _CACHED_SENSOR, _CACHED_RATES
    model_path = DATA_DIR / "sensor_model.json"
    trained = load_sensor(model_path) if model_path.exists() else train_sensor()
    _CACHED_SENSOR = ClassifierSensor.from_trained(trained)
    _CACHED_RATES = trained.rates
    return _CACHED_SENSOR, _CACHED_RATES


def _load_best_weights() -> EvalWeights:
    weights_path = DATA_DIR / "tuned_weights.json"
    if weights_path.exists():
        try:
            return load_tuned_weights(weights_path)
        except (OSError, ValueError, KeyError):
            return DEFAULT_WEIGHTS
    return DEFAULT_WEIGHTS


@dataclass(frozen=True, slots=True)
class MatchAgents:
    """Bundled AI agents and sensor for one live game session."""

    attacker: AttackerAgent
    defender: DefenderAgent
    sensor: Sensor
    sensor_rates: SensorConfig
    attacker_label: str
    defender_label: str


def build_match_agents(
    difficulty: DifficultyTier,
    agent_rng: np.random.Generator,
    *,
    use_classifier_sensor: bool = True,
) -> MatchAgents:
    """Factory creating real Phase 3-9 AI agents + Phase 5 Naive Bayes sensor by difficulty."""
    sensor: Sensor
    if use_classifier_sensor:
        sensor, rates = _get_classifier_sensor_and_rates()
    else:
        rates = load_sensor_rates()
        sensor = RateSensor(rates)

    weights = _load_best_weights()
    if difficulty == "easy":
        return MatchAgents(
            attacker=GreedyAttacker(),
            defender=StaticDefender(),
            sensor=sensor,
            sensor_rates=rates,
            attacker_label="Greedy A* Attacker",
            defender_label="Static Defender",
        )
    if difficulty == "hard":
        return MatchAgents(
            attacker=MinimaxAttacker(max_depth=4, time_budget=0.45, weights=weights),
            defender=BeliefCSPDefender(rates=rates),
            sensor=sensor,
            sensor_rates=rates,
            attacker_label="Minimax Alpha-Beta (d=4)",
            defender_label="Belief + CSP (MAC) Defender",
        )
    # Default "standard": fast responsive Minimax (d=3) + BeliefCSPDefender
    return MatchAgents(
        attacker=MinimaxAttacker(max_depth=3, time_budget=0.30, weights=weights),
        defender=BeliefCSPDefender(rates=rates),
        sensor=sensor,
        sensor_rates=rates,
        attacker_label="Minimax Alpha-Beta (d=3)",
        defender_label="Belief + CSP (MAC) Defender",
    )


def build_random_match_agents(agent_rng: np.random.Generator) -> MatchAgents:
    """Fallback random agents for baseline comparisons."""
    return MatchAgents(
        attacker=RandomAttacker(agent_rng),
        defender=RandomDefender(agent_rng),
        sensor=RateSensor(CONFIG.sensors),
        sensor_rates=CONFIG.sensors,
        attacker_label="Random Attacker",
        defender_label="Random Defender",
    )


def attacker_telemetry_summary(attacker: AttackerAgent) -> str | None:
    """Read live Minimax Alpha-Beta search stats (read-only) for HUD diagnostics."""
    if isinstance(attacker, MinimaxAttacker) and attacker.last_report is not None:
        rep = attacker.last_report
        return (
            f"MINIMAX d={rep.depth} | NODES:{rep.nodes} | "
            f"PRUNED:{rep.cutoffs} ({rep.seconds * 1000:.0f}ms)"
        )
    return None


def build_ai_tutorial_caption(
    network: Network,
    turn: int,
    side: Side,
    belief: tuple[float, ...],
    alerts: tuple[bool, ...],
    attacker: AttackerAgent,
    defender: DefenderAgent,
    node_codes: tuple[str, ...],
) -> tuple[str, str]:
    """Generate live AI-driven tutorial captions explaining every technique in plain English."""
    del network
    alert_count = sum(1 for flag in alerts if flag)
    top_idx = max(range(len(belief)), key=lambda i: belief[i]) if belief else 0
    top_pct = int(round((belief[top_idx] if belief else 0.0) * 100))
    top_code = node_codes[top_idx] if top_idx < len(node_codes) else f"M{top_idx}"

    if turn <= 1:
        return (
            "PHASE 15 // AI OVERVIEW",
            (
                f"Naive Bayes sensors raised {alert_count} alert(s). Bayesian belief puts "
                f"{top_code} at {top_pct}%. Attacker plans with A* + Minimax Alpha-Beta; "
                "Defender plans resources with a MAC CSP solver."
            ),
        )

    if isinstance(attacker, MinimaxAttacker) and attacker.last_report is not None:
        rep = attacker.last_report
        ab_note = (
            f"Minimax Alpha-Beta searched {rep.nodes} nodes (depth {rep.depth}, "
            f"{rep.cutoffs} cutoffs) over A* cheapest routes."
        )
    else:
        ab_note = "Attacker uses A* shortest-path search to evaluate cheapest routes to DB-01."

    if isinstance(defender, BeliefCSPDefender) and defender.last_stats is not None:
        st = defender.last_stats
        csp_note = (
            f"CSP MAC solver planned defenses in {st.assignments_tried} assignments "
            f"({st.backtracks} backtracks)."
        )
    else:
        csp_note = "CSP MAC planner allocates patches, firewalls, and honeypots across top routes."

    if side is Side.DEFENDER:
        return (
            f"TURN {turn:02d} // DEFENDER AI BRIEFING",
            (
                f"Naive Bayes flagged {alert_count} host(s); Bayesian belief ranks "
                f"{top_code} highest ({top_pct}%). {csp_note}"
            ),
        )
    return (
        f"TURN {turn:02d} // ATTACKER AI BRIEFING",
        f"{ab_note} Naive Bayes alerts will update the Defender's Bayesian heatmap next.",
    )


def run_post_game_coach(history: list[tuple[GameState, Action]]) -> CoachReport:
    """Run the Phase 9 local-search Post-Game Coach on a finished game's move history."""
    return analyze_defender_game(history, config=CONFIG)
