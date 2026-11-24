from __future__ import annotations

from agents.belief_defender import load_sensor_rates
from config import CONFIG, BeliefConfig, CSPConfig, GameConfig, SensorConfig
from core.actions import Action, Firewall, Honeypot, Pass, Patch
from core.network import Edge
from core.observation import DefenderObservation
from csp.model import build_defense_csp
from csp.solver import MAC_MRV_LCV, SolverConfig, SolverStats, solve_csp
from prob.belief import BeliefTracker
from search.heuristics import UNREACHABLE
from search.problem import build_problem
from search.uninformed import uniform_cost


class BeliefCSPDefender:
    """Defender combining Phase 6 Bayesian belief tracking with Phase 7 CSP resource planning.

    Every turn:
      1. Updates the Bayesian belief vector from `DefenderObservation`.
      2. Formulates and solves the global resource-placement CSP using the latest belief.
      3. Executes the highest-urgency legal action from the solved CSP plan.
    """

    name = "belief csp defender"

    def __init__(
        self,
        rates: SensorConfig | None = None,
        solver_config: SolverConfig = MAC_MRV_LCV,
        game: GameConfig = CONFIG.game,
        belief_cfg: BeliefConfig = CONFIG.belief,
        csp_cfg: CSPConfig = CONFIG.csp,
    ) -> None:
        self._rates = rates
        self._solver_config = solver_config
        self._game = game
        self._belief_cfg = belief_cfg
        self._csp_cfg = csp_cfg
        self._tracker: BeliefTracker | None = None
        self.last_belief: tuple[float, ...] = ()
        self.last_plan: tuple[Action, ...] = ()
        self.last_stats: SolverStats | None = None

    def _ensure_tracker(self, observation: DefenderObservation) -> BeliefTracker:
        if self._tracker is None or self._tracker.network is not observation.network:
            rates = self._rates if self._rates is not None else load_sensor_rates()
            self._tracker = BeliefTracker(
                observation.network,
                rates,
                game=self._game,
                config=self._belief_cfg,
            )
        return self._tracker

    def choose_action(self, observation: DefenderObservation) -> Action:
        tracker = self._ensure_tracker(observation)
        belief = tracker.step(observation)
        self.last_belief = belief

        stock = observation.stock
        if stock.patches == 0 and stock.firewalls == 0 and stock.honeypots == 0:
            self.last_plan = ()
            self.last_stats = None
            return Pass()

        problem = build_defense_csp(
            observation.network,
            stock,
            patched=observation.patched,
            firewalls=observation.firewalls,
            honeypots=observation.honeypots,
            belief=belief,
            routes_to_cover=self._csp_cfg.routes_to_cover,
            game=self._game,
        )
        result = solve_csp(problem, self._solver_config)
        self.last_stats = result.stats

        if result.assignment is None:
            # Graceful fallback: relax route coverage if the graph is tightly constrained
            relaxed = build_defense_csp(
                observation.network,
                stock,
                patched=observation.patched,
                firewalls=observation.firewalls,
                honeypots=observation.honeypots,
                belief=belief,
                routes_to_cover=0,
                game=self._game,
            )
            result = solve_csp(relaxed, self._solver_config)
            self.last_stats = result.stats

        if result.assignment is None:
            self.last_plan = ()
            return Pass()

        legal = set(observation.legal_actions)
        planned = tuple(
            act
            for act in result.assignment.values()
            if isinstance(act, Action) and act in legal
        )
        self.last_plan = planned
        if not planned:
            return Pass()

        return self._pick_most_urgent(observation, tracker, planned)

    def _pick_most_urgent(
        self,
        observation: DefenderObservation,
        tracker: BeliefTracker,
        planned: tuple[Action, ...],
    ) -> Action:
        suspects = tracker.top_k(3)
        belief = tracker.probabilities
        base_cost = self._expected_cost(
            observation, suspects, observation.patched, observation.firewalls
        )
        net = observation.network

        best_action: Action = planned[0]
        best_score = -1.0

        for action in planned:
            if isinstance(action, Firewall):
                after = self._expected_cost(
                    observation,
                    suspects,
                    observation.patched,
                    observation.firewalls | {(action.a, action.b)},
                )
                gain = after - base_cost
                # Prioritize cutting links near the jewel or right in front of the intruder
                proximity = 1.5 * (belief[action.a] + belief[action.b])
                jewel_bonus = 0.8 if net.jewel in (action.a, action.b) else 0.0
                score = gain * 2.2 + proximity + jewel_bonus
            elif isinstance(action, Patch):
                after = self._expected_cost(
                    observation,
                    suspects,
                    observation.patched | {action.machine},
                    observation.firewalls,
                )
                gain = after - base_cost
                jewel_bonus = 1.0 if action.machine == net.jewel else 0.0
                score = gain * 2.0 + 1.2 * belief[action.machine] + jewel_bonus
            elif isinstance(action, Honeypot):
                # Reward traps placed directly on the intruder's likely next step
                next_hit = sum(
                    belief[nbr] / max(1, len(net.neighbours(nbr)))
                    for nbr in net.neighbours(action.machine)
                )
                score = next_hit * self._game.honeypot_delay_turns * 3.0
            else:
                score = -1.0

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _expected_cost(
        self,
        observation: DefenderObservation,
        suspects: tuple[tuple[int, float], ...],
        patched: frozenset[int],
        firewalls: frozenset[Edge],
    ) -> float:
        total_weight = sum(prob for _, prob in suspects)
        if total_weight <= 0.0:
            return 0.0
        expected = 0.0
        for machine, prob in suspects:
            problem = build_problem(
                observation.network,
                machine,
                None,
                0,
                frozenset({machine}),
                patched,
                firewalls,
                self._game,
            )
            result = uniform_cost(problem)
            cost = float(UNREACHABLE if result.cost is None else result.cost)
            expected += (prob / total_weight) * cost
        return expected
