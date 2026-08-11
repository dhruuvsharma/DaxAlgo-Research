from __future__ import annotations

import json
import unittest
from pathlib import Path


PLAN_PATH = Path(__file__).with_name("transformer_dqn_architecture_plan.json")


class TransformerDqnPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    def test_status_forbids_accuracy_profit_and_deployment_claims(self) -> None:
        self.assertEqual(self.plan["schema_version"], 1)
        self.assertEqual(
            self.plan["status"], "proposed_not_implemented_not_validated"
        )
        claims = " ".join(self.plan["non_claims"]).lower()
        for required in ("accuracy", "economic edge", "live trading", "profitability"):
            self.assertIn(required, claims)

    def test_actions_are_unambiguous_target_positions(self) -> None:
        actions = self.plan["architecture"]["actions"]
        self.assertIn("target positions", actions["semantics"].lower())
        self.assertEqual(
            [(item["id"], item["target_position"]) for item in actions["values"]],
            [(-1, -1.0), (0, 0.0), (1, 1.0)],
        )
        self.assertIn("current target", actions["execution_rule"].lower())

    def test_chronology_is_fold_local_and_holdout_is_not_replayed(self) -> None:
        design = json.dumps(self.plan["validation_design"], sort_keys=True).lower()
        for required in (
            "five rolling",
            "purge and embargo",
            "fold-local",
            "ten dqn seeds",
            "physically sealed",
        ):
            self.assertIn(required, design)
        self.assertIn("cannot initialize", design)
        negative_controls = " ".join(self.plan["negative_controls"]).lower()
        self.assertIn("holdout data never enter replay", negative_controls)

    def test_environment_selects_exogenous_counterfactual_replay(self) -> None:
        environment = self.plan["architecture"]["environment"]
        simulator = environment["simulator_class"].lower()
        self.assertIn("exogenous historical counterfactual simulator", simulator)
        self.assertIn("fixed small notional", simulator)
        self.assertIn("capacity", environment["market_impact_scope"].lower())

    def test_reward_separates_training_shape_from_raw_net_pnl(self) -> None:
        reward = self.plan["architecture"]["reward"]
        accounting = reward["equity_accounting"].lower()
        for cost in ("bid/ask", "fees", "slippage", "funding", "terminal"):
            self.assertIn(cost, accounting)
        evaluation = reward["evaluation_rule"].lower()
        self.assertIn("raw net pnl", evaluation)
        self.assertIn("separately", evaluation)
        forbidden = " ".join(reward["forbidden_rewards"]).lower()
        self.assertIn("future maximum price", forbidden)
        self.assertIn("holdout performance", forbidden)

    def test_program_and_confirmation_gates_are_complete(self) -> None:
        phases = self.plan["training_program"]
        self.assertEqual([phase["phase"] for phase in phases], list(range(7)))
        self.assertTrue(all(phase["exit"] for phase in phases))
        gates = {item["gate"]: item["condition"] for item in self.plan["go_no_go_gates"]}
        self.assertIn("best frozen non-RL policy", gates["incremental_rl_value"])
        self.assertIn("five rolling folds", gates["stability"])
        self.assertIn("one pre-registered champion", gates["confirmation"].lower())
        self.assertIn("60", gates["confirmation"])


if __name__ == "__main__":
    unittest.main()
