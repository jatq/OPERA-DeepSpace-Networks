import unittest

from experimental.run_v10_10_lns2_audit import lns2_decision, lns2_policy
from experimental.run_v10_9_milp_scaling import _strong_baseline_decision, build_congested_scenario
from experimental.v10_3_optioncast_eval import decision_plan_value
from experimental.v10_3_optioncast_model import observation_ids, radio_conflict
from experimental.v10_4_mismatch_eval import full_option_portfolio


class V1010Lns2AuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = build_congested_scenario(100, 10, 20)

    def test_lns2_is_feasible_and_never_worse_than_one_swap_in_plan(self) -> None:
        observation_id = sorted(observation_ids(self.scenario))[0]
        one_swap = _strong_baseline_decision(self.scenario, observation_id)
        lns2, evaluations = lns2_decision(self.scenario, observation_id)
        portfolio = full_option_portfolio(self.scenario)
        self.assertGreater(evaluations, 0)
        self.assertGreaterEqual(
            decision_plan_value(self.scenario, portfolio, lns2, observation_id) + 1e-9,
            decision_plan_value(self.scenario, portfolio, one_swap, observation_id),
        )
        contacts = {item.contact_id: item for item in self.scenario.contacts}
        selected = [contacts[item.contact_id] for item in lns2.activations]
        self.assertFalse(any(
            radio_conflict(left, right)
            for index, left in enumerate(selected) for right in selected[index + 1:]
        ))

    def test_policy_is_complete_and_deterministic(self) -> None:
        left = lns2_policy(self.scenario)
        right = lns2_policy(self.scenario)
        self.assertEqual(left.policy, right.policy)
        self.assertEqual(set(left.policy), set(observation_ids(self.scenario)))


if __name__ == "__main__":
    unittest.main()
