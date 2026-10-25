"""Small exact, causal evaluator for the audited V10.3 OPTIONCAST model.

The important separation is deliberately visible in the API: policies are
indexed by observations ``Z`` and are *planned* with ``P_plan``; a fixed
policy is subsequently scored with ``P_score(W,Z)``.  Only
``PerfectInfoOracle`` has a W-indexed map.  This module is a Stage-0 exact
evaluator, not a heuristic or a formal-suite runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Mapping, Sequence

from experimental.v10_3_optioncast_model import (
    PROBABILITY_SCALE,
    CAPAuthorizationLedger,
    CandidateContact,
    CONTROL_MESSAGE_CAPACITY_UNITS,
    CONTROL_MESSAGE_ENERGY_MILLIUNITS,
    CONTROL_MESSAGE_PRICE_MILLIUNITS,
    ContractKind,
    ContractLedger,
    OptioncastScenario,
    ExerciseBeacon,
    PreboundGrant,
    T0ContractChoice,
    Task,
    VcmMode,
    World,
    beacon_reaches_source,
    contact_can_serve_task,
    contact_capacity_units,
    exercise_fee,
    likelihood,
    net_value_milliunits,
    nominal_contact_capacity_units,
    observation_ids,
    posterior_world_weights,
    radio_conflict,
    validate_scenario,
    validate_prebound_grants,
    validate_t0_contract_choices,
)


COMMIT_ONLY_DYNAMIC_EXACT = "CommitOnlyDynamicExact"
COMMIT_PORTFOLIO_DYNAMIC_EXACT = "CommitPortfolioDynamicExact"
OPTION_STATIC_EXACT = "OptionStaticExact"
FORECAST_OPTION_THEN_CAP = "ForecastOptionThenCAP"
CAP_THEN_FORECAST_OPTION = "CAPThenForecastOption"
PREBOUND_GRANT_OPTION = "PreboundGrantOption"
OPTIONCAST_CAP_EXACT = "OPTIONCASTCAPExact"
PERFECT_INFO_ORACLE = "PerfectInfoOracle"

DEPLOYABLE_METHODS = (
    COMMIT_ONLY_DYNAMIC_EXACT, COMMIT_PORTFOLIO_DYNAMIC_EXACT,
    OPTION_STATIC_EXACT, FORECAST_OPTION_THEN_CAP, CAP_THEN_FORECAST_OPTION,
    PREBOUND_GRANT_OPTION, OPTIONCAST_CAP_EXACT,
)

# The same explicit limits apply to every named deployable method.  A limit
# raises instead of silently replacing an exact baseline with a heuristic.
MAX_PORTFOLIOS = 8_192
MAX_T1_DECISIONS = 120_000

@dataclass(frozen=True, order=True)
class ContactActivation:
    contact_id: str
    mode_id: str


@dataclass(frozen=True, order=True)
class TaskAssignment:
    task_id: str
    contact_id: str


@dataclass(frozen=True)
class T1Decision:
    """A t1 action.  Omitting an option means release; fees are not refunded."""

    activations: tuple[ContactActivation, ...]
    assignments: tuple[TaskAssignment, ...]


@dataclass(frozen=True)
class PolicyEvaluation:
    method: str
    portfolio: tuple[T0ContractChoice, ...]
    decisions_by_observation: Mapping[str, T1Decision]
    expected_net_milliunits: float
    expected_gross_value: float
    expected_t0_fees: float
    expected_exercise_fees: float
    net_value_ppm: float
    net_value_pp: float
    world_net_values: Mapping[str, float]
    expected_receipt_grants_consumed: float
    candidate_noncompletion_rate: float
    authorization_waste_rate: float
    violations: tuple[str, ...] = ()

    @property
    def expected_value(self) -> float:
        """Compatibility name; values are net milliunits, not a success rate."""
        return self.expected_net_milliunits


@dataclass(frozen=True)
class OracleEvaluation:
    """EVPI-only bound with one t0 portfolio and W-indexed t1 decisions.

    This is intentionally not a :class:`PolicyEvaluation`: the oracle sees
    W at t1 and hence is not a deployable Z-indexed policy.
    """

    method: str
    portfolio: tuple[T0ContractChoice, ...]
    decisions_by_world: Mapping[str, T1Decision]
    expected_net_milliunits: float
    expected_gross_value: float
    expected_t0_fees: float
    expected_exercise_fees: float
    net_value_ppm: float
    net_value_pp: float
    world_net_values: Mapping[str, float]
    expected_receipt_grants_consumed: float
    candidate_noncompletion_rate: float
    authorization_waste_rate: float


@dataclass(frozen=True)
class PairedScenarioMetric:
    """One seed's paired VBS comparison; no seeds are evaluated implicitly."""

    scenario_id: str
    proposed_net_value_pp: float
    baseline_net_value_pp: float
    gain_pp: float
    proposed_net_value_ppm: float
    baseline_net_value_ppm: float
    gain_ppm: float


def _maps(s: OptioncastScenario) -> tuple[dict[str, CandidateContact], dict[str, Task], dict[str, VcmMode]]:
    return ({x.contact_id: x for x in s.contacts}, {x.task_id: x for x in s.tasks}, {x.mode_id: x for x in s.vcm_modes})


def _key(value: object) -> str:
    return repr(value)


def _best(items: Sequence[object], score: Callable[[object], float]) -> object:
    if not items:
        raise ValueError("exact action universe is empty")
    # Deterministic and data-independent for tied numerical values.
    return max(items, key=lambda item: (score(item), _key(item)))


def enumerate_portfolios(
    scenario: OptioncastScenario, *, allow_commit: bool = True, allow_option: bool = True
) -> tuple[tuple[T0ContractChoice, ...], ...]:
    """Enumerate legal t0 contract choices before Z or W exists."""
    validate_scenario(scenario)
    alternatives: list[tuple[T0ContractChoice | None, ...]] = []
    for contact in sorted(scenario.contacts, key=lambda x: x.contact_id):
        choices: list[T0ContractChoice | None] = [None]
        if allow_option:
            choices.append(T0ContractChoice(contact.contact_id, ContractKind.OPTION))
        if allow_commit:
            choices.extend(T0ContractChoice(contact.contact_id, ContractKind.COMMIT, mode.mode_id) for mode in scenario.vcm_modes)
        alternatives.append(tuple(choices))
    result: list[tuple[T0ContractChoice, ...]] = []
    for raw in product(*alternatives):
        portfolio = tuple(sorted((x for x in raw if x is not None), key=lambda x: x.contact_id))
        try:
            validate_t0_contract_choices(portfolio, scenario)
        except ValueError:
            continue
        result.append(portfolio)
        if len(result) > MAX_PORTFOLIOS:
            raise RuntimeError("portfolio enumeration exceeded frozen exact state cap")
    return tuple(sorted(set(result), key=_key))


def _active_sets(
    portfolio: Sequence[T0ContractChoice], s: OptioncastScenario, *, commit_activation_optional: bool = False,
) -> tuple[tuple[ContactActivation, ...], ...]:
    contacts, _, modes = _maps(s)
    per_contract: list[tuple[ContactActivation | None, ...]] = []
    for choice in portfolio:
        if choice.kind is ContractKind.COMMIT:
            assert choice.commit_mode_id is not None
            # A commitment is an irrevocable RF reservation.  It must be
            # activated at t1, even if no payload is ultimately delivered;
            # otherwise a commitment could be silently released to make room
            # for an overlapping option, which is not the advertised control.
            activation = ContactActivation(choice.contact_id, choice.commit_mode_id)
            per_contract.append((None, activation) if commit_activation_optional else (activation,))
        else:
            per_contract.append((None,) + tuple(ContactActivation(choice.contact_id, mode.mode_id) for mode in s.vcm_modes))
    output: list[tuple[ContactActivation, ...]] = []
    for raw in product(*per_contract):
        active = tuple(sorted((x for x in raw if x is not None), key=lambda x: x.contact_id))
        exercise_cost = sum(exercise_fee(contacts[x.contact_id], x.mode_id) for x in active if any(
            p.contact_id == x.contact_id and p.kind is ContractKind.OPTION for p in portfolio
        ))
        energy = sum(contacts[x.contact_id].duration_epochs * modes[x.mode_id].energy_milliunits_per_epoch for x in active)
        selected = [contacts[x.contact_id] for x in active]
        if exercise_cost > s.exercise_budget_milliunits or energy > s.energy_budget_milliunits:
            continue
        if any(radio_conflict(left, right) for index, left in enumerate(selected) for right in selected[index + 1:]):
            continue
        output.append(active)
    return tuple(sorted(output, key=_key))


@dataclass
class _PreboundGrantLedger:
    """Independent t0-fixed authorization path for the prebinding controls.

    Unlike CAP, every listed grant consumes one of B immediately at t0.  A
    later failed delivery leaves that slot locked; no task/contact/mode can
    inherit it.  This class deliberately does not instantiate or delegate to
    ``CAPAuthorizationLedger`` so the two controls have genuinely different
    state machines.
    """

    scenario: OptioncastScenario
    contract_ledger: ContractLedger
    grants_by_task: Mapping[str, PreboundGrant]
    completed_task_ids: set[str]
    beacons_by_task: dict[str, ExerciseBeacon]
    pending_bytes_by_contract: dict[str, int]
    completed_bytes_by_contract: dict[str, int]
    control_messages_by_contract: dict[str, int]

    @classmethod
    def from_grants(
        cls, scenario: OptioncastScenario, grants: Sequence[PreboundGrant], contract_ledger: ContractLedger,
    ) -> "_PreboundGrantLedger":
        validate_prebound_grants(grants, scenario)
        return cls(
            scenario, contract_ledger, {grant.task_id: grant for grant in grants}, set(), {}, {}, {}, {},
        )

    @property
    def locked_final_grants(self) -> int:
        return len(self.grants_by_task)

    @property
    def completed_final_grants(self) -> int:
        return len(self.completed_task_ids)

    def prebound_beacon(self, assignment: TaskAssignment, activation: ContactActivation, task: Task) -> ExerciseBeacon:
        grant = self.grants_by_task.get(assignment.task_id)
        if grant is None:
            raise ValueError("prebound path rejects an unreserved task")
        if (grant.contact_id, grant.mode_id) != (assignment.contact_id, activation.mode_id):
            raise ValueError("prebound grant cannot be rebound after t0")
        if assignment.task_id in self.completed_task_ids:
            raise ValueError("prebound task is one-shot")
        record = self.contract_ledger.active_contract(assignment.contact_id)
        if record.mode_id != activation.mode_id or record.contract_id != f"contract:{assignment.contact_id}":
            raise ValueError("prebound beacon requires the purchased active contract and its fixed mode")
        if assignment.task_id in self.beacons_by_task:
            raise ValueError("prebound beacon is one-shot")
        capacity = nominal_contact_capacity_units(record.contact, next(
            mode for mode in self.scenario.vcm_modes if mode.mode_id == activation.mode_id
        ))
        used = (
            self.pending_bytes_by_contract.get(record.contract_id, 0)
            + self.completed_bytes_by_contract.get(record.contract_id, 0)
            + self.control_messages_by_contract.get(record.contract_id, 0) * CONTROL_MESSAGE_CAPACITY_UNITS
        )
        if used + task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS > capacity:
            raise ValueError("prebound beacon exceeds exercised contact capacity")
        self.contract_ledger.reserve_control_message()
        # This is a delivery token for the same t1 control-plane assumption
        # used by CAP, not a CAP ExerciseBeacon issued by a capsule ledger.
        beacon = ExerciseBeacon(
            beacon_id=f"prebound:{grant.grant_id}", capsule_id=f"prebound:{grant.grant_id}",
            task_id=task.task_id, contract_id=f"contract:{assignment.contact_id}",
            contact_id=assignment.contact_id, mode_id=activation.mode_id,
            byte_cap_units=task.payload_units, issued_epoch=self.scenario.t0_epoch, counter=0,
        )
        self.beacons_by_task[assignment.task_id] = beacon
        self.pending_bytes_by_contract[record.contract_id] = self.pending_bytes_by_contract.get(record.contract_id, 0) + task.payload_units
        self.control_messages_by_contract[record.contract_id] = self.control_messages_by_contract.get(record.contract_id, 0) + 1
        return beacon

    def object_commit(self, assignment: TaskAssignment, activation: ContactActivation, task: Task, *, byte_count_units: int) -> None:
        grant = self.grants_by_task.get(assignment.task_id)
        if grant is None or assignment.task_id in self.completed_task_ids:
            raise ValueError("prebound grant is unavailable")
        if (grant.contact_id, grant.mode_id) != (assignment.contact_id, activation.mode_id):
            raise ValueError("prebound grant cannot be rebound after t0")
        if byte_count_units != task.payload_units:
            raise ValueError("prebound object size differs from its t0 task")
        record = self.contract_ledger.active_contract(assignment.contact_id)
        if record.mode_id != activation.mode_id or record.contract_id != f"contract:{assignment.contact_id}":
            raise ValueError("prebound object requires the purchased active contract and its fixed mode")
        if assignment.task_id not in self.beacons_by_task:
            raise ValueError("prebound object needs an issued beacon")
        capacity = nominal_contact_capacity_units(record.contact, next(
            mode for mode in self.scenario.vcm_modes if mode.mode_id == activation.mode_id
        ))
        pending_other = self.pending_bytes_by_contract.get(record.contract_id, 0) - task.payload_units
        total = (
            self.completed_bytes_by_contract.get(record.contract_id, 0)
            + pending_other
            + task.payload_units
            + self.control_messages_by_contract.get(record.contract_id, 0) * CONTROL_MESSAGE_CAPACITY_UNITS
        )
        if total > capacity:
            raise ValueError("prebound ObjectCommit exceeds exercised contact capacity")
        self.pending_bytes_by_contract[record.contract_id] = pending_other
        self.completed_bytes_by_contract[record.contract_id] = self.completed_bytes_by_contract.get(record.contract_id, 0) + task.payload_units
        self.completed_task_ids.add(assignment.task_id)


def enumerate_t1_decisions(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    *,
    prebound: Sequence[PreboundGrant] | None = None,
    allowed_task_ids: frozenset[str] | None = None,
    commit_activation_optional: bool = False,
) -> tuple[T1Decision, ...]:
    """All nominally feasible t1 actions for one fixed t0 portfolio.

    CAP can advertise more than B candidates.  B is enforced only by the
    atomic receipt ledger in each realised world, exactly as the protocol
    specifies.  A prebound control instead restricts candidate assignments.
    """
    contacts, tasks, modes = _maps(scenario)
    portfolio_by_id = {item.contact_id: item for item in portfolio}
    # ``None`` selects CAP.  An empty tuple selects the prebound path with no
    # locked grants, which must not accidentally fall back to CAP.
    is_prebound = prebound is not None
    prebound_items = tuple(prebound or ())
    prebound_by_task = {item.task_id: item for item in prebound_items}
    if len(prebound_by_task) != len(prebound_items) or len(prebound_items) > scenario.final_grant_budget:
        raise ValueError("invalid t0 prebound grant set")
    output: list[T1Decision] = []
    if allowed_task_ids is not None and not allowed_task_ids <= set(tasks):
        raise ValueError("task subset contains an undeclared task")
    for active in _active_sets(portfolio, scenario, commit_activation_optional=commit_activation_optional):
        active_by_id = {item.contact_id: item for item in active}
        options_per_task: list[tuple[TaskAssignment | None, ...]] = []
        for task in sorted(scenario.tasks, key=lambda x: x.task_id):
            candidates: list[TaskAssignment | None] = [None]
            if allowed_task_ids is not None and task.task_id not in allowed_task_ids:
                options_per_task.append(tuple(candidates))
                continue
            for activation in active:
                contact = contacts[activation.contact_id]
                if not contact_can_serve_task(contact, task):
                    continue
                if nominal_contact_capacity_units(contact, modes[activation.mode_id]) < task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS:
                    continue
                capsule = next(p for p in scenario.policy_capsules if p.task_id == task.task_id)
                if not capsule.permits(contact_id=contact.contact_id, mode_id=activation.mode_id, epoch=scenario.t1_epoch, contract_version=scenario.contract_version):
                    continue
                bound = prebound_by_task.get(task.task_id)
                if bound is not None and (bound.contact_id != contact.contact_id or bound.mode_id != activation.mode_id):
                    continue
                if bound is None and is_prebound:
                    continue
                candidates.append(TaskAssignment(task.task_id, contact.contact_id))
            options_per_task.append(tuple(candidates))
        for raw in product(*options_per_task):
            assignments = tuple(sorted((x for x in raw if x is not None), key=lambda x: x.task_id))
            used: dict[str, int] = {}
            for assignment in assignments:
                used[assignment.contact_id] = (
                    used.get(assignment.contact_id, 0)
                    + tasks[assignment.task_id].payload_units
                    + CONTROL_MESSAGE_CAPACITY_UNITS
                )
            if any(used[cid] > nominal_contact_capacity_units(contacts[cid], modes[active_by_id[cid].mode_id]) for cid in used):
                continue
            active_energy = sum(
                contacts[activation.contact_id].duration_epochs
                * modes[activation.mode_id].energy_milliunits_per_epoch
                for activation in active
            )
            if active_energy + len(assignments) * CONTROL_MESSAGE_ENERGY_MILLIUNITS > scenario.energy_budget_milliunits:
                continue
            decision = T1Decision(active, assignments)
            output.append(decision)
            if len(output) > MAX_T1_DECISIONS:
                raise RuntimeError("t1 enumeration exceeded frozen exact state cap")
    return tuple(sorted(set(output), key=_key))


def enumerate_prebound_grants(
    scenario: OptioncastScenario, portfolio: Sequence[T0ContractChoice]
) -> tuple[tuple[PreboundGrant, ...], ...]:
    """All legal t0 grant bindings for the receipt-budget control.

    This is deliberately a separate t0 universe: a later Z observation may
    choose exercise/release only *within* these task/contact/mode bindings.
    """
    contacts, _, _ = _maps(scenario)
    contract_by_contact = {choice.contact_id: choice for choice in portfolio}
    candidates: list[PreboundGrant] = []
    for task in sorted(scenario.tasks, key=lambda item: item.task_id):
        for contact_id, choice in sorted(contract_by_contact.items()):
            contact = contacts[contact_id]
            if not contact_can_serve_task(contact, task):
                continue
            mode_ids = (choice.commit_mode_id,) if choice.kind is ContractKind.COMMIT else tuple(mode.mode_id for mode in scenario.vcm_modes)
            for mode_id in mode_ids:
                if mode_id is None or nominal_contact_capacity_units(contact, next(m for m in scenario.vcm_modes if m.mode_id == mode_id)) < task.payload_units:
                    continue
                candidates.append(PreboundGrant(f"pre:{task.task_id}:{contact_id}:{mode_id}", task.task_id, contact_id, mode_id))
    output: list[tuple[PreboundGrant, ...]] = [()]
    for size in range(1, scenario.final_grant_budget + 1):
        for group in combinations(candidates, size):
            if len({item.task_id for item in group}) != len(group):
                continue
            try:
                validate_prebound_grants(group, scenario)
            except ValueError:
                continue
            output.append(tuple(sorted(group, key=lambda item: item.grant_id)))
            if len(output) > MAX_T1_DECISIONS:
                raise RuntimeError("prebound enumeration exceeded frozen exact state cap")
    return tuple(sorted(set(output), key=_key))


def _fees(portfolio: Sequence[T0ContractChoice], decision: T1Decision, s: OptioncastScenario) -> tuple[int, int]:
    contacts, _, _ = _maps(s)
    t0 = sum(contacts[x.contact_id].commit_fee_milliunits if x.kind is ContractKind.COMMIT else contacts[x.contact_id].option_fee_milliunits for x in portfolio)
    by_id = {x.contact_id: x for x in portfolio}
    t1 = (
        sum(exercise_fee(contacts[x.contact_id], x.mode_id) for x in decision.activations if by_id[x.contact_id].kind is ContractKind.OPTION)
        + len(decision.assignments) * CONTROL_MESSAGE_PRICE_MILLIUNITS
    )
    return t0, t1


def _realise(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    decision: T1Decision,
    world: World,
    *,
    prebound: Sequence[PreboundGrant] | None = None,
    commit_activation_optional: bool = False,
) -> tuple[int, int, int, int]:
    """Execute the real protocol state machines in one truth world.

    Returns gross value, net value, consumed receipt grants, and number of
    assigned candidates.  W is intentionally only consumed here, after the
    Z-indexed policy decision has already been fixed.
    """
    contacts, tasks, modes = _maps(scenario)
    t0_fee, t1_fee = _fees(portfolio, decision, scenario)
    contract = ContractLedger(scenario, portfolio)
    active = {x.contact_id: x for x in decision.activations}
    # Monotonic contract transitions: absent held options are released, but
    # their fees remain in contract.t0_spend_milliunits.
    for choice in portfolio:
        if choice.kind is ContractKind.COMMIT and choice.contact_id not in active and not commit_activation_optional:
            raise ValueError("a committed contact must occupy its t1 RF reservation")
        if choice.contact_id not in active and choice.kind is ContractKind.OPTION:
            contract.release(choice.contact_id, epoch=scenario.t1_epoch)
        elif choice.contact_id in active and choice.kind is ContractKind.OPTION:
            contract.exercise(choice.contact_id, active[choice.contact_id].mode_id, epoch=scenario.t1_epoch)
        elif choice.contact_id in active and choice.kind is ContractKind.COMMIT:
            contract.activate_commit(choice.contact_id, epoch=scenario.t1_epoch)
    # CAP and prebinding are separate authorization paths.  CAP spends B only
    # at a successful ObjectCommit; prebinding locks its own B slots at t0.
    cap_ledger = CAPAuthorizationLedger.from_scenario(scenario, contract) if prebound is None else None
    prebound_ledger = _PreboundGrantLedger.from_grants(scenario, prebound, contract) if prebound is not None else None
    beacons: dict[TaskAssignment, ExerciseBeacon] = {}
    for assignment in decision.assignments:
        activation = active[assignment.contact_id]
        task = tasks[assignment.task_id]
        if cap_ledger is not None:
            capsule = next(x for x in scenario.policy_capsules if x.task_id == task.task_id)
            beacons[assignment] = cap_ledger.issue_exercise_beacon(
                capsule.capsule_id, assignment.contact_id, byte_cap_units=task.payload_units,
                epoch=scenario.t1_epoch, counter=capsule.initial_counter,
            )
        else:
            assert prebound_ledger is not None
            beacons[assignment] = prebound_ledger.prebound_beacon(assignment, activation, task)

    # The source sees only an arriving beacon.  Deterministic priority makes
    # both capacity scheduling and receipt races reproducible without a W
    # input at the t1 policy node.
    candidates = sorted(decision.assignments, key=lambda a: (-tasks[a.task_id].value, a.task_id, a.contact_id))
    used_actual: dict[str, int] = {}
    gross = 0
    for assignment in candidates:
        beacon = beacons[assignment]
        assert hasattr(beacon, "contact_id")
        if not beacon_reaches_source(beacon, world, scenario):
            continue
        contact = contacts[assignment.contact_id]
        activation = active[assignment.contact_id]
        capacity = contact_capacity_units(contact, modes[activation.mode_id], world)
        needed = tasks[assignment.task_id].payload_units + CONTROL_MESSAGE_CAPACITY_UNITS
        if used_actual.get(contact.contact_id, 0) + needed > capacity:
            continue
        if cap_ledger is not None and cap_ledger.consumed_final_grants >= scenario.final_grant_budget:
            continue
        used_actual[contact.contact_id] = used_actual.get(contact.contact_id, 0) + needed
        try:
            if cap_ledger is not None:
                cap_ledger.object_commit(
                    beacon.beacon_id, object_digest=f"obj:{assignment.task_id}:{assignment.contact_id}",
                    byte_count_units=tasks[assignment.task_id].payload_units, epoch=contact.start_epoch, counter=1,
                )
            else:
                assert prebound_ledger is not None
                prebound_ledger.object_commit(
                    assignment, activation, tasks[assignment.task_id], byte_count_units=tasks[assignment.task_id].payload_units,
                )
        except ValueError:
            # A legal policy must never rely on a failed commit.  This branch
            # is retained so malformed hand scenarios fail closed.
            continue
        gross += tasks[assignment.task_id].value
    completed = cap_ledger.consumed_final_grants if cap_ledger is not None else prebound_ledger.completed_final_grants
    return gross, net_value_milliunits(gross, t0_contract_fees=t0_fee, exercise_fees=t1_fee, scenario=scenario), completed, len(decision.assignments)


def decision_plan_value(
    scenario: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    decision: T1Decision,
    observation_id: str,
    *,
    prebound: Sequence[PreboundGrant] | None = None,
    commit_activation_optional: bool = False,
) -> float:
    """Planner's E[net | Z] using only declared P_plan(W|Z)."""
    weights = posterior_world_weights(scenario, observation_id, planning=True)
    return sum(
        weights[world.world_id]
        * _realise(
            scenario, portfolio, decision, world, prebound=prebound,
            commit_activation_optional=commit_activation_optional,
        )[1]
        for world in scenario.worlds
    )


def _plan_policy_value(
    s: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    policy: Mapping[str, T1Decision],
    *,
    prebound: Sequence[PreboundGrant] | None = None,
    commit_activation_optional: bool = False,
) -> float:
    return sum(
        world.probability_units / PROBABILITY_SCALE * likelihood(s, world, z, planning=True)
        * decision_plan_value(
            s, portfolio, policy[z], z, prebound=prebound,
            commit_activation_optional=commit_activation_optional,
        )
        for world in s.worlds for z in observation_ids(s)
    )


def _score_policy(
    method: str,
    s: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    policy: Mapping[str, T1Decision],
    *,
    prebound: Sequence[PreboundGrant] | None = None,
    commit_activation_optional: bool = False,
) -> PolicyEvaluation:
    issues: list[str] = []
    if set(policy) != set(observation_ids(s)):
        issues.append("deployable policy must be indexed by exactly the declared Z values")
    world_net: dict[str, float] = {}
    expected_net = expected_gross = expected_t1 = 0.0
    expected_consumed = expected_candidates = 0.0
    for world in s.worlds:
        conditional_net = 0.0
        for z in observation_ids(s):
            gross, net, consumed, candidates = _realise(
                s, portfolio, policy[z], world, prebound=prebound,
                commit_activation_optional=commit_activation_optional,
            )
            mass = likelihood(s, world, z, planning=False)
            conditional_net += mass * net
            expected_net += world.probability_units / PROBABILITY_SCALE * mass * net
            expected_gross += world.probability_units / PROBABILITY_SCALE * mass * gross
            expected_t1 += world.probability_units / PROBABILITY_SCALE * mass * _fees(portfolio, policy[z], s)[1]
            expected_consumed += world.probability_units / PROBABILITY_SCALE * mass * consumed
            expected_candidates += world.probability_units / PROBABILITY_SCALE * mass * candidates
        world_net[world.world_id] = conditional_net
    t0 = _fees(portfolio, T1Decision((), ()), s)[0]
    denominator = sum(task.value for task in s.tasks) * s.value_milliunits_per_point
    ppm = expected_net / denominator * 1_000_000 if denominator else 0.0
    # CAP's ReceiptOnlyGrant does not exist until ObjectCommit succeeds.  A
    # missed beacon or a failed link is therefore a candidate noncompletion,
    # never authorization waste.  Prebound's irreversible t0 waste is
    # calculated separately in _choose_prebound_portfolio.
    noncompletion = 0.0 if expected_candidates == 0 else max(0.0, expected_candidates - expected_consumed) / expected_candidates
    return PolicyEvaluation(
        method, tuple(portfolio), dict(policy), expected_net, expected_gross,
        float(t0), expected_t1, ppm, ppm / 10_000, world_net,
        expected_consumed, noncompletion, 0.0, tuple(issues),
    )


def _policy_for_portfolio(
    s: OptioncastScenario,
    portfolio: Sequence[T0ContractChoice],
    *,
    dynamic: bool,
    prebound: Sequence[PreboundGrant] | None = None,
    phase_order: str = "joint",
    allowed_task_ids: frozenset[str] | None = None,
    commit_activation_optional: bool = False,
) -> Mapping[str, T1Decision]:
    decisions = enumerate_t1_decisions(
        s, portfolio, prebound=prebound, allowed_task_ids=allowed_task_ids,
        commit_activation_optional=commit_activation_optional,
    )
    if dynamic:
        if phase_order == "joint":
            return {
                z: _best(
                    decisions,
                    lambda d, z=z: decision_plan_value(
                        s, portfolio, d, z, prebound=prebound,
                        commit_activation_optional=commit_activation_optional,
                    ),
                )
                for z in observation_ids(s)
            }
        if phase_order == "capacity_then_cap":
            # Phase 1 deliberately ignores task values/authorization and
            # selects a contact/mode plan by posterior capacity per price.
            contacts, _, modes = _maps(s)
            output: dict[str, T1Decision] = {}
            for z in observation_ids(s):
                activation_sets = tuple(sorted({d.activations for d in decisions}, key=_key))
                weights = posterior_world_weights(s, z, planning=True)
                def capacity_score(active: tuple[ContactActivation, ...]) -> float:
                    # A commitment paid at t0 has no exercise fee.  Charging
                    # it again here would bias this split baseline against
                    # portfolios that use commitments.
                    option_by_contact = {
                        item.contact_id for item in portfolio if item.kind is ContractKind.OPTION
                    }
                    option_fees = sum(
                        exercise_fee(contacts[a.contact_id], a.mode_id)
                        for a in active if a.contact_id in option_by_contact
                    )
                    expected_capacity = sum(
                        weights[w.world_id] * sum(contact_capacity_units(contacts[a.contact_id], modes[a.mode_id], w) for a in active)
                        for w in s.worlds
                    )
                    return expected_capacity * s.value_milliunits_per_point - option_fees
                selected = _best(activation_sets, capacity_score)
                candidates = tuple(d for d in decisions if d.activations == selected)
                output[z] = _best(
                    candidates,
                    lambda d, z=z: decision_plan_value(
                        s, portfolio, d, z, prebound=prebound,
                        commit_activation_optional=commit_activation_optional,
                    ),
                )
            return output
        raise ValueError("unknown audited phase order")
    decision = _best(
        decisions,
        lambda d: sum(
            w.probability_units / PROBABILITY_SCALE
            * _realise(
                s, portfolio, d, w, prebound=prebound,
                commit_activation_optional=commit_activation_optional,
            )[1]
            for w in s.worlds
        ),
    )
    return {z: decision for z in observation_ids(s)}


def _choose_portfolio(
    s: OptioncastScenario,
    *,
    allow_commit: bool,
    allow_option: bool,
    dynamic: bool,
    method: str,
    commit_activation_optional: bool = False,
) -> PolicyEvaluation:
    candidates: list[tuple[tuple[T0ContractChoice, ...], Mapping[str, T1Decision]]] = []
    for portfolio in enumerate_portfolios(s, allow_commit=allow_commit, allow_option=allow_option):
        policy = _policy_for_portfolio(
            s, portfolio, dynamic=dynamic, commit_activation_optional=commit_activation_optional,
        )
        candidates.append((portfolio, policy))
    chosen = _best(
        candidates,
        lambda item: _plan_policy_value(
            s, item[0], item[1], commit_activation_optional=commit_activation_optional,
        ),
    )
    portfolio, policy = chosen
    return _score_policy(
        method, s, portfolio, policy, commit_activation_optional=commit_activation_optional,
    )


def _choose_prebound_portfolio(s: OptioncastScenario, *, method: str, task_first: bool) -> PolicyEvaluation:
    """t0 prebinding control: B grants cannot be reassigned after Z.

    ``PreboundGrantOption`` anticipates the full t1 subproblem when selecting
    immutable t0 bindings.  It is deliberately distinct from
    ``CAPThenForecastOption``: this control locks B at t0 and never calls the
    CAP receipt-only authorization path at t1.
    """
    candidates: list[tuple[tuple[T0ContractChoice, ...], tuple[PreboundGrant, ...], Mapping[str, T1Decision]]] = []
    for portfolio in enumerate_portfolios(s, allow_commit=True, allow_option=True):
        for grants in enumerate_prebound_grants(s, portfolio):
            policy = _policy_for_portfolio(s, portfolio, dynamic=True, prebound=grants)
            candidates.append((portfolio, grants, policy))
            if len(candidates) > MAX_T1_DECISIONS:
                raise RuntimeError("prebound portfolio search exceeded frozen exact state cap")
    def phase_one_score(item: tuple[tuple[T0ContractChoice, ...], tuple[PreboundGrant, ...], Mapping[str, T1Decision]]) -> float:
        portfolio, grants, policy = item
        if not task_first:
            return _plan_policy_value(s, portfolio, policy, prebound=grants)
        t0_fee = _fees(portfolio, T1Decision((), ()), s)[0]
        task_value = sum(next(task.value for task in s.tasks if task.task_id == grant.task_id) * s.value_milliunits_per_point for grant in grants)
        return task_value - t0_fee
    portfolio, _grants, policy = _best(candidates, phase_one_score)
    result = _score_policy(method, s, portfolio, policy, prebound=_grants)
    # In the prebound control, unused locked grant slots are authorization
    # waste even if CAP's receipt state machine correctly rejects no object.
    bound = len(_grants)
    waste = 0.0 if bound == 0 else max(0.0, bound - result.expected_receipt_grants_consumed) / bound
    return PolicyEvaluation(**{**result.__dict__, "authorization_waste_rate": waste})


def _cap_priority_task_subset(s: OptioncastScenario) -> frozenset[str]:
    """The audited CAP-first phase-one rule.

    At t1 the controller first freezes at most B task capsules, ordered by
    higher payload value, then earlier deadline, then immutable task id.  It
    sees no contact, VCM, or truth outcome in this phase; only the later
    exact phase may exercise/release purchased options and bind a route.  CAP
    is still the authorization state machine: no final B slot is consumed
    unless the selected task reaches ObjectCommit.
    """
    ordered = sorted(s.tasks, key=lambda task: (-task.value, task.deadline_epoch, task.task_id))
    return frozenset(task.task_id for task in ordered[: s.final_grant_budget])


def _cap_then_forecast_policy(
    s: OptioncastScenario, portfolio: Sequence[T0ContractChoice],
) -> Mapping[str, T1Decision]:
    """CAP task selection precedes the exact contact/mode phase at every Z."""
    subset = _cap_priority_task_subset(s)
    # The deterministic capsule-priority set is deliberately frozen before
    # any route action.  The returned policy remains Z-indexed because the
    # second phase optimizes the conditional exercise/VCM decision at Z.
    return _policy_for_portfolio(s, portfolio, dynamic=True, allowed_task_ids=subset)


def evaluate_method(scenario: OptioncastScenario, method: str) -> PolicyEvaluation:
    """Evaluate one method.  Formal 31-seed scoring is intentionally external."""
    if method not in DEPLOYABLE_METHODS + (PERFECT_INFO_ORACLE,):
        raise ValueError(f"unknown V10.3 method: {method}")
    if method == COMMIT_ONLY_DYNAMIC_EXACT:
        return _choose_portfolio(scenario, allow_commit=True, allow_option=False, dynamic=True, method=method)
    if method == COMMIT_PORTFOLIO_DYNAMIC_EXACT:
        # This is intentionally stronger than CommitOnly: it pays for the
        # same t0 commitment portfolio but can elect one nonconflicting subset
        # at t1.  It cannot refund, change a committed VCM, or use an
        # unpurchased contact.
        return _choose_portfolio(
            scenario, allow_commit=True, allow_option=False, dynamic=True,
            method=method, commit_activation_optional=True,
        )
    if method == OPTION_STATIC_EXACT:
        return _choose_portfolio(scenario, allow_commit=True, allow_option=True, dynamic=False, method=method)
    if method == FORECAST_OPTION_THEN_CAP:
        candidates: list[tuple[tuple[T0ContractChoice, ...], Mapping[str, T1Decision]]] = []
        for portfolio in enumerate_portfolios(scenario):
            policy = _policy_for_portfolio(scenario, portfolio, dynamic=True, phase_order="capacity_then_cap")
            candidates.append((portfolio, policy))
        portfolio, policy = _best(candidates, lambda item: _plan_policy_value(scenario, item[0], item[1]))
        return _score_policy(method, scenario, portfolio, policy)
    if method == CAP_THEN_FORECAST_OPTION:
        candidates: list[tuple[tuple[T0ContractChoice, ...], Mapping[str, T1Decision]]] = []
        for portfolio in enumerate_portfolios(scenario):
            policy = _cap_then_forecast_policy(scenario, portfolio)
            candidates.append((portfolio, policy))
        portfolio, policy = _best(candidates, lambda item: _plan_policy_value(scenario, item[0], item[1]))
        return _score_policy(method, scenario, portfolio, policy)
    if method == PREBOUND_GRANT_OPTION:
        return _choose_prebound_portfolio(scenario, method=method, task_first=False)
    if method == OPTIONCAST_CAP_EXACT:
        return _choose_portfolio(scenario, allow_commit=True, allow_option=True, dynamic=True, method=method)
    raise AssertionError("method dispatch must be exhaustive")


def perfect_information_oracle(scenario: OptioncastScenario) -> OracleEvaluation:
    """Return the EVPI upper bound without giving W to the t0 portfolio step.

    For each *fixed* t0 portfolio, the oracle may choose a different legal
    t1 decision after observing W.  The portfolio is then selected by its
    prior-weighted W-score.  This makes it a valid upper bound on the single
    reference exact Z-policy, while keeping oracle information out of t0.
    """
    candidates: list[tuple[tuple[T0ContractChoice, ...], Mapping[str, T1Decision]]] = []
    for portfolio in enumerate_portfolios(scenario):
        decisions = enumerate_t1_decisions(scenario, portfolio)
        by_world = {
            world.world_id: _best(decisions, lambda d, world=world: _realise(scenario, portfolio, d, world)[1])
            for world in scenario.worlds
        }
        candidates.append((portfolio, by_world))
    portfolio, decisions_by_world = _best(
        candidates,
        lambda item: sum(
            world.probability_units / PROBABILITY_SCALE
            * _realise(scenario, item[0], item[1][world.world_id], world)[1]
            for world in scenario.worlds
        ),
    )
    expected_net = expected_gross = expected_t1 = 0.0
    expected_completed = expected_candidates = 0.0
    world_net: dict[str, float] = {}
    for world in scenario.worlds:
        gross, net, completed, candidates_count = _realise(
            scenario, portfolio, decisions_by_world[world.world_id], world,
        )
        weight = world.probability_units / PROBABILITY_SCALE
        world_net[world.world_id] = net
        expected_net += weight * net
        expected_gross += weight * gross
        expected_t1 += weight * _fees(portfolio, decisions_by_world[world.world_id], scenario)[1]
        expected_completed += weight * completed
        expected_candidates += weight * candidates_count
    t0 = _fees(portfolio, T1Decision((), ()), scenario)[0]
    denominator = sum(task.value for task in scenario.tasks) * scenario.value_milliunits_per_point
    ppm = expected_net / denominator * 1_000_000 if denominator else 0.0
    return OracleEvaluation(
        PERFECT_INFO_ORACLE, tuple(portfolio), dict(decisions_by_world), expected_net,
        expected_gross, float(t0), expected_t1, ppm, ppm / 10_000, world_net,
        expected_completed,
        0.0 if expected_candidates == 0 else max(0.0, expected_candidates - expected_completed) / expected_candidates,
        0.0,
    )


def evaluate_all(scenario: OptioncastScenario) -> dict[str, PolicyEvaluation]:
    """No 31-seed run occurs here; this evaluates one supplied scenario."""
    return {method: evaluate_method(scenario, method) for method in DEPLOYABLE_METHODS}


def paired_metrics(
    scenarios: Sequence[OptioncastScenario], *, proposed_method: str = OPTIONCAST_CAP_EXACT, baseline_method: str = COMMIT_ONLY_DYNAMIC_EXACT
) -> tuple[PairedScenarioMetric, ...]:
    """Return supplied-scenario paired metrics only; this never builds/runs a suite.

    The caller must pass the preregistered 31 score scenarios after an
    independent audit.  Pairing is at scenario/seed level, not at the four
    correlated worlds inside a scenario.
    """
    rows: list[PairedScenarioMetric] = []
    for scenario in scenarios:
        proposed = evaluate_method(scenario, proposed_method)
        baseline = evaluate_method(scenario, baseline_method)
        rows.append(PairedScenarioMetric(
            scenario.scenario_id, proposed.net_value_pp, baseline.net_value_pp,
            proposed.net_value_pp - baseline.net_value_pp,
            proposed.net_value_ppm, baseline.net_value_ppm,
            proposed.net_value_ppm - baseline.net_value_ppm,
        ))
    return tuple(rows)


__all__ = [
    "COMMIT_ONLY_DYNAMIC_EXACT", "COMMIT_PORTFOLIO_DYNAMIC_EXACT", "OPTION_STATIC_EXACT", "FORECAST_OPTION_THEN_CAP", "CAP_THEN_FORECAST_OPTION", "PREBOUND_GRANT_OPTION", "OPTIONCAST_CAP_EXACT", "PERFECT_INFO_ORACLE", "DEPLOYABLE_METHODS", "CONTROL_MESSAGE_CAPACITY_UNITS", "CONTROL_MESSAGE_ENERGY_MILLIUNITS", "CONTROL_MESSAGE_PRICE_MILLIUNITS", "ContactActivation", "TaskAssignment", "T1Decision", "PolicyEvaluation", "OracleEvaluation", "PairedScenarioMetric", "enumerate_portfolios", "enumerate_prebound_grants", "enumerate_t1_decisions", "decision_plan_value", "evaluate_method", "perfect_information_oracle", "evaluate_all", "paired_metrics",
]
