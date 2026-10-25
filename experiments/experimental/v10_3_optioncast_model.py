"""Auditable primitives for V10.3 OPTIONCAST-CAP.

This module deliberately models a *real* t0 contract choice: a contact is
not purchased, committed, or held as a non-refundable option.  An option can
be exercised once at t1 or released once; only exercise creates transmission
capacity.  CAP is represented independently as a fail-closed, atomic ledger.
No policy evaluator lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isclose
from typing import Mapping, Sequence


DATA_UNIT_BYTES = 1_000
PROBABILITY_SCALE = 10_000
T0_EPOCH = 0
T1_EPOCH = 5
FIRST_CONTACT_EPOCH = 8
FINAL_GRANT_BUDGET_B = 2
# A control-plane authorization message is not free: whether it is produced
# through CAP at t1 or carried by the prebound authorization path, it consumes
# the same contact capacity, spacecraft energy, and priced service unit.  The
# evaluator imports these protocol constants; it must not redefine them.
CONTROL_MESSAGE_CAPACITY_UNITS = 1
CONTROL_MESSAGE_ENERGY_MILLIUNITS = 2
CONTROL_MESSAGE_PRICE_MILLIUNITS = 2


@dataclass(frozen=True)
class VcmMode:
    mode_id: str
    capacity_units_per_epoch: int
    required_esn0_cdb: int
    energy_milliunits_per_epoch: int


@dataclass(frozen=True)
class CandidateContact:
    contact_id: str
    station_id: str
    start_epoch: int
    end_epoch: int
    clear_sky_esn0_cdb: int
    capacity_units: int
    commit_fee_milliunits: int
    option_fee_milliunits: int
    exercise_fee_milliunits_by_mode: Mapping[str, int]
    station_slot_units: int = 1
    radio_id: str = "spacecraft-rf-1"

    @property
    def duration_epochs(self) -> int:
        return self.end_epoch - self.start_epoch


@dataclass(frozen=True)
class Task:
    task_id: str
    release_epoch: int
    deadline_epoch: int
    payload_units: int
    value: int


@dataclass(frozen=True)
class World:
    """Exogenous scoring truth W; policies never receive this object."""

    world_id: str
    probability_units: int
    weather_good: bool
    station_healthy: Mapping[str, bool]
    # ``None`` means the t1 ExerciseBeacon never reaches the source.  A
    # finite epoch can still violate the physical propagation lower bound or
    # the contact-specific source-arrival deadline.
    beacon_arrival_epoch_by_contact: Mapping[str, int | None]


@dataclass(frozen=True)
class ForecastObservation:
    """A finite observation Z available at t1, never a truth-world label."""

    observation_id: str
    weather_signal_good: bool
    station_health_signal: Mapping[str, bool]


@dataclass(frozen=True)
class ForecastModel:
    weather_accuracy_units: int
    station_health_accuracy_units: int


@dataclass(frozen=True)
class PolicyCapsule:
    """Signed non-final authorization policy.  It carries no object digest."""

    capsule_id: str
    task_id: str
    contact_whitelist: tuple[str, ...]
    mode_whitelist: tuple[str, ...]
    expires_epoch: int
    contract_version: str
    initial_counter: int

    def permits(self, *, contact_id: str, mode_id: str, epoch: int, contract_version: str) -> bool:
        return (
            contact_id in self.contact_whitelist
            and mode_id in self.mode_whitelist
            and epoch <= self.expires_epoch
            and contract_version == self.contract_version
        )


class ContractKind(str, Enum):
    NONE = "none"
    COMMIT = "commit"
    OPTION = "option"


class ContractState(str, Enum):
    NONE = "none"
    COMMITTED = "committed"
    OPTION_HELD = "option_held"
    EXERCISED = "exercised"
    RELEASED = "released"
    EXPIRED = "expired"


@dataclass(frozen=True)
class T0ContractChoice:
    contact_id: str
    kind: ContractKind
    # A committed contact fixes its VCM at t0.  Options choose VCM only when
    # exercised, which is the contract's conditional action.
    commit_mode_id: str | None = None


@dataclass
class ContractRecord:
    contract_id: str
    contact: CandidateContact
    kind: ContractKind
    state: ContractState
    t0_fee_milliunits: int
    contract_version: str
    mode_id: str | None = None
    exercise_fee_milliunits: int = 0
    exercised_epoch: int | None = None


@dataclass(frozen=True)
class ExerciseBeacon:
    beacon_id: str
    capsule_id: str
    task_id: str
    contract_id: str
    contact_id: str
    mode_id: str
    byte_cap_units: int
    issued_epoch: int
    counter: int


@dataclass(frozen=True)
class ObjectCommit:
    task_id: str
    object_digest: str
    beacon_id: str
    contact_id: str
    mode_id: str
    byte_count_units: int
    counter: int


@dataclass(frozen=True)
class PreboundGrant:
    """t0 control: unlike CAP, this irreversibly occupies one B slot."""

    grant_id: str
    task_id: str
    contact_id: str
    mode_id: str
    issued_epoch: int = T0_EPOCH


@dataclass(frozen=True)
class OptioncastScenario:
    scenario_id: str
    contacts: tuple[CandidateContact, ...]
    tasks: tuple[Task, ...]
    worlds: tuple[World, ...]
    forecast_observations: tuple[ForecastObservation, ...]
    # P_score(Z|W), used only for scoring the fixed policy.
    score_likelihood_by_world: Mapping[str, Mapping[str, float]]
    # P_plan(Z|W), the policy's declared model.  It may intentionally differ.
    plan_likelihood_by_world: Mapping[str, Mapping[str, float]]
    score_forecast_model: ForecastModel
    plan_forecast_model: ForecastModel
    vcm_modes: tuple[VcmMode, ...]
    policy_capsules: tuple[PolicyCapsule, ...]
    station_slot_limits: Mapping[str, int]
    contract_budget_milliunits: int
    exercise_budget_milliunits: int
    option_count_budget: int
    energy_budget_milliunits: int
    final_grant_budget: int = FINAL_GRANT_BUDGET_B
    uplink_latency_epochs: int = 1
    validation_latency_epochs: int = 1
    value_milliunits_per_point: int = 100
    t0_epoch: int = T0_EPOCH
    t1_epoch: int = T1_EPOCH
    contract_version: str = "optioncast-v10.3"


def contacts_overlap(left: CandidateContact, right: CandidateContact) -> bool:
    return left.start_epoch < right.end_epoch and right.start_epoch < left.end_epoch


def radio_conflict(left: CandidateContact, right: CandidateContact) -> bool:
    return left.radio_id == right.radio_id and contacts_overlap(left, right)


def contact_can_serve_task(contact: CandidateContact, task: Task) -> bool:
    """Whole-contact eligibility; partial-window use is deliberately disallowed."""
    return task.release_epoch <= contact.start_epoch and contact.end_epoch <= task.deadline_epoch


def effective_esn0_cdb(contact: CandidateContact, world: World) -> int:
    if not world.station_healthy.get(contact.station_id, False):
        return -10_000
    return contact.clear_sky_esn0_cdb if world.weather_good else contact.clear_sky_esn0_cdb - 250


def contact_capacity_units(contact: CandidateContact, mode: VcmMode, world: World) -> int:
    if effective_esn0_cdb(contact, world) < mode.required_esn0_cdb:
        return 0
    return min(contact.capacity_units, contact.duration_epochs * mode.capacity_units_per_epoch)


def nominal_contact_capacity_units(contact: CandidateContact, mode: VcmMode) -> int:
    return min(contact.capacity_units, contact.duration_epochs * mode.capacity_units_per_epoch)


def exercise_cutoff_epoch(scenario: OptioncastScenario, contact: CandidateContact) -> int:
    """Latest issue time that leaves time for one uplink and validation."""
    return contact.start_epoch - scenario.uplink_latency_epochs - scenario.validation_latency_epochs


def beacon_arrival_deadline_epoch(scenario: OptioncastScenario, contact: CandidateContact) -> int:
    """Latest physical source-arrival time that leaves validation time."""
    return contact.start_epoch - scenario.validation_latency_epochs


def beacon_reaches_source(beacon: ExerciseBeacon, world: World, scenario: OptioncastScenario) -> bool:
    """Scoring-side delivery predicate; it exposes no future truth to policy."""
    arrival = world.beacon_arrival_epoch_by_contact.get(beacon.contact_id)
    contact = _contact_by_id(scenario, beacon.contact_id)
    return (
        arrival is not None
        and beacon.issued_epoch + scenario.uplink_latency_epochs <= arrival
        and arrival <= beacon_arrival_deadline_epoch(scenario, contact)
    )


def net_value_milliunits(
    gross_delivery_value: int, *, t0_contract_fees: int, exercise_fees: int, scenario: OptioncastScenario
) -> int:
    """Frozen net-value formula used by a later evaluator, without a score run."""
    return gross_delivery_value * scenario.value_milliunits_per_point - t0_contract_fees - exercise_fees


def exercise_fee(contact: CandidateContact, mode_id: str) -> int:
    try:
        return contact.exercise_fee_milliunits_by_mode[mode_id]
    except KeyError as exc:
        raise ValueError(f"contact {contact.contact_id} does not offer mode {mode_id}") from exc


def observation_ids(scenario: OptioncastScenario) -> tuple[str, ...]:
    return tuple(item.observation_id for item in scenario.forecast_observations)


def likelihood(scenario: OptioncastScenario, world: World, observation_id: str, *, planning: bool) -> float:
    table = scenario.plan_likelihood_by_world if planning else scenario.score_likelihood_by_world
    return table[world.world_id][observation_id]


def posterior_world_weights(scenario: OptioncastScenario, observation_id: str, *, planning: bool = True) -> dict[str, float]:
    table = scenario.plan_likelihood_by_world if planning else scenario.score_likelihood_by_world
    normalizer = sum(world.probability_units / PROBABILITY_SCALE * table[world.world_id][observation_id] for world in scenario.worlds)
    if normalizer <= 0.0:
        raise ValueError("cannot condition on a zero-probability observation")
    return {
        world.world_id: world.probability_units / PROBABILITY_SCALE * table[world.world_id][observation_id] / normalizer
        for world in scenario.worlds
    }


def _maps(scenario: OptioncastScenario) -> tuple[dict[str, CandidateContact], dict[str, VcmMode], dict[str, Task]]:
    return (
        {item.contact_id: item for item in scenario.contacts},
        {item.mode_id: item for item in scenario.vcm_modes},
        {item.task_id: item for item in scenario.tasks},
    )


def _contact_by_id(scenario: OptioncastScenario, contact_id: str) -> CandidateContact:
    contacts, _, _ = _maps(scenario)
    try:
        return contacts[contact_id]
    except KeyError as exc:
        raise ValueError("unknown contact") from exc


def validate_t0_contract_choices(choices: Sequence[T0ContractChoice], scenario: OptioncastScenario) -> None:
    """Validate fees, station reservation slots, and t0 contract semantics."""
    contacts, modes, _ = _maps(scenario)
    if len({choice.contact_id for choice in choices}) != len(choices):
        raise ValueError("at most one t0 contract per contact")
    if any(choice.contact_id not in contacts for choice in choices):
        raise ValueError("contract choice references an undeclared contact")
    if any(choice.kind is ContractKind.NONE for choice in choices):
        raise ValueError("none is represented by omitting the contact from choices")
    if any(choice.kind is ContractKind.COMMIT and choice.commit_mode_id not in modes for choice in choices):
        raise ValueError("a committed contact must fix an offered VCM mode at t0")
    if any(choice.kind is ContractKind.OPTION and choice.commit_mode_id is not None for choice in choices):
        raise ValueError("an option cannot pre-select its exercise VCM")
    if any(choice.kind is ContractKind.COMMIT and choice.commit_mode_id not in contacts[choice.contact_id].exercise_fee_milliunits_by_mode for choice in choices):
        raise ValueError("commit mode must be offered by that contact")
    fee = sum(
        contacts[choice.contact_id].commit_fee_milliunits if choice.kind is ContractKind.COMMIT
        else contacts[choice.contact_id].option_fee_milliunits
        for choice in choices
    )
    if fee > scenario.contract_budget_milliunits:
        raise ValueError("t0 contract budget exceeded")
    if sum(choice.kind is ContractKind.OPTION for choice in choices) > scenario.option_count_budget:
        raise ValueError("option-count budget exceeded")
    # Stations cannot sell more overlapping candidate slots than declared.  An
    # option still occupies a station reservation slot, even before exercise.
    for station_id, limit in scenario.station_slot_limits.items():
        selected = [contacts[item.contact_id] for item in choices if contacts[item.contact_id].station_id == station_id]
        boundaries = sorted({point for item in selected for point in (item.start_epoch, item.end_epoch)})
        for epoch in boundaries:
            occupied = sum(item.station_slot_units for item in selected if item.start_epoch <= epoch < item.end_epoch)
            if occupied > limit:
                raise ValueError("station reservation-slot limit exceeded")


class ContractLedger:
    """Monotonic t0/t1 contract state machine with non-refundable option fees."""

    def __init__(self, scenario: OptioncastScenario, choices: Sequence[T0ContractChoice]) -> None:
        validate_t0_contract_choices(choices, scenario)
        self.scenario = scenario
        contacts, modes, _ = _maps(scenario)
        self._modes = modes
        self.records: dict[str, ContractRecord] = {}
        self.exercise_spend_milliunits = 0
        self.energy_spend_milliunits = 0
        self.control_message_spend_milliunits = 0
        self.control_message_energy_milliunits = 0
        for choice in choices:
            contact = contacts[choice.contact_id]
            state = ContractState.COMMITTED if choice.kind is ContractKind.COMMIT else ContractState.OPTION_HELD
            fee = contact.commit_fee_milliunits if choice.kind is ContractKind.COMMIT else contact.option_fee_milliunits
            self.records[choice.contact_id] = ContractRecord(
                contract_id=f"contract:{choice.contact_id}", contact=contact, kind=choice.kind, state=state,
                t0_fee_milliunits=fee, contract_version=scenario.contract_version, mode_id=choice.commit_mode_id,
            )

    @property
    def t0_spend_milliunits(self) -> int:
        return sum(record.t0_fee_milliunits for record in self.records.values())

    def _assert_radio_available(self, candidate: ContractRecord) -> None:
        for existing in self.records.values():
            if existing.contract_id == candidate.contract_id or existing.state is not ContractState.EXERCISED:
                continue
            if radio_conflict(existing.contact, candidate.contact):
                raise ValueError("single spacecraft RF conflict at exercise")

    def exercise(self, contact_id: str, mode_id: str, *, epoch: int) -> ContractRecord:
        if epoch < self.scenario.t1_epoch:
            raise ValueError("an option cannot be exercised before t1")
        record = self.records.get(contact_id)
        if record is None or record.kind is not ContractKind.OPTION or record.state is not ContractState.OPTION_HELD:
            raise ValueError("only one held option may be exercised once")
        if epoch > exercise_cutoff_epoch(self.scenario, record.contact):
            raise ValueError("option exercise misses its uplink/validation cutoff")
        mode = self._modes.get(mode_id)
        if mode is None:
            raise ValueError("undeclared VCM mode")
        fee = exercise_fee(record.contact, mode_id)
        energy = record.contact.duration_epochs * mode.energy_milliunits_per_epoch
        if self.exercise_spend_milliunits + fee > self.scenario.exercise_budget_milliunits:
            raise ValueError("exercise-fee budget exceeded")
        if self.energy_spend_milliunits + energy > self.scenario.energy_budget_milliunits:
            raise ValueError("VCM energy budget exceeded")
        self._assert_radio_available(record)
        self.exercise_spend_milliunits += fee
        self.energy_spend_milliunits += energy
        record.mode_id = mode_id
        record.exercise_fee_milliunits = fee
        record.exercised_epoch = epoch
        record.state = ContractState.EXERCISED
        return record

    def activate_commit(self, contact_id: str, *, epoch: int) -> ContractRecord:
        """Activate an irrevocable commitment; it has no t1 exercise fee."""
        if epoch < self.scenario.t1_epoch:
            raise ValueError("commit activation cannot precede the decision guard")
        record = self.records.get(contact_id)
        if record is None or record.kind is not ContractKind.COMMIT or record.state is not ContractState.COMMITTED:
            raise ValueError("only an unactivated commitment can activate")
        if epoch > exercise_cutoff_epoch(self.scenario, record.contact):
            raise ValueError("commit activation misses its uplink/validation cutoff")
        assert record.mode_id is not None
        mode = self._modes[record.mode_id]
        energy = record.contact.duration_epochs * mode.energy_milliunits_per_epoch
        if self.energy_spend_milliunits + energy > self.scenario.energy_budget_milliunits:
            raise ValueError("VCM energy budget exceeded")
        self._assert_radio_available(record)
        self.energy_spend_milliunits += energy
        record.exercised_epoch = epoch
        record.state = ContractState.EXERCISED
        return record

    def release(self, contact_id: str, *, epoch: int) -> ContractRecord:
        if epoch < self.scenario.t1_epoch:
            raise ValueError("an option cannot release before t1")
        record = self.records.get(contact_id)
        if record is None or record.kind is not ContractKind.OPTION or record.state is not ContractState.OPTION_HELD:
            raise ValueError("only an unexercised option may be released")
        # Deliberately do not alter t0_spend_milliunits: option fees never refund.
        record.state = ContractState.RELEASED
        return record

    def active_contract(self, contact_id: str) -> ContractRecord:
        record = self.records.get(contact_id)
        if record is None or record.state is not ContractState.EXERCISED or record.mode_id is None:
            raise ValueError("contact has no exercised/active contract")
        return record

    def reserve_control_message(self) -> None:
        """Atomically account for one authorization control message.

        Pricing is included in net-value accounting, while energy is a hard
        spacecraft budget.  Keeping both counters in the contract ledger
        makes CAP and the independent prebound path obey the same physical
        resource rule instead of treating their beacons as free metadata.
        """
        if self.energy_spend_milliunits + CONTROL_MESSAGE_ENERGY_MILLIUNITS > self.scenario.energy_budget_milliunits:
            raise ValueError("control-message energy budget exceeded")
        self.energy_spend_milliunits += CONTROL_MESSAGE_ENERGY_MILLIUNITS
        self.control_message_energy_milliunits += CONTROL_MESSAGE_ENERGY_MILLIUNITS
        self.control_message_spend_milliunits += CONTROL_MESSAGE_PRICE_MILLIUNITS


@dataclass
class _CapsuleState:
    policy: PolicyCapsule
    next_counter: int
    beacon_id: str | None = None


@dataclass
class CAPAuthorizationLedger:
    """Fail-closed CAP ledger: ObjectCommit atomically spends one final grant."""

    scenario: OptioncastScenario
    contract_ledger: ContractLedger
    capsules: dict[str, _CapsuleState]
    beacons: dict[str, ExerciseBeacon] = field(default_factory=dict)
    commits_by_task: dict[str, ObjectCommit] = field(default_factory=dict)
    commits_by_digest: dict[str, ObjectCommit] = field(default_factory=dict)
    commits_by_beacon: dict[str, ObjectCommit] = field(default_factory=dict)
    committed_bytes_by_contract: dict[str, int] = field(default_factory=dict)
    consumed_final_grants: int = 0

    @classmethod
    def from_scenario(cls, scenario: OptioncastScenario, contract_ledger: ContractLedger) -> "CAPAuthorizationLedger":
        states = {policy.capsule_id: _CapsuleState(policy, policy.initial_counter) for policy in scenario.policy_capsules}
        return cls(scenario, contract_ledger, states)

    def issue_exercise_beacon(
        self, capsule_id: str, contact_id: str, *, byte_cap_units: int, epoch: int, counter: int
    ) -> ExerciseBeacon:
        state = self.capsules.get(capsule_id)
        if state is None or state.beacon_id is not None:
            raise ValueError("a PolicyCapsule can issue exactly one ExerciseBeacon")
        if counter != state.next_counter:
            raise ValueError("CAP counter replay or rollback")
        record = self.contract_ledger.active_contract(contact_id)
        assert record.mode_id is not None
        policy = state.policy
        if epoch < self.scenario.t1_epoch or not policy.permits(
            contact_id=contact_id, mode_id=record.mode_id, epoch=epoch, contract_version=record.contract_version
        ):
            raise ValueError("beacon violates policy capsule whitelist or expiry")
        capacity = nominal_contact_capacity_units(record.contact, self._mode(record.mode_id))
        if byte_cap_units <= 0 or byte_cap_units + CONTROL_MESSAGE_CAPACITY_UNITS > capacity:
            raise ValueError("beacon byte cap exceeds exercised contact capacity")
        existing = self.committed_bytes_by_contract.get(record.contract_id, 0)
        reserved = sum(
            item.byte_cap_units + CONTROL_MESSAGE_CAPACITY_UNITS
            for beacon_id, item in self.beacons.items()
            if item.contract_id == record.contract_id and beacon_id not in self.commits_by_beacon
        )
        if existing + reserved + byte_cap_units + CONTROL_MESSAGE_CAPACITY_UNITS > capacity:
            raise ValueError("beacon allocations exceed exercised capacity")
        self.contract_ledger.reserve_control_message()
        beacon = ExerciseBeacon(
            beacon_id=f"beacon:{capsule_id}:{counter}", capsule_id=capsule_id, task_id=policy.task_id,
            contract_id=record.contract_id, contact_id=contact_id, mode_id=record.mode_id,
            byte_cap_units=byte_cap_units, issued_epoch=epoch, counter=counter,
        )
        self.beacons[beacon.beacon_id] = beacon
        state.beacon_id = beacon.beacon_id
        state.next_counter += 1
        return beacon

    def object_commit(
        self, beacon_id: str, *, object_digest: str, byte_count_units: int, epoch: int, counter: int) -> ObjectCommit:
        beacon = self.beacons.get(beacon_id)
        if beacon is None:
            raise ValueError("ObjectCommit needs a valid ExerciseBeacon")
        state = self.capsules[beacon.capsule_id]
        if counter != state.next_counter:
            raise ValueError("CAP counter replay or rollback")
        task = next((item for item in self.scenario.tasks if item.task_id == beacon.task_id), None)
        if task is None:
            raise ValueError("beacon references no declared task")
        if not object_digest or byte_count_units != task.payload_units or byte_count_units > beacon.byte_cap_units:
            raise ValueError("invalid object digest or byte count")
        record = self.contract_ledger.active_contract(beacon.contact_id)
        if epoch < record.contact.start_epoch or epoch > record.contact.end_epoch:
            raise ValueError("ObjectCommit lies outside the exercised contact window")
        if beacon.task_id in self.commits_by_task or object_digest in self.commits_by_digest:
            raise ValueError("task and object digest commits are one-shot")
        if self.consumed_final_grants >= self.scenario.final_grant_budget:
            raise ValueError("ReceiptOnlyGrant budget exhausted")
        used = self.committed_bytes_by_contract.get(record.contract_id, 0)
        pending = sum(
            item.byte_cap_units + CONTROL_MESSAGE_CAPACITY_UNITS
            for pending_beacon_id, item in self.beacons.items()
            if item.contract_id == record.contract_id and pending_beacon_id not in self.commits_by_beacon
            and pending_beacon_id != beacon_id
        )
        if used + byte_count_units + CONTROL_MESSAGE_CAPACITY_UNITS + pending > nominal_contact_capacity_units(record.contact, self._mode(beacon.mode_id)):
            raise ValueError("ObjectCommit exceeds exercised contact capacity")
        # Mutations occur only after every validation above: this is the atomic
        # grant-consumption point required by CAP.
        commit = ObjectCommit(beacon.task_id, object_digest, beacon_id, beacon.contact_id, beacon.mode_id, byte_count_units, counter)
        self.commits_by_task[beacon.task_id] = commit
        self.commits_by_digest[object_digest] = commit
        self.commits_by_beacon[beacon_id] = commit
        # Persist both payload and the authorization message that enabled it.
        # Otherwise a later beacon issued after this commit could reuse the
        # already-consumed control-message unit and overbook the contact.
        self.committed_bytes_by_contract[record.contract_id] = (
            used + byte_count_units + CONTROL_MESSAGE_CAPACITY_UNITS
        )
        self.consumed_final_grants += 1
        state.next_counter += 1
        return commit

    def _mode(self, mode_id: str) -> VcmMode:
        for mode in self.scenario.vcm_modes:
            if mode.mode_id == mode_id:
                return mode
        raise ValueError("unknown VCM mode")


@dataclass(frozen=True)
class StructuralOptionWitness:
    task_id: str
    option_contact_id: str
    mode_id: str
    positive_world_ids: tuple[str, ...]


def structural_option_witnesses(scenario: OptioncastScenario) -> tuple[StructuralOptionWitness, ...]:
    """Conservative task-level predicate from the preregistration document.

    It does not score a policy.  It asks whether no legal *single-contact*
    t0 commitment can serve a task, while a budgeted option can be exercised
    causally after t1 and delivers in at least one positive-probability world.

    This is intentionally stricter than an arithmetic fee/capacity predicate:
    each claimed world executes a fresh ``ContractLedger`` and
    ``CAPAuthorizationLedger`` path.  It must pass t0 reservation budgets,
    the t1 cutoff, RF and energy checks, actual beacon arrival, actual-world
    capacity, atomic B-slot consumption, and whole-contact deadline checks.
    In particular, a nominally capacious contact with no delivered beacon is
    not a structural option witness.
    """
    def route_delivers(
        *, task: Task, contact: CandidateContact, mode: VcmMode, kind: ContractKind, world: World
    ) -> bool:
        """Execute one minimal causal route without consulting the evaluator.

        The function deliberately creates fresh ledgers for each truth world.
        Its use of ``world`` is scoring-side only: it verifies that a route
        selected from declared task/contact/mode data can actually finish in a
        positive-probability world.  It never constructs a W-indexed policy.
        """
        choice = T0ContractChoice(
            contact.contact_id, kind,
            commit_mode_id=mode.mode_id if kind is ContractKind.COMMIT else None,
        )
        try:
            contracts = ContractLedger(scenario, (choice,))
            if kind is ContractKind.COMMIT:
                contracts.activate_commit(contact.contact_id, epoch=scenario.t1_epoch)
            else:
                contracts.exercise(contact.contact_id, mode.mode_id, epoch=scenario.t1_epoch)
            cap = CAPAuthorizationLedger.from_scenario(scenario, contracts)
            capsule = next(item for item in scenario.policy_capsules if item.task_id == task.task_id)
            beacon = cap.issue_exercise_beacon(
                capsule.capsule_id, contact.contact_id, byte_cap_units=task.payload_units,
                epoch=scenario.t1_epoch, counter=cap.capsules[capsule.capsule_id].next_counter,
            )
            if not beacon_reaches_source(beacon, world, scenario):
                return False
            if contact_capacity_units(contact, mode, world) < task.payload_units + CONTROL_MESSAGE_CAPACITY_UNITS:
                return False
            cap.object_commit(
                beacon.beacon_id,
                object_digest=f"structural:{task.task_id}:{contact.contact_id}:{mode.mode_id}:{world.world_id}",
                byte_count_units=task.payload_units,
                epoch=contact.start_epoch,
                counter=cap.capsules[capsule.capsule_id].next_counter,
            )
            return cap.consumed_final_grants == 1
        except (StopIteration, ValueError):
            return False

    witnesses: list[StructuralOptionWitness] = []
    capsules_by_task = {item.task_id: item for item in scenario.policy_capsules}
    for task in scenario.tasks:
        policy = capsules_by_task.get(task.task_id)
        if policy is None:
            continue
        commit_possible = False
        option_witness: StructuralOptionWitness | None = None
        for contact in scenario.contacts:
            if not contact_can_serve_task(contact, task) or contact.contact_id not in policy.contact_whitelist:
                continue
            for mode in scenario.vcm_modes:
                if mode.mode_id not in policy.mode_whitelist or mode.mode_id not in contact.exercise_fee_milliunits_by_mode:
                    continue
                commit_worlds = tuple(
                    world.world_id for world in scenario.worlds
                    if world.probability_units > 0 and route_delivers(
                        task=task, contact=contact, mode=mode, kind=ContractKind.COMMIT, world=world
                    )
                )
                if commit_worlds:
                    commit_possible = True
                option_worlds = tuple(
                    world.world_id for world in scenario.worlds
                    if world.probability_units > 0 and route_delivers(
                        task=task, contact=contact, mode=mode, kind=ContractKind.OPTION, world=world
                    )
                )
                if option_worlds:
                    option_witness = StructuralOptionWitness(task.task_id, contact.contact_id, mode.mode_id, option_worlds)
        if not commit_possible and option_witness is not None:
            witnesses.append(option_witness)
    return tuple(witnesses)


def validate_prebound_grants(grants: Sequence[PreboundGrant], scenario: OptioncastScenario) -> None:
    """Validate the t0-fixed control and its irreversible B-slot consumption."""
    contacts, modes, tasks = _maps(scenario)
    if len(grants) > scenario.final_grant_budget:
        raise ValueError("prebound grants exceed the same ReceiptOnlyGrant budget B")
    if len({item.grant_id for item in grants}) != len(grants) or len({item.task_id for item in grants}) != len(grants):
        raise ValueError("prebound grant ids and tasks must be one-shot")
    for grant in grants:
        if grant.issued_epoch != scenario.t0_epoch:
            raise ValueError("a prebound grant must irreversibly lock B at t0")
        if grant.task_id not in tasks or grant.contact_id not in contacts or grant.mode_id not in modes:
            raise ValueError("prebound grant references undeclared items")
        if not contact_can_serve_task(contacts[grant.contact_id], tasks[grant.task_id]):
            raise ValueError("prebound grant requires whole-contact task eligibility")
        policy = next(item for item in scenario.policy_capsules if item.task_id == grant.task_id)
        if not policy.permits(contact_id=grant.contact_id, mode_id=grant.mode_id, epoch=scenario.t0_epoch, contract_version=scenario.contract_version):
            raise ValueError("prebound grant violates the same CAP whitelist")


def validate_scenario(scenario: OptioncastScenario) -> None:
    if not 4 <= len(scenario.contacts) <= 6 or not 6 <= len(scenario.tasks) <= 8:
        raise ValueError("V10.3 Stage-0 requires 4--6 contacts and 6--8 tasks")
    if scenario.t0_epoch != T0_EPOCH or scenario.t1_epoch != T1_EPOCH:
        raise ValueError("the Stage-0 decision epochs are frozen")
    if any(contact.start_epoch <= scenario.t1_epoch for contact in scenario.contacts):
        raise ValueError("all contacts must occur after the t1 guard")
    if any(exercise_cutoff_epoch(scenario, contact) < scenario.t1_epoch for contact in scenario.contacts):
        raise ValueError("every contact must leave time for t1 uplink and validation")
    if len({item.contact_id for item in scenario.contacts}) != len(scenario.contacts):
        raise ValueError("contact ids must be unique")
    if len({item.task_id for item in scenario.tasks}) != len(scenario.tasks):
        raise ValueError("task ids must be unique")
    if len({item.capsule_id for item in scenario.policy_capsules}) != len(scenario.policy_capsules):
        raise ValueError("PolicyCapsule ids must be unique")
    if {item.task_id for item in scenario.policy_capsules} != {item.task_id for item in scenario.tasks}:
        raise ValueError("exactly one PolicyCapsule is required for every task")
    if sum(world.probability_units for world in scenario.worlds) != PROBABILITY_SCALE or any(world.probability_units <= 0 for world in scenario.worlds):
        raise ValueError("world probabilities must be strictly positive and sum to scale")
    contacts, modes, tasks = _maps(scenario)
    if any(contact.duration_epochs <= 0 or contact.capacity_units <= 0 or contact.station_slot_units <= 0 for contact in scenario.contacts):
        raise ValueError("contacts need positive duration, capacity, and station slots")
    if any(contact.station_id not in scenario.station_slot_limits or scenario.station_slot_limits[contact.station_id] <= 0 for contact in scenario.contacts):
        raise ValueError("every station needs a positive slot limit")
    if any(contact.commit_fee_milliunits <= 0 or contact.option_fee_milliunits <= 0 for contact in scenario.contacts):
        raise ValueError("contract fees must be positive")
    if any(set(contact.exercise_fee_milliunits_by_mode) != set(modes) or any(fee <= 0 for fee in contact.exercise_fee_milliunits_by_mode.values()) for contact in scenario.contacts):
        raise ValueError("each contact needs a positive exercise fee for every VCM mode")
    if any(task.payload_units <= 0 or task.release_epoch >= task.deadline_epoch for task in scenario.tasks):
        raise ValueError("tasks need positive payloads and valid windows")
    if any(set(world.beacon_arrival_epoch_by_contact) != set(contacts) for world in scenario.worlds):
        raise ValueError("every truth world must define an independent beacon-delivery outcome per contact")
    if any(mode.capacity_units_per_epoch <= 0 or mode.energy_milliunits_per_epoch <= 0 for mode in scenario.vcm_modes):
        raise ValueError("VCM modes need positive capacity and energy")
    if (scenario.contract_budget_milliunits <= 0 or scenario.exercise_budget_milliunits <= 0
            or scenario.energy_budget_milliunits <= 0 or scenario.option_count_budget < 0
            or scenario.final_grant_budget <= 0 or scenario.uplink_latency_epochs < 0
            or scenario.validation_latency_epochs < 0 or scenario.value_milliunits_per_point <= 0):
        raise ValueError("all budgets must be valid")
    if any(
        policy.task_id not in tasks
        or not policy.contact_whitelist
        or not policy.mode_whitelist
        or set(policy.contact_whitelist) - set(contacts)
        or set(policy.mode_whitelist) - set(modes)
        or policy.expires_epoch < scenario.t1_epoch
        or policy.contract_version != scenario.contract_version
        or policy.initial_counter < 0
        for policy in scenario.policy_capsules
    ):
        raise ValueError("invalid PolicyCapsule whitelist, version, expiry, or counter")
    ids = set(observation_ids(scenario))
    if not ids or len(ids) != len(scenario.forecast_observations):
        raise ValueError("Z outcomes must be nonempty and unique")
    world_ids = {world.world_id for world in scenario.worlds}
    for table in (scenario.score_likelihood_by_world, scenario.plan_likelihood_by_world):
        if set(table) != world_ids:
            raise ValueError("P_score/P_plan require one row per truth world")
        for row in table.values():
            if set(row) != ids or any(value <= 0.0 or value >= 1.0 for value in row.values()) or not isclose(sum(row.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("each likelihood row must have full support and sum to one")
    for observation_id in ids:
        if not posterior_world_weights(scenario, observation_id, planning=False) or not posterior_world_weights(scenario, observation_id, planning=True):
            raise ValueError("every observation must have positive marginal probability")


__all__ = [
    "DATA_UNIT_BYTES", "PROBABILITY_SCALE", "T0_EPOCH", "T1_EPOCH", "FIRST_CONTACT_EPOCH", "FINAL_GRANT_BUDGET_B",
    "VcmMode", "CandidateContact", "Task", "World", "ForecastObservation", "ForecastModel", "PolicyCapsule",
    "ContractKind", "ContractState", "T0ContractChoice", "ContractRecord", "ExerciseBeacon", "ObjectCommit", "PreboundGrant",
    "OptioncastScenario", "ContractLedger", "CAPAuthorizationLedger", "StructuralOptionWitness", "contacts_overlap",
    "radio_conflict", "contact_can_serve_task", "effective_esn0_cdb", "contact_capacity_units", "nominal_contact_capacity_units",
    "exercise_fee", "exercise_cutoff_epoch", "beacon_arrival_deadline_epoch", "beacon_reaches_source", "net_value_milliunits", "observation_ids", "likelihood",
    "posterior_world_weights", "validate_t0_contract_choices", "validate_prebound_grants", "structural_option_witnesses", "validate_scenario",
]
