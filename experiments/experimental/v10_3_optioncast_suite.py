"""Frozen, score-blind V10.3 OPTIONCAST-CAP scenario generator.

The generator never imports an evaluator and never filters a seed by a policy
result.  Calendar/prices, workload, truth, beacon delivery, and observations
use separately named deterministic random streams.
"""

from __future__ import annotations

import hashlib
import random
from itertools import product

from experimental.v10_3_optioncast_model import (
    FIRST_CONTACT_EPOCH,
    PROBABILITY_SCALE,
    CandidateContact,
    ForecastModel,
    ForecastObservation,
    OptioncastScenario,
    PolicyCapsule,
    Task,
    VcmMode,
    World,
    validate_scenario,
)


SUITE_VERSION = "v10.3-optioncast-cap-scoreblind-1"
SCORE_SEEDS = tuple(range(31))
# Development seeds are intentionally disjoint from the frozen score set.
DEVELOPMENT_SEEDS = tuple(range(10_000, 10_005))
SEEDS = SCORE_SEEDS
STATIONS = ("dss-a", "dss-b", "dss-c")
VCM_MODES = (
    VcmMode("robust-qpsk", capacity_units_per_epoch=2, required_esn0_cdb=400, energy_milliunits_per_epoch=10),
    VcmMode("fast-8psk", capacity_units_per_epoch=4, required_esn0_cdb=660, energy_milliunits_per_epoch=18),
)


def _rng(seed: int, stream: str) -> random.Random:
    digest = hashlib.sha256(f"{SUITE_VERSION}:{seed}:{stream}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _contacts(seed: int) -> tuple[CandidateContact, ...]:
    rng = _rng(seed, "calendar-prices")
    count = 4 + rng.randrange(3)
    # The first pair creates a natural cross-station single-RF conflict.  The
    # remaining windows are calendar draws, not fitted to tasks or outcomes.
    starts = [FIRST_CONTACT_EPOCH + rng.randrange(3), FIRST_CONTACT_EPOCH + 1 + rng.randrange(3)]
    starts.extend(rng.randrange(FIRST_CONTACT_EPOCH + 4, 28) for _ in range(count - 2))
    starts.sort()
    contacts: list[CandidateContact] = []
    for index, start in enumerate(starts):
        duration = rng.choice((3, 4, 5))
        commit_fee = rng.choice((90, 105, 125, 145, 165))
        option_fee = rng.choice((15, 20, 25, 30, 35))
        exercise_fees = {
            "robust-qpsk": rng.choice((30, 40, 50, 60)),
            "fast-8psk": rng.choice((45, 55, 65, 75)),
        }
        contacts.append(CandidateContact(
            contact_id=f"c{seed:02d}-{index}", station_id=STATIONS[index % len(STATIONS)],
            start_epoch=start, end_epoch=start + duration, clear_sky_esn0_cdb=rng.choice((480, 560, 640, 720, 780)),
            capacity_units=rng.choice((6, 8, 10, 12, 14)), commit_fee_milliunits=commit_fee,
            option_fee_milliunits=option_fee, exercise_fee_milliunits_by_mode=exercise_fees,
        ))
    return tuple(contacts)


def _tasks(seed: int) -> tuple[Task, ...]:
    """Workload stream deliberately has no access to the contact calendar."""
    rng = _rng(seed, "tasks")
    output: list[Task] = []
    for index in range(6 + rng.randrange(3)):
        release = rng.randrange(0, 18)
        output.append(Task(
            task_id=f"t{seed:02d}-{index}", release_epoch=release,
            deadline_epoch=release + rng.choice((8, 10, 12, 14, 16)),
            payload_units=rng.choice((2, 3, 4, 5, 6)), value=rng.randrange(4, 15),
        ))
    return tuple(output)


def _capsules(seed: int, tasks: tuple[Task, ...], contacts: tuple[CandidateContact, ...]) -> tuple[PolicyCapsule, ...]:
    # The policy whitelist is intentionally broad and generated mechanically;
    # no task is assigned a bespoke "good" contact.
    contact_ids = tuple(item.contact_id for item in contacts)
    mode_ids = tuple(item.mode_id for item in VCM_MODES)
    return tuple(PolicyCapsule(
        capsule_id=f"cap{seed:02d}-{task.task_id}", task_id=task.task_id,
        contact_whitelist=contact_ids, mode_whitelist=mode_ids, expires_epoch=40,
        contract_version="optioncast-v10.3", initial_counter=0,
    ) for task in tasks)


def _worlds(seed: int, contacts: tuple[CandidateContact, ...]) -> tuple[World, ...]:
    truth_rng = _rng(seed, "truth")
    delivery_rng = _rng(seed, "beacon-delivery")
    stations = tuple(sorted({item.station_id for item in contacts}))
    worlds: list[World] = []
    for index in range(4):
        # At least one all-clear world and several independently generated
        # degradations provide positive success and failure probability.
        weather_good = True if index == 0 else bool(truth_rng.randrange(2))
        healthy = {station: True if index == 0 else truth_rng.random() < 0.76 for station in stations}
        if index == 2:
            healthy[stations[truth_rng.randrange(len(stations))]] = False
        arrivals: dict[str, int | None] = {}
        for contact in contacts:
            arrival_deadline = contact.start_epoch - 1
            if index == 0:
                arrivals[contact.contact_id] = arrival_deadline
            else:
                draw = delivery_rng.random()
                arrivals[contact.contact_id] = None if draw < 0.16 else arrival_deadline if draw < 0.76 else arrival_deadline + 1
        worlds.append(World(
            world_id=f"w{index}", probability_units=PROBABILITY_SCALE // 4,
            weather_good=weather_good, station_healthy=healthy, beacon_arrival_epoch_by_contact=arrivals,
        ))
    return tuple(worlds)


def _observations_and_channel(
    seed: int, worlds: tuple[World, ...], contacts: tuple[CandidateContact, ...], *, mismatch: str
) -> tuple[tuple[ForecastObservation, ...], dict[str, dict[str, float]], dict[str, dict[str, float]], ForecastModel, ForecastModel]:
    rng = _rng(seed, "observations")
    stations = tuple(sorted({item.station_id for item in contacts}))
    observations = tuple(
        ForecastObservation(
            observation_id=f"z{index}", weather_signal_good=weather_signal,
            station_health_signal=dict(zip(stations, station_signals)),
        )
        for index, (weather_signal, station_signals) in enumerate(product((False, True), product((False, True), repeat=len(stations))))
    )
    score_model = ForecastModel(weather_accuracy_units=8_000 + rng.randrange(801), station_health_accuracy_units=8_200 + rng.randrange(601))
    if mismatch == "matching":
        plan_model = ForecastModel(score_model.weather_accuracy_units, score_model.station_health_accuracy_units)
    elif mismatch == "moderate":
        plan_model = ForecastModel(max(5_500, score_model.weather_accuracy_units - 1_300), max(5_500, score_model.station_health_accuracy_units - 1_100))
    elif mismatch == "severe":
        # A frozen anti-calibration control: P_plan assigns each signal bit a
        # directionally reversed likelihood relative to the scoring channel.
        # This remains a full-support stochastic channel (Z never names W),
        # rather than a truth-label leak.
        plan_model = ForecastModel(3_800, 3_500)
    elif mismatch == "noinfo":
        # The null-information control must remove information from both the
        # physical scoring channel and the policy's planning channel.  With
        # 0.5 accuracy for every independent bit, every Z has the same
        # probability in every W, so observing Z leaves the prior unchanged.
        score_model = ForecastModel(5_000, 5_000)
        plan_model = ForecastModel(5_000, 5_000)
    else:
        raise ValueError("mismatch must be matching, moderate, severe, or noinfo")

    def channel(model: ForecastModel) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for world in worlds:
            row: dict[str, float] = {}
            for observation in observations:
                probability = (model.weather_accuracy_units if world.weather_good == observation.weather_signal_good
                               else PROBABILITY_SCALE - model.weather_accuracy_units) / PROBABILITY_SCALE
                for station in stations:
                    accuracy = model.station_health_accuracy_units
                    probability *= (accuracy if world.station_healthy[station] == observation.station_health_signal[station]
                                    else PROBABILITY_SCALE - accuracy) / PROBABILITY_SCALE
                row[observation.observation_id] = probability
            table[world.world_id] = row
        return table

    return observations, channel(score_model), channel(plan_model), score_model, plan_model


def make_scenario(seed: int, *, mismatch: str = "matching") -> OptioncastScenario:
    contacts = _contacts(seed)
    tasks = _tasks(seed)
    worlds = _worlds(seed, contacts)
    observations, score_likelihoods, plan_likelihoods, score_model, plan_model = _observations_and_channel(seed, worlds, contacts, mismatch=mismatch)
    scenario = OptioncastScenario(
        scenario_id=f"optioncast-v103-{seed:02d}-{mismatch}", contacts=contacts, tasks=tasks, worlds=worlds,
        forecast_observations=observations, score_likelihood_by_world=score_likelihoods,
        plan_likelihood_by_world=plan_likelihoods, score_forecast_model=score_model, plan_forecast_model=plan_model,
        vcm_modes=VCM_MODES, policy_capsules=_capsules(seed, tasks, contacts),
        station_slot_limits={station: 1 for station in STATIONS}, contract_budget_milliunits=105,
        exercise_budget_milliunits=150, option_count_budget=2, energy_budget_milliunits=150,
        final_grant_budget=2, uplink_latency_epochs=1, validation_latency_epochs=1,
    )
    validate_scenario(scenario)
    return scenario


def build_suite(*, mismatch: str = "matching", seeds: tuple[int, ...] = SCORE_SEEDS) -> tuple[OptioncastScenario, ...]:
    return tuple(make_scenario(seed, mismatch=mismatch) for seed in seeds)


def suite_digest(suite: tuple[OptioncastScenario, ...]) -> str:
    return hashlib.sha256(repr(suite).encode("utf-8")).hexdigest()


__all__ = [
    "SUITE_VERSION", "SCORE_SEEDS", "DEVELOPMENT_SEEDS", "SEEDS", "STATIONS", "VCM_MODES", "make_scenario", "build_suite", "suite_digest",
]
