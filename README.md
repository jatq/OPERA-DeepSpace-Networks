# OPERA: Option-Based Planning and Execution with Recourse and Commit-Gated Authorization

This repository contains the experiment code and frozen evidence for
**OPERA**, a two-stage planning framework for deep-space contact acquisition,
forecast-conditioned execution, indivisible task loading, and commit-gated
authorization.

The software accompanies a research manuscript under review. It is research
code, not operational flight or ground-segment software.

## What the code implements

- composite none, commit, and option contract portfolios at `t0`;
- public-forecast-conditioned decisions at `t1`;
- VCM mode selection, contact activation, task loading, and authorization
  consumption under shared budgets;
- exact small-instance evaluation, an expected-capacity MILP projection,
  Sequential-1Swap, Hybrid, and budgeted LNS-2 search;
- price-budget sensitivity experiments, equal-wall-clock scalability
  comparisons, statistical summaries, and integrity manifests.

Development identifiers such as `OPTIONCAST-CAP`, `OPTIONCASTCAPExact`, and
`CAPAuthorizationLedger` remain in executable code and frozen JSON evidence
for byte-level reproducibility. OPERA is the publication name.

## Repository layout

```text
experiments/
  experimental/               core model, generators, solvers, and runners
  q2_strengthening/           price-budget and full two-stage experiments
evidence/
  reference_exact/            31-seed exact reference results
  e1/                         1,440 price-budget records
  e2_equal_budget/            90 paired scalability instances
  e3_full_two_stage_exploratory/
  DATA_PROVENANCE.md          external timing-data provenance and scope
```

`evidence/` contains frozen row-level outputs and summaries. The external SOHO
and InSight source files are not redistributed; `evidence/DATA_PROVENANCE.md`
records their public sources, hashes, counts, and segmentation rules.

## Environment

The original frozen evidence was produced with Python 3.10 on Linux. The
reproduction path was also exercised with Python 3.12 on Windows.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
export PYTHONPATH=experiments
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
$env:PYTHONPATH = "experiments"
```

## Tests

```bash
python -m pytest -q \
  experiments/q2_strengthening/test_e1_pricing.py \
  experiments/q2_strengthening/test_e1_run.py \
  experiments/q2_strengthening/test_e2_two_stage_scaling.py \
  experiments/experimental/test_run_v10_10_lns2_audit.py \
  experiments/experimental/test_equal_budget_benchmark.py
```

## Reproduction commands

Reference exact comparison:

```bash
python -m experimental.run_reference_exact \
  --output-dir results/reference-exact
```

Equal-budget post-observation comparison:

```bash
python -m experimental.run_equal_budget_benchmark \
  --formal --budget-seconds 1.0 \
  --output results/e2-equal-budget/summary.json
```

E1 price-budget sensitivity:

```bash
python -m q2_strengthening.e1_run \
  --formal --output-dir results/e1-formal
```

Exploratory full two-stage search:

```bash
python -m q2_strengthening.e2_two_stage_scaling \
  --formal --budget-seconds 30 \
  --rows-output results/e3-full-two-stage/rows.jsonl \
  --summary-output results/e3-full-two-stage/summary.json
```

Formal runners preserve the scenario as the statistical unit. Observation
branches and latent worlds are not treated as independent samples. The
equal-budget and full-two-stage runners use fresh subprocesses for peak-memory
isolation and enforce one numerical thread.

## Interpretation boundaries

- LNS-2, Hybrid, and the exploratory outer search are bounded heuristics, not
  claimed globally optimal solvers.
- The external replay uses measured timing with controlled capacity, task,
  price, failure, forecast, and authorization overlays. It does not estimate
  operational DSN throughput.
- Smoke tests validate interfaces, constraints, and output paths; they are not
  evidence of comparative performance.
- The authorization model is a planning and state-machine model. It is not a
  deployment-level Byzantine-security proof.

## Integrity

`SHA256SUMS.txt` records SHA-256 digests for the public repository files except
the checksum file itself. To verify the checkout:

```bash
sha256sum -c SHA256SUMS.txt
```

On Windows, verify individual files with:

```powershell
Get-FileHash .\experiments\experimental\v10_3_optioncast_model.py -Algorithm SHA256
```

Compare the result with the corresponding line in `SHA256SUMS.txt`.

## Citation

Citation metadata is provided in `CITATION.cff`. Until the manuscript is
published, cite this repository and the manuscript title.

## License

The code and derived evidence in this repository are released under the MIT
License. See `LICENSE`.

## Contact

Corresponding author: Shengli Zhang, Shenzhen University
(`zsl@szu.edu.cn`).
