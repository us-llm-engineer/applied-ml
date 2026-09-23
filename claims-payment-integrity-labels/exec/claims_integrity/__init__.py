"""Public surface of the claims-integrity package.

Stack note: the CCEM core is a *derived computational adaptation* -- a
numpy/scipy linear-softmax classifier with per-reviewer column-stochastic
confusion matrices, preserving the paper's Eq. (2)/(6) and its permutation
ambiguity.  It is not a claim that the source paper requires a neural
architecture.

See ``tests/INDEX.md`` for the behavioural contract exercised by this
package's frozen test suite.
"""

from __future__ import annotations

from .ccem import (
    align_posteriors,
    best_label_permutation,
    ccem_objective,
    ccem_observed_probability,
    fit_ccem,
    run_ccem_identifiability_experiment,
    vote_posterior,
)
from .config import CLASS_NAMES, CONFIG, PROVENANCE, canonical_config
from .experiments import (
    capacity_study,
    calibration_falsifier_study,
    dollars_study,
    fit_scorers,
    scorec_certificate_control,
    scrc_exchangeable_control,
    wilson_interval,
)
from .figures import (
    build_all,
    capacity_rows,
    dollars_rows,
    identifiability_rows,
    render_capacity,
    render_dollars,
    render_identifiability,
    write_figure,
)
from .ledger import config_hash, record_run, write_ledger
from .monitors import agreement_drift_monitor, pairwise_agreement, queue_volume_monitor
from .scoring import (
    build_features,
    ccem_posterior,
    fit_naive_scorer,
    fit_noise_aware_scorer,
    identify_clean_class,
    no_leakage_probe,
    overpayment_score,
    score_matrix,
    three_way_split,
)
from .selective import (
    acceptance_rate,
    bounded_loss,
    clopper_pearson_lcb,
    compare_calibration_sampling,
    empirical_bernstein_slack,
    scorec_certificate,
    scrc_inductive_policy,
    selected_risk,
    select_scorc_policy,
    utility,
)
from .synthetic import generate_synthetic_claims, make_claim_stream, stream_digest

PACKAGE_VERSION = "1.0.0"

__all__ = [
    "PACKAGE_VERSION",
    "CLASS_NAMES",
    "CONFIG",
    "PROVENANCE",
    "canonical_config",
    "record_run",
    "config_hash",
    "write_ledger",
    "generate_synthetic_claims",
    "make_claim_stream",
    "stream_digest",
    "ccem_observed_probability",
    "ccem_objective",
    "best_label_permutation",
    "align_posteriors",
    "vote_posterior",
    "fit_ccem",
    "run_ccem_identifiability_experiment",
    "selected_risk",
    "acceptance_rate",
    "bounded_loss",
    "utility",
    "select_scorc_policy",
    "scrc_inductive_policy",
    "scorec_certificate",
    "clopper_pearson_lcb",
    "empirical_bernstein_slack",
    "compare_calibration_sampling",
    "build_features",
    "no_leakage_probe",
    "three_way_split",
    "fit_naive_scorer",
    "fit_noise_aware_scorer",
    "score_matrix",
    "ccem_posterior",
    "identify_clean_class",
    "overpayment_score",
    "fit_scorers",
    "capacity_study",
    "dollars_study",
    "calibration_falsifier_study",
    "scrc_exchangeable_control",
    "scorec_certificate_control",
    "wilson_interval",
    "agreement_drift_monitor",
    "queue_volume_monitor",
    "pairwise_agreement",
    "identifiability_rows",
    "capacity_rows",
    "dollars_rows",
    "render_identifiability",
    "render_capacity",
    "render_dollars",
    "write_figure",
    "build_all",
]
