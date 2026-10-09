"""NB2 walkthrough package: a synthetic task with known latent truth.

Submodules (import them explicitly; nothing is re-exported here so that the
package stays import-safe for NB2, NB3 and the pytest suite):

- ``task``        synthetic teacher--student task + its known optimum
- ``preprocess``  train-only fit/evaluation split, scaler, hashes
- ``incidents``   seeded telemetry traces + the no-hindsight online diagnosis
- ``seed_audit``  real paired-seed sweep and the seed-aware summary
- ``pipeline``    the interface NB3 reuses (published JSON, cached dataset)

Every number published by this package is a project choice derived here, not a
value quoted from the three papers.
"""
