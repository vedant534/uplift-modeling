# Rehearsal run — not confirmatory

This directory preserves the first implementation run and its protocol exactly
as they existed before the strict final-evaluation governance audit.

The numerical calculations and source-index disjointness checks passed, but the
run is **not eligible for confirmatory claims** because:

- the final-evaluation frame was materialized before validation decisions were
  frozen;
- the freeze artifact could be overwritten and the same final sample replayed;
- the outcome and several executable protocol semantics were not fail-closed
  protocol fields.

The active workflow supersedes these artifacts with a versioned protocol, a
new final sample that excludes this rehearsal sample, staged development/final
loading, and immutable one-shot claim and completion receipts.

