# Presentation layer

This directory contains interview-facing material derived from the sealed
confirmatory metrics in `reports/metrics/`. It presents an **offline causal
targeting and policy-evaluation case study**, not a deployable system. The
sealed evidence remains unchanged; `../ERRATA.md` documents legacy labels.

## Deliverables

- `Uplift_Modeling_Interview_Case_Study.pptx`: five-slide, decision-first deck.
- `../output/pdf/uplift_modeling_interview_case_study.pdf`: one-page A4 case
  study for interview sharing.
- `figures/`: record-labelled PNG exports of the four diagnostic chart slides.

## Decision story

- Primary hypothetical scenario: contribution margin 100, cost 0.50 per
  assigned target, and 100,000 eligible records.
- Economic candidate set: `response`, `t_learner`, and `target_nobody`.
- The economic candidate scope was manually defined during protocol design and
  frozen before final evaluation. S-learner was not excluded through an
  executable screening test.
- Validation selected response at 2%; that policy and budget were frozen before
  the disjoint final sample was loaded. This was the best evaluated
  validation-grid decision under the stated hypothetical scenario, not a
  universal business optimum.
- Final incremental net value versus target nobody: 5,109.98 hypothetical
  currency units, with a conditional row-level paired-bootstrap 95% CI of
  1,815.61 to 8,171.56.
- T-learner did not satisfy the predeclared superiority rule, so response was
  retained as the benchmark champion among the evaluated fixed linear baseline
  models.

Response, S, and T used fixed SGD-logistic configurations. This experiment did
not establish superiority over tuned nonlinear uplift models, causal forests,
X/R/DR-learners, or every possible policy. S-learner remains visible in
diagnostics; at the separately evaluated 20% budget it was rejected as a
response challenger, which is not a global inferiority finding.

Intervals condition on sampled records, fitted models, frozen rankings, and the
frozen cutoff. They exclude model-refitting, policy-selection, and latent
advertiser/experiment cluster uncertainty. Clustered resampling would be
preferred if valid cluster identifiers were available; no artificial or
pseudo-clusters should be invented.

The primary estimator is empirical-marginal-propensity inverse-probability
weighting (`e_hat = mean(T)`). Historical `HT`, `ht`, and
`horvitz_thompson` labels mean HT-style IPW using that empirically estimated
marginal treatment propensity, not classical design-probability HT. Hájek
remains the separately labelled sensitivity estimate.

## Terminology

The source supplies records with anonymized features but no durable unique
person identifier. Presentation copy therefore uses:

- `eligible records targeted`, not `users targeted`;
- `per 1,000 eligible records` for population-normalized cumulative gain;
- `per 1,000 targeted records` for budget-specific comparisons; and
- `per 1,000 records in decile` for within-decile estimates.

S-learner reports an encoded feature width of `p = 5,230`; its estimator design
`[Z, T, Z*T]` contains `2p + 1 = 10,461` columns.

## Future deployment work

The case study is not production-ready. Deployment would require a serialized
encoder/model bundle, feature contract, deterministic top-k tie-breaking,
batch-scoring entry point, monitoring, and prospective online validation. None
of those components is built in this correction pass.

## Governance boundary

The original `reports/` artifacts and governed executable files are not edited
or regenerated. Presentation assets read those frozen outputs and live in a
separate namespace so that terminology and narrative can improve without
silently rewriting the evidence chain.
