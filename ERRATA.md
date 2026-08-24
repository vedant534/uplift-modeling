# Evaluation Terminology Errata v1

**Issued:** 2026-08-18  
**Applies to:** confirmatory evaluation `conversion_final_v2_20260818`

This addendum corrects interpretation and terminology without changing the
estimator mathematics, numerical results, final sample, frozen decisions,
receipt chain, or sealed figures. The immutable protocol, freeze, access claim,
completion receipt, code manifest, and 26 receipt-covered outputs retain their
historical bytes and hashes.

## Corrected interpretation

1. **Confidence intervals.** All reported bootstrap intervals are conditional
   row-level paired-bootstrap confidence intervals. They condition on the
   sampled records, fitted models, frozen rankings, and frozen cutoff. They do
   not include model-refitting, policy-selection, or latent advertiser or
   experiment cluster uncertainty. Clustered resampling would be preferred if
   valid cluster identifiers were available. No artificial or pseudo-clusters
   should be invented.

2. **Primary estimator.** The primary estimator uses
   `e_hat = mean(T)` and should be called empirical-marginal-propensity
   inverse-probability weighting, or HT-style IPW using the empirically
   estimated marginal treatment propensity. It is not classical
   design-probability Horvitz-Thompson. Sealed fields containing `ht` or
   `horvitz_thompson` are legacy names. The separately reported Hajek estimate
   remains a sensitivity estimate.

3. **S-learner candidate scope.** The confirmatory economic candidate set was
   manually configured as response, T-learner, and target-nobody during
   protocol design and frozen before final evaluation. S-learner was not
   excluded through an executable screening threshold. Its weaker result as a
   response challenger at the separately evaluated 20% budget is
   budget-specific and does not prove global inferiority.

4. **Champion and budget claims.** Response is the benchmark champion among the
   evaluated fixed linear baseline models. Response, S, and T used fixed
   SGD-logistic configurations. The experiment did not establish superiority
   over tuned nonlinear uplift models, causal forests, X/R/DR-learners, or
   every possible policy. The 2% policy was the best evaluated validation-grid
   decision under the stated hypothetical scenario, not a universal business
   optimum.

5. **S-learner dimensionality.** The encoded covariate width is `p = 5,230`.
   The S-learner estimator design `[Z, T, Z*T]` has
   `2p + 1 = 10,461` columns. The sealed `encoded_feature_count = 5,230` field
   describes encoded width only.

6. **Records, not users.** Dataset rows are source records, not verified unique
   users. Read historical `users targeted` as `eligible records targeted`, and
   `per 1,000 eligible users` as `per 1,000 eligible records`. Sealed figures
   are not regenerated solely to change those labels.

7. **Offline scope.** This is an offline causal targeting and policy-evaluation
   case study, not a production-ready or deployable policy. Deployment would
   require a serialized encoder/model bundle, feature contract, deterministic
   top-k tie-breaking, batch-scoring entry point, monitoring, and prospective
   online validation. Those components are outside this correction pass.

## Historical artifact mapping

- `decision_protocol.json`, the freeze, claim, receipt, `reports/metrics/`, and
  `reports/figures/` are preserved because they are hash-bound evidence.
- Machine-readable names such as `primary_estimator: horvitz_thompson`,
  `policy_value_estimator: horvitz_thompson`, `ht`, and
  `development_rejected_candidate: s_learner` are historical contract labels;
  the corrected interpretations above govern prose use.
- Interview-facing README, PowerPoint, PDF, and their future-generation sources
  are unsealed narrative artifacts and may be corrected without changing the
  completed evaluation.
