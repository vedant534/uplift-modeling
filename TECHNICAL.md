# Technical Evaluation and Governance Notes

## Governed result

All three decision-analysis improvements are implemented and the version-2
confirmatory evaluation is complete. The immutable
[`completion receipt`](reports/metrics/final-evaluation-conversion_final_v2_20260818.receipt.json)
verifies the protocol, frozen validation decisions, one-shot final-access claim,
executable-code manifest, and every published result.

This is an **offline causal targeting and policy-evaluation case study**. The
sealed v2 evidence is preserved verbatim; [`ERRATA.md`](ERRATA.md) records the
post-evaluation terminology corrections and the legacy names that remain in
immutable files.

The old 20% headline remains exploratory: this project snapshot contains no
dated evidence that 20% was chosen before those results were inspected. The
first remediation attempt is also excluded because an integration audit found
that its final frame had been materialized before validation decisions were
frozen. It is preserved as an invalid rehearsal under
[`audit/rehearsal_20260817/`](audit/rehearsal_20260817/).

For the predeclared primary scenario—100 hypothetical currency units of
contribution margin per conversion, 0.50 per assigned target, and 100,000
eligible records—validation selected the **response policy at 2%**. On the new
disjoint final sample:

| Frozen final estimate | Result |
|---|---:|
| Assigned targets | 2,000 |
| Estimated incremental conversions | 61.10 |
| Incremental net value vs target nobody | 5,109.98 |
| Conditional row-level paired-bootstrap 95% CI | [1,815.61, 8,171.56] |
| Break-even cost per assigned target | 3.0550 |
| Break-even-cost 95% CI | [1.4078, 4.5858] |

These are hypothetical scenario units, not observed advertiser profit. The 2%
choice is the **best evaluated validation-grid decision under this frozen
hypothetical scenario**, not a universal business optimum.

All intervals are **conditional row-level paired-bootstrap confidence
intervals**. They condition on the sampled records, fitted models, frozen
rankings, and frozen cutoff. They do not include model-refitting,
policy-selection, or latent advertiser/experiment cluster uncertainty.
Clustered resampling would be preferred if valid cluster identifiers were
available; none are available here, and no artificial or pseudo-clusters should
be invented.

## Decision conclusions

### 1. The 20% choice was not validated

No provenance in this snapshot establishes an ex-ante 20% decision. The active
[`decision protocol`](decision_protocol.json) therefore selected policy and
budget only on the development sample's validation partition, froze the result,
and then evaluated it once on a previously untouched 400,000-row sample. The
selection grid was 1%–100% plus `target_nobody`; final outcomes were never used
to change the frozen choice.

The confirmatory economic candidate scope was manually defined during protocol
design as `{response, T-learner, target_nobody}` and frozen before final
evaluation; **S-learner was not excluded through an executable screening
test**. At the separately evaluated legacy 20% operating point on development
validation, response produced 1.055 cumulative HT-style IPW incremental
conversions per 1,000 eligible records, compared with 0.890 for T and 0.751 for
S. That result supports rejecting S as a response challenger at 20%; it does not
establish global inferiority across budgets or economic scenarios.

Fresh 20% estimates are retained only as continuity diagnostics. They later
corroborated the budget-specific ordering: response delivered 5.025 HT-style IPW incremental
conversions per 1,000 targeted records, compared with 4.047 for T and 3.820 for
S. Paired differences versus response were negative for both T (−0.978, 95% CI
[−1.752, −0.163]) and S (−1.205, 95% CI [−2.134, −0.189]). These final-sample
diagnostics did not determine the pre-final candidate set.

### 2. Champion–challenger decision

Response is the **benchmark champion among the evaluated fixed linear baseline
models**, and T-learner is the challenger. Response, S, and T used fixed
SGD-logistic configurations. This experiment did not establish superiority over
tuned nonlinear uplift models, causal forests, X/R/DR-learners, every possible
policy, or a production system.

For the frozen 2% primary decision, T minus response incremental net value was
−1,778.18 (95% CI [−3,460.54, 33.52]); the Hájek sensitivity estimate was also
negative (−1,777.01). The predeclared superiority rule therefore did not pass:
**retain response; superiority of T was not demonstrated**. This is not an
equivalence or noninferiority claim.

Top-k membership diagnostics show how different the two policies are:

| Targeted share | Jaccard | Common records | Response only | T only | Switched decisions | T-only minus response-only HT-style IPW uplift/1,000 (95% CI) |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.6958 | 32,826 | 7,174 | 7,174 | 14,348 | −12.759 [−21.832, −3.099] |
| 20% | 0.7342 | 67,740 | 12,260 | 12,260 | 24,520 | −6.380 [−11.376, −1.115] |

The data identifies source records, not durable unique people, so these counts
are records/targeting decisions rather than verified distinct users. Jaccard
and switch-cohort statistics are diagnostics; they are not extra promotion
tests. The exact switched-cohort HT-style IPW decomposition is verified in the
report.

Promotion requires the paired 95% lower confidence bound for T-minus-response
incremental net value to exceed zero and Hájek to agree in direction. Even if
that offline gate passed, it would authorize only a prospective randomized
online test—not automatic deployment.

### 3. Cost- and margin-aware selection

For contribution margin `M`, assigned-target cost `c`, eligible population `N`,
budget `q`, and causal gain rate `g` versus target nobody:

```text
incremental net value = N * (M*g - c*q)
break-even cost       = M*g/q       (undefined when q = 0)
```

Every validation scenario explicitly includes `target_nobody` with zero
budget, gain, cost, and incremental net value. The globally selected frozen
response decisions were:

The manually configured economic candidate set was exactly
`{response, T-learner, target_nobody}`. Response was the benchmark champion,
T-learner was the retained model challenger, and `target_nobody` was the
explicit no-action baseline rather than a scored model. S-learner remained in
ranking and continuity diagnostics. Its exclusion from economic optimization
was a protocol-design scope choice frozen before final evaluation, not the
output of a coded screening threshold.

| Scenario (M=100; N=100,000) | Cost/target | Validation-selected budget | Final net value (95% CI) | Break-even cost (95% CI) |
|---|---:|---:|---:|---:|
| Primary reference | 0.50 | 2% | 5,109.98 [1,815.61, 8,171.56] | 3.0550 [1.4078, 4.5858] |
| Zero targeting cost | 0.00 | 44% | 10,493.07 [6,534.44, 14,285.29] | 0.2385 [0.1485, 0.3247] |
| Low targeting cost | 0.10 | 11% | 8,468.02 [4,772.55, 12,070.86] | 0.8698 [0.5339, 1.1974] |
| High targeting cost | 1.00 | 2% | 4,109.98 [815.61, 7,171.56] | 3.0550 [1.4078, 4.5858] |

All currency, margin, cost, and population inputs are explicitly hypothetical
because Criteo v2.1 supplies none of them. Cost means cost per **assigned
target**, not per impression, click, or realized exposure. With heterogeneous
record-level margins or costs, ranking should instead use expected individual
net uplift.

## Ranking and integrity diagnostics

On the fresh final sample, response had the strongest overall ranking metric:

| Policy | AUUC | Qini coefficient (95% CI) |
|---|---:|---:|
| Response | 0.001025 | 0.000482 [0.000298, 0.000660] |
| T-learner | 0.000848 | 0.000306 [0.000116, 0.000490] |
| S-learner | 0.000785 | 0.000243 [0.000050, 0.000426] |
| Expected random | 0.000543 | 0 |

The final treatment share was 0.8498 and the naive intention-to-treat ATE was
0.001086. The treatment-prediction diagnostic AUC was 0.5047; its logistic loss
did not improve over the constant baseline. Targeting scores were exactly
invariant to treatment-column perturbation; the model feature contract also
excludes outcomes. Every fitted model reported convergence. The primary
estimator is **empirical-marginal-propensity inverse-probability weighting**:
`e_hat = mean(T)`. For historical continuity it may be described as HT-style
IPW using the empirically estimated marginal treatment propensity, but it is
not classical design-probability Horvitz–Thompson. The separate Hájek estimate
remains a directional sensitivity estimator. Immutable fields such as `ht` and
`horvitz_thompson` are legacy names and are not rewritten.

For S-learner dimensionality, the encoded covariate width is `p = 5,230`. The
estimator design `[Z, T, Z*T]` therefore has `2p + 1 = 10,461` columns. The
sealed `encoded_feature_count = 5,230` field refers only to encoded feature
width, not the full S-learner estimator design.

## Fail-closed sample and artifact chain

The governed order was:

1. Validate protocol fields and exact executable hashes.
2. Plan source indices only; do not materialize final records.
3. Load development only, fit on its training partition, and select on its
   validation partition.
4. Exclusively create and fsync the validation-decision freeze.
5. Exclusively create and verify the one-shot final-access claim.
6. Only then load the pinned final sample and evaluate frozen choices.
7. Hash every published output into an immutable completion receipt and verify
   the protocol/freeze/claim/code/output chain.

| Sample role | Rows | Seed | Selected-index SHA-256 | Access status |
|---|---:|---:|---|---|
| Development | 2,000,000 | 42 | `32a6a6f0...5959fe7c` | Loaded before freeze |
| Invalid rehearsal | 400,000 | 20,260,817 | `03cfd456...c50c886` | Reserved; not loaded in v2 |
| Fresh final | 400,000 | 20,260,818 | `ce993fa6...b4c867b` | Loaded only after claim |

The development-plus-rehearsal exclusion union contains 2,400,000 records with
digest `09bb916c...d07ea49`. Final overlap with that union is exactly zero.

Immutable chain:

- protocol SHA-256: `9444cfafa2029f09c0770aa166197939282cbfa69b529b2d02652662c887024d`;
- [`freeze`](reports/metrics/final-evaluation-conversion_final_v2_20260818.freeze.json): `c866f865ffb8449b3d7ed772a0e4c7f4a2246c284143854a35b9b7cb6a0515e3`;
- [`access claim`](reports/metrics/final-evaluation-conversion_final_v2_20260818.claim.json): `ffe6dacd43327177430c4e13ea2b682c34c0f945d92b8eddf953c3eb1ca9f178`;
- [`completion receipt`](reports/metrics/final-evaluation-conversion_final_v2_20260818.receipt.json): `8d363c2099918d0b42b67b246f573ddec75d715c04020c40012be4d8a2a0f679`.

Replay is deliberately blocked. Re-running the same evaluation ID now refuses
before evaluation rather than overwriting the freeze, claim, or receipt. Any
future confirmatory run requires a new evaluation ID and a newly reserved final
sample that excludes all previously accessed final records.

## Reports and verification

Key machine-readable outputs:

- [`result_card.json`](reports/metrics/result_card.json)
- [`promotion_decision.json`](reports/metrics/promotion_decision.json)
- [`champion_challenger_diagnostics.csv`](reports/metrics/champion_challenger_diagnostics.csv)
- [`validation_economic_grid.csv`](reports/metrics/validation_economic_grid.csv)
- [`final_frozen_economic_decisions.csv`](reports/metrics/final_frozen_economic_decisions.csv)
- [`ranking_metrics.csv`](reports/metrics/ranking_metrics.csv)
- [`source_metadata.csv`](reports/metrics/source_metadata.csv)
- [`experiment_summary.json`](reports/metrics/experiment_summary.json)

Figures include the
[`validation economic curves`](reports/figures/economic_net_value_validation.png),
[`Qini curves`](reports/figures/qini_curve.png), and
[`policy-value curves`](reports/figures/policy_value.png).

The presentation-facing layer is deliberately versioned outside the sealed
report directory. It includes the corrected decision-first
[`interview deck`](presentation/Uplift_Modeling_Interview_Case_Study.pptx), the
[`one-page case study`](output/pdf/uplift_modeling_interview_case_study.pdf),
and [`record-labelled figure exports`](presentation/figures/). These artifacts
read the frozen metrics without replacing or rewriting governed evidence.

The complete post-run suite passes **63/63** tests, including output receipt
verification, exact report-contract checks, sample disjointness, economic
identities, promotion logic, bootstrap transformations, and governance
tamper/replay cases:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The completed, non-repeatable invocation is recorded in
[`experiment_summary.json`](reports/metrics/experiment_summary.json). It used
1,000 final bootstrap repetitions and 100 validation repetitions.

## Limitations

- The benchmark estimates intention-to-treat effects of randomized assignment.
- Conditional row-level paired bootstraps condition on sampled records, fitted
  models, frozen rankings, and the frozen cutoff; they exclude refitting,
  policy-selection, and latent advertiser/experiment cluster uncertainty.
- Valid advertiser or experiment cluster identifiers are unavailable. Clustered
  resampling would be preferred if they existed; pseudo-clusters must not be
  fabricated.
- The validation budget grid is finite, so results identify the best evaluated
  point rather than a continuous optimum.
- Constant margin and assigned-target cost are scenario assumptions, not facts
  recovered from this dataset.
- Criteo's privacy-preserving subsampling prevents direct extrapolation to the
  advertiser's original population.
- Rows are source records, not verified unique users; targeting counts and
  per-1,000 quantities use eligible records as the denominator.
- This project is offline and not production-ready or deployable. Deployment
  would require a serialized encoder/model bundle, a feature contract,
  deterministic top-k tie-breaking, a batch-scoring entry point, monitoring,
  and prospective online validation. Those components are future work, not
  part of this correction pass.
- The immutable file chain assumes a trusted, quiescent local workspace; it is
  not designed to resist a hostile process concurrently replacing directories.
