# Dataset

This project uses the corrected, unbiased v2.1 release of the
[Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).
The data is not committed to this repository.

The experiment downloads the file automatically when its configured local path
does not exist. To place it manually instead, download:

```text
https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
```

and save it as:

```text
data/criteo-research-uplift-v2.1.csv.gz
```

The loader verifies the published SHA256 before accepting an automatic download:

```text
2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc
```

The source contains `f0` through `f11`, `treatment`, `conversion`, `visit`, and
`exposure`. The detailed Criteo v2 benchmark paper identifies `f0`, `f2`, `f7`
and `f10` as continuous and the other eight features as categorical, even
though every anonymized feature is serialized as a float. Models use only
`f0` through `f11`; categorical modalities are sparsely one-hot encoded.
`treatment` is the randomized intervention, while `exposure` is post-treatment
and is never used as a model feature.

Every experiment records the actual source hash, source row count, sampling
method, seed, selected row-index digest, and selected index range under
`reports/metrics/source_metadata.csv`. The frozen workflow draws:

- 2,000,000 development rows uniformly without replacement from all 13,979,592
  source rows;
- a pinned 400,000-row rehearsal sample that was previously accessed and is now
  reserved from confirmatory reuse; and
- a new 400,000-row final-evaluation sample uniformly without replacement from
  the complement of the union of development and every reserved sample.

Index planning reads source identity/schema/count but returns no row contents.
Development is loaded in a dedicated stage. Only after validation decisions are
exclusively frozen and an immutable one-shot access claim is verified does a
second staged load return the pinned final frame. Runtime provenance records the
full exclusion-union digest and asserts a final/union overlap count of zero. The
experiment does not use a prefix sample or permit final rows to participate in
model fitting, policy selection, or budget selection.

The dataset is released under CC BY-NC-SA 4.0. Cite the Criteo benchmark paper
listed on the dataset page when publishing results, and review the license before
redistributing data or derived artifacts.
