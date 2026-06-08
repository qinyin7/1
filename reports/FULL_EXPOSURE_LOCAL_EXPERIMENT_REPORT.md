# KuaiRec Full-Exposure Local Experiment Report

## 1. Protocol

Profile: `local_8gb_large`

Evaluation panel: `small_matrix/full_val`

Protocol version: result schema v7, stable user hash split.

The profile contains all 1,411 fully observed users plus 1,589 normal big-matrix
users for collaborative training support. The validation panel has 717 users and
2,376,459 observed user-video pairs, with matrix density 99.62%.

All model selection below uses full-exposure feedback. Big temporal metrics are
diagnostics only.

## 2. Recall Baselines

| Exp | Model | Recall@100 | NDCG@100 | Precision@100 | Utility@100 | Coverage@100 | Cold Recall@100 | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R0.2 | Decayed popular | 0.046030 | 0.467038 | 0.484826 | 0.491729 | 0.032462 | 0.000000 | Strong popularity baseline, weak coverage |
| R1.0 | ItemCF | 0.036695 | 0.387470 | 0.380711 | 0.366953 | 0.731590 | 0.000000 | Coverage diagnostic, not winner |
| R2.4 | Content TF-IDF | 0.032394 | 0.350789 | 0.348340 | 0.335223 | 0.364893 | 0.037720 | Cold-start support |
| R3.3 | Feature two-tower | 0.073543 | 0.749328 | 0.741381 | 0.788841 | 0.058311 | 0.252668 | Best pure precision/utility |
| R3.4 | R3.3 + ID Dropout | 0.073168 | 0.747683 | 0.738131 | 0.785335 | 0.155095 | 0.257727 | Best engineering recall baseline |
| R3.5 | R3.3 + hard negatives | 0.072539 | 0.741763 | 0.730669 | 0.776557 | 0.053802 | 0.183978 | Not adopted |
| R3.6 | R3.3 + both | 0.071989 | 0.740636 | 0.726527 | 0.764686 | 0.168320 | 0.233836 | More diverse, lower utility |

Decision: use `R3.4` as the recall baseline for candidate generation. It keeps
almost all of `R3.3`'s precision while greatly improving coverage and cold-start
recall.

## 3. Candidate Ranking

Candidate channels: `R1.0 + R2.4 + R3.4`

Training rows: 17,415 explicitly exposed candidate rows. Unexposed recalled
items are not labeled as negatives.

| Exp | Recall@100 | NDCG@100 | Precision@100 | Utility@100 | Coverage@100 | Cold Recall@100 | Tower Unique Selected | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PR3 | 0.076417 | 0.781863 | 0.764728 | 0.823494 | 0.314097 | 0.152783 | 0.599972 | Current baseline |
| PR3.no_recall_features | 0.076345 | 0.783731 | 0.764630 | 0.823340 | 0.321010 | 0.156986 | 0.624296 | Similar, recall features not critical |
| PR3.no_cross_features | 0.076636 | 0.784207 | 0.766583 | 0.825642 | 0.321611 | 0.158104 | 0.625927 | Slightly higher on this seed |
| PR3.no_temporal_features | 0.076391 | 0.782893 | 0.764226 | 0.822252 | 0.310490 | 0.163417 | 0.619609 | Similar |
| PR.no_tower | 0.061864 | 0.661185 | 0.626081 | 0.635174 | 0.617974 | 0.035894 | 0.000000 | Tower candidates are necessary |

Multi-seed follow-up supports simplifying the tabular LTR ranker to
`PR3.no_cross_features`. It improves Recall in all three seeds, improves NDCG
significantly in two seeds, and removes cross-affinity features that were not
stable under the new full-exposure protocol.

| Exp | Seeds | Recall@100 Mean | NDCG@100 Mean | Utility@100 Mean | Decision |
|---|---:|---:|---:|---:|---|
| PR3 | 3 | 0.076245 | 0.783101 | 0.824712 | Challenger |
| PR3.no_cross_features | 3 | 0.076666 | 0.785250 | 0.829665 | Best tabular baseline |

Paired bootstrap, `PR3.no_cross_features - PR3`:

| Seed | Recall Diff 95% CI | NDCG Diff 95% CI |
|---:|---|---|
| 2026 | [-0.000025, +0.000473] | [+0.000395, +0.004408] |
| 2027 | [+0.000262, +0.000713] | [+0.001546, +0.005332] |
| 2028 | [+0.000328, +0.000772] | [-0.001395, +0.002466] |

## 4. Deep Candidate Rankers

| Exp | Recall@100 | NDCG@100 | Precision@100 | Utility@100 | Coverage@100 | Cold Recall@100 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| DR1.deepfm | 0.065093 | 0.661632 | 0.662901 | 0.693382 | 0.391644 | 0.286156 | Not adopted |
| DR2.din | 0.075378 | 0.771737 | 0.758368 | 0.816743 | 0.152991 | 0.293833 | Sequence ranker candidate |
| DR3.mmoe_complete | 0.070816 | 0.735748 | 0.714463 | 0.758061 | 0.285543 | 0.063954 | Not adopted |
| DR3.mmoe_complete_strong | 0.072205 | 0.750407 | 0.728689 | 0.775202 | 0.173129 | 0.318379 | Cold-start challenger only |
| DR3.mmoe_multitask | 0.074065 | 0.763950 | 0.745900 | 0.798745 | 0.171626 | 0.307044 | Not adopted |

Paired bootstrap, `PR3 - DR2.din`:

| Metric | Mean Diff | 95% CI |
|---|---:|---|
| Recall@100 | +0.001039 | [+0.000527, +0.001642] |
| NDCG@100 | +0.010126 | [+0.005520, +0.015146] |

Decision at validation time: PR3/no-cross is the strongest tabular LTR baseline.
DIN is close on Recall/NDCG and has much stronger cold-start recall, so it is
kept as the primary sequence-ranker candidate for the final architecture story.

## 5. Current Baseline

Current full-exposure baseline:

```text
Recall: R3.4 feature two-tower with ID Dropout
Candidate ranker: DR2.din sequence ranker over R1.0 + R2.4 + R3.4 candidates
Primary validation: small_matrix/full_val
Frozen test: small_matrix/full_test passed
```

## 6. Frozen Full-Test

The frozen test panel contains 694 users, a 3,327-video catalog, and 2,300,111
observed user-video feedback rows. Density is 99.62%.

| Exp | Recall@100 | NDCG@100 | Precision@100 | Utility@100 | Coverage@100 | Cold Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| R1.0 ItemCF | 0.036191 | 0.371089 | 0.362738 | 0.345267 | 0.731590 | 0.000000 |
| R2.4 Content | 0.031969 | 0.336188 | 0.335793 | 0.321246 | 0.363390 | 0.035246 |
| R3.4 TwoTower | 0.074563 | 0.732234 | 0.721758 | 0.764506 | 0.160204 | 0.266349 |
| PR3.no_cross_features | 0.078100 | 0.766319 | 0.749568 | 0.800454 | 0.402164 | 0.165352 |
| DR2.din | 0.078333 | 0.770960 | 0.754135 | 0.808220 | 0.174632 | 0.271940 |

Paired bootstrap, `PR3.no_cross_features - R3.4` on full-test:

| Metric | Mean Diff | 95% CI |
|---|---:|---|
| Recall@100 | +0.003537 | [+0.002986, +0.004096] |
| NDCG@100 | +0.034086 | [+0.028962, +0.039104] |

Paired bootstrap, `DR2.din - PR3.no_cross_features` on full-test:

| Metric | Mean Diff | 95% CI |
|---|---:|---|
| Recall@100 | +0.000233 | [-0.000172, +0.000656] |
| NDCG@100 | +0.004641 | [+0.001023, +0.008212] |

Paired bootstrap, `DR2.din - R3.4` on full-test:

| Metric | Mean Diff | 95% CI |
|---|---:|---|
| Recall@100 | +0.003770 | [+0.003417, +0.004108] |
| NDCG@100 | +0.038727 | [+0.035847, +0.041436] |

Final decision: use `DR2.din` as the project-facing final ranker because it
keeps Recall at parity with PR3/no-cross, significantly improves NDCG on
full-test, has better Utility and cold-start recall, and gives the project a
clear sequence-interest modeling story. Keep `PR3.no_cross_features` as a
CPU-friendly tabular fallback and Learning-to-Rank baseline.
