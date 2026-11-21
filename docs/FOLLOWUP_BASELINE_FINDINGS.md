# Expanded saved-results audit, 21 November 2026

Computed by `bash run_followup.sh review` on the historical `out_full_new` data
and all five stages of `out_transfer_final`. The result contains 3,675 test
prediction evaluations, 990 source-versus-target-honest threshold comparisons,
and no missing prediction files. These rows reuse datasets and training seeds;
they are not 3,675 independent experiments.

## Richer inputs do not rescue the BB84 blind spots

For within-BB84 models with full inputs, the mean across five training seeds:

| Model / representation | PNS recall | Blocking recall | Loss-manipulation recall | Honest FAR |
|---|---:|---:|---:|---:|
| Logistic / raw | 3.3% | 6.7% | 6.7% | 3.3% |
| RF / raw | 2.7% | 6.7% | 2.0% | 0.0% |
| MLP / raw | 0.7% | 10.0% | 3.3% | 0.0% |
| AE / latent | 1.3% | 4.0% | 1.3% | 0.0% |
| JEPA / latent | 3.3% | 4.0% | 0.7% | 1.3% |
| AE / combined | 0.0% | 4.0% | 3.3% | 0.0% |
| JEPA / combined | 0.7% | 3.3% | 1.3% | 0.0% |

QBER-only gives 0% PNS recall and 3.3% recall for both blocking and loss
manipulation at 3.3% honest FAR. Adding support to common features also leaves
these attack families weak. Common-feature boosting increases some detection
but with higher false alarms (e.g. 16.7% blocking recall at 10% FAR).
These are descriptive comparisons at each fitted threshold, not matched-FAR
superiority tests. Failure with full inputs means one should investigate the
simulator distributions, finite measurement support and attack mechanisms, not
assume that merely adding those measurements resolves the problem.

## Target-honest calibration identifies a large threshold problem

Below, model-seed outcomes are averaged over logistic, MLP, AE and JEPA in the
primary common/standard stage. The model, normalizer and temperature are fixed.
Only the threshold changes using the target's disjoint `cal_B` honest sessions.
This is target-honest adaptation, explicitly not strict zero-shot.

| Direction | Source-threshold FAR | Adapted FAR | Source recall | Adapted recall | AUROC |
|---|---:|---:|---:|---:|---:|
| BB84 -> BKM07 | 97.2% | 5.2% | 99.5% | 61.9% | 0.834 |
| BB84 -> E91 | 37.3% | 4.7% | 63.1% | 34.2% | 0.710 |
| BKM07 -> BB84 | 0.0% | 2.3% | 10.0% | 17.6% | 0.591 |
| BKM07 -> E91 | 0.8% | 3.2% | 31.7% | 36.4% | 0.712 |
| E91 -> BB84 | 0.0% | 0.2% | 11.2% | 17.0% | 0.566 |
| E91 -> BKM07 | 10.3% | 5.0% | 69.2% | 60.6% | 0.831 |

BB84 -> BKM07 has useful ranking but a severely shifted source threshold.
In contrast, recalibration does not fix weak ranking into BB84. Original and
adapted thresholds use the exact same test scores; AUROC is unchanged by
construction. The historical honest test population remains 30 per protocol,
so these descriptive mean rates do not establish a target FAR guarantee.

## Where to inspect exact outputs

Under `out_followup_20261121/saved_review`:

- `all_saved_evaluations.csv`, `all_saved_per_attack.csv`,
  `all_saved_attack_summary.csv`: every available historical test comparison.
- `bb84_within_all_controls.csv`, `bb84_*.png`: complete BB84 controls.
- `threshold_adaptation.csv`, `threshold_adaptation_per_attack.csv`,
  `threshold_direction_summary.csv`, `threshold_adaptation.png`: calibration audit.
- `coverage.json`: audited stages, counts and skipped-file inventory.

The targeted confirmatory training and fresh dataset experiments are separate
stages. Passing execution tests does not establish their scientific outcome.
See `FOLLOWUP_STUDY.md` for the preselected contrast, label costs, data sizes,
resumption and service commands.
