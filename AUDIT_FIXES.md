# QKD audit corrections and result provenance

This branch corrects BKM07 SIFT bit bookkeeping in both `qkd_pipeline.py` and the
corresponding `changed.py` scalar/vector simulators. Bob's recorded classical bit
now remains the measurement outcome when a faulty preparation flips only the
outgoing photon. Operational key QBER compares Bob's recorded bit with Alice's
measurement on the returned photon. `qber_zsr` remains a compatibility alias for
that same statistic; it is not an independent measurement.

**The committed `out/` tables were generated before this correction.** In
particular, every old BKM07 score is historical and must not be cited as a
measurement from the corrected code. Regenerate using a fresh output directory,
for example `python qkd_pipeline.py all --out out_corrected`, before reporting
new scores. The command is expensive. No new full-scale run is claimed here.

New generated data include `audit_manifest.json` with split run IDs, observation
hash duplication counts, and hashes of the source and saved arrays. `load_data`
rejects arrays produced by a different version of `qkd_pipeline.py`. The raw
records and checkpoints are large and should be published separately with their
hashes if independent prediction replay is required.

Focused checks: `python qkd_pipeline.py selftest`. A controlled preparation-fault
test checks that Bob retains his measured bit, the outgoing preparation fault
affects the returned key, and window key QBER matches the disclosed-bit error.

Remaining research tasks are separate from these corrections: calibrate on more
honest runs, isolate each noise/distance shift, audit duplicate raw records on
the regenerated dataset, validate detector and optical assumptions, and rerun
all comparisons. No claim about new detector reliability or paper readiness is
made by this code change alone.
