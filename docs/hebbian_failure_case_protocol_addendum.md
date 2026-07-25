# Hebbian failure-case protocol addendum

Date: 2026-07-25

Decision: **BRANCH D — FREEZE AS FAILURE-CASE BASELINE**

## Frozen classification

`COMMON-MODE UPDATE REMOVAL: NOT SUFFICIENT`

The original convolutional Oja + WTA seed-42 configuration is retained only as
a `health-gate failure-case baseline`. The output-filter update-centering
candidate is a preserved negative result and is not eligible to replace it.

## Permitted use

- Reproduce Q4 frozen-snapshot update-mechanism measurements.
- Replicate the failure across preregistered seeds after separate approval.
- Compare failure severity without calling the configuration healthy.
- Preserve all original and candidate artifacts, hashes, and gate decisions.

## Prohibited use

- Do not describe either configuration as a health-passing Hebbian baseline.
- Do not enter Stage 3 or start formal seeds 0–4 without further approval.
- Do not add another repair candidate under this decision.
- Do not change performance or representation-health thresholds.
- Do not access test data for selection or mechanism repair.
- Do not treat the 50 fixed Q4 batches as independent seeds.
- Do not start dimension or architecture sweeps.

## Next authorized decision

The only recommended next task is to design, preregister, and obtain approval
for a formal multi-seed **failure replication** protocol. That protocol must
retain the failure-case label and must not claim that a healthy Hebbian
configuration has been obtained.
