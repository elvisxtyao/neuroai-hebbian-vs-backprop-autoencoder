# Phase 0-v1 BP Team Confirmation

Status: Pending BP teammate response

Last updated: 2026-07-21

## 1. Purpose

This file records external confirmation that the BP implementation follows the
same comparison contract as the Hebbian implementation. The Hebbian maintainer
must not mark this confirmation complete on behalf of the BP teammate.

## 2. Message to send to the BP teammate

Please review `docs/protocols/PHASE0_STANDARD_V1.md` and confirm the following for the BP
baseline commit/run:

- [ ] It uses the shared `ConvAutoencoder` rather than a copied model class.
- [ ] Encoder/decoder parameter names and shapes match the shared model.
- [ ] The untrained paired model hash matches the Hebbian run for the same seed.
- [ ] It uses `mnist_split_v1.npz` and the same train/validation/test loaders.
- [ ] Inputs remain in `[0,1]` without z-score normalization.
- [ ] Representation training uses reconstruction MSE and does not use labels.
- [ ] The BP autoencoder trains for 10 epochs with the documented Adam setup.
- [ ] The best checkpoint is selected by validation reconstruction MSE only.
- [ ] The encoder is frozen for the shared standardized linear probe.
- [ ] It uses the shared evaluation and result schema.
- [ ] Test metrics are not used for tuning or checkpoint selection.
- [ ] Any deviation from `phase0-v1` is listed below.

Suggested response:

```text
I confirm that BP commit <commit> and run <run_id> are phase0-v1 compliant.
Reviewed by: <name>
Date: <YYYY-MM-DD>
Deviations: none / <list each deviation>
Evidence links or paths: <links>
```

## 3. Recorded response

| Field | Value |
|---|---|
| Reviewer | Pending |
| Date | Pending |
| BP commit | Pending |
| BP run ID | Pending |
| Compliance result | Pending |
| Evidence | Pending |

## 4. Deviations and resolution

| ID | Deviation | Affected comparison | Resolution | Status |
|---|---|---|---|---|
| — | No response recorded yet | Phase 0 external compliance gate | Send the template above | Pending |

## 5. Completion rule

`P0-TEAM-01` is complete when the template has been sent and the communication
location/date is recorded. `P0-TEAM-02` is complete only when a real response,
commit, run ID, and deviations are recorded here.
