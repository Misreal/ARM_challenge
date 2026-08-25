# Findings

Results that came out of measurement rather than design. Each one records what was expected,
what the device said, and what changed as a consequence.

## The sensitivity prediction lost to the device

`custom_cnn` was built as a fixture with planted sensitivities, and `EXPECTED_SENSITIVITY` in
`src/models/custom_cnn.py` recorded the expected ranking **before** the Pi sweep ran: stem and
stage1 robust, stage2 fragile under per-tensor weights but recovered by per-channel, stage3
fragile regardless. The reasoning was that quantization error accumulates with depth, so the
deepest block should suffer most.

The device disagreed. Measured `recovery_share` — each group's share of total quantization
damage — on the Pi 5:

| Rank | per-tensor | per-channel |
| --- | --- | --- |
| 0 | **stem** 0.544 | **stage2** 0.312 |
| 1 | stage2 0.321 | **stem** 0.186 |
| 2 | stage3 0.052 | stage3 0.130 |
| 3 | stage1 0.028 | classifier 0.052 |
| 4 | classifier 0.009 | stage1 0.051 |

The prediction was inverted where it mattered. **The stem — the first convolution — is the most
damaging group to quantize, not stage3.** That is the well-known result: the first conv has few
channels with widely differing ranges, so a single per-tensor scale fits it badly, and every
downstream activation inherits the error. It is why the standard recipe keeps the first conv in
FP32. The analyzer reproduced conventional wisdom; the planted key contradicted it.

`stage1` was predicted robust and measured robust, in both schemes. That half held.

### Why the fixture did not work as an answer key

Quantization sensitivity is a property of the trained weight and activation distributions, not
of the architecture. Channel counts and depths are choosable; the dynamic ranges that emerge
from training are not. So a sensitivity "plant" is a prediction wearing a fixture's clothes, and
this one was wrong. The project has no independent oracle for the analyzer as a result — the
strongest evidence it works is that it recovered the textbook answer on a model where the
authors expected something else.

### A second, separate problem

`test_stage2_is_largely_recovered_by_per_channel` compares stage2's *share* under per-channel
against half its share under per-tensor. Shares are normalized, so that assertion cannot
distinguish "stage2 recovered" from "everything else got worse". It tests the wrong quantity for
the question it asks, independent of whether the prediction was right.

### What changed

Nothing in the key or the assertions — PLAN.md DO-NOT #14 forbids editing either to match
observed output, and doing so would convert a prediction into a postdiction. The five failing
cases in `tests/test_sensitivity_answer_key.py` are marked `xfail(strict=True)`, so the suite is
green, the claims stay scored, and a future run that starts passing is reported as `XPASS`
rather than passing silently.

The measured ranking, not the key, is what feeds the cost-benefit join and the search-space
reduction. No pinning decision was ever taken from the prediction.
