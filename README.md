# Deployment optimizer for ARM edge devices

There is no single best way to deploy a neural network. How you quantize it, which layers you
leave alone, and how you configure the runtime trade speed against size against memory against
accuracy, and the trade-offs are hardware-specific: **a smaller model is frequently not a faster
one on ARM**.

This searches those settings automatically, measures every candidate on a real Raspberry Pi 5, and
returns the set of configurations that nothing else beats on all four counts at once.

> Optuna finds the best *training* configuration. This finds the best *deployment* configuration.

---

## Look at the results first

Open **`artifacts/dashboard/index.html`** in a browser. No install, no hardware, no network. It
lists every measured campaign; click one to explore its trade space, rank the four objectives by
what you care about, and read the exact recipe for the winning configuration.

Three campaigns ship measured on a Pi 5. Their fastest members, against their own FP32 exports:

| Model | Fastest member | Speedup | Smallest | Pareto set |
| --- | --- | --- | --- | --- |
| ResNet-18 | 3.36 ms | 5.0x | 11.3 MB from 44.9 MB | 8 configurations |
| MobileNetV2 | 1.48 ms | 2.5x | 2.5 MB from 9.4 MB | 5 configurations |
| Custom CNN | 1.77 ms | 3.9x | 1.8 MB from 7.1 MB | 13 configurations |

Every figure on every page is extracted from the result JSONs at build time. The templates contain
no numbers, so a page cannot drift from what was measured.

## Run it yourself, without a Raspberry Pi

```bash
pip install -r requirements.txt
python -m src.app run --model resnet18_cifar --mock
```

That runs all ten stages against a simulated device and writes a new run with its own page. It
takes about a minute. **Every number it produces is invented** -- the run is stamped `mock` and the
page carries a banner saying so. It exists to prove the pipeline works end to end, not to tell you
anything about performance.

```bash
python -m src.app list          # every run and whether it has a page
python -m scripts.build_index   # rebuild the landing page
```

## Run it for real

You need a Raspberry Pi 5 on 64-bit Bookworm with the official PSU and an active cooler. See
[`docs/raspberry-pi_setup.md`](docs/raspberry-pi_setup.md), then write a `pi_target.json` with your
host, user and key path.

```bash
sudo bash scripts/pi_prepare.sh            # on the Pi, after every boot
python -m src.app run --model resnet18_cifar
```

Stages are skipped if they are already done, so an interrupted overnight campaign resumes where it
stopped rather than starting over. `--only <stage>` reruns one stage; `--force` redoes work.

## Bring your own model

```bash
python -m src.import_onnx --onnx your_model.onnx --name your_model --seal-test
python -m src.app run --model your_model
```

The model must be a **CIFAR-100 classifier with a static `(1, 3, 32, 32)` input and 100 output
classes**. The importer refuses anything else, and says why. Two refusals are worth knowing about
in advance:

- **224x224 models are rejected.** The evaluation bundle is 32x32 uint8 pixels. Accepting a larger
  input would mean resizing, and a resize implemented differently from the one used at training
  time is a well-known silent accuracy killer. CIFAR at its native size makes that impossible, and
  the pipeline keeps it that way.
- **Graphs whose nodes carry no module paths are rejected.** Per-layer sensitivity analysis groups
  ONNX nodes back into blocks using the module path `torch.onnx.export` writes into each node name.
  A graph exported some other way may arrive with names that carry no structure, and a sensitivity
  ranking over one undifferentiated mega-group measures nothing. Exporting a PyTorch checkpoint
  yourself is the reliable route.

## How it works

Eleven stages, declared as data in [`src/pipeline.py`](src/pipeline.py):

| Stage | What it does |
| --- | --- |
| `baseline` | Validate the graph, freeze its FP32 accuracy |
| `quant_baselines` | Score fp32, dynamic INT8, and both static INT8 recipes |
| `sensitivity` | Probe how much damage quantizing each block does |
| `group_cost` | Measure what excluding each block costs in latency |
| `cost_benefit` | Join each block's benefit to its price |
| `sentinel` | Repeat one config to measure the device's own noise floor |
| `search_space` | Pin the blocks whose price the noise floor cannot resolve, keep the rest |
| `study` | Sensitivity-constrained hybrid search, measured on the device |
| `finalists` | Re-measure the shortlist five times and name the deployment choices |
| `final_test` | Score the front once on the sealed test set |
| `dashboard` | Render that run's page |

**Sensitivity analysis before search** is the part that distinguishes this from running global INT8
quantization. Rather than quantizing everything or hand-picking the usual first and last layers, it
measures which blocks actually suffer, prices each exclusion in milliseconds on the device, and
searches only where the trade is real.

**The search is hybrid, and usually exhaustive.** Once the reduction has pinned the blocks whose
price is settled, what is left is small — 8 to 16 precision vectors on the bundled models. The
planner enumerates that space exactly, one factor at a time off a canonical recipe, so each round
reads as *what does this one knob buy*. Only a space too large to enumerate falls back to a
constrained NSGA-II, which no model bundled here reaches.

### What "measured" means here

- **Latency, memory and size come from the device.** Never from FLOPs or parameter counts. A
  candidate is measured in its own process so peak RSS is attributable, with the CPU governor
  pinned, a temperature gate before each timed section, and any run flagged invalid if the Pi
  throttled during it.
- **The test set is scored once, after the search has finished choosing.** Training, optimization
  and calibration each use a separate split of the training data. Nothing about the search ever
  reads the test set, because repeatedly querying it to steer optimization is how a headline number
  becomes meaningless.
- **Accuracy is a hard filter, not an objective to trade away.** Candidates below
  `baseline - budget` are rejected regardless of how fast they are.
- **Differences smaller than the measurement's own noise are called ties.** Repeating one identical
  configuration eight times over half an hour gave a 2.55% spread, so the page refuses to rank two
  candidates apart on less than that, and shows which of your priorities actually broke the tie.
  That band does more than format the page: it decides which block costs count as real, and a block
  whose price cannot be told from zero is searched rather than pinned. Guessing wrong in the pinning
  direction is silent and permanent; guessing wrong in the searching direction costs one more
  candidate. The finalists are then measured five more times, interleaved, so the recommendation
  does not rest on a single pass.

## Layout

```
src/app.py           the entry point
src/pipeline.py      the ten stages, as data
src/runs.py          what a run is and where its files live
src/import_onnx.py   bring your own model
src/quant/           quantization, config hashing, node-to-block grouping
src/sensitivity/     per-block damage probes and the cost-benefit join
src/search/          the reduction, the planner, the campaign and the sealed test
src/bench/           the device agent and the SSH driver that feeds it
src/portable/        torch-free code that also runs on the Pi
runs/                one directory per campaign, each with its own page
```

## Known limitations

- **CIFAR-100 only.** Another dataset would need its own splits, calibration set and evaluation
  bundles. That is a design job, not a configuration flag.
- **Pruning is deliberately out of scope.** Zeroing weights shrinks a parameter count without
  making ARM inference faster; real gains need structured pruning plus a graph rebuild.
- **Accuracy screening does not work on this dev machine.** The host CPU has AVX2 but no VNNI, so
  ONNX Runtime's INT8 path accumulates in 16 bits and saturates, which makes per-channel
  quantization score *worse* than per-tensor -- impossible as a property of quantization, and a
  measurement of the host rather than the model. All quantization accuracy is therefore measured on
  the Pi, whose Cortex-A76 has the dot-product instructions that avoid it.
- **The noise floor was measured on the FP32 configuration at 16.4 ms**, and the Pareto front sits
  near 3.4 ms. Applying the same percentage there assumes the spread scales with the work done. It
  probably mostly does, but it is an assumption, and the honest fix is a second sentinel near the
  front's latency.
