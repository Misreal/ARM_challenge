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

## What you need before you start

**The device.** A Raspberry Pi 5 running **64-bit Raspberry Pi OS Bookworm**, with the official
27 W PSU and an active cooler. The PSU and cooler are not optional extras: an undervolted or
throttling Pi measures its own thermal state rather than your model, and the harness refuses to
record a run where that happened.

**The host.** Any machine with Python 3.11 and an SSH client. It never needs a GPU unless you
intend to train a model from scratch.

**The model.** A **CIFAR-100 classifier**, exported to ONNX with:

| Requirement | Why |
| --- | --- |
| Static `(1, 3, 32, 32)` input | Dynamic axes break static INT8 calibration and invalidate the latency method |
| 100 output classes | The evaluation bundles are CIFAR-100 |
| Exported by `torch.onnx.export` | Node names must carry module paths, or per-block sensitivity has nothing to group by |

The importer checks all three and refuses with a reason rather than producing a meaningless
ranking. **224x224 models are rejected** — the evaluation bundle is 32x32 uint8 pixels, and
resizing at evaluation time in a way that differs from training is a well-known silent accuracy
killer.

## How to run it on a Raspberry Pi

### 1. Set up the host — about 10 minutes

```bash
git clone https://github.com/Misreal/ARM_challenge.git
cd ARM_challenge
pip install -r requirements.txt
pytest
```

Five failures in `tests/test_sensitivity_answer_key.py` are expected. They are a recorded
disagreement between a planted answer key and what the device actually measured, kept deliberately
rather than edited away.

### 2. Set up the Pi — about 30–45 minutes, once per device

Follow [`docs/raspberry-pi_setup.md`](docs/raspberry-pi_setup.md), **Sections 1 through 5**:

| Section | What it does |
| --- | --- |
| §1 | Network and first SSH login |
| §2 | System update and base tools |
| §3 | Create the `~/armopt` virtual environment |
| §4 | Install `requirements-pi.txt` — pinned, and deliberately torch-free |
| §5 | Verify ONNX Runtime imports and reports the right version |

§6 covers benchmark hygiene. Read it. It is what separates a measurement from a number.

The ONNX Runtime version on the Pi must match the host exactly. Different kernels shift both
accuracy and latency, so a mismatch quietly compares two different things.

### 3. Connect the two — about 5 minutes

Set up SSH key authentication (§12), then copy the template and fill in **your own** device:

```bash
cp pi_target.example.json pi_target.json
```

Every value in the template is a placeholder that must be replaced:

| Field | What to put | Required |
| --- | --- | --- |
| `host` | Your Pi's IP address or hostname | yes |
| `user` | Your username **on the Pi**, not on your laptop | yes |
| `remote_root` | Where this repo lives on the Pi, e.g. `/home/<you>/arm_challenge` | yes |
| `python` | The **venv** interpreter from §3, e.g. `/home/<you>/armopt/bin/python` | recommended |
| `port` | SSH port; leave `22` unless you changed it | no |
| `identity_file` | Path to a specific private key, or `null` for your SSH default | no |

`python` matters more than it looks. It must point at the `~/armopt` venv, **not** the Pi's system
`python3` — §5 of the setup guide has you verify exactly this. The system interpreter either lacks
ONNX Runtime entirely or carries a different version, and a version mismatch between host and
device silently compares two different things.

`pi_target.json` is gitignored, so it never arrives with a clone and your host details never get
committed. The template is the only version in the repository. If you would rather not write a file
at all, `ARMOPT_PI_HOST`, `ARMOPT_PI_USER`, `ARMOPT_PI_ROOT`, `ARMOPT_PI_PYTHON` and
`ARMOPT_PI_KEY` override it from the environment.

Confirm the link before spending device time on anything:

```bash
python -m src.bench.remote --push-code --check
```

That reports the governor, clock, temperature and throttle mask. If it passes, the host can drive
the Pi.

### 4. Build the evaluation bundles — about 5 minutes

Required once, whichever model you bring. The bundles are gitignored because they are derived data.

```bash
python -m src.data.splits --create      # downloads CIFAR-100 and writes the split
python -m src.data.export_pi_data       # optimization and calibration bundles
python -m src.data.export_test_bundle   # the sealed test bundle
```

The split is three-way and strictly separated: the search never sees the test set, and calibration
never sees the data the search is scored on.

### 5. Bring in a model

Trained checkpoints and ONNX graphs are build outputs, not source, so they are not in the
repository. Pick whichever route suits you.

**Option A — download the models these results were measured on.** The three FP32 exports are
attached to the [v1.0 release](https://github.com/Misreal/ARM_challenge/releases/tag/v1.0):

```bash
curl -L -O https://github.com/Misreal/ARM_challenge/releases/download/v1.0/resnet18_cifar.onnx
python -m src.import_onnx --onnx resnet18_cifar.onnx --name resnet18_repro --seal-test
```

| Asset | Size |
| --- | --- |
| `resnet18_cifar.onnx` | 42.8 MB |
| `mobilenetv2_cifar.onnx` | 8.9 MB |
| `custom_cnn.onnx` | 6.8 MB |

Import under a **new name**, as above. The bundled reports already carry sealed test scores, and
the importer refuses to overwrite one — that guard is what stops the sealed split from being
scored twice.

**Option B — bring your own model.** Same command, your graph:

```bash
python -m src.import_onnx --onnx your_model.onnx --name your_model --seal-test
```

It must meet the three requirements above. The importer checks them and refuses with a reason.

### 6. Run the campaign

```bash
sudo bash scripts/pi_prepare.sh        # on the Pi, after every boot
python -m src.app run --model your_model
```

`pi_prepare.sh` pins the CPU governor. It needs `sudo`, so it cannot be done for you from the host,
and the harness refuses to measure a device whose governor is unpinned.

Stages are skipped if already done, so an interrupted overnight campaign resumes where it stopped.
`--only <stage>` reruns one stage; `--force` redoes work.

### 7. Read the results

```bash
python -m src.app list          # every run and whether it has a page
```

Open `runs/<run_id>/dashboard/index.html` for that campaign, or
`artifacts/dashboard/index.html` for the landing page listing all of them. Both are rebuilt
automatically at the end of a run.

## How it works

Twelve stages, declared as data in [`src/pipeline.py`](src/pipeline.py):

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
| `index` | Refresh the landing page that lists every run |

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
src/pipeline.py      the twelve stages, as data
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

## Trying the pipeline without a Raspberry Pi

If you have no device and only want to see the machinery work:

```bash
pip install -r requirements.txt
python -m src.app run --model resnet18_cifar --mock
```

That walks every stage against a simulated device in about a minute and writes a run with its own
page. **Every number it produces is invented.** The run is stamped `mock`, the page carries a
banner saying so, and `.gitignore` keeps simulated runs out of the repository. It exists to prove
the pipeline runs end to end — it tells you nothing about performance, and nothing it outputs
belongs in a comparison.

## Licence

MIT — see [`LICENSE`](LICENSE).
