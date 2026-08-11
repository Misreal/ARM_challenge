# Demo video script

Target length **3 to 4 minutes**. The repo is the deliverable; the video's job is to prove it runs
and to make the idea land in the first twenty seconds.

Record at 1920x1080. Keep the terminal font large enough to read at half size, because most people
will watch this in a small window.

---

## 0:00 to 0:20 — the problem, on screen not in words

Open on the landing page (`artifacts/dashboard/index.html`), three run cards visible.

> "Take a trained model and put it on a Raspberry Pi. How do you quantize it? Which layers do you
> leave alone? How many threads? Every answer trades speed against size against memory against
> accuracy, and the right answer depends on the hardware. A smaller model is often *not* a faster
> one on ARM."

Click into the ResNet-18 run. Let the Pareto scatter land.

> "This searches those settings automatically, measures every candidate on a real Pi 5, and gives
> you the set that nothing else beats."

## 0:20 to 1:10 — the page does the arguing

Stay on the ResNet-18 page.

1. Point at the front: **8 configurations**, 3.36 ms at the fast end, five times the FP32 export.
2. Drag **Model size** to the top of the priority list. The table reorders live.
3. Drag **Top-1** to the top. A different configuration wins.

> "There is no single winner, so the page does not pretend there is. You say what you care about,
> it tells you which configuration wins under *your* priorities, and which of your priorities broke
> the tie."

4. Hover a row with the `≈` marker.

> "These two are closer than the device's own repeat noise. Ranking them apart would be reading
> tea leaves, so it says so."

5. Scroll to the selected configuration panel and read one recipe aloud.

> "And this is the output you actually deploy: static per-channel INT8, uint8 activations, minmax
> calibration on 512 images, first convolution and classifier left in FP32, four threads."

## 1:10 to 1:50 — what makes it more than global INT8

Scroll to the sensitivity and cost-benefit panel.

> "Before searching, it measures how much damage quantizing each block does, and separately what
> excluding that block *costs* in milliseconds on the device. Some blocks are fragile and free to
> protect. Those are the wins. The search then only spends trials where the trade is real."

Point at the global baselines on the scatter.

> "Global dynamic INT8, global per-tensor, global per-channel, all measured through the identical
> harness. Per-tensor is faster but fails the accuracy budget outright."

## 1:50 to 2:30 — it runs, and you can prove it

Cut to a terminal.

```bash
python -m src.app run --model resnet18_cifar --mock
```

Let the stage list scroll. Do not speed this up; watching ten stages tick past is the point.

> "One command runs the whole pipeline. No Pi needed for this one -- it is running against a
> simulator so anyone can try the repo -- and the page it produces is stamped as simulated, because
> a fake number that looks measured is worse than no number."

Open the page it wrote, show the banner, close it.

Run it again.

> "Run it twice and it skips everything already done. An overnight campaign that dies at trial 30
> resumes at trial 30."

## 2:30 to 3:10 — bring your own model, on real hardware

Show the Pi on camera, briefly. Then:

```bash
python -m src.import_onnx --onnx your_model.onnx --name your_model --seal-test
python -m src.app run --model your_model
```

Time-lapse the campaign. Land on the finished page.

> "Any CIFAR-100 model with a 32 by 32 input. The importer validates the graph, builds the
> quantization-ready variant, recovers the block structure, and measures its baseline. Then the
> same pipeline runs on it."

## 3:10 to 3:40 — the honesty slide

Static text on screen, read over it:

- Latency, memory and size measured on the device, never estimated from FLOPs
- Test set scored once, after the search finished choosing
- Accuracy a hard filter, not a soft objective
- Differences smaller than the measured repeat noise reported as ties
- Every figure on every page extracted from result files, never typed in

> "The easiest way to win a benchmark is to measure it wrong. So: nothing is estimated, the test
> set is touched once, and the pages are generated from the measurements rather than written by
> hand."

## Close

Back to the landing page.

> "Clone it, open the page, and it is all there."

---

## Shot list

| Shot | Source | Notes |
| --- | --- | --- |
| Landing page | `artifacts/dashboard/index.html` | Full screen, light theme reads better on video |
| ResNet-18 trade space | that run's page | Have the scatter already rendered before recording |
| Priority reorder | same page | Rehearse; the reorder is the single best moment |
| Mock run | terminal | Full width, 16pt or larger |
| Resumed run | terminal | Same window, immediately after |
| The Pi | camera | Five seconds is plenty, with the cooler visible |
| Real campaign | terminal, time-lapsed | 8x or so |
| Honesty slide | static | Plain text, no animation |

## Before recording

```bash
python -m src.app list                                   # confirm the three runs are present
python -m scripts.build_index                            # landing page current
rm -rf scratch/runs                                      # so the mock run starts clean on camera
```

Check the terminal has no absolute paths on screen that reveal anything personal, and that
`pi_target.json` is not visible in any directory listing you record.
