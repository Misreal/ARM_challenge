# Raspberry Pi 5 — Operator Setup Guide

From "I just plugged the Pi in" to "this Pi can produce a benchmark number I'm willing to
put in a results table."

## Who this is for and how to read it

This guide assumes you have already flashed the SD card and booted the Pi once, so SSH and
WiFi already work. It does **not** cover flashing.

**The single most important convention in this document:** every code block is labelled with
where you type it.

- ```powershell``` blocks → type these in **PowerShell on your Windows laptop**.
- ```bash``` blocks → type these in the **SSH session on the Pi** (i.e. after you have
  connected and the prompt shows your Pi username and hostname, e.g. `you@yourpi:~ $`).

If you ever paste a `bash` block into PowerShell you will get a confusing error like
`sudo : The term 'sudo' is not recognized`. That is the tell that you are in the wrong shell.

**Placeholders used throughout this doc** — fill these in once for your own setup and swap
them in wherever you see them:

| Placeholder | What it means | Example |
| --- | --- | --- |
| `<PI_USER>` | Your login username on the Pi | `pi` |
| `<PI_HOST>` | The Pi's hostname | `raspberrypi` |
| `<PI_IP>` | The Pi's IP address on your network | `192.168.1.42` |
| `<PROJECT_ROOT>` | Local path to this repo on your laptop | `C:\Users\you\code\ARM_challenge` |

## The device you are working with

Fill in this table with facts about *your* Pi before you start. Don't assume the values from
someone else's setup — hostnames, IPs, and even the OS image can differ.

| Property | Value |
| --- | --- |
| Board | Raspberry Pi 5 |
| OS | Raspberry Pi OS **64-bit Lite** recommended (headless, no desktop) |
| Desktop | None. Headless, terminal only. |
| Python | Whatever ships as the system `python3` on your image (check with `python3 --version`) |
| Hostname | `<PI_HOST>` → mDNS name `<PI_HOST>.local` |
| Login user | `<PI_USER>`, password authentication over SSH (or key-based, if you set that up) |
| Network | However your Pi and laptop reach each other — see Section 1.1 |
| Pi IP | `<PI_IP>` |

Two notes on that table that matter later:

**Python version parity matters.** Whatever Python version the Pi ships (e.g. 3.11.x), your
PC-side conda environment for this project should match the same major/minor version. That
parity means the same `onnxruntime` wheel series (e.g. `cp311`) installs on both machines, so
PC-side and Pi-side numbers come from comparable builds.

**If you changed the Pi's hostname away from the OS default** (`raspberrypi`), remember that
any generic instructions elsewhere on the internet assuming the default hostname won't match
your device.

## Section 1 — Power on and connect

### 1.1 SSH in

```powershell
# Laptop (PowerShell). Connects to the Pi's shell over the network.
ssh <PI_USER>@<PI_IP>
```

You will be prompted for the password (nothing appears as you type — that's normal Unix
behaviour, not a frozen terminal). Correct output looks roughly like:

```
Linux <PI_HOST> 6.x.x-rpt-rpi-2712 #1 SMP PREEMPT ... aarch64

The programs included with the Debian GNU/Linux system are free software; ...
Last login: ...
<PI_USER>@<PI_HOST>:~ $
```

The prompt is your confirmation: your user, your Pi's hostname, current directory `~` (your
home directory).

Sanity check that you are on the right machine and OS:

```bash
# Pi. Instant. Confirms hostname, architecture, and OS release.
hostname
uname -m          # expect: aarch64   (64-bit ARM — required for the onnxruntime wheel)
cat /etc/os-release | head -2
python3 --version
```

If `uname -m` prints `armv7l` instead of `aarch64`, you are on a 32-bit OS. There is no
aarch64 onnxruntime wheel for that and the project does not support it — you would have to
reflash. This should not happen with a 64-bit image, but it's a 5-second check that saves an
hour of confusing pip errors.

### If this goes wrong

**`ssh: connect to host <PI_IP> port 22: Connection timed out`**

The most likely cause is that the IP changed — this is especially common on a phone hotspot,
where addresses are handed out in connection order and can be reassigned between boots.

Two ways to re-find the Pi:

1. **Router or hotspot's connected-device list.** Most routers and phone hotspot settings
   screens show connected device names/IPs.
2. **Monitor.** Connect a monitor to the Pi via micro-HDMI (the Pi 5 uses micro-HDMI, and the
   port nearest the USB-C power connector is HDMI0 — use that one). The console shows the login
   prompt. Without a USB keyboard attached, this is **display-only** — you can read boot
   messages, error text, and the IP if it's printed, but you cannot type. Still useful: a Pi
   that fails to get an IP will say so on that screen.
3. **Sweep your subnet from the laptop** as a last resort (adjust the range to match your
   network):

```powershell
# Laptop (PowerShell). Pings a range of addresses; adjust to your subnet.
2..254 | ForEach-Object { $ip = "192.168.1.$_"; if (Test-Connection $ip -Count 1 -Quiet) { Write-Output "alive: $ip" } }
```

Then `ssh <PI_USER>@<the address that answered>`.

**`WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!`**

SSH remembers a fingerprint (host key) per IP address in your local `known_hosts` file. Because
IPs can get recycled between devices (especially on hotspots), you may eventually connect to
`<PI_IP>` and find a *different* machine's key there — SSH loudly refuses, because that pattern
is also what a man-in-the-middle attack looks like.

On a private network with a Pi you physically own, the benign explanation is overwhelmingly
likely. Remove the stale entry:

```powershell
# Laptop (PowerShell). Deletes the remembered key for that one IP. Instant.
ssh-keygen -R <PI_IP>
```

Then reconnect and answer `yes` when asked to accept the new fingerprint.

**`Permission denied, please try again.`** — the password was wrong, or you typed a different
username than `<PI_USER>`.

## Section 2 — First-time system prep (one time only)

Everything in this section is done **once**, on a fresh OS install. Skip it on later sessions.

### 2.1 Update the system

```bash
# Pi. Refreshes the package index, then upgrades every installed package.
sudo apt update && sudo apt full-upgrade -y
```

**Budget real time for this.** It is typically several hundred megabytes. Over a slow or
metered connection this can be **20+ minutes**.

Correct output ends with something like `0 upgraded, 0 newly installed, 0 to remove` on an
already-current system, or a long list of `Setting up <package> ...` lines followed by the
prompt returning.

> **If your SSH session drops mid-upgrade**, the upgrade process on the Pi may be killed
> partway through, leaving the package database half-configured. Reconnect and run
> `sudo dpkg --configure -a` followed by `sudo apt full-upgrade -y` again.

### 2.2 Install the base tools

```bash
# Pi. Installs pip, venv, git, and tmux. ~1-2 minutes.
sudo apt install -y python3-pip python3-venv git tmux
```

`tmux` is installed just incase you'll use it for the longer benchmark runs.

Verify:

```bash
# Pi. Instant.
python3 -m venv --help > /dev/null && echo "venv OK"
git --version
tmux -V
```

Expect `venv OK`, a git version line, and a tmux version line.

## Section 3 — Create the Python virtual environment

```bash
# Pi. Creates the venv at ~/armopt, then activates it.
python3 -m venv ~/armopt
source ~/armopt/bin/activate
```

After activation your prompt gains a prefix:

```
(armopt) <PI_USER>@<PI_HOST>:~ $
```

That `(armopt)` is the whole signal. If it isn't there, you are using system Python and any
`pip install` will either fail or land in the wrong place.

Confirm the venv is really in charge:

```bash
# Pi. Should print a path inside your home directory, NOT /usr/bin/python3.
which python3
python3 --version
```

### Re-activation is required every session

The venv is not "installed" — it's activated per shell. **Every new SSH session starts
deactivated.** Run this each time:

```bash
# Pi. Run at the start of every SSH session.
source ~/armopt/bin/activate
```

Optional convenience — auto-activate on login by appending the line to your shell startup file:

```bash
# Pi. Appends the activation line to ~/.bashrc so every new shell activates automatically.
echo 'source ~/armopt/bin/activate' >> ~/.bashrc
```

Trade-off worth understanding: this is convenient, but it means you are *always* in the venv,
including when you're doing unrelated system maintenance. Some people prefer explicit
activation precisely because the `(armopt)` prefix is a visible reminder of which Python is
running. Either choice is defensible; just know which one you made.

## Section 4 — Install the pinned dependencies

### 4.1 Why the versions are hard-pinned

Look at `requirements-pi.txt`:

```
numpy>=1.26
onnx==1.22.0
onnxruntime==1.27.0
psutil>=5.9
```

`onnx` and `onnxruntime` use `==` (exactly this version), not `>=` (this or newer). That is a
**methodology requirement**.

Updating ONNX Runtime versions modifies underlying kernel routines, changing both the speed and numerical output of the exact same .onnx model.


### 4.2 NEVER install torch on this Pi

**Do not run `pip install torch` or `pip install torchvision` on the Pi.** 


### 4.3 Get the requirements file onto the Pi

You need `requirements-pi.txt` on the device. Simplest route — copy it from the laptop:

```powershell
# Laptop (PowerShell). Copies the single file into the Pi's home directory. ~1 second.
scp <PROJECT_ROOT>\requirements-pi.txt <PI_USER>@<PI_IP>:~/
```

(See Section 7 for more on `scp`.)

Alternatively, write it by hand on the Pi with `nano requirements-pi.txt`, paste the four lines
above, then `Ctrl+O`, `Enter`, `Ctrl+X` to save and exit. Only do this if you copy the pins
exactly.

### 4.4 Install

```bash
# Pi. Make sure the prompt shows (armopt) first!
source ~/armopt/bin/activate
pip install --upgrade pip
pip install -r ~/requirements-pi.txt
```

Expect **3–10 minutes**, dominated by downloading the onnxruntime wheel (tens of MB). Correct
output ends with a line like:

```
Successfully installed numpy-1.26.x onnx-1.22.0 onnxruntime-1.27.0 protobuf-... psutil-...
```

### If this goes wrong

**`error: externally-managed-environment`** — you are not in the venv. The prompt is missing
`(armopt)`. Run `source ~/armopt/bin/activate` and retry.

## Section 5 — Verify the install

```bash
# Pi, inside (armopt). Instant. Prints the ORT version and the available backends.
python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

You may see one or two harmless warning lines above the actual output, e.g.:

```
[W:onnxruntime:Default, device_discovery.cc:283 GetGpuDevices] Failed to detect devices under "/sys/class/drm/card0" ...
```

This is ORT probing for a GPU device and not finding one the way it expects on the Pi's
VideoCore graphics — irrelevant, since this project only uses `CPUExecutionProvider`. Ignore it.

Correct output (the two lines that matter) looks like one of:

```
1.27.0
['XnnpackExecutionProvider', 'CPUExecutionProvider']
```

```
1.27.0
['CPUExecutionProvider']
```

```
1.27.0
['AzureExecutionProvider', 'CPUExecutionProvider']
```

Read that output carefully — it answers two different questions.

**The version must match the PC exactly.** If it doesn't, stop and fix it before collecting any
data (Section 4.1 explains why).

**An "execution provider" (EP)** is a backend ORT can dispatch operators to. As long as `CPUExecutionProvider` is in the list, you're good — that's the only backend this project uses, and the one all headline numbers are measured on. Anything else in the list
(`AzureExecutionProvider`, `XnnpackExecutionProvider`, etc.) is extra and can be ignored.

Also verify the other imports actually load:

```bash
# Pi, inside (armopt). Instant. Catches a missing onnx/psutil before you need them mid-run.
python3 -c "import numpy, onnx, psutil; print('numpy', numpy.__version__); print('onnx', onnx.__version__); print('psutil', psutil.__version__)"
```

`numpy` is only pinned `>=1.26` (not `==`), so seeing a numpy 2.x on the Pi is fine as long as
your PC-side environment is also on numpy 2.x — the two just need to be broadly consistent with
each other, not byte-identical, since numpy isn't part of the strict `==` pin in Section 4.1.

## Section 6 — Benchmark hygiene (this is what makes the numbers trustworthy)

Everything up to here was installation. This section is the actual methodology. A Pi that
"works" can still produce numbers that are pure noise, and the failure is silent — you get
plausible-looking milliseconds that are really measuring temperature and voltage.

The governing idea: **when you time a model, the model must be the only thing that changed.**
Every item below removes a hidden variable.

### 6.1 Power supply and cooling — check, don't assume

Two physical checks you must perform on the actual hardware:

**Is the official 27 W USB-C PSU connected?** The Pi 5 negotiates its power budget with the
supply. An underpowered charger (a phone brick, a laptop USB port, a generic 5V/2A supply)
doesn't produce a dramatic failure — it produces *silent undervoltage throttling*, where the
firmware quietly reduces the clock to stay within budget. Your latency numbers then encode
which charger you grabbed that day. Look at the brick and confirm it is the official 27 W unit.

**Is active cooling attached?** The Pi 5 runs a Cortex-A76 at 2.4 GHz and will thermally
throttle under a sustained inference load without a fan or heatsink. Physically look at the
board: is the Active Cooler (or an equivalent fan/heatsink) mounted and its fan connector
plugged into the small 4-pin JST header near the USB ports? Note that the Active Cooler's fan
is thermostatically controlled by the firmware — it may run slowly or not spin at all while the
board is cool and idle. That's normal, not a sign it's disconnected; what matters is that it's
physically mounted and wired in, so it ramps up once the board heats up under load. If you
cannot confirm the mount, do not collect latency data yet — a passively-cooled Pi under a long
benchmark is measuring its own heat soak.

I can't verify either of these remotely. They're yours to check.

### 6.2 The throttling flag — `vcgencmd get_throttled`

This is the single most valuable command in this guide.

```bash
# Pi. Instant. Reports whether the firmware has hit any power or thermal limit.
vcgencmd get_throttled
```

Correct output:

```
throttled=0x0
```

`0x0` means: no undervoltage, no frequency capping, no thermal throttling — not now, and not at
any point since boot. **Any nonzero value means the run is contaminated and the numbers should
be discarded.**

The value is a bitfield. The bits you'll actually see:

| Bit | Meaning |
| --- | --- |
| 0 (`0x1`) | Under-voltage **right now** |
| 1 (`0x2`) | ARM frequency capped **right now** |
| 2 (`0x4`) | Currently throttled |
| 3 (`0x8`) | Soft temperature limit active |
| 16 (`0x10000`) | Under-voltage **has occurred** since boot |
| 17 (`0x20000`) | Frequency capping **has occurred** since boot |
| 18 (`0x40000`) | Throttling **has occurred** since boot |
| 19 (`0x80000`) | Soft temperature limit **has occurred** since boot |

The low bits are live state; the high bits (`0x1____`) are sticky "this happened at some point
since boot" flags. So `throttled=0x50000` means "under-voltage and throttling both happened
earlier, but not at this instant" — still a discard.

**Use it as a bracket around every benchmark:**

```bash
# Pi. Check BEFORE the run.
vcgencmd get_throttled

# ... run your benchmark ...

# Pi. Check AFTER the run.
vcgencmd get_throttled
```

Why both? The sticky bits persist since boot, so a nonzero reading *before* tells you the
machine was already unhealthy. A `0x0` before and a nonzero after tells you the throttling
happened *during your measurement* — which is exactly the case where the numbers look normal
but are wrong. Log both readings alongside every result.

To clear the sticky bits, reboot (`sudo reboot`). There is no "reset flags" command.

### 6.3 Temperature — `vcgencmd measure_temp`

```bash
# Pi. Instant. Current SoC temperature.
vcgencmd measure_temp
```

Output looks like `temp=42.3'C`.

The Pi begins soft-throttling in the low 80s °C and hard-throttles above that. But the point is
not just to avoid the limit — it's that **an inference benchmark started on a hot Pi runs at a
different clock than the same benchmark on a cool Pi**, even below the throttle threshold.

Practical gate: before starting a timed run, watch the temperature until it settles at a stable
idle value with cooling attached (typically somewhere in the 35–50 °C range depending on your
room; the number matters less than the *stability*). Use the same starting condition for every
candidate you intend to compare.

Watch it live during a run from a second SSH session:

```bash
# Pi. Reprints temp and clock every 2 seconds. Ctrl+C to stop.
watch -n 2 "vcgencmd measure_temp; vcgencmd measure_clock arm"
```

`measure_clock arm` returns hertz — 2400000000 is the full 2.4 GHz. Seeing it sag mid-run is
throttling in action. At idle, a lower reading (e.g. ~1.5 GHz) is normal too, if the CPU
governor hasn't been pinned yet (see 6.4) — the `ondemand` governor deliberately runs slower
when there's no load.

### 6.4 CPU governor — pin the clock

Linux runs a **CPU frequency governor**: a policy that decides what clock speed to run at.
The default is typically `ondemand`, which ramps the clock up under load and back down
when idle.

For interactive use that's ideal. For benchmarking it is a **hidden variable**: two identical
models can time differently purely because the governor happened to ramp at different moments,
and short runs are the worst affected because they may finish before the clock has fully ramped
up. You want the clock pinned so the only difference between candidates is the model.

Check the current governor:

```bash
# Pi. Instant. Prints the active governor for core 0.
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

Expect `ondemand` initially. See what's available:

```bash
# Pi. Instant.
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
```

Set it to `performance` (which pins the maximum clock). Two ways:

**Option A — `cpufrequtils` (nicer, and gives you a persistent default):**

```bash
# Pi. ~30 seconds to install.
sudo apt install -y cpufrequtils

# Set all cores to performance for this boot.
sudo cpufreq-set -g performance

# Verify (prints the current policy per core).
cpufreq-info | grep -i "governor"
```

`cpufreq-info` always prints a boilerplate line like `The governor "performance" may decide
which speed to use` for each core when the governor is set correctly — that's normal output,
not a warning.

To make it survive reboots, `cpufrequtils` reads `/etc/default/cpufrequtils`:

```bash
# Pi. Sets the boot-time default governor.
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
```

**Option B — write the sysfs file directly (no extra package):**

```bash
# Pi. Writes 'performance' to every CPU core's governor file. Instant.
# Not persistent across reboots.
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Note `sudo tee` rather than `sudo echo ... >`: the `>` redirection is performed by *your*
shell, which is unprivileged, so `sudo echo x > /protected/file` fails with "Permission
denied". `tee` is the program that does the writing, and `sudo` applies to it. Also note
`tee` needs a destination *argument* — piping into bare `sudo tee` with nothing after it just
echoes back to your screen and writes nowhere.

Confirm afterwards:

```bash
# Pi. Should print 'performance' once per core (4 lines on a Pi 5).
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Re-check this after every reboot** unless you set the persistent option. It is easy to forget
and it silently degrades a whole day of measurements.

### 6.5 Where the Pi physically sits

A Pi in a closed backpack measures the backpack.

Latency, RAM, and thermal measurements must be done with the Pi **stationary, in open air, with
the active cooler running and the real 27 W PSU attached**. Not on a bed, not under a jacket,
not in a bag "just for a quick run." Restricted airflow raises the steady-state temperature,
which lowers the sustained clock, which inflates latency — and none of that appears in the
output file. It just looks like a slower model.

**Accuracy work is different and it's worth understanding why.** Accuracy is a function of the
arithmetic the model performs, and that arithmetic is bit-identical regardless of clock speed
or temperature. A throttled Pi computes the same logits as a cool one; it just takes longer. So
accuracy evaluation runs can be done anywhere, and remain valid. Only the *timing* and
*thermal* measurements are environment-sensitive.

That distinction is genuinely useful for planning: you can batch the environment-sensitive work
into deliberate stationary sessions and treat accuracy sweeps as fill-in work.

## Section 7 — Getting the project onto the Pi

The code (`src/`) and data artifacts need to land on the Pi in a directory layout that mirrors
the project's expectations: everything in `src/quant/*.py` looks for `artifacts/onnx/`,
`artifacts/pi_data/`, and `artifacts/reports/` **relative to whatever directory you run the
Python command from** — not any arbitrary folder name. Pick one project directory on the Pi and
be consistent about running everything from inside it.

### 7.1 Create the directory layout on the Pi

```bash
# Pi. One time. Creates the project root and the three artifact subfolders it expects.
mkdir -p ~/ARM_challenge/artifacts/onnx ~/ARM_challenge/artifacts/pi_data ~/ARM_challenge/artifacts/reports
```

### 7.2 Copy the source code

```powershell
# Laptop (PowerShell). Copies the whole src/ tree. -r is recursive.
scp -r <PROJECT_ROOT>\src <PI_USER>@<PI_IP>:~/ARM_challenge/
```

### 7.3 Copy the evaluation data bundle

```powershell
# Laptop (PowerShell). Lands the uint8 optval/calib arrays + manifest.
scp -r <PROJECT_ROOT>\artifacts\pi_data <PI_USER>@<PI_IP>:~/ARM_challenge/artifacts/
```

### 7.4 Copy a model's ONNX files

**You need both files per model, not just one.** `src/quant/quantize.py` resolves two paths per
model: `<model>.onnx` (the deployable FP32 graph, used for the `fp32` baseline) and
`<model>_quant_ready.onnx` (the shape-inferred/fused graph the dynamic and static quantized
variants are actually built from). Copying only one produces a `build_failed: Missing
artifacts/onnx/<model>...onnx` error the first time you try to run anything.

```powershell
# Laptop (PowerShell). Repeat for each model you plan to benchmark:
# resnet18_cifar, mobilenetv2_cifar, custom_cnn
scp <PROJECT_ROOT>\artifacts\onnx\<model>.onnx <PI_USER>@<PI_IP>:~/ARM_challenge/artifacts/onnx/
scp <PROJECT_ROOT>\artifacts\onnx\<model>_quant_ready.onnx <PI_USER>@<PI_IP>:~/ARM_challenge/artifacts/onnx/
```

Verify what landed:

```bash
# Pi. Lists files with human-readable sizes so you can confirm nothing truncated.
ls -lh ~/ARM_challenge/artifacts/onnx ~/ARM_challenge/artifacts/pi_data
```

**Transfers over a slow or metered connection are worth being deliberate about.** Copy the
specific models you're about to benchmark, not the whole `artifacts/` tree. If a transfer is
taking minutes, that's the connection, not a hang — but it is also a signal you're moving more
than you need.

For repeated syncing of a directory that mostly hasn't changed, `rsync` only transfers the
differences and is much kinder to a slow link:

```powershell
# Laptop (PowerShell). Requires rsync available on the laptop side.
# -a preserves attributes, -v is verbose, -z compresses in transit.
rsync -avz <PROJECT_ROOT>/artifacts/pi_data/ <PI_USER>@<PI_IP>:~/ARM_challenge/artifacts/pi_data/
```

If `rsync` isn't installed on Windows, plain `scp` is fine — just be selective.

### 7.5 Copy the scripts directory

`scripts/pi_prepare.sh` (Section 12.3, run after every boot) lives outside `src/`, so `scp -r src`
in 7.2 does not bring it along. Copy it once, the same way:

```powershell
# Laptop (PowerShell). Copies scripts/, including pi_prepare.sh.
scp -r <PROJECT_ROOT>\scripts <PI_USER>@<PI_IP>:~/ARM_challenge/
```

Skipping this step is what produces `bash: scripts/pi_prepare.sh: No such file or directory` the
first time you try to run it.

### If this goes wrong

**`scp: ... : No such file or directory`** — usually the *destination* directory doesn't exist
on the Pi yet. `scp` will not create intermediate directories. Run the `mkdir -p` from 7.1
first.

**Permission denied on the destination** — you're writing outside your home directory. Stick to
paths under `~/`.

**Running `scp` and getting `ssh: Could not resolve hostname d: Name or service not known`
(or similar)** — you ran the `scp` command inside the Pi's SSH session instead of in a
PowerShell window on the laptop. `scp` for pushing files to the Pi always runs on the laptop
side; check your prompt before running it.

**`build_failed: Missing artifacts/onnx/<model>.onnx`** when running a quant script — you copied
`<model>_quant_ready.onnx` but forgot the plain `<model>.onnx` (or vice versa). See 7.4 — you
need both.

## Section 8 — "Ready for trial" checklist

Tick every box before collecting a single campaign number.

- [ ] **SSH works** — `ssh <PI_USER>@<PI_IP>` gets you to a shell prompt on the Pi
- [ ] **VPN client isn't blocking LAN traffic** (if you use one on the laptop)
- [ ] **venv activates** — `source ~/armopt/bin/activate` and the prompt shows `(armopt)`
- [ ] **ORT version matches the PC's environment exactly**
- [ ] **`CPUExecutionProvider` is present** in `ort.get_available_providers()`
- [ ] **XNNPACK presence/absence recorded** in project notes, with the date, either way
- [ ] **`onnx`, `numpy`, `psutil` all import** cleanly
- [ ] **No torch on the Pi** — `pip list | grep -i torch` returns nothing
- [ ] **`vcgencmd get_throttled` returns `throttled=0x0`**
- [ ] **Governor is `performance`** on all cores
- [ ] **Active cooler attached and its cable connected** (physically verified — it may not be
      spinning at idle, that's normal)
- [ ] **Official 27 W USB-C PSU in use** (physically verified)
- [ ] **Pi is stationary in open air**, not in a bag or under anything
- [ ] **Idle temperature is stable** via `vcgencmd measure_temp`
- [ ] **`src/` and both ONNX files per model copied**, matching the layout in Section 7
- [ ] **`pi_data` copied** and verified with `ls -lh`

### The first real run is a methodology validation, not a result

Once the checklist is clean, **do not** start collecting campaign data. Do this first.

**The test:** take one FP32 model — any of the baselines, it doesn't matter which — and
benchmark it **5 separate times, spread over roughly an hour if you can manage it**. Not five
repetitions inside one script invocation; five genuinely separate runs with real gaps between
them, so that thermal state, governor behaviour, and any background system activity get a
chance to differ. Record the median latency from each.

If you compress the gaps to save time, understand what you're trading away: the whole point of
spreading trials out is giving drift a chance to actually happen, so a hidden rig problem can
surface. A compressed test that passes tells you less than a spread-out one that passes — it
could still hide a problem that only shows up over a longer session. If you do compress it,
still leave at least a minute or two between trials rather than running them back-to-back, and
watch `vcgencmd get_throttled` / `measure_temp` closely.

**The acceptance criterion:** the five medians must agree within about **3%** of each other.

**Why medians rather than means:** a single OS scheduling hiccup produces one enormous outlier
that drags a mean noticeably while barely moving a median. You want a statistic that describes
the typical iteration, not one that a stray interrupt can hijack.

**What it means if they don't agree:** the problem is your **measurement rig**, not the model.
Nothing about the model changed across those five runs, so any spread larger than a few percent
is pure apparatus noise. If your rig can't reproduce itself on an *identical* model, it cannot
possibly resolve genuine differences between *different* candidates — and the whole campaign
would be comparing noise. Common culprits, in the order worth checking:

1. Governor drifted back to `ondemand` (did you reboot?).
2. Throttling — check `get_throttled` after each of the five runs, not just at the end.
3. Thermal drift — the Pi got progressively hotter across the hour; cooling is inadequate.
4. Background load — something else on the Pi is consuming CPU. Check with `top` or `htop`.
5. Insufficient warmup inside the benchmark itself — the first iterations include one-time
   costs (memory allocation, kernel selection, cache warming) that shouldn't count.

**Fix the rig, then re-run the five-trial test until it passes.** No campaign numbers before it
does. This is the least glamorous hour of the project and it's the one that determines whether
any of the later results mean anything.

**Practical tip for saving each trial's report:** the benchmark script overwrites its own output
file on every run, so if you're timing the same model five times in a row, copy the report out
between trials or it'll be gone:

```bash
# Pi. Run right after each trial, before starting the next one. Adjust the number each time.
cp artifacts/reports/<model>_latency_aarch64.json artifacts/reports/trial_1_<model>_latency_aarch64.json
```

This copying is only needed when repeating the *same* model/command multiple times, like this
validation step — normal campaign runs across different models don't collide with each other,
since the report filenames already include the model name.

## Section 9 — Your first real trial

The rig is validated and the project files are in place per Section 7. The three valid
`--model` values are `resnet18_cifar`, `mobilenetv2_cifar`, and `custom_cnn`.

Run the three commands below **in this order, per model**. It is deliberately cheap-to-expensive
and blocker-first: run 1 is fast and can invalidate an entire branch of the project, so it goes
first. Do not reorder them to get to the exciting latency numbers sooner.

Before starting, activate the venv and move into the project directory:

```bash
# Pi. Start of every session.
source ~/armopt/bin/activate
cd ~/ARM_challenge
ls artifacts/onnx artifacts/pi_data
```

> **Use `tmux` for the longer runs.** The Pi needs no internet once the models and `pi_data`
> are local — it only needed the network to receive them. So you can start a run inside `tmux`,
> disconnect the laptop, and walk away; the Pi keeps working. Start with `tmux` (installed back
> in Section 2.2), launch the command, detach with `Ctrl+B` then `D`, and reattach later with
> `tmux attach`. Without tmux, closing the SSH session kills the run. Remember tmux gives you a
> fresh shell, so re-run `source ~/armopt/bin/activate` and `cd ~/ARM_challenge` inside it too.

### Run 1 — Saturation probe (do this first)

```bash
# Pi, inside (armopt), from ~/ARM_challenge. The single highest-value run on this device.
python -m src.quant.saturation_probe --model resnet18_cifar
```

**The question it answers:** is your PC's INT8 accuracy data an artifact of the PC's CPU, or a
real property of quantization?

**Background.** Many consumer x86 CPUs (anything without VNNI — e.g. AMD Zen 1–3, older Intel)
cause ONNX Runtime's U8S8 INT8 path to accumulate in 16 bits and **saturate** — intermediate
sums exceed what the accumulator can hold and get clipped. A common symptom: static
**per-channel** quantization scores *worse* than static **per-tensor** on the PC. That result is
impossible as a property of quantization itself — per-channel is strictly more expressive than
per-tensor, so it cannot be inherently worse. It must be a hardware artifact if you see it.

The Pi 5's Cortex-A76 implements ARMv8.2 dot-product instructions (`SDOT`/`UDOT`), which
accumulate into 32 bits and should not saturate — but that's a hypothesis until this probe runs
on the device.

**How to read the result:**

- **Per-channel now beats per-tensor on ARM** → the hypothesis holds. Your PC numbers were a
  hardware artifact, per-channel quantization behaves as theory predicts on the deployment
  target, and a mixed-precision search built on these accuracy numbers rests on solid ground.
  Proceed.
- **Per-channel still loses to per-tensor on ARM** → something deeper is wrong, and it is not
  the accumulator width. Do **not** start a search campaign. The per-channel branch needs
  rethinking first (calibration data, the quantization config, or the export itself), because a
  search built on a broken accuracy signal optimizes toward the wrong models.

**Runtime:** the shortest of the three — it evaluates a diagnostic subset, not the full set.

**Note on environment:** this is *accuracy* work. Accuracy is bit-identical regardless of clock
speed or temperature (Section 6.5), so this probe stays valid even on a throttled or
battery-powered Pi. It is the one run you can legitimately do in a less-than-ideal setting.

### Run 2 — Quantization baselines

```bash
# Pi, inside (armopt), from ~/ARM_challenge. Produces the honest ARM accuracy table.
python -m src.quant.baselines --model resnet18_cifar
```

**The question it answers:** what does each quantization strategy actually cost in accuracy, on
the hardware you deploy to?

It evaluates four configurations — **fp32**, **dynamic**, **static per-tensor**, and **static
per-channel** — over the optval images. This table **replaces** any PC-side table entirely, if
your PC CPU lacks VNNI (Section 6's saturation story) — the PC's per-channel numbers aren't
trustworthy there; the Pi's are.

**Prerequisite:** `artifacts/pi_data/` must already be on the device (Section 7). If it isn't,
this run fails immediately on a missing-file error rather than wasting your time.

**Runtime:** this is the **slowest of the three** — it runs thousands of inferences across four
configurations. Good candidate for `tmux`.

Note the optval set is deliberately *not* the CIFAR-100 test set. The test set is sealed for
final evaluation only and was never exported to the Pi.

### Run 3 — Latency measurement

```bash
# Pi, inside (armopt), from ~/ARM_challenge. Bracket this one with throttle checks.
vcgencmd get_throttled                        # want throttled=0x0
python -m src.quant.measure --model resnet18_cifar
vcgencmd get_throttled                        # want throttled=0x0 again
```

**The question it answers:** how fast is this model, actually — the first **admissible** latency
numbers in the project.

This is the run that genuinely depends on the clean rig from Section 6. Accuracy work tolerates
a hot or throttled Pi; this does not. Governor pinned to `performance`, cooling running, real
27 W PSU, Pi stationary in open air, `get_throttled` clean on **both** sides of the run. If the
post-run check is nonzero, discard the result and fix the rig.

**How admissibility is enforced in code.** This isn't an honour system — `measure.py` mechanizes
it:

```python
ADMISSIBLE_MACHINES = ("aarch64", "arm64")
```

Every report is stamped with an `admissible` boolean that is true only when
`platform.machine()` is in that tuple — i.e. only on ARM. The output filename is also suffixed
with `platform.machine()`, so:

- the Pi writes `artifacts/reports/<model>_latency_aarch64.json`
- a PC writes `artifacts/reports/<model>_latency_amd64.json`

The two **coexist and can never overwrite each other**, and you can tell them apart at a glance
from the filename alone. That design exists because a PC latency number silently clobbering a Pi
one would be an invisible, unrecoverable methodology failure.

**Expect `"admissible": true` in the output JSON.** Treat that flag as the gate: if it is
`false`, the number may not be quoted anywhere — not in a table, not in a presentation, not in
a conversation. Check it:

```bash
# Pi. Confirms the admissibility flag without opening the whole file.
grep admissible artifacts/reports/resnet18_cifar_latency_aarch64.json
```

#### What to watch for in run 3

On a VNNI-less PC, dynamic quantization can measure dramatically slower than FP32 — that is not
a plausible property of dynamic quantization in general; it is almost certainly a PC-CPU-path
artifact, and it is worth checking whether it disappears on ARM.

So look specifically at the dynamic row:

- **Dynamic is competitive on the Pi** → confirmed as a PC artifact. Nothing further needed.
- **Dynamic is still catastrophically slow on the Pi** → that is a **real finding**, and a
  valuable one. It means dynamic quantization is dead as a search dimension on this hardware,
  and you want to know that on day one rather than after burning search budget exploring a
  branch that can never win. Record it and drop dynamic from the search space.

Either outcome is useful. This is the kind of result that is only obtainable by measuring on the
target.

### Then repeat for the other two models

Run all three commands for **one model first**, end to end, to prove the path works before
scaling up. Once that's clean, copy over the ONNX files for the next model (Section 7.4) and
repeat the same three commands for it.

That's **nine reports** total (3 models × 3 runs) once you've done all three. When all nine
exist, the Pi has finished its on-device measurement job: every downstream decision now rests on
measured ARM numbers rather than PC estimates, and later phases of the project (sensitivity
analysis, search) can begin.

## Section 10 — Shutting down properly

**Do not yank the power cable on a running Pi.**

Linux keeps filesystem writes buffered in RAM and flushes them to storage in batches — it's a
large performance win, but it means the SD card's on-disk state briefly lags what the OS thinks
is written. Cutting power mid-flush can leave the filesystem inconsistent: at best a slow
fsck-on-boot, at worst a corrupted card and a reflash. SD cards are noticeably less forgiving
of this than SSDs.

Correct shutdown:

```bash
# Pi. Flushes buffers, stops services, halts the system. ~10 seconds.
sudo shutdown -h now
```

Your SSH session will drop with something like `Connection to <PI_IP> closed by remote
host.` — that is the expected, correct outcome, not an error.

**Then watch the board.** The green activity LED blinks while storage is still being written.
**Wait until it stops blinking entirely** (a few seconds after the session drops), then wait
another couple of seconds for margin, and only then unplug the USB-C cable.

To reboot instead of shutting down:

```bash
# Pi. Restarts. Back on the network in ~30-60 seconds.
sudo reboot
```

After any reboot, remember: the venv is deactivated, the governor may have reset to `ondemand`,
and the `get_throttled` sticky bits are cleared. Re-do the relevant checklist items.

## Section 11 — Command quick reference

### Laptop (PowerShell)

| Task | Command |
| --- | --- |
| Connect to the Pi | `ssh <PI_USER>@<PI_IP>` |
| Clear a stale host key | `ssh-keygen -R <PI_IP>` |
| Find the Pi if the IP moved | `2..254 \| ForEach-Object { $ip = "192.168.1.$_"; if (Test-Connection $ip -Count 1 -Quiet) { Write-Output "alive: $ip" } }` (adjust subnet) |
| Copy the source tree to the Pi | `scp -r <PROJECT_ROOT>\src <PI_USER>@<PI_IP>:~/ARM_challenge/` |
| Copy a directory to the Pi | `scp -r <local-dir> <PI_USER>@<PI_IP>:~/ARM_challenge/` |
| Copy one file to the Pi | `scp <local-file> <PI_USER>@<PI_IP>:~/ARM_challenge/artifacts/onnx/` |

### Pi (bash)

| Task | Command |
| --- | --- |
| Activate the venv | `source ~/armopt/bin/activate` |
| Move to the project root | `cd ~/ARM_challenge` |
| Confirm architecture | `uname -m` (expect `aarch64`) |
| Check ORT version + providers | `python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"` |
| Confirm no torch is installed | `pip list \| grep -i torch` (expect no output) |
| **Throttle/undervoltage flag** | `vcgencmd get_throttled` (want `throttled=0x0`) |
| Temperature | `vcgencmd measure_temp` |
| Current ARM clock | `vcgencmd measure_clock arm` |
| Live temp + clock monitor | `watch -n 2 "vcgencmd measure_temp; vcgencmd measure_clock arm"` |
| Read CPU governor | `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| Set governor to performance | `sudo cpufreq-set -g performance` |
| Set governor (no extra package) | `echo performance \| sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| Start a persistent session | `tmux` (detach: `Ctrl+B` then `D`; reattach: `tmux attach`) |
| Check running processes | `top` (press `q` to quit) |
| Free RAM / disk | `free -h` / `df -h` |
| Run saturation probe | `python -m src.quant.saturation_probe --model <model>` |
| Run accuracy baselines | `python -m src.quant.baselines --model <model>` |
| Run latency measurement | `python -m src.quant.measure --model <model>` |
| Shut down | `sudo shutdown -h now` (wait for the LED to stop, then unplug) |
| Reboot | `sudo reboot` |

---

## Section 12 — Unattended operation (Phase 3 automation)

Sections 1–11 describe driving the Pi **by hand**. That is how every number in
`artifacts/reports_pi/` was produced, and it does not survive contact with Phase 6, which
needs on the order of 300 measurements across three models. This section switches the Pi
from "a machine you log into" to "a machine your laptop calls".

Three things have to be true before that works.

### 12.1 Key-based SSH (mandatory, not a convenience)

With password authentication an unattended run does not fail — it **hangs**, waiting at a
prompt nobody will answer, until a timeout fires hours later. The driver therefore forces
`BatchMode=yes`, which turns a missing key into an immediate error instead. That makes key
auth a requirement.

```powershell
# Laptop (PowerShell). Creates a key only if you don't already have one. Instant.
if (-not (Test-Path "$env:USERPROFILE\.ssh\id_ed25519")) { ssh-keygen -t ed25519 -C "armopt-pi" }
```

```powershell
# Laptop (PowerShell). Appends your public key to the Pi's authorized_keys.
# This is the ONE time you still type the Pi password.
$key = Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
ssh <PI_USER>@<PI_IP> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$key' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

```powershell
# Laptop (PowerShell). Must print 'key-auth-ok' WITHOUT prompting for a password.
ssh -o BatchMode=yes <PI_USER>@<PI_IP> "echo key-auth-ok"
```

If that last command says `Permission denied (publickey)`, the key did not land. Check
permissions on the Pi: `~/.ssh` must be `700` and `authorized_keys` must be `600`, or sshd
silently ignores the file.

### 12.2 `pi_target.json` — where the laptop looks for the device

Copy the template in the project root on the **laptop** and fill in your own device:

```powershell
# Laptop (PowerShell). The copy is gitignored; the template is the only version in the repo.
copy pi_target.example.json pi_target.json
```

The template carries every field with a placeholder value, and the README's table says which
ones are required. Two that are worth calling out here:

- `remote_root` is the directory from Section 7.1, e.g. `/home/<PI_USER>/ARM_challenge`.
- `python` points **inside the venv**, e.g. `/home/<PI_USER>/armopt/bin/python`. The driver
  does not run `source activate`; it invokes the venv interpreter directly, which is
  equivalent and cannot be forgotten. The system `python3` either lacks ONNX Runtime or
  carries a different version, and a version mismatch silently compares two different things.

`ARMOPT_PI_HOST`, `ARMOPT_PI_USER`, `ARMOPT_PI_ROOT`, `ARMOPT_PI_PYTHON` and `ARMOPT_PI_KEY`
override the file from the environment if you would rather not write one.

### 12.3 `pi_prepare.sh` — run after every boot

The CPU governor resets to the image default on reboot, so pinning it is not one-time setup:
it is the first thing done in every benchmarking session.

```bash
# Pi. Run once per boot, before any measurement. Instant.
cd ~/ARM_challenge && sudo bash scripts/pi_prepare.sh
```

It pins `performance` on every core, verifies the readback, prints clock/temp/throttle, and
exits non-zero if the board is throttling or undervolted. The agent (`src/bench/agent.py`)
**refuses to measure** if this was skipped — deliberately, because a campaign run under
`ondemand` measures the frequency scheduler and there is no way to tell from the numbers
afterwards.

### 12.4 Driving it from the laptop

```powershell
# Laptop (PowerShell). Refresh the Pi's copy of src/, then probe the device. ~5 seconds.
python -m src.bench.remote --push-code --check
```

That prints the device's governor, clock, temperature and throttle mask, and warns if the
governor is not pinned. Once it looks right:

```powershell
# Laptop (PowerShell). Measures fp32 at 1, 2 and 4 threads. Results cached by config hash.
python -m src.bench.remote --model resnet18_cifar --configs fp32 --threads 1 2 4
```

Each measurement runs as **two** processes on the Pi: one that quantizes and caches the
artifact, and one that loads and times it. That split is what makes peak RAM attributable —
a process that builds *and* times a candidate reports the calibrator's memory, not the
model's.

### 12.5 What the driver refuses to do

| Situation | Behaviour | Why |
| --- | --- | --- |
| Governor not `performance` | Refuses to measure | The timing would describe the scheduler |
| Throttle bit set during a run | Records the result, marks it inadmissible | It describes thermodynamics, not the model |
| Timings drift between halves | Marks it inadmissible | The device was still warming up |
| SSH link drops | Retries 3× with backoff | Transport failures are transient |
| Quantization fails | Returns the failure, no retry | Deterministic — retrying costs Pi time to reach the same answer |
| Same config asked for twice | Serves from cache | Nothing is ever measured twice |
| Inadmissible result | **Not** cached | An environmental fact must not be replayed as a candidate's number |
