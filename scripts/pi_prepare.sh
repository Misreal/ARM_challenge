#!/usr/bin/env bash
# Put the Pi into a state where a latency number means something.
#
# Run this on the Pi, as root, after every boot:
#
#     sudo bash scripts/pi_prepare.sh
#
# The governor resets to the image default on reboot, so this is not a one-time
# setup step -- it is the first thing done in every benchmarking session. The
# agent (`src/bench/agent.py`) refuses to measure if it was skipped, which is
# the point: a campaign run under `ondemand` measures the frequency scheduler,
# and there is no way to tell that from the numbers afterwards.
#
# Exits non-zero if the device is not fit to benchmark, so it can gate a script.

set -euo pipefail

TARGET_GOVERNOR="performance"
MAX_START_TEMP_C=60

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: must run as root (writing scaling_governor needs it). Use: sudo bash $0" >&2
    exit 1
fi

echo "== Pinning CPU governor to '${TARGET_GOVERNOR}'"
pinned=0
for governor_file in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    [[ -w "${governor_file}" ]] || continue
    echo "${TARGET_GOVERNOR}" > "${governor_file}"
    pinned=$((pinned + 1))
done

if [[ "${pinned}" -eq 0 ]]; then
    echo "ERROR: no writable scaling_governor found under /sys. This kernel does not" >&2
    echo "       expose cpufreq, so clock speed cannot be pinned or verified." >&2
    exit 1
fi
echo "   pinned ${pinned} core(s)"

echo "== Verifying"
observed="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
if [[ "${observed}" != "${TARGET_GOVERNOR}" ]]; then
    echo "ERROR: governor reads back as '${observed}', not '${TARGET_GOVERNOR}'." >&2
    exit 1
fi

clock_khz="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)"
printf '   governor  %s\n' "${observed}"
printf '   clock     %d MHz\n' "$((clock_khz / 1000))"

temp_milli="$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)"
temp_c=$((temp_milli / 1000))
printf '   temp      %d C\n' "${temp_c}"

# Bit 0 = under-voltage now, bit 1 = ARM frequency capped, bit 2 = throttled now,
# bit 3 = soft temperature limit. Anything non-zero in the low nibble means the
# board is not delivering full performance right now -- almost always the PSU.
if command -v vcgencmd >/dev/null 2>&1; then
    throttled="$(vcgencmd get_throttled)"
    printf '   throttle  %s\n' "${throttled}"
    mask="${throttled#throttled=}"
    if [[ "$(( mask & 0xF ))" -ne 0 ]]; then
        echo "ERROR: the board is throttling or undervolted right now (${throttled})." >&2
        echo "       Check the official 27 W USB-C PSU and the active cooler before measuring." >&2
        exit 1
    fi
else
    echo "   throttle  vcgencmd not installed -- throttle state cannot be verified" >&2
fi

if [[ "${temp_c}" -gt "${MAX_START_TEMP_C}" ]]; then
    echo "NOTE: ${temp_c} C is above the ${MAX_START_TEMP_C} C start gate; the agent will" >&2
    echo "      wait for it to fall before timing anything." >&2
fi

echo "== Ready"
