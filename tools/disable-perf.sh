#!/bin/sh

echo 3 > /proc/sys/kernel/perf_event_paranoid
echo 1 > /proc/sys/kernel/kptr_restrict
echo 1 > /proc/sys/kernel/nmi_watchdog
