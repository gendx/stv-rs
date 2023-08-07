#!/bin/sh

echo -1 > /proc/sys/kernel/perf_event_paranoid
echo 0 > /proc/sys/kernel/kptr_restrict
echo 0 > /proc/sys/kernel/nmi_watchdog
