# Runbook: Rollback

Trigger on quality/performance/failure regression. Freeze promotion, identify bad revision, restore immutable known-good revision, verify readiness, smoke, health/latency, preserve timestamps and artifacts, then analyze.

Evidence: detected time, rollback start, known-good ready/healthy time, reason, bad/restored revisions.
