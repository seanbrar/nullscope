# Benchmark Results

Python: 3.14.2
Platform: darwin (macOS)

## Reference Measurements

| Benchmark | ns/call |
| ----------- | --------- |
| `time.perf_counter()` | ~80 |
| Empty function call | ~33 |
| Minimal context manager | ~120 |

---

## v0.1.0 Baseline (2026-02-04)

### Disabled Mode (No-op)

| Benchmark | ns/call |
| ----------- | --------- |
| scope entry/exit | 180.4 |
| 3 nested scopes | 485.6 |
| timed() decorator creation | 126.6 |
| timed() decorated call | 34.5 |
| metric() call | 89.9 |

### Enabled Mode

| Benchmark | ns/call |
| ----------- | --------- |
| scope entry/exit | 3,033.5 |
| 3 nested scopes | 9,054.4 |
| 5 nested scopes | 15,611.3 |
| timed() decorated call | 2,908.7 |
| metric() call | 760.8 |
| scope with 2 metadata keys | 3,326.2 |

---

## v0.2.0 Optimizations (2026-02-04)

Changes:

- Replaced `@contextmanager` with explicit `_Scope` class (`__enter__`/`__exit__`)
- Identity function singleton for no-op `timed()` (eliminates lambda allocation)
- Single tuple creation per scope entry
- Optimized `metric()` to avoid redundant `.join()` calls

### Disabled Mode (No-op)

| Benchmark | v0.1.0 | v0.2.0 | Change |
| ----------- | -------- | -------- | -------- |
| scope entry/exit | 180.4 | 181.4 | — |
| 3 nested scopes | 485.6 | 478.8 | -1% |
| timed() decorator creation | 126.6 | 88.7 | **-30%** |
| timed() decorated call | 34.5 | 32.9 | -5% |
| metric() call | 89.9 | 80.9 | -10% |

### Enabled Mode

| Benchmark | v0.1.0 | v0.2.0 | Change |
| ----------- | -------- | -------- | -------- |
| scope entry/exit | 3,033.5 | 2,531.1 | **-17%** |
| 5 nested scopes | 15,611.3 | 13,160.2 | **-16%** |
| timed() decorated call | 2,908.7 | 2,295.6 | **-21%** |
| metric() call | 760.8 | 691.6 | **-9%** |
| scope with 2 metadata keys | 3,326.2 | 2,346.1 | **-29%** |

---

## Future Optimization Candidates

1. **Validation caching**: Cache validated scope names in a set for repeated calls
2. **Empty metadata short-circuit**: Skip dict merge when user metadata is empty
3. **Disabled scope overhead**: Still ~60 ns above minimal CM baseline (acceptable)
