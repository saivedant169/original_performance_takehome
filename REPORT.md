# Performance Optimization Report

## 1. Objective

The goal was to optimize the take-home kernel in `perf_takehome.py` for the
official submission benchmark:

- `forest_height=10`
- `rounds=16`
- `batch_size=256`

Success is measured by cycle count from `tests/submission_tests.py`, while
preserving correctness and keeping `tests/` unchanged.

## 2. Final Result

Validation was run with:

```bash
python3 tests/submission_tests.py
```

Observed result:

- Tests: `Ran 9 tests ... OK`
- Cycle count: `1329`
- Reported speedup: `111.16177577125659x` over the baseline constant used by
  the test suite (`BASELINE = 147734`)

Integrity check:

```bash
git diff origin/main tests/
```

Output is empty, so the tests directory is unchanged.

## 3. Design Strategy

The implementation focuses on reducing the most expensive operations and then
packing independent work more tightly in the VLIW schedule.

Core ideas:

- Build one flat operation stream and schedule globally with dependency-aware
  bundling (`pack_into_bundles`), instead of relying on local ordering.
- Replace many memory gathers at shallow tree levels with preloaded node vectors
  and `vselect` trees.
- Reduce vector-ALU pressure by fusing compatible hash patterns into
  `multiply_add`.
- Use ALU lanes for simple lane-wise work (XOR, parity/offset prep) to reserve
  VALU bandwidth for operations that truly need it.
- Tile work by blocks and rounds to increase instruction-level parallelism while
  staying within scratch limits.

## 4. What Changed in the Kernel

### A. Global list scheduling

The kernel first emits a flat list of `(engine, slot)` operations, then
schedules into earliest legal bundles subject to:

- engine slot limits (`SLOT_LIMITS`)
- read-after-write readiness
- write-after-write / write-after-read hazards

This improves packing across rounds and blocks, not just within one local
sequence.

### B. Preload-and-select for levels 0 to 3

Nodes `0..14` are preloaded once (scalar load + `vbroadcast`).  
For tree levels `0`, `1`, `2`, and `3`, the kernel uses bit tests and `vselect`
graphs to choose node values without per-lane gather loads.

Because level is computed as `round % (forest_height + 1)`, these shallow levels
recur in later rounds too, so the preload benefit is reused.

### C. Hash-stage fusion

When a hash stage matches the pattern equivalent to:

`(v + c1) + (v << k)`

it is emitted as a single vector:

`v = multiply_add(v, (1 + 2^k), c1)`

This removes extra intermediate vector operations and lowers VALU demand.

### D. Better ALU/VALU role split

- Lane-wise XOR (`value ^= node`) is emitted as ALU lane ops.
- Index update computes lane offsets in ALU and then performs one vector
  `multiply_add` for `idx = 2*idx + offset`.

This keeps expensive VALU slots available for fused hash work.

### E. Tiling and scratch layout

The kernel keeps full-batch index/value buffers in scratch and processes work in
tiles:

- `tile_w = 17` blocks
- `tile_r = 13` rounds

Each tile slot gets dedicated temporary vectors (`node`, `t0..t3`) to maintain
parallel independent chains for scheduling.

### F. Low-overhead initialization

Small constants are built from zero-initialized scratch with ALU/flow ops where
possible, reducing load-slot usage during setup.

## 5. Correctness and Validation

Correctness is enforced in `tests/submission_tests.py` by comparing kernel output
with `reference_kernel2`. The current implementation passes:

- 8 repeated correctness executions
- all benchmark threshold tests

Total: **9/9 tests passing**.

## 6. Scope and Tradeoffs

This solution is benchmark-focused and intentionally tuned for the fixed
submission configuration (`10, 16, 256`).  
The optimization choices prioritize cycle count for that target over broad
generality.

## 7. Reproducible Commands

```bash
# 1) Ensure tests directory is untouched
git diff origin/main tests/

# 2) Run official benchmark/correctness suite
python3 tests/submission_tests.py
```

Expected current cycle result in this workspace: **1329**.
