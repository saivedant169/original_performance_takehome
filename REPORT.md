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
- Cycle count: `1183`
- Reported speedup: `124.88081149619612x` over the baseline constant used by
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
- Replace gathers at shallow tree levels with preloaded node vectors and
  `vselect` trees.
- Reduce vector-ALU pressure by fusing compatible hash patterns into
  `multiply_add`.
- Keep indices in one-based form in scratch (`1..n_nodes`), so parent/child
  transitions become cheap arithmetic and level selection can reuse carried
  branch bits.
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

### D. One-based index update with carried branch bits

- Input indices are shifted to one-based immediately after load.
- For non-wrap rounds, update is:
  `j = 2*j + (val & 1)` (with `multiply_add` on vectors).
- At wrap (`lvl == forest_height`), index is reset to `1` via `vbroadcast`.
- Low-level branch bits are carried through context vectors (`node/t2/t3`) and
  reused by shallow-level `vselect` logic, reducing recomputation.

### E. Constant and pointer setup tightening

- A larger set of frequently used small constants (`5,6,7,8,9,10,11,12,13,14,16,19,33,4097`)
  is synthesized arithmetically from `sc[0]`/`sc[1]`, reducing load-engine use.
- Input/value pointers are derived from `p_forest` with `add_imm`:
  `p_forest_m1 = p_forest - 1`, `p_indices = p_forest + n_nodes`,
  `p_values = p_forest + n_nodes + batch_size`.
- Gather addresses for deep levels use `vp_forest = broadcast(p_forest - 1)`,
  matching the one-based index convention.

### F. Tiling and scratch layout

The kernel keeps full-batch index/value buffers in scratch and processes work in
tiles:

- `tile_w = 16` blocks
- `tile_r = 13` rounds
- `strip_r = 13` (defaulted from `tile_r`)

Each tile slot gets dedicated temporary vectors (`node`, `t0..t3`) to maintain
parallel independent chains for scheduling.

### G. Latest safe refinement

- In the phase-`2d` preload loop, value vectors are loaded before index vectors.
- The `off_reg` increment is moved before the lane-wise index `+1` fixups.
- Unused vector constants (`vc_1`, `vc_3`, `vc_4`, `vc_7`) are removed.

This combination produced the additional safe drop from `1184` to `1183`.

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

Recent exploration note:

- Additional schedule and micro-op experiments were run after reaching `1183`,
  but none improved on `1183` without correctness loss or regression. Current
  stable best in this workspace remains `1183`.

## 7. Reproducible Commands

```bash
# 1) Ensure tests directory is untouched
git diff origin/main tests/

# 2) Run official benchmark/correctness suite
python3 tests/submission_tests.py
```

Expected current cycle result in this workspace: **1183**.
