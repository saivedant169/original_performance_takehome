import random
import unittest
from collections import defaultdict
from typing import Optional

from problem import (
    HASH_STAGES,
    N_CORES,
    SCRATCH_SIZE,
    SLOT_LIMITS,
    VLEN,
    DebugInfo,
    Engine,
    Input,
    Machine,
    Tree,
    build_mem_image,
    reference_kernel,
    reference_kernel2,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec_slice(base: int, n: int = VLEN) -> range:
    """Return the scratch addresses that make up a vector starting at *base*."""
    return range(base, base + n)


def _deps(engine: str, slot: tuple) -> tuple[list[int], list[int]]:
    """
    Return (reads, writes) scratch-address sets for a single slot so the
    scheduler can track data-flow without executing anything.
    """
    reads: list[int] = []
    writes: list[int] = []

    if engine == "alu":
        _op, dst, src_a, src_b = slot
        reads = [src_a, src_b]
        writes = [dst]

    elif engine == "valu":
        if len(slot) == 3 and slot[0] == "vbroadcast":
            _, dst, src = slot
            reads = [src]
            writes = list(_vec_slice(dst))
        elif len(slot) == 5 and slot[0] == "multiply_add":
            _, dst, a, b, c = slot
            reads = list(_vec_slice(a)) + list(_vec_slice(b)) + list(_vec_slice(c))
            writes = list(_vec_slice(dst))
        elif len(slot) == 4:
            _op, dst, lhs, rhs = slot
            reads = list(_vec_slice(lhs)) + list(_vec_slice(rhs))
            writes = list(_vec_slice(dst))
        else:
            raise NotImplementedError(f"Unhandled valu slot: {slot}")

    elif engine == "load":
        if slot[0] == "load":
            _, dst, addr = slot
            reads = [addr]
            writes = [dst]
        elif slot[0] == "vload":
            _, dst, addr = slot
            reads = [addr]
            writes = list(_vec_slice(dst))
        elif slot[0] == "const":
            writes = [slot[1]]
        elif slot[0] == "load_offset":
            _, dst, addr, _lane = slot
            reads = [addr]
            writes = [dst]
        else:
            raise NotImplementedError(f"Unhandled load slot: {slot}")

    elif engine == "store":
        if slot[0] == "store":
            _, addr, src = slot
            reads = [addr, src]
        elif slot[0] == "vstore":
            _, addr, src = slot
            reads = [addr] + list(_vec_slice(src))
        else:
            raise NotImplementedError(f"Unhandled store slot: {slot}")

    elif engine == "flow":
        if slot[0] == "select":
            _, dst, cond, a, b = slot
            reads = [cond, a, b]
            writes = [dst]
        elif slot[0] == "add_imm":
            _, dst, src, _imm = slot
            reads = [src]
            writes = [dst]
        elif slot[0] == "vselect":
            _, dst, cond, a, b = slot
            reads = (
                list(_vec_slice(cond))
                + list(_vec_slice(a))
                + list(_vec_slice(b))
            )
            writes = list(_vec_slice(dst))
        elif slot[0] in ("halt", "pause", "trace_write", "jump", "jump_indirect",
                         "cond_jump", "cond_jump_rel", "coreid"):
            pass
        else:
            raise NotImplementedError(f"Unhandled flow slot: {slot}")

    return reads, writes


def pack_into_bundles(
    ops: list[tuple[str, tuple]],
) -> list[dict[str, list[tuple]]]:
    """
    Greedy list-scheduler: place each operation into the earliest cycle where
    (a) all its source operands are ready and (b) the target engine still has
    free slots.  Produces compact VLIW bundles respecting SLOT_LIMITS.
    """
    bundles: list[dict[str, list[tuple]]] = []
    slot_usage: list[dict[str, int]] = []
    # cycle in which an address is ready to be read
    avail: dict[int, int] = defaultdict(int)
    # last cycle that wrote an address (WAW hazard)
    last_wr: dict[int, int] = defaultdict(lambda: -1)
    # last cycle that read an address (WAR hazard)
    last_rd: dict[int, int] = defaultdict(lambda: -1)

    def _grow(cyc: int) -> None:
        while len(bundles) <= cyc:
            bundles.append({})
            slot_usage.append(defaultdict(int))

    def _earliest_free(engine: str, not_before: int) -> int:
        cyc = not_before
        cap = SLOT_LIMITS[engine]
        while True:
            _grow(cyc)
            if slot_usage[cyc][engine] < cap:
                return cyc
            cyc += 1

    for eng, slot in ops:
        reads, writes = _deps(eng, slot)

        # Data-flow: cannot start before all inputs are written
        ready = max((avail[r] for r in reads), default=0)
        # Hazards: WAW and WAR
        for w in writes:
            ready = max(ready, last_wr[w] + 1, last_rd[w])

        cyc = _earliest_free(eng, ready)
        _grow(cyc)
        bundles[cyc].setdefault(eng, []).append(slot)
        slot_usage[cyc][eng] += 1

        for r in reads:
            if last_rd[r] < cyc:
                last_rd[r] = cyc
        for w in writes:
            last_wr[w] = cyc
            avail[w] = cyc + 1

    return [b for b in bundles if b]


# ---------------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------------

class KernelBuilder:
    """
    Builds a VLIW program for the vectorized Merkle-tree kernel.

    Strategy
    --------
    * Represent the whole program as a flat list of (engine, slot) pairs and
      then hand it to *pack_into_bundles* for cycle-accurate scheduling.
    * Pre-load tree nodes 0-14 into broadcast vectors so levels 0-3 need no
      memory gathers at all (only vselect to pick the right node).
    * Levels 4+ still require per-element gathers from the forest array.
    * All scalar constants are built from a zero-initialized scratch cell to
      avoid wasting load slots on small values.
    * Vector constants are allocated first and filled with vbroadcast so
      every arithmetic operation can use a vector operand directly.
    """

    def __init__(self):
        self.instrs: list[dict] = []
        self._scratch_ptr: int = 0
        self._scratch_dbg: dict[int, tuple[str, int]] = {}
        # dedup maps: value -> scratch address
        self._scalar_cache: dict[int, int] = {}
        self._vec_cache: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Scratch allocator
    # ------------------------------------------------------------------

    def _alloc(self, length: int = 1, tag: Optional[str] = None) -> int:
        addr = self._scratch_ptr
        if tag is not None:
            self._scratch_dbg[addr] = (tag, length)
        self._scratch_ptr += length
        assert self._scratch_ptr <= SCRATCH_SIZE, "Scratch space exhausted"
        return addr

    def _alloc_vec(self, tag: Optional[str] = None) -> int:
        return self._alloc(VLEN, tag)

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=self._scratch_dbg)

    # ------------------------------------------------------------------
    # Constant helpers (deduplicating)
    # ------------------------------------------------------------------

    def _scalar(self, val: int, pending: list, tag: Optional[str] = None) -> int:
        """Return the scratch address for a scalar constant, creating it lazily."""
        if val not in self._scalar_cache:
            addr = self._alloc(tag=tag or f"sc_{val}")
            pending.append(("load", ("const", addr, val)))
            self._scalar_cache[val] = addr
        return self._scalar_cache[val]

    def _vec(self, val: int, pending: list, tag: Optional[str] = None) -> int:
        """Return the scratch address for a broadcast vector constant."""
        if val not in self._vec_cache:
            sc = self._scalar(val, pending)
            addr = self._alloc_vec(tag or f"vc_{val}")
            pending.append(("valu", ("vbroadcast", addr, sc)))
            self._vec_cache[val] = addr
        return self._vec_cache[val]

    # ------------------------------------------------------------------
    # Main kernel builder
    # ------------------------------------------------------------------

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        tile_w: int = 16,   # how many VLEN-blocks to process together
        tile_r: int = 13,   # how many rounds to tile together
        strip_r: Optional[int] = None,  # rounds per block-stripe inside tile
    ) -> None:
        """
        Emit all operations for the kernel and schedule them into VLIW bundles.

        Parameters
        ----------
        forest_height : depth of the Merkle tree (levels 0 .. forest_height).
        n_nodes       : total nodes in the forest (= len(forest.values)).
        batch_size    : number of input elements (must be a multiple of VLEN).
        rounds        : number of hash rounds to execute.
        tile_w        : width of the block tile (VLEN-sized groups processed together).
        tile_r        : depth of the round tile.
        """
        assert batch_size % VLEN == 0
        if strip_r is None:
            strip_r = tile_r
        assert 1 <= strip_r <= tile_r

        # ----------------------------------------------------------------
        # Memory layout constants (fixed by build_mem_image)
        # ----------------------------------------------------------------
        HDR_FOREST_VALUES = 7          # mem[7] = forest values pointer
        HDR_INP_INDICES   = 7 + n_nodes
        HDR_INP_VALUES    = 7 + n_nodes + batch_size

        # ----------------------------------------------------------------
        # Phase 1 – allocate ALL scratch regions up front so addresses are
        # stable before any ops are appended.
        # ----------------------------------------------------------------

        # Two generic temporary scalar registers reused throughout.
        tmp0 = self._alloc(tag="tmp0")
        tmp1 = self._alloc(tag="tmp1")

        # Pointers into the memory image (scalar).
        p_forest  = self._alloc(tag="p_forest")
        p_forest_m1 = self._alloc(tag="p_forest_m1")
        p_indices = self._alloc(tag="p_indices")
        p_values  = self._alloc(tag="p_values")

        # ---- small scalar constants ----
        # Allocate all constant cells first so const_map is populated before
        # we allocate vector cells (avoids gaps in address space).
        for v in [0, 1, 2, 3, 4, 7, 8]:
            self._alloc(tag=f"sc_{v}")
            self._scalar_cache[v] = self._scratch_ptr - 1

        # ---- vector constants ----
        for v in [2]:
            self._alloc_vec(tag=f"vc_{v}")
            self._vec_cache[v] = self._scratch_ptr - VLEN

        # ---- forest pointer vector (broadcast of p_forest scalar) ----
        vp_forest = self._alloc_vec(tag="vp_forest")

        # ---- preloaded node scalars + vectors (nodes 0 .. N_PRELOAD-1) ----
        N_PRELOAD = 15
        node_sc = [self._alloc(tag=f"nd_sc_{i}") for i in range(N_PRELOAD)]
        node_vc = [self._alloc_vec(tag=f"nd_vc_{i}") for i in range(N_PRELOAD)]

        # Make sure every node index 0..N_PRELOAD-1 has a scalar constant.
        for i in range(N_PRELOAD):
            if i not in self._scalar_cache:
                self._alloc(tag=f"sc_{i}")
                self._scalar_cache[i] = self._scratch_ptr - 1

        # ---- hash-stage constants (scalars + vectors) ----
        hs_vec1  = []   # first-operand vector for each hash stage
        hs_vec3  = []   # third-operand vector (or None if fused multiply_add)
        hs_mulv  = []   # fused multiply_add multiplier vector (or None)

        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if val1 not in self._scalar_cache:
                self._alloc(tag=f"sc_{val1}")
                self._scalar_cache[val1] = self._scratch_ptr - 1
            if val1 not in self._vec_cache:
                self._alloc_vec(tag=f"vc_{val1}")
                self._vec_cache[val1] = self._scratch_ptr - VLEN
            hs_vec1.append(self._vec_cache[val1])

            if op1 == "+" and op2 == "+" and op3 == "<<":
                # Fuse into multiply_add: val = val * (1 + 2^val3) + val1
                mval = 1 + (1 << val3)
                if mval not in self._scalar_cache:
                    self._alloc(tag=f"sc_{mval}")
                    self._scalar_cache[mval] = self._scratch_ptr - 1
                if mval not in self._vec_cache:
                    self._alloc_vec(tag=f"vc_{mval}")
                    self._vec_cache[mval] = self._scratch_ptr - VLEN
                hs_mulv.append(self._vec_cache[mval])
                hs_vec3.append(None)
            else:
                if val3 not in self._scalar_cache:
                    self._alloc(tag=f"sc_{val3}")
                    self._scalar_cache[val3] = self._scratch_ptr - 1
                if val3 not in self._vec_cache:
                    self._alloc_vec(tag=f"vc_{val3}")
                    self._vec_cache[val3] = self._scratch_ptr - VLEN
                hs_vec3.append(self._vec_cache[val3])
                hs_mulv.append(None)

        # ---- working buffers per tile slot ----
        n_blocks = batch_size // VLEN
        idx_base = self._alloc(n_blocks * VLEN, tag="idx_buf")
        val_base = self._alloc(n_blocks * VLEN, tag="val_buf")
        off_reg  = self._alloc(tag="off_reg")

        # Per-tile scratch contexts: each group slot gets 5 temp vectors.
        ctxs = []
        for i in range(tile_w):
            ctxs.append({
                "node": self._alloc_vec(tag=f"ctx{i}_node"),
                "t0":   self._alloc_vec(tag=f"ctx{i}_t0"),
                "t1":   self._alloc_vec(tag=f"ctx{i}_t1"),
                "t2":   self._alloc_vec(tag=f"ctx{i}_t2"),
                "t3":   self._alloc_vec(tag=f"ctx{i}_t3"),
            })

        # ----------------------------------------------------------------
        # Phase 2 – build the flat operations list
        # ----------------------------------------------------------------
        ops: list[tuple[str, tuple]] = []

        # Shorthand aliases for frequently used addresses.
        sc = self._scalar_cache
        vc = self._vec_cache

        # ---- 2a: emit all const loads (independent → scheduler packs 2/cyc) ----
        ops.append(("load", ("const", p_forest,  HDR_FOREST_VALUES)))

        for v, addr in sc.items():
            if v in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 14, 16, 19, 33, 4097):
                continue   # built from zero-init or arithmetic below
            ops.append(("load", ("const", addr, v)))

        # Build small constants arithmetically from zero-initialised sc[0].
        ops.append(("flow", ("add_imm", sc[1], sc[0], 1)))
        ops.append(("alu",  ("+", sc[2], sc[1], sc[1])))
        ops.append(("alu",  ("+", sc[3], sc[2], sc[1])))
        ops.append(("alu",  ("+", sc[4], sc[2], sc[2])))
        ops.append(("alu",  ("+", sc[5],  sc[4],  sc[1])))
        ops.append(("alu",  ("+", sc[6],  sc[4],  sc[2])))
        ops.append(("alu",  ("+", sc[7],  sc[4],  sc[3])))
        ops.append(("alu",  ("+", sc[8],  sc[4],  sc[4])))
        ops.append(("alu",  ("+", sc[9],  sc[8],  sc[1])))
        ops.append(("alu",  ("+", sc[10], sc[8],  sc[2])))
        ops.append(("alu",  ("+", sc[11], sc[8],  sc[3])))
        ops.append(("alu",  ("+", sc[12], sc[8],  sc[4])))
        ops.append(("alu",  ("+", sc[13], sc[12], sc[1])))
        ops.append(("alu",  ("+", sc[14], sc[12], sc[2])))
        ops.append(("alu",  ("+", sc[16], sc[8],  sc[8])))
        ops.append(("alu",  ("+", sc[19], sc[16], sc[3])))
        ops.append(("alu",  ("+", sc[33], sc[16], sc[16])))
        ops.append(("alu",  ("+", sc[33], sc[33], sc[1])))
        ops.append(("alu",  ("<<", sc[4097], sc[1], sc[12])))
        ops.append(("alu",  ("+",  sc[4097], sc[4097], sc[1])))

        ops.append(("flow", ("add_imm", p_forest_m1, p_forest, -1)))
        ops.append(("flow", ("add_imm", p_indices,   p_forest, n_nodes)))
        ops.append(("flow", ("add_imm", p_values,    p_forest, n_nodes + batch_size)))

        # ---- 2b: broadcast all vector constants (6 valu slots/cycle) ----
        ops.append(("valu", ("vbroadcast", vp_forest, p_forest_m1)))
        for v, addr in vc.items():
            ops.append(("valu", ("vbroadcast", addr, sc[v])))

        # ---- 2c: preload tree nodes 0..N_PRELOAD-1 ----
        ping = tmp0
        pong = tmp1
        for ni in range(N_PRELOAD):
            # Alternate between two temp regs to allow address and load to
            # be scheduled independently.
            reg = ping if ni % 2 == 0 else pong
            ops.append(("alu",  ("+",          reg,         p_forest, sc[ni])))
            ops.append(("load", ("load",        node_sc[ni], reg)))
            ops.append(("valu", ("vbroadcast",  node_vc[ni], node_sc[ni])))

        # ---- 2d: load input indices and values ----
        for blk in range(n_blocks):
            ops.append(("alu",  ("+",     tmp0, p_values,  off_reg)))
            ops.append(("load", ("vload", val_base + blk * VLEN, tmp0)))
            ops.append(("alu",  ("+",     tmp0, p_indices, off_reg)))
            ops.append(("load", ("vload", idx_base + blk * VLEN, tmp0)))
            ops.append(("alu",  ("+",     off_reg, off_reg, sc[8])))
            for lane in range(VLEN):
                ops.append(
                    ("alu", ("+", idx_base + blk * VLEN + lane,
                             idx_base + blk * VLEN + lane, sc[1]))
                )

        # ---- 2e: tiled main loop ----
        for grp_start in range(0, n_blocks, tile_w):
            for rnd_start in range(0, rounds, tile_r):
                rnd_end = min(rounds, rnd_start + tile_r)
                for seg_start in range(rnd_start, rnd_end, strip_r):
                    seg_end = min(rnd_end, seg_start + strip_r)
                    for gi in range(tile_w):
                        blk = grp_start + gi
                        if blk >= n_blocks:
                            break
                        ctx   = ctxs[gi]
                        i_vec = idx_base + blk * VLEN
                        v_vec = val_base + blk * VLEN

                        for rnd in range(seg_start, seg_end):
                            lvl = rnd % (forest_height + 1)
                            self._emit_level(
                                ops, lvl, forest_height,
                                i_vec, v_vec, ctx,
                                vp_forest, node_vc, sc, vc,
                            )
                            self._emit_hash(
                                ops, v_vec, ctx,
                                hs_vec1, hs_vec3, hs_mulv,
                            )
                            if rnd != rounds - 1:
                                self._emit_idx_update(
                                    ops, lvl, forest_height, rnd,
                                    i_vec, v_vec, ctx, sc, vc,
                                )

        # ---- 2f: write results back to memory ----
        ops.append(("flow", ("add_imm", tmp0, p_values, 0)))
        for blk in range(n_blocks):
            ops.append(("store", ("vstore", tmp0, val_base + blk * VLEN)))
            if blk != n_blocks - 1:
                ops.append(("alu", ("+", tmp0, tmp0, sc[8])))

        # ----------------------------------------------------------------
        # Phase 3 – schedule everything into VLIW bundles
        # ----------------------------------------------------------------
        self.instrs.extend(pack_into_bundles(ops))

    # ------------------------------------------------------------------
    # Operation emitters
    # ------------------------------------------------------------------

    def _emit_level(
        self,
        ops: list,
        lvl: int,
        forest_height: int,
        i_vec: int,
        v_vec: int,
        ctx: dict,
        vp_forest: int,
        node_vc: list,
        sc: dict,
        vc: dict,
    ) -> None:
        """Emit the XOR-with-node operations for one level of the Merkle tree."""

        def xor_with(nvec: int) -> None:
            """Emit VLEN ALU XOR ops: v_vec ^= nvec (lane-wise)."""
            for lane in range(VLEN):
                ops.append(
                    ("alu", ("^", v_vec + lane, v_vec + lane, nvec + lane))
                )

        if lvl == 0:
            xor_with(node_vc[0])

        elif lvl == 1:
            # bit0 from prior lvl0 update is kept in ctx["node"].
            ops.append(("flow", ("vselect", ctx["t0"], ctx["node"],
                                 node_vc[2], node_vc[1])))
            xor_with(ctx["t0"])

        elif lvl == 2:
            # bit0 from lvl0 is in node, bit1 from lvl1 is in t2.
            ops.append(("flow", ("vselect", ctx["t0"],   ctx["t2"],
                                 node_vc[4], node_vc[3])))
            ops.append(("flow", ("vselect", ctx["t1"],   ctx["t2"],
                                 node_vc[6], node_vc[5])))
            ops.append(("flow", ("vselect", ctx["t0"],   ctx["node"],
                                 ctx["t1"],  ctx["t0"])))
            xor_with(ctx["t0"])

        elif lvl == 3:
            # bit0, bit1, bit2 are carried in node/t2/t3.
            ops.append(("flow", ("vselect", ctx["t0"],   ctx["t3"],
                                 node_vc[8],  node_vc[7])))
            ops.append(("flow", ("vselect", ctx["t1"],   ctx["t3"],
                                 node_vc[10], node_vc[9])))
            ops.append(("flow", ("vselect", ctx["t0"],   ctx["t2"],
                                 ctx["t1"],   ctx["t0"])))

            ops.append(("flow", ("vselect", ctx["t1"],   ctx["t3"],
                                 node_vc[12], node_vc[11])))
            ops.append(("flow", ("vselect", ctx["t3"],   ctx["t3"],
                                 node_vc[14], node_vc[13])))
            ops.append(("flow", ("vselect", ctx["t1"],   ctx["t2"],
                                 ctx["t3"],   ctx["t1"])))
            ops.append(("flow", ("vselect", ctx["t0"],   ctx["node"],
                                 ctx["t1"],   ctx["t0"])))
            xor_with(ctx["t0"])

        else:
            # Deep levels: gather from memory using the forest pointer vector.
            for lane in range(VLEN):
                ops.append(
                    ("alu", ("+", ctx["t0"] + lane,
                             vp_forest + lane, i_vec + lane))
                )
                ops.append(
                    ("load", ("load", ctx["node"] + lane, ctx["t0"] + lane))
                )
            xor_with(ctx["node"])

    def _emit_hash(
        self,
        ops: list,
        v_vec: int,
        ctx: dict,
        hs_vec1: list,
        hs_vec3: list,
        hs_mulv: list,
    ) -> None:
        """Emit the six-stage hash update for one element block."""
        for hi, (_op1, _v1, op2, _op3, _v3) in enumerate(HASH_STAGES):
            mv = hs_mulv[hi]
            if mv is not None:
                # Fused: v = v * mv + hs_vec1[hi]
                ops.append(("valu", ("multiply_add", v_vec, v_vec,
                                     mv, hs_vec1[hi])))
            else:
                # Two-op form: v = (v op1 c1) op2 (v op3 c3)
                ops.append(("valu", (_op1, ctx["t0"], v_vec, hs_vec1[hi])))
                ops.append(("valu", (_op3, ctx["t1"], v_vec, hs_vec3[hi])))
                ops.append(("valu", (op2,  v_vec, ctx["t0"], ctx["t1"])))

    def _emit_idx_update(
        self,
        ops: list,
        lvl: int,
        forest_height: int,
        rnd: int,
        i_vec: int,
        v_vec: int,
        ctx: dict,
        sc: dict,
        vc: dict,
    ) -> None:
        """Emit the index-update step after the hash."""
        if lvl == forest_height:
            # Wrap in one-based form: reset j to 1.
            ops.append(("valu", ("vbroadcast", i_vec, sc[1])))
        else:
            # One-based update: j = j * 2 + (val & 1)
            if lvl == 0:
                bit_vec = ctx["node"]
            elif lvl == 1:
                bit_vec = ctx["t2"]
            elif lvl == 2:
                bit_vec = ctx["t3"]
            else:
                bit_vec = ctx["t0"]

            for lane in range(VLEN):
                ops.append(
                    ("alu", ("&", bit_vec + lane, v_vec + lane, sc[1]))
                )
            ops.append(
                ("valu", ("multiply_add", i_vec, i_vec,
                          vc[2], bit_vec))
            )


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
) -> int:
    print(f"Testing forest_height={forest_height}, rounds={rounds}, batch_size={batch_size}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp    = Input.generate(forest, batch_size, rounds)
    mem    = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace: dict = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints

    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        vp = ref_mem[6]
        assert (
            machine.mem[vp : vp + len(inp.values)]
            == ref_mem[vp : vp + len(inp.values)]
        ), f"Value mismatch on round {i}"
        ip = ref_mem[5]
        if prints:
            print("got idx:", machine.mem[ip : ip + len(inp.indices)])
            print("exp idx:", ref_mem[ip : ip + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for _ in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values  == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    do_kernel_test(10, 16, 256)
