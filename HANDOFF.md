# RTL golden vector work — handoff

Context for picking this up on another machine. The goal is an RTL
implementation of the HBP decoder taken far enough to report PnR numbers
(area, timing, power) in a paper.

---

## 1. Setup on a new machine

Python 3.11 is required by the README.

```bash
conda create -n gari-nms python=3.11 -y
conda activate gari-nms
pip install ldpc stim sinter
```

Known good versions: ldpc 2.4.1, stim 1.16.0, sinter 1.16.0.

**Then rebuild the shared library. This is not optional.**

```bash
cd my_decoders && make clean && make && cd ..
```

`my_decoders/hbplib_v2.so` is tracked in git but it is a build artifact, and
the committed copy is a **macOS arm64** binary. It will not load on Linux.
Rebuilding leaves the repo dirty on that file — **do not commit it**, or macOS
users break. Gitignoring it is a sensible cleanup nobody has done yet.

**Run everything from the repository root.** The ctypes wrapper loads
`my_decoders/hbplib_v2.so` by relative path.

Sanity check:

```bash
python -c "from my_decoders.hbplib_wrapper_v2 import ldpc_dec_msaa_quantum_serial_big_matrix_c; print('ok')"
```

---

## 2. What was established

### There is no GPU acceleration, and adding it is a rewrite

No CUDA, cupy, torch or OpenCL anywhere in the Python or C sources. The
Makefile emits CPU code only (`-O3 -fopenmp`). Any GPUs on the host sit idle.

The only speed knob is the OpenMP thread count, the **12th positional
argument** of `stim_batched_data_v2.py`. The script defaults to 4.

Measured on 2× Xeon Gold 6530 (64 physical cores, 128 SMT threads), 16384
shots at d=6:

| threads | 1 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| decode | 11.81s | 2.72s | 0.70s | **0.44s** | 0.44s |

64 is the sweet spot; SMT adds nothing. Re-measure on new hardware — the
optimum is the physical core count, not 64 as such.

Note the guard in `hbp_decoder_v2.c`: if `ensemble_size * N` exceeds 512 MB per
thread the decoder silently halves the thread count, and above 2 GB it aborts.
Watch the `DEBUG: Memory validation` line at large `d` with large ensembles.

### Scale of the problem (d=6, the smallest code available)

| | |
|---|---|
| H | 4464 × 20196 |
| edges (nnz) | 46080 |
| row degree | 2 … 35, mean 10.3 |
| column degree | 1 … 7 |
| `hshape0` / `dxshape0` | 432 / 180 |

Memory an RTL decoder needs (etar ×2 + lambda + H index ROM):

| LLR width | 8-bit | 12-bit | 16-bit |
|---|---|---|---|
| total | 1.59 Mbit | 2.04 Mbit | 2.49 Mbit |

d=12 is substantially larger.

### Things that help the hardware

- **Only the syndrome varies per shot.** `llr` is a constant vector across all
  shots, so priors and the parity check adjacency are ROM. The DUT interface
  is literally syndrome in, correction out.
- **`alpha` = 0.96875 = 31/32**, so normalisation is `x - (x>>5)`, no
  multiplier. The other candidates commented out in the C are 15/16, 63/64,
  127/128 — all shift-friendly.
- **The schedule PRNG is xorshift32**, a few gates. Generating the row order on
  the fly in hardware is far cheaper than the 16 MB ROM image the dumper emits.

### Things that will bite

- **Row degree is very irregular** (2 to 35, mean 10.3). A fixed 35-wide check
  node unit idles most of the time, so a folded or serial CNU is effectively
  forced. This is itself a publishable angle.
- **The reference is double precision.** Bit-exact comparison against
  fixed-point RTL is impossible until a fixed-point model exists.
- **The schedule is serial and lambda is updated in place**, so a quantisation
  error propagates to the next row within the same iteration. Combined with
  early stopping keying off `lambda[idx] < 0`, one flipped sign changes the
  iteration count. **A differing final output therefore does not prove a bug** —
  this is the whole reason the trace hooks exist.

---

## 3. What is built

Three commits on `claude/gpu-acceleration-check-a0014d`. The first is merged to
`main` as PR #1; the other two are not yet in a PR.

### `a4c2aad` — five trace hooks (merged, PR #1)

`my_decoders/hbp_trace.h` plus instrumentation in
`ldpc_dec_msaa_quantum_serial_big_matrix_c`, driven by an `HbpTrace*` that is
`NULL` on every existing call path.

| hook | site | file |
|---|---|---|
| 1 | `lambda` after init | `h1_init_lambda.hex` |
| 2 | `row_indices` per iteration | `h2_row_order.hex` |
| 3 | per-row `lambda`/`etar` slice | `h3_row_lambda.hex`, `h3_row_etar.hex` |
| 4 | end-of-iteration snapshot | `h4_iter_lambda.hex`, `h4_iter_etar.hex` |
| 5 | parity count and stop decision | `h5_parity.txt` |

Verified: traced and untraced runs return bit-identical `Hdec`, `lambda` and
iteration count, and batched decode time is unchanged (0.41s vs 0.44s on 64
threads). The batched OpenMP path passes `NULL` explicitly.

Hooks 3 and 4 use `etar`'s own layout — stride `Hrows_c` = 35, unused slots
zero filled — so a testbench indexes them as a plain 2D array.

### `cc2a8dd` — `dump_static_config.py`

What the RTL holds in ROM.

```bash
python dump_static_config.py --d 6 --p 0.002 --iters 400
```

Emits `cfg_hrows.hex`, `cfg_hcols.hex`, `cfg_row_deg.hex`, `cfg_col_deg.hex`,
`cfg_llr.hex`, `cfg_row_order.hex`, `cfg_meta.txt` into `vectors/config/`.

`random_permutation` is reimplemented in Python bit for bit against the C
xorshift32, **including the fact that the decoder permutes its index arrays in
place, so shuffles are cumulative across iterations**. Getting that wrong would
silently desynchronise the RTL. The script therefore finishes by decoding one
shot with hook 2 on and asserting its generated schedule matches what the C
decoder actually walks.

### `5d8a051` — `dump_io_vectors.py`

Per-shot golden problems from the stim DEM sampler.

```bash
python dump_io_vectors.py --d 6 --p 0.002 --shots 16384 --n 8 --trace
```

Emits per shot: `in_detectors.hex`, `in_syndrome_pm1.hex`, `out_hdec.hex`,
`out_lambda.hex`, `info.txt`, plus the hook files when `--trace` is given.

**Shot selection is the point.** Mean iteration count at d=6 p=0.002 is 1.42,
so the first N shots exercise nothing. `--pick mix` takes failures plus the
slowest shots. A 16384 shot run yields 9 failures including one that runs to
the 400 iteration limit without converging — the hardest case the RTL has to
reproduce.

Validated three ways: the same syndromes through the production batched path
(ensemble of one) agree on iteration counts and logical errors 9 against 9;
reading the emitted files back and decoding reproduces `out_hdec.hex`,
`out_lambda.hex` and the iteration count bit-identically for all 8 vectors; and
the schedule cross-check above.

`load_problem()` lives in `dump_static_config.py` and both scripts share it, so
the circuit, DEM and prior setup cannot drift between the two dumps.

Output formats, consistent across every file: floats are the raw IEEE754
double bit pattern in 16 hex digits, integers are 8 hex digits, one word per
line, for `$readmemh`. In SystemVerilog use `$bitstoreal` until a fixed point
width is chosen. `Hrows`/`Hcols` are **1-indexed with 0 as terminator**.
`Syn_x = 1 - 2*detector`, so −1 prints as `ffffffff`.

---

## 4. Roadmap

| | stage | status |
|---|---|---|
| **0** | float reference vectors | **done** |
| 1 | fixed-point model + word width sweep | **next** |
| 2 | RTL, verified bit-exact against the model | |
| 3 | synthesis → PnR (area, timing) | |
| 4 | gate-level sim + SAIF power | |

### Why Phase 1 comes before any RTL

The paper's independent variable is word width, and area scales with it almost
linearly (the table above). The question "how few bits before LER degrades"
**cannot be answered in RTL** — one LER point needs tens of thousands of shots,
which RTL simulation cannot deliver. It has to be a fast fixed-point C or
Python model, and that same model then becomes the bit-exact golden reference
that makes RTL verification tractable.

So Phase 1 is: reimplement the kernel in fixed point, sweep the width, plot LER
against the float baseline that Phase 0 just established, and pick the width.
Only then is the RTL worth writing.

Phase 4 wants *short* vectors — gate-level simulation is orders of magnitude
slower than RTL, so a handful of shots is the limit.

---

## 5. Gotchas

- **Rebuild the `.so` after every checkout**, and never commit it.
- **Run from the repository root.**
- **Cap `trace_max_iters`.** Hooks 3 and 4 together are about 11 MB per
  iteration at d=6. Uncapped at 400 iterations that is multiple GB per shot.
- `vectors/` is gitignored — `cfg_row_order.hex` alone is 16 MB.
- `meta.txt`'s `iters_run` is the raw C loop counter, so it reads `Nloop+1`
  when the decoder never converged. The Python return value is clamped to
  `Nloop`.
- **Tracing is not thread safe** — it holds `FILE*` state. Single-shot path
  only, never inside the OpenMP batch.
- `stim_batched_data_v2.py` skips a run entirely if its output CSV already has
  `max_errs` errors. Delete the CSV to force a re-run.
- If git fails with `OpenSSL version mismatch. Built against 30000020, you have
  30600010`, something has put a conda env's `lib` on `LD_LIBRARY_PATH` and
  git is loading the wrong `libcrypto`. Use `env -u LD_LIBRARY_PATH git ...`,
  and scope the variable to the env that needs it via its `activate.d` script
  rather than exporting it globally in `.bashrc`.
