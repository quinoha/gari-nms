"""Dump the decoder's static configuration as $readmemh images for RTL.

Everything here is fixed for a given (code, noise model, p, prior type): the
parity check adjacency, the prior LLRs and the row visiting schedule. Only the
syndrome changes from shot to shot, so these files are what an RTL decoder
holds in ROM and the per-shot vectors are generated separately.

The circuit, DEM and prior pipeline is reproduced exactly as in
stim_batched_data_v2.py (see "priors" there); keep the two in step if that
computation ever changes.

Floating point words are the raw IEEE754 double bit pattern in 16 hex digits,
matching my_decoders/hbp_trace.h. Integer words are 8 hex digits.

Run from the repository root:
    python dump_static_config.py --d 6 --p 0.002 --iters 400
"""

import argparse
import os
import pickle
import struct

import numpy as np
import stim
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices

from helper_functions import bb_circuit


# --- writers --------------------------------------------------------------

def write_f64(path, values):
    with open(path, 'w') as f:
        for v in np.asarray(values, dtype=np.float64).ravel():
            f.write('%016x\n' % struct.unpack('>Q', struct.pack('>d', v))[0])
    return len(np.asarray(values).ravel())


def write_u32(path, values):
    with open(path, 'w') as f:
        for v in np.asarray(values).ravel():
            f.write('%08x\n' % (int(v) & 0xFFFFFFFF))
    return len(np.asarray(values).ravel())


# --- schedule -------------------------------------------------------------

def random_permutation(arr, seed):
    """In-place xorshift32 Fisher-Yates, bit for bit as random_permutation()
    in my_decoders/hbp_decoder_v2.c. The array is permuted in place and the
    decoder reuses it across iterations, so the shuffles are cumulative."""
    n = len(arr)
    if n <= 1:
        return arr
    state = seed & 0xFFFFFFFF
    if state == 0:
        state = 1
    for i in range(n - 1, 0, -1):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        j = state % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def build_schedule(rows, hshape0, dxshape0, cs, rs, iters):
    """Reproduce the row order the decoder walks, one row per word, for each
    of the first `iters` iterations. Mirrors the switch in the decoder loop."""
    row_indices = list(range(rows))
    bottom = [i + hshape0 for i in range(rows - hshape0)]
    a = [i for i in range(dxshape0)]
    b = [i + dxshape0 for i in range(hshape0 - dxshape0)]
    ab = [i for i in range(hshape0)]

    out = np.zeros((iters, rows), dtype=np.int64)
    for loop in range(iters):
        if cs == 1:
            if rs:
                random_permutation(ab, rs + loop)
            if rs or loop == 0:
                row_indices = bottom + ab
        elif cs == 2:
            if rs:
                random_permutation(a, rs + loop)
                random_permutation(b, rs + loop)
            if rs or loop == 0:
                row_indices = bottom + a + b
        elif cs == 3:
            if rs:
                random_permutation(b, rs + loop)
                random_permutation(a, rs + loop)
            if rs or loop == 0:
                # the decoder writes b at hshape0 then a at (rows - dxshape0)
                merged = list(bottom) + list(b) + list(a)
                row_indices = merged
        else:
            if rs:
                random_permutation(row_indices, rs + loop)
        out[loop] = row_indices
    return out


# --- setup, mirroring stim_batched_data_v2.py -----------------------------

def load_problem(d, p, prt):
    noise_model = 'CL_both'
    _, code = bb_circuit(d, p=0, p1=0, p2=0, p3=0, p4=0, r=d)
    fname = f'data/circuits/{code.name}_d{d}_{noise_model}_{p}.stim'
    circuit = stim.Circuit.from_file(fname)

    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True,
                                       ignore_decomposition_failures=True)
    matrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)

    with open(f'data/circuits/{code.name}_d{d}_{noise_model}_matrices.pkl', 'rb') as fh:
        xz = pickle.load(fh)
    with open(f'data/circuits/{code.name}_d{d}_{noise_model}_hcols_hrows.pkl', 'rb') as fh:
        Hcols, Hrows = pickle.load(fh)

    h = matrices.check_matrix.toarray().astype(np.int8)
    big_cols = Hcols.shape[0]
    mx, nx = xz.dx.shape
    mz, nz = xz.dz.shape

    # priors, verbatim from stim_batched_data_v2.py
    priors_big = np.zeros(big_cols, dtype='float')
    priors_ab = np.zeros(nx + nz, dtype='float')
    for i in range(nx):
        priors_ab[i] = np.sum(matrices.priors[np.where(xz.i_dx_in_hx == i)[0]])
    for i in range(nz):
        priors_ab[nx + i] = np.sum(matrices.priors[np.where(xz.i_dz_in_hz == i)[0]])

    if prt == 0:
        priors_big[:nx + nz] = 0.5  # zero llr
    elif prt == 2:
        priors_big[:nx + nz] = priors_ab
    else:
        raise NotImplementedError(f'prior type {prt}')

    priors_big[nx + nz:2 * nx + nz] = matrices.priors[xz.i_hx_only]
    priors_big[2 * nx + nz:2 * (nx + nz)] = matrices.priors[xz.i_hz_only]
    priors_big[2 * (nx + nz):] = matrices.priors[xz.i_hy_only]

    llr = np.log((1 - priors_big) / priors_big)
    return dict(code=code, Hrows=Hrows, Hcols=Hcols, llr=llr,
                hshape0=h.shape[0], dxshape0=mx, nx=nx, nz=nz)


# --- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--d', type=int, default=6, help='code distance')
    ap.add_argument('--p', type=float, default=0.002, help='physical error rate')
    ap.add_argument('--prt', type=int, default=0, choices=[0, 2], help='prior type')
    ap.add_argument('--rs', type=int, default=1, help='random schedule seed, 0 disables shuffling')
    ap.add_argument('--cs', type=int, default=2, help='custom schedule: 0 plain, 1 bottom-AB, 2 bottom-A-B, 3 bottom-B-A')
    ap.add_argument('--iters', type=int, default=400, help='how many iterations of schedule to emit')
    ap.add_argument('--alpha', type=float, default=0.96875, help='min-sum normalisation factor')
    ap.add_argument('--out', default='vectors/config', help='output directory')
    ap.add_argument('--no-verify', action='store_true', help='skip the cross-check against the C hook')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'building d={args.d} p={args.p} prt={args.prt} cs={args.cs} rs={args.rs} ...')
    prob = load_problem(args.d, args.p, args.prt)

    Hrows, Hcols, llr = prob['Hrows'], prob['Hcols'], prob['llr']
    rows, stride = Hrows.shape
    nvars, col_stride = Hcols.shape
    row_deg = (Hrows > 0).sum(1)
    col_deg = (Hcols > 0).sum(1)
    nnz = int(row_deg.sum())

    bad = ~np.isfinite(llr)
    if bad.any():
        print(f'  WARNING: {bad.sum()} non-finite llr entries, these cannot be '
              f'represented in fixed point and need clipping before RTL use')

    sched = build_schedule(rows, prob['hshape0'], prob['dxshape0'],
                           args.cs, args.rs, args.iters)

    o = args.out
    counts = {
        'cfg_hrows.hex':     write_u32(f'{o}/cfg_hrows.hex', Hrows),
        'cfg_hcols.hex':     write_u32(f'{o}/cfg_hcols.hex', Hcols),
        'cfg_row_deg.hex':   write_u32(f'{o}/cfg_row_deg.hex', row_deg),
        'cfg_col_deg.hex':   write_u32(f'{o}/cfg_col_deg.hex', col_deg),
        'cfg_llr.hex':       write_f64(f'{o}/cfg_llr.hex', llr),
        'cfg_row_order.hex': write_u32(f'{o}/cfg_row_order.hex', sched),
    }

    with open(f'{o}/cfg_meta.txt', 'w') as f:
        f.write(
            f'code={prob["code"].name}\nd={args.d}\np={args.p}\n'
            f'prior_type={args.prt}\nalpha={args.alpha!r}\n'
            f'schedule={args.cs}\nschedule_seed={args.rs}\n'
            f'rows={rows}\nnvars={nvars}\n'
            f'row_stride={stride}\ncol_stride={col_stride}\n'
            f'nnz={nnz}\nmax_row_degree={int(row_deg.max())}\n'
            f'max_col_degree={int(col_deg.max())}\n'
            f'hshape0={prob["hshape0"]}\ndxshape0={prob["dxshape0"]}\n'
            f'schedule_iters={args.iters}\n'
            f'index_base=1 (0 terminates a row/column)\n'
            f'float_word=ieee754_double_hex\nint_word=uint32_hex\n')

    print(f'\nwrote to {o}/')
    for name, n in counts.items():
        size = os.path.getsize(f'{o}/{name}') / 1e6
        print(f'  {name:20s} {n:>10,} words  {size:6.2f} MB')
    print(f'  {"cfg_meta.txt":20s}')
    print(f'\nrows={rows} nvars={nvars} nnz={nnz} '
          f'row_deg {int(row_deg.min())}..{int(row_deg.max())} '
          f'col_deg {int(col_deg.min())}..{int(col_deg.max())}')

    if not args.no_verify:
        verify_schedule(prob, args, sched)


def verify_schedule(prob, args, sched):
    """Decode one shot with only the row-order hook on and check that the
    schedule generated here is what the C decoder actually walks."""
    from my_decoders.hbplib_wrapper_v2 import (
        ldpc_dec_msaa_quantum_serial_big_matrix_c as dec, HBP_TR_ORDER)
    import tempfile

    n_check = min(4, args.iters)
    rows = prob['Hrows'].shape[0]
    rng = np.random.default_rng(0)
    syn = np.where(rng.random(rows) < 0.5, -1, 1).astype(np.int8)

    with tempfile.TemporaryDirectory() as td:
        dec(llr=prob['llr'], Nloop=n_check - 1,
            Hrows=prob['Hrows'], Hcols=prob['Hcols'],
            alpha=args.alpha, Syn_x=syn, random_order=args.rs,
            hshape=(prob['hshape0'],), dxshape=(prob['dxshape0'],),
            custom_random_schedule_HBP=args.cs, early_stopping=False,
            trace_dir=td, trace_flags=HBP_TR_ORDER, trace_shot=0,
            trace_max_iters=n_check)
        got = np.loadtxt(f'{td}/shot0000_h2_row_order.hex', dtype=str)

    got = np.array([int(x, 16) for x in np.atleast_1d(got)]).reshape(-1, rows)
    n = min(len(got), n_check)
    ok = np.array_equal(got[:n], sched[:n])
    print(f'\nschedule cross-check against the C decoder over {n} iterations: '
          f'{"MATCH" if ok else "MISMATCH"}')
    if not ok:
        first = np.argmax((got[:n] != sched[:n]).any(1))
        print(f'  first differing iteration: {first}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
