"""Dump per-shot golden I/O vectors for RTL verification.

Syndromes come from the stim DEM sampler, the same source the LER runs use, so
these are real decoding problems rather than synthetic ones. For each selected
shot this writes the detector pattern going in and the correction, final LLRs
and iteration count coming out.

The static side of the problem (parity check adjacency, prior LLRs, row
schedule) is emitted separately by dump_static_config.py, whose loader this
script shares so the two cannot drift apart.

Shot selection matters. Shots that converge in one iteration exercise almost
nothing, while the shots that decide the logical error rate are the slow ones
and the failures, so --pick defaults to a mix of both.

Floating point words are the raw IEEE754 double bit pattern in 16 hex digits,
integers are 8 hex digits, both matching dump_static_config.py.

Run from the repository root:
    python dump_io_vectors.py --d 6 --p 0.002 --shots 4096 --n 8
"""

import argparse
import os

import numpy as np

from dump_static_config import load_problem, write_f64, write_u32
from my_decoders.hbplib_wrapper_v2 import (
    ldpc_dec_msaa_quantum_serial_big_matrix_c as decode,
    ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler as decode_batched,
    HBP_TR_ALL)


def sample_syndromes(prob, shots, seed):
    """Sample detector data and lay it out over the big matrix rows, exactly as
    stim_batched_data_v2.py does before calling the decoder."""
    det, obs, _ = prob['dem'].compile_sampler(seed=seed).sample(shots=shots, bit_packed=False)
    det = det.astype(np.int8)
    obs = obs.astype(np.int8)

    det_index, mx, m = prob['det_index'], prob['mx'], prob['m']
    big = np.zeros((shots, prob['big_rows']), dtype=np.int8)
    big[:, :mx] = det[:, np.where(det_index == 1)[0]]   # X detectors see Z errors
    big[:, mx:m] = det[:, np.where(det_index == 3)[0]]  # Z detectors see X errors
    return big, obs


def logical_error(prob, hdec, obs_row):
    """Whether this correction leaves a logical error, matching the check the
    batched decoder does in C on corr_dz = correction[nx:]."""
    l_dz = prob['l_dz']
    corr_dz = hdec[prob['nx']:prob['nx'] + l_dz.shape[1]]
    return bool(np.any((l_dz @ corr_dz + obs_row) % 2))


def decode_shot(prob, args, syn_row, trace_dir=None, shot_id=0):
    return decode(
        llr=prob['llr'], Nloop=args.iters,
        Hrows=prob['Hrows'], Hcols=prob['Hcols'],
        alpha=args.alpha, Syn_x=syn_row,
        random_order=args.rs,
        hshape=(prob['hshape0'],), dxshape=(prob['dxshape0'],),
        custom_random_schedule_HBP=args.cs, early_stopping=True,
        trace_dir=trace_dir, trace_flags=HBP_TR_ALL,
        trace_shot=shot_id, trace_max_iters=args.trace_max_iters)


def pick_shots(iters, fails, n, how):
    """Choose which shots are worth emitting."""
    fail_idx = np.flatnonzero(fails)
    slow_idx = np.argsort(-iters)
    if how == 'fail':
        return fail_idx[:n]
    if how == 'slow':
        return slow_idx[:n]
    if how == 'first':
        return np.arange(min(n, len(iters)))
    # mix: failures first, then the slowest shots that are not already in
    out = list(fail_idx[:max(1, n // 2)])
    for i in slow_idx:
        if len(out) >= n:
            break
        if i not in out:
            out.append(int(i))
    return np.array(out[:n], dtype=int)


def cross_check(prob, args, syn, obs, iters, fails):
    """Decode the same syndromes through the production batched path with an
    ensemble of one, which reduces to the single-shot call, and confirm the
    iteration counts and logical errors agree. This is what validates that the
    correction is being sliced and scored the way the C code does."""
    log_out, it_out = decode_batched(
        llr_batch=prob['llr'], llr_ab_batch=prob['llr_ab'],
        batch_size=syn.shape[0], Nloop=args.iters,
        Hrows=prob['Hrows'], Hcols=prob['Hcols'],
        alpha=args.alpha, Syn_x_batch=syn,
        random_order=args.rs,
        hshape=(prob['hshape0'],), dxshape=(prob['dxshape0'],),
        ensemble_size=1, prob_or_itr='i',
        nx=prob['nx'], nz=prob['nz'], mx=prob['mx'], m=prob['m'],
        dz=prob['dz'], l_dz=prob['l_dz'],
        custom_random_schedule_HBP=args.cs, early_stopping=True,
        num_threads=args.threads)

    batched_fail = np.any((log_out + obs) % 2, axis=1)
    same_iters = np.array_equal(np.minimum(iters, args.iters), it_out)
    same_fail = np.array_equal(fails, batched_fail)
    print(f'\ncross-check against the batched production path:')
    print(f'  iteration counts agree : {same_iters}')
    print(f'  logical errors agree   : {same_fail} '
          f'({fails.sum()} vs {batched_fail.sum()} failures)')
    return same_iters and same_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--d', type=int, default=6)
    ap.add_argument('--p', type=float, default=0.002)
    ap.add_argument('--prt', type=int, default=0, choices=[0, 2])
    ap.add_argument('--rs', type=int, default=1)
    ap.add_argument('--cs', type=int, default=2)
    ap.add_argument('--iters', type=int, default=400, help='decoder iteration limit (Nloop)')
    ap.add_argument('--alpha', type=float, default=0.96875)
    ap.add_argument('--shots', type=int, default=4096, help='how many shots to sample and decode')
    ap.add_argument('--n', type=int, default=8, help='how many shots to emit as vectors')
    ap.add_argument('--pick', default='mix', choices=['mix', 'fail', 'slow', 'first'])
    ap.add_argument('--seed', type=int, default=0, help='DEM sampler seed')
    ap.add_argument('--trace', action='store_true', help='also run the five hooks on the emitted shots')
    ap.add_argument('--trace-max-iters', type=int, default=2, dest='trace_max_iters',
                    help='cap traced iterations, hooks 3 and 4 are about 11 MB each per iteration')
    ap.add_argument('--threads', type=int, default=64, help='threads for the cross-check only')
    ap.add_argument('--out', default='vectors/io')
    ap.add_argument('--no-verify', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'building d={args.d} p={args.p} prt={args.prt} cs={args.cs} rs={args.rs} ...')
    prob = load_problem(args.d, args.p, args.prt)

    print(f'sampling {args.shots} shots and decoding ...')
    syn_bits, obs = sample_syndromes(prob, args.shots, args.seed)
    syn_pm1 = (1 - 2 * syn_bits).astype(np.int8)

    iters = np.zeros(args.shots, dtype=np.int32)
    fails = np.zeros(args.shots, dtype=bool)
    hdecs = {}
    for i in range(args.shots):
        hdec, it, _ = decode_shot(prob, args, syn_pm1[i])
        iters[i] = it
        fails[i] = logical_error(prob, hdec, obs[i])
        hdecs[i] = hdec

    conv = iters < args.iters
    print(f'  converged {conv.sum()}/{args.shots}, '
          f'logical failures {fails.sum()}, '
          f'iterations mean {iters.mean():.2f} max {iters.max()}')

    if not args.no_verify and not cross_check(prob, args, syn_pm1, obs, iters, fails):
        raise SystemExit('cross-check failed, not emitting vectors')

    chosen = pick_shots(iters, fails, args.n, args.pick)
    print(f'\nemitting {len(chosen)} shots ({args.pick}) to {args.out}/')

    for rank, shot in enumerate(chosen):
        shot = int(shot)
        d = f'{args.out}/shot{rank:04d}'
        os.makedirs(d, exist_ok=True)

        hdec = hdecs[shot]
        if args.trace:
            hdec, it2, lam = decode_shot(prob, args, syn_pm1[shot], trace_dir=d, shot_id=rank)
            assert it2 == iters[shot], 'tracing perturbed the decode'
        else:
            hdec, _, lam = decode_shot(prob, args, syn_pm1[shot])

        write_u32(f'{d}/in_detectors.hex', syn_bits[shot])
        write_u32(f'{d}/in_syndrome_pm1.hex', syn_pm1[shot])
        write_u32(f'{d}/out_hdec.hex', hdec)
        write_f64(f'{d}/out_lambda.hex', lam)
        with open(f'{d}/info.txt', 'w') as f:
            f.write(f'source_shot={shot}\niterations={iters[shot]}\n'
                    f'converged={bool(conv[shot])}\nlogical_error={bool(fails[shot])}\n'
                    f'detectors_set={int(syn_bits[shot].sum())}\n'
                    f'correction_weight={int(hdec.sum())}\n'
                    f'traced={args.trace}\n')
        print(f'  shot{rank:04d}  src={shot:<6d} iters={iters[shot]:<5d} '
              f'fail={str(bool(fails[shot])):<5s} dets={int(syn_bits[shot].sum()):<4d} '
              f'weight={int(hdec.sum())}')

    with open(f'{args.out}/io_summary.csv', 'w') as f:
        f.write('vector,source_shot,iterations,converged,logical_error,detectors_set\n')
        for rank, shot in enumerate(chosen):
            shot = int(shot)
            f.write(f'shot{rank:04d},{shot},{iters[shot]},{int(conv[shot])},'
                    f'{int(fails[shot])},{int(syn_bits[shot].sum())}\n')

    with open(f'{args.out}/io_meta.txt', 'w') as f:
        f.write(f'd={args.d}\np={args.p}\nprior_type={args.prt}\n'
                f'schedule={args.cs}\nschedule_seed={args.rs}\n'
                f'alpha={args.alpha!r}\nNloop={args.iters}\n'
                f'sampler_seed={args.seed}\nshots_sampled={args.shots}\n'
                f'vectors_emitted={len(chosen)}\npick={args.pick}\n'
                f'detector_words={prob["big_rows"]}\ncorrection_words={len(prob["llr"])}\n'
                f'syndrome_mapping=Syn_x = 1 - 2*detector, so -1 prints as ffffffff\n'
                f'float_word=ieee754_double_hex\nint_word=uint32_hex\n')
    print(f'\nwrote {args.out}/io_summary.csv and io_meta.txt')


if __name__ == '__main__':
    main()
