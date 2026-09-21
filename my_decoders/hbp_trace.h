// Opt-in observation hooks for generating RTL golden vectors.
//
// The decoder takes an HbpTrace* that is NULL on every normal run, including
// the batched OpenMP path, so the default behaviour and performance are
// unchanged. Tracing writes files and keeps FILE* state, so a single HbpTrace
// must only ever be used from one thread at a time.
//
// All floating point words are dumped as the raw IEEE754 double bit pattern in
// 16 hex digits, one per line, ready for $readmemh. Read them back in
// SystemVerilog with $bitstoreal({hi,lo}) until a fixed point width is chosen.

#ifndef HBP_TRACE_H
#define HBP_TRACE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>

// Hook selection bits, one per instrumented site.
#define HBP_TR_INIT   (1u << 0)  // hook 1: lambda straight after initialisation
#define HBP_TR_ORDER  (1u << 1)  // hook 2: row_indices once the schedule is fixed
#define HBP_TR_ROW    (1u << 2)  // hook 3: per-row lambda / etar slice
#define HBP_TR_ITER   (1u << 3)  // hook 4: end-of-iteration lambda / etar snapshot
#define HBP_TR_PARITY (1u << 4)  // hook 5: parity count and early-stop decision
#define HBP_TR_ALL    (0x1fu)

typedef struct {
    uint32_t flags;
    int      shot;       // only used to name the files
    int      max_iters;  // stop tracing once loop reaches this, <= 0 means no cap
    int      rows;       // Hrows_r, recorded for the meta file
    int      row_stride; // Hrows_c, the fixed stride of every per-row dump
    int      nvars;      // N
    FILE*    f_init;
    FILE*    f_order;
    FILE*    f_row_lam;
    FILE*    f_row_eta;
    FILE*    f_iter_lam;
    FILE*    f_iter_eta;
    FILE*    f_parity;
    FILE*    f_meta;
} HbpTrace;

static inline void hbp_tr_put_f64(FILE* f, double v) {
    union { double d; uint64_t u; } c;
    c.d = v;
    fprintf(f, "%016" PRIx64 "\n", c.u);
}

static inline void hbp_tr_put_u32(FILE* f, uint32_t v) {
    fprintf(f, "%08" PRIx32 "\n", v);
}

// True while this iteration is still inside the traced window.
static inline int hbp_tr_on(const HbpTrace* t, uint32_t bit, int loop) {
    return t && (t->flags & bit) && (t->max_iters <= 0 || loop < t->max_iters);
}

static inline FILE* hbp_tr_open(const char* dir, int shot, const char* name) {
    char path[640];
    snprintf(path, sizeof(path), "%s/shot%04d_%s", dir, shot, name);
    FILE* f = fopen(path, "w");
    if (!f) fprintf(stderr, "hbp_trace: could not open %s\n", path);
    return f;
}

// Creates a trace writing into an existing directory. Returns NULL on failure,
// which the decoder treats as "no tracing", so a failed trace never aborts a run.
__attribute__((visibility("default")))
HbpTrace* hbp_trace_create(const char* dir, uint32_t flags,
                                         int shot, int max_iters) {
    if (!dir) return NULL;
    HbpTrace* t = calloc(1, sizeof(HbpTrace));
    if (!t) return NULL;

    t->flags     = flags;
    t->shot      = shot;
    t->max_iters = max_iters;

    if (flags & HBP_TR_INIT)   t->f_init    = hbp_tr_open(dir, shot, "h1_init_lambda.hex");
    if (flags & HBP_TR_ORDER)  t->f_order   = hbp_tr_open(dir, shot, "h2_row_order.hex");
    if (flags & HBP_TR_ROW) {
                               t->f_row_lam = hbp_tr_open(dir, shot, "h3_row_lambda.hex");
                               t->f_row_eta = hbp_tr_open(dir, shot, "h3_row_etar.hex");
    }
    if (flags & HBP_TR_ITER) {
                               t->f_iter_lam = hbp_tr_open(dir, shot, "h4_iter_lambda.hex");
                               t->f_iter_eta = hbp_tr_open(dir, shot, "h4_iter_etar.hex");
    }
    if (flags & HBP_TR_PARITY) t->f_parity  = hbp_tr_open(dir, shot, "h5_parity.txt");

    t->f_meta = hbp_tr_open(dir, shot, "meta.txt");
    return t;
}

__attribute__((visibility("default")))
void hbp_trace_destroy(HbpTrace* t) {
    if (!t) return;
    if (t->f_init)     fclose(t->f_init);
    if (t->f_order)    fclose(t->f_order);
    if (t->f_row_lam)  fclose(t->f_row_lam);
    if (t->f_row_eta)  fclose(t->f_row_eta);
    if (t->f_iter_lam) fclose(t->f_iter_lam);
    if (t->f_iter_eta) fclose(t->f_iter_eta);
    if (t->f_parity)   fclose(t->f_parity);
    if (t->f_meta)     fclose(t->f_meta);
    free(t);
}

// Hook 1: the decoder state the RTL must hold coming out of reset.
static inline void hbp_tr_init(HbpTrace* t, const double* lambda, int N) {
    if (!t->f_init) return;
    for (int i = 0; i < N; ++i) hbp_tr_put_f64(t->f_init, lambda[i]);
}

// Hook 2: the order rows are visited in this iteration. The schedule is
// reshuffled per iteration, so the RTL has to follow the very same order.
static inline void hbp_tr_order(HbpTrace* t, int loop,
                                const int* row_indices, int rows) {
    (void)loop;
    if (!t->f_order) return;
    for (int i = 0; i < rows; ++i) hbp_tr_put_u32(t->f_order, (uint32_t)row_indices[i]);
}

// Hook 3: the lambda and etar slice belonging to one row, immediately after
// that row has been processed. Both are written at the fixed stride row_stride
// with unused slots zero filled, mirroring how etar is laid out in memory, so a
// testbench can index them as a plain 2D array.
static inline void hbp_tr_row(HbpTrace* t, int loop, const double* lambda,
                              const int* hrow, const double* etar_row, int stride) {
    (void)loop;
    if (!t->f_row_lam || !t->f_row_eta) return;
    for (int c = 0; c < stride; ++c) {
        int idx = hrow[c];
        hbp_tr_put_f64(t->f_row_lam, idx == 0 ? 0.0 : lambda[idx - 1]);
        hbp_tr_put_f64(t->f_row_eta, idx == 0 ? 0.0 : etar_row[c]);
    }
}

// Hook 4: the whole decoder state at the iteration boundary.
static inline void hbp_tr_iter(HbpTrace* t, int loop, const double* lambda, int N,
                               const double* etar, int etar_len) {
    (void)loop;
    if (t->f_iter_lam) for (int i = 0; i < N; ++i)        hbp_tr_put_f64(t->f_iter_lam, lambda[i]);
    if (t->f_iter_eta) for (int i = 0; i < etar_len; ++i) hbp_tr_put_f64(t->f_iter_eta, etar[i]);
}

// Hook 5: how many checks still fail and whether this iteration stops the loop.
static inline void hbp_tr_parity(HbpTrace* t, int loop, int parity_t, int stop) {
    if (!t->f_parity) return;
    fprintf(t->f_parity, "loop=%d parity_violations=%d stop=%d\n", loop, parity_t, stop);
}

// Written once the run is over so the offsets describe what actually got traced.
static inline void hbp_tr_finish(HbpTrace* t, int iters_traced, int iters_run) {
    if (!t->f_meta) return;
    fprintf(t->f_meta,
            "shot=%d\n"
            "flags=0x%02x\n"
            "nvars=%d\n"
            "rows=%d\n"
            "row_stride=%d\n"
            "iters_run=%d\n"
            "iters_traced=%d\n"
            "word=ieee754_double_hex\n"
            "h1_init_lambda=%d words\n"
            "h2_row_order=%d words per iteration\n"
            "h3_row_lambda=%d words per iteration (rows * row_stride)\n"
            "h3_row_etar=%d words per iteration (rows * row_stride)\n"
            "h4_iter_lambda=%d words per iteration\n"
            "h4_iter_etar=%d words per iteration (rows * row_stride)\n",
            t->shot, t->flags, t->nvars, t->rows, t->row_stride,
            iters_run, iters_traced,
            t->nvars, t->rows,
            t->rows * t->row_stride, t->rows * t->row_stride,
            t->nvars, t->rows * t->row_stride);
}

#endif  // HBP_TRACE_H
