/*
 * Standalone profiling program for e8d4_fused_kernel_t.
 * Measures clock64() at phase boundaries inside the kernel.
 *
 * Phases:
 *   A: codebook load  (global/L2 → SMEM)
 *   B: main N-loop    (decode + accumulate)
 *   C: warp reduction (register → SMEM)
 *   D: correction     (CSR sparse + scale + output)
 *
 * Compile:
 *   nvcc -O3 -arch=sm_75 -o profile_e8d4 profile_e8d4.cu
 * Run:
 *   ./profile_e8d4
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_N  256
#define N_WARPS  8   // BLOCK_N / 32

// Per-block timing: 5 timestamps (4 gaps = 4 phases)
struct BlockTiming {
    long long t0;  // kernel entry
    long long t1;  // after codebook load + __syncthreads
    long long t2;  // after main N-loop
    long long t3;  // after warp reduction → SMEM
    long long t4;  // after correction + output
};

__global__ void e8d4_profile_kernel(
    const int16_t* __restrict__ Q,
    const __half*  __restrict__ codes,
    const __half*  __restrict__ g,
    const __half*  __restrict__ h,
    const __half*  __restrict__ x,
    const __half*  __restrict__ O_vals,
    const int*     __restrict__ O_col,
    const int*     __restrict__ O_row_ptr,
    __half*        __restrict__ y,
    int M, int N,
    BlockTiming*   timings
) {
    __shared__ __half codes_s[512 * 4];
    __shared__ float  red[8][N_WARPS];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m    = blockIdx.x;

    long long t0, t1, t2, t3, t4;

    // ── Phase A: codebook load ──────────────────────────────────────────────
    if (tid == 0) t0 = clock64();

    #pragma unroll 8
    for (int i = tid; i < 512 * 4; i += BLOCK_N)
        codes_s[i] = codes[i];
    __syncthreads();

    if (tid == 0) t1 = clock64();

    // ── Phase B: main N-loop ────────────────────────────────────────────────
    float acc[8] = {};

    for (int n = tid; n < N; n += BLOCK_N) {
        const float hx = __half2float(__ldg(h + n)) * __half2float(__ldg(x + n));

        const int16_t q   = Q[(size_t)m * N + n];
        const int     idx = ((int)q + 32768) & 0xFFFF;
        if (idx == 65535) continue;

        const int ct     = (idx >> 14) & 3;
        const int idi    = (idx >> 7)  & 0x7F;
        const int idj    =  idx        & 0x7F;
        const int ci_off = (ct * 128 + idi) * 4;
        const int cj_off = (ct * 128 + idj) * 4;

        for (int k = 0; k < 4; k++) {
            acc[k]   += __half2float(codes_s[ci_off + k]) * hx;
            acc[k+4] += __half2float(codes_s[cj_off + k]) * hx;
        }
    }

    if (tid == 0) t2 = clock64();

    // ── Phase C: warp reduction ─────────────────────────────────────────────
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        for (int k = 0; k < 8; k++)
            acc[k] += __shfl_down_sync(0xFFFFFFFF, acc[k], offset);

    if (lane == 0)
        for (int k = 0; k < 8; k++)
            red[k][warp] = acc[k];
    __syncthreads();

    if (tid == 0) t3 = clock64();

    // ── Phase D: correction + scale + output ────────────────────────────────
    if (tid < 8) {
        const int k = tid;

        float val = 0.f;
        for (int w = 0; w < N_WARPS; w++) val += red[k][w];

        const int start = O_row_ptr[m];
        const int end   = O_row_ptr[m + 1];
        for (int i = start; i < end; i++) {
            const int   col = O_col[i];
            const float hx  = __half2float(__ldg(h + col))
                             * __half2float(__ldg(x + col));
            val += __half2float(__ldg(O_vals + i * 8 + k)) * hx;
        }

        val *= __half2float(__ldg(g + m * 8 + k));
        y[m * 8 + k] = __float2half(val);
    }

    if (tid == 0) t4 = clock64();

    // Store timestamps (thread 0 only)
    if (tid == 0) {
        timings[m].t0 = t0;
        timings[m].t1 = t1;
        timings[m].t2 = t2;
        timings[m].t3 = t3;
        timings[m].t4 = t4;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static void fill_random_fp16(__half* p, int n) {
    for (int i = 0; i < n; i++)
        p[i] = __float2half((float)rand() / RAND_MAX * 2.f - 1.f);
}

static void fill_random_int16(int16_t* p, int n) {
    for (int i = 0; i < n; i++) {
        // ~1% critical weights (65535-mapped), rest random codebook index
        if (rand() % 100 == 0) {
            p[i] = (int16_t)32767;  // 32767 + 32768 = 65535 → critical
        } else {
            // valid index 0..65534, mapped via (q+32768)&0xFFFF
            int idx = rand() % 65534;  // 0..65533
            p[i] = (int16_t)(idx - 32768);
        }
    }
}

// ── Benchmark one (M, N) shape ───────────────────────────────────────────────

static void bench(int M, int N, const char* label) {
    const int M8 = M * 8;

    // Allocate host buffers
    int16_t* h_Q      = (int16_t*)malloc((size_t)M * N * sizeof(int16_t));
    __half*  h_codes  = (__half*)malloc(512 * 4 * sizeof(__half));
    __half*  h_g      = (__half*)malloc(M8 * sizeof(__half));
    __half*  h_h      = (__half*)malloc(N  * sizeof(__half));
    __half*  h_x      = (__half*)malloc(N  * sizeof(__half));

    fill_random_int16(h_Q, M * N);
    fill_random_fp16(h_codes, 512 * 4);
    fill_random_fp16(h_g, M8);
    fill_random_fp16(h_h, N);
    fill_random_fp16(h_x, N);

    // Build a simple CSR for ~1% critical weights
    int* h_row_ptr = (int*)calloc(M + 1, sizeof(int));
    // count critical weights per row
    for (int m = 0; m < M; m++) {
        int cnt = 0;
        for (int n = 0; n < N; n++) {
            int idx = ((int)h_Q[(size_t)m * N + n] + 32768) & 0xFFFF;
            if (idx == 65535) cnt++;
        }
        h_row_ptr[m + 1] = h_row_ptr[m] + cnt;
    }
    int n_crit = h_row_ptr[M];
    int*    h_col    = (int*)   malloc(n_crit * sizeof(int));
    __half* h_ovals  = (__half*)malloc(n_crit * 8 * sizeof(__half));
    {
        int ci = 0;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int idx = ((int)h_Q[(size_t)m * N + n] + 32768) & 0xFFFF;
                if (idx == 65535) {
                    h_col[ci] = n;
                    fill_random_fp16(h_ovals + ci * 8, 8);
                    ci++;
                }
            }
        }
    }

    // Allocate device buffers
    int16_t* d_Q; __half* d_codes; __half* d_g; __half* d_h; __half* d_x;
    int*     d_row_ptr; int* d_col; __half* d_ovals; __half* d_y;
    BlockTiming* d_timings;

    CUDA_CHECK(cudaMalloc(&d_Q,        (size_t)M * N * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_codes,    512 * 4 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_g,        M8 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_h,        N  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_x,        N  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col,      n_crit * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ovals,    n_crit * 8 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y,        M8 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_timings,  M * sizeof(BlockTiming)));

    CUDA_CHECK(cudaMemcpy(d_Q,       h_Q,       (size_t)M * N * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_codes,   h_codes,   512 * 4 * sizeof(__half),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g,       h_g,       M8 * sizeof(__half),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h,       h_h,       N  * sizeof(__half),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,       h_x,       N  * sizeof(__half),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (M + 1) * sizeof(int),           cudaMemcpyHostToDevice));
    if (n_crit > 0) {
        CUDA_CHECK(cudaMemcpy(d_col,   h_col,   n_crit * sizeof(int),            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ovals, h_ovals, n_crit * 8 * sizeof(__half),     cudaMemcpyHostToDevice));
    }

    // Warmup (3 runs)
    dim3 grid(M, 1), block(BLOCK_N);
    for (int i = 0; i < 3; i++) {
        e8d4_profile_kernel<<<grid, block>>>(
            d_Q, d_codes, d_g, d_h, d_x,
            d_ovals, d_col, d_row_ptr, d_y,
            M, N, d_timings);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Profile run
    e8d4_profile_kernel<<<grid, block>>>(
        d_Q, d_codes, d_g, d_h, d_x,
        d_ovals, d_col, d_row_ptr, d_y,
        M, N, d_timings);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch timing results
    BlockTiming* h_timings = (BlockTiming*)malloc(M * sizeof(BlockTiming));
    CUDA_CHECK(cudaMemcpy(h_timings, d_timings, M * sizeof(BlockTiming), cudaMemcpyDeviceToHost));

    // Get SM clock frequency (cycles → μs)
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    double clk_ghz = prop.clockRate / 1e6;  // kHz → GHz

    // Aggregate: use median block to avoid outliers
    // For simplicity, compute averages over all blocks
    double sum_A = 0, sum_B = 0, sum_C = 0, sum_D = 0, sum_total = 0;
    for (int m = 0; m < M; m++) {
        BlockTiming* t = &h_timings[m];
        sum_A     += (double)(t->t1 - t->t0);
        sum_B     += (double)(t->t2 - t->t1);
        sum_C     += (double)(t->t3 - t->t2);
        sum_D     += (double)(t->t4 - t->t3);
        sum_total += (double)(t->t4 - t->t0);
    }
    sum_A /= M; sum_B /= M; sum_C /= M; sum_D /= M; sum_total /= M;

    // Convert cycles → μs
    double us_A = sum_A / (clk_ghz * 1000.0);
    double us_B = sum_B / (clk_ghz * 1000.0);
    double us_C = sum_C / (clk_ghz * 1000.0);
    double us_D = sum_D / (clk_ghz * 1000.0);
    double us_T = sum_total / (clk_ghz * 1000.0);

    printf("%-22s  M=%4d N=%5d  n_crit=%5d\n", label, M, N, n_crit);
    printf("  A. codebook load  : %7.2f μs  (%5.1f%%)\n", us_A, 100.*us_A/us_T);
    printf("  B. main N-loop    : %7.2f μs  (%5.1f%%)\n", us_B, 100.*us_B/us_T);
    printf("  C. warp reduction : %7.2f μs  (%5.1f%%)\n", us_C, 100.*us_C/us_T);
    printf("  D. correction+out : %7.2f μs  (%5.1f%%)\n", us_D, 100.*us_D/us_T);
    printf("  Total (avg/block) : %7.2f μs\n\n", us_T);

    // Cleanup
    free(h_Q); free(h_codes); free(h_g); free(h_h); free(h_x);
    free(h_row_ptr); free(h_col); free(h_ovals); free(h_timings);
    cudaFree(d_Q); cudaFree(d_codes); cudaFree(d_g); cudaFree(d_h); cudaFree(d_x);
    cudaFree(d_row_ptr); cudaFree(d_col); cudaFree(d_ovals); cudaFree(d_y);
    cudaFree(d_timings);
}

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s  (SM clock: %.0f MHz,  SMs: %d)\n\n",
           prop.name, prop.clockRate / 1e3, prop.multiProcessorCount);

    srand(42);

    bench(512,  4096,  "q/o/down 4k→4k");
    bench(128,  4096,  "k/v_proj 4k→1k");
    bench(1792, 4096,  "gate/up  4k→14k");
    bench(512,  14336, "down     14k→4k");

    return 0;
}
