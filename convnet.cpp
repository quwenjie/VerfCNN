
#include "convnet.h"
#include "hyrax.hpp"
#include "logup.hpp"
#include "tools.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include<iostream>
#include<vector>
#include <mcl/bn.hpp>
using namespace mcl::bn;
using namespace std;
const int C0=3;
#define FXP_VALUE 6
#define get_output_dim(input_dim,kernel_size,stride,pad) (input_dim +2*pad-kernel_size)/ stride+1

FILE* file=NULL;

void mat_mult(const int *mat_l, const int *mat_r, int *result, const unsigned int K, const unsigned int M)
{       
    unsigned int n, k, m;
    unsigned int row, col;
    int accumulator;

    for (m = 0; m < M; m++)
    {
        
            accumulator = 0;
            for (k = 0; k < K; k++)
            {
                accumulator += mat_l[k] * mat_r[k*M + m];
            }
            result[m] = accumulator;
        
    }
}
void conv2d(const int *x, const int *w, ll *y, int N, int C_in, int C_out, int H, int W, int H_new, int W_new,
    int k_size_h, int k_size_w, int stride_h, int stride_w, int pad_h, int pad_w)
{

    static int Y=0;
    ++Y;
    int n_i, c_out_j, c_in_i; /* sample and channels */
    int n, m; /* kernel iterations */
    int i, j; /* output image iteration */
    int *y_new=new int[H_new*W_new];
    memset(y_new,0,sizeof(int)*H_new*W_new);
    for (c_out_j = 0; c_out_j < C_out; c_out_j++)
    {
        for (i = 0; i < H_new; i++)
        {
            for (j = 0; j < W_new; j++)
            {
                int output_idx_y = i * W_new + j;
                ll S = 0;
                for (c_in_i = 0; c_in_i < C_in; c_in_i++)
                {
                    int in_sum=0;
                    for (n = 0; n < k_size_h; n++)
                    {
                        for (m = 0; m < k_size_w; m++)
                        {
                            int x_i = i * stride_h + n - pad_h;
                            int x_j = j * stride_w + m - pad_w;
                            int x_value = 0; // Default to 0 for padding
                            if (x_i >= 0 && x_i < H && x_j >= 0 && x_j < W)
                            {
                                x_value = x[c_in_i * H * W+ x_i * W + x_j];
                            }
                            //x[c_in_i][i+n-1][j+m-1]
                            //w[c_out_j][c_in_i][n][m]
                            int w_value = w[c_out_j * C_in * k_size_h * k_size_w + c_in_i * k_size_h * k_size_w + n * k_size_w + m];
                            S+= x_value * w_value;
                            if(c_in_i==0 && c_out_j==0)
                                y_new[i*W_new+j]+=x_value * w_value;
                            //y[c_out_j][i][j]
                        }
                    }
                    //assert(S<(1<<30));
                    
                }

                y[c_out_j * H_new * W_new + i * W_new + j] = S;
            }
        }
    }

}    
G1* g_shared;
G1 G_shared;
G1* conv_output_commit,*relu_input_commit,*relu_output_commit,*relu_f_commit,*relu_t_commit;
double total_conv_time_sec = 0.0;
double total_relu_time_sec = 0.0;
double total_pooling_time_sec = 0.0;

// Specialized open for sparse W: computes LT only on non-zero entries.
static void open_w_sparse(ll* w, Fr* r, Fr eval, G1& G, G1* g, G1* comm, int l, const vector<size_t>& nz_lin_idx)
{
    int halfl = l / 2;
    int rownum = (1 << halfl), colnum = (1 << (l - halfl));
    Fr* L = get_eq(r, halfl);
    Fr* R = get_eq(r + halfl, l - halfl);
    vector<Fr> LT(colnum);
    for (int i = 0; i < colnum; ++i) 
        LT[i] = 0;

    for (size_t p = 0; p < nz_lin_idx.size(); ++p)
    {
        const size_t lin = nz_lin_idx[p];
        LT[lin / static_cast<size_t>(rownum)] += L[lin % static_cast<size_t>(rownum)] * Fr(w[lin]);
    }

    G1 tprime = perdersen_commit_classic(comm, L, rownum);
    prove_dot_product(tprime, G * eval, R, LT.data(), eval, g, G, colnum);
    delete[] L;
    delete[] R;
}

void conv2d_prover(const int *x, const int *ww, const ll *output,
    int C_in, int C_out, int H, int W, int H_new, int W_new,
    int k_size_h, int k_size_w, int stride_h, int stride_w, int pad_h, int pad_w)
{
    static int prover_call_id = 0;
    ++prover_call_id;
    // This checker follows the same assumptions as testconv.py:
    // square input, square kernel, stride=1, and "same" output shape.
    assert(H == W);
    assert(k_size_h == k_size_w);
    assert(stride_h == 1 && stride_w == 1);
    assert(H_new == H && W_new == W);

    const int padded_h = H + 2 * pad_h;
    const int padded_w = W + 2 * pad_w;
    const int len_x = padded_h * padded_w;
    const int len_w = padded_w * k_size_h;

    
    const size_t PADD_C = next_pow2(static_cast<size_t>(C_in));
    const size_t PADD_D = next_pow2(static_cast<size_t>(C_out));
    const size_t PADD_X = next_pow2(static_cast<size_t>(len_x));
    const size_t PADD_W = next_pow2(static_cast<size_t>(len_w));
    const size_t PADD_Y = next_pow2(static_cast<size_t>(len_x + len_w));

    vector<long long> X(PADD_C * PADD_X, 0);
    vector<long long> W1(PADD_D * PADD_C * PADD_W, 0);
    vector<long long> Y(PADD_D * PADD_Y, 0);

    // Build per-channel reversed padded X.
    for (int ci = 0; ci < C_in; ++ci)
    {
        for (int pi = 0; pi < padded_h; ++pi)
        {
            for (int pj = 0; pj < padded_w; ++pj)
            {
                const size_t flat = static_cast<size_t>(ci) * PADD_X
                    + static_cast<size_t>(pi * padded_w + pj);
                if (pi < pad_h || pj < pad_w || pi >= padded_h - pad_h || pj >= padded_w - pad_w)
                {
                    X[flat] = 0;
                }
                else
                {
                    const int src_i = H - 1 - (pi - pad_h);
                    const int src_j = W - 1 - (pj - pad_w);
                    const size_t xidx = static_cast<size_t>(ci) * static_cast<size_t>(H * W)
                        + static_cast<size_t>(src_i * W + src_j);
                    X[flat] = x[xidx];
                }
            }
        }
    }

    // Build W for 1D convolution by spacing each kernel row with padded_w.
    for (int co = 0; co < C_out; ++co)
    {
        for (int ci = 0; ci < C_in; ++ci)
        {
            for (int i = 0; i < k_size_h; ++i)
            {
                for (int j = 0; j < k_size_w; ++j)
                {
                    const size_t widx = static_cast<size_t>(co) * static_cast<size_t>(C_in * k_size_h * k_size_w)
                        + static_cast<size_t>(ci) * static_cast<size_t>(k_size_h * k_size_w)
                        + static_cast<size_t>(i * k_size_w + j);
                    const size_t Widx = (static_cast<size_t>(co) * PADD_C + static_cast<size_t>(ci))
                        * PADD_W
                        + static_cast<size_t>(i * padded_w + j);
                    W1[Widx] = ww[widx];
                }
            }
        }
    }

    // Mark active j positions in W1 (skip padded-zero columns in later field ops).
    vector<size_t> active_w_j;
    active_w_j.reserve(PADD_W);
    for (size_t j = 0; j < PADD_W; ++j)
    {
        bool nonzero = false;
        for (size_t d = 0; d < static_cast<size_t>(C_out) && !nonzero; ++d)
        {
            for (size_t c = 0; c < PADD_C; ++c)
            {
                if (W1[(d * PADD_C + c) * PADD_W + j] != 0)
                {
                    nonzero = true;
                    break;
                }
            }
        }
        if (nonzero) active_w_j.push_back(j);
    }
    vector<size_t> w_sparse_idx;
    w_sparse_idx.reserve(active_w_j.size() * static_cast<size_t>(C_out) * static_cast<size_t>(PADD_C));
    for (size_t lin = 0; lin < W1.size(); ++lin)
    {
        if (W1[lin] != 0) 
            w_sparse_idx.push_back(lin);
    }
    // Multi-channel 1D convolution: C*n^2 with C*D*(n*m) => D*(n^2+n*m-1).
    // All axes use power-of-two padding.
    for (size_t co = 0; co < PADD_D; ++co)
    {
        for (size_t ci = 0; ci < PADD_C; ++ci)
        {
            for (int i = 0; i < len_x; ++i)
            {
                for (int j = 0; j < len_w; ++j)
                {
                    Y[co * PADD_Y +i + j] +=X[ci * PADD_X + i] *W1[(co * PADD_C + ci) * PADD_W + j];
                }
            }
        }
    }
    // Fr-domain consistency:
    // For each d in [0, C_out), check
    //   sum_k Y[d][k] * alpha^k
    // = sum_c (sum_i X[c][i] * alpha^i) * (sum_j W1[d][c][j] * alpha^j)
    

    vector<Fr> X_fr(X.size()), W1_fr(W1.size()), Y_fr(Y.size());
    for (size_t i = 0; i < X.size(); ++i) 
        X_fr[i]=X[i];
    for (size_t i = 0; i < W1.size(); ++i) 
        W1_fr[i]=W1[i];
    for (size_t i = 0; i < Y.size(); ++i) 
        Y_fr[i]=Y[i];

    const size_t y_domain = static_cast<size_t>(C_out) * PADD_Y;
    vector<unsigned char> used_y_idx(y_domain, 0);
    vector<int> mapped_y_idx;
    mapped_y_idx.reserve(static_cast<size_t>(C_out) * static_cast<size_t>(H_new * W_new));
    for (int co = 0; co < C_out; ++co)
    {
        for (int xi = 0; xi < H_new; ++xi)
        {
            for (int yi = 0; yi < W_new; ++yi)
            {
                    const size_t out_idx = static_cast<size_t>(co) * static_cast<size_t>(H_new * W_new)
                        + static_cast<size_t>(xi * W_new + yi);
                    const size_t y_pos = static_cast<size_t>(padded_h - 1 - xi) * static_cast<size_t>(padded_w)
                        + static_cast<size_t>(padded_w - 1 - yi);
                    assert(y_pos < PADD_Y);
                    assert(static_cast<long long>(output[out_idx]) == Y[static_cast<size_t>(co) * PADD_Y + y_pos]);
                    const size_t y_lin = static_cast<size_t>(co) * PADD_Y + y_pos;
                    assert(y_lin < y_domain);
                    assert(used_y_idx[y_lin] == 0);
                    used_y_idx[y_lin] = 1;
                    mapped_y_idx.push_back(static_cast<int>(y_lin));
            }
        }
    }
    vector<int> idx_Y;
    idx_Y.reserve(y_domain);
    for (size_t y_lin = 0; y_lin < y_domain; ++y_lin)
        idx_Y.push_back(static_cast<int>(y_lin));
    // K part: remaining Y indices not mapped to OUTPUT.
    vector<ll> K;
    
    K.reserve(y_domain - mapped_y_idx.size());
    vector<int> idx_K;
    
    idx_K.reserve(y_domain - mapped_y_idx.size());
    for (size_t y_lin = 0; y_lin < y_domain; ++y_lin)
    {
        if (used_y_idx[y_lin])
            continue;
        K.push_back(Y[y_lin]);
        idx_K.push_back(static_cast<int>(y_lin));
    }

    // Pad K / idx_K length to power-of-two: K pad with 0, idx_K pad with 1.
    const size_t K_orig_size = K.size();
    const size_t K_padded_size = next_pow2(K_orig_size);
    if (K_padded_size > K_orig_size)
    {
        K.resize(K_padded_size, 0);
        idx_K.resize(K_padded_size, 1);
    }
    const int l_i = log2_(PADD_X);
    const int l_k = log2_(PADD_Y);
    const int l_c = log2_(PADD_C);
    const int l_d = log2_(PADD_D);
    const int l_j = log2_(PADD_W);
    const int l_X = l_i + l_c;
    const int l_W = l_j + l_c + l_d;
    const int l_Y = l_k + l_d;
    const int l_out = log2_(mapped_y_idx.size());
    const int l_kvals = log2_(K.size());
    //weight is committed in preprocessing
    G1* comm_W = prover_commit(reinterpret_cast<ll*>(W1.data()), g_shared, l_W, 1);   

    auto conv_t0 = std::chrono::high_resolution_clock::now();
    G1* comm_X = prover_commit(X.data(), g_shared, l_X, 1);
    G1* comm_Y = prover_commit(Y.data(), g_shared, l_Y, 1);
    conv_output_commit = prover_commit(const_cast<ll*>(output), g_shared, l_out, 1);
    G1* comm_kvals = prover_commit(K.data(), g_shared, l_kvals, 1);
    Fr alpha;
    alpha.setByCSPRNG();
    const size_t max_deg = max(PADD_Y, max(PADD_X, PADD_W));
    vector<Fr> alpha_pow(max_deg);
    alpha_pow[0]=1;
    for (size_t i = 1; i < max_deg; ++i)
        alpha_pow[i] = alpha_pow[i - 1] * alpha;

    const int lg_c_out = log2_(static_cast<size_t>(C_out));
    vector<Fr> r(lg_c_out);
    for (int i = 0; i < lg_c_out; ++i) 
        r[i].setByCSPRNG();
    vector<Fr> eq_D = build_tilde_eq_vector(static_cast<size_t>(C_out), r);


    Fr * L=new Fr[C_out];
    Fr S1=0;
    for (size_t d = 0; d < static_cast<size_t>(C_out); ++d)
    {
        L[d]=0;
        for (size_t k = 0; k < PADD_Y; ++k)
        {
            L[d] += Y_fr[d * PADD_Y + k] * alpha_pow[k];
        }
        S1+=L[d]*eq_D[d];
    }

    // Build YP[k] = sum_d Y[d][k] * eq_D[d], then prove
    // S1 = sum_k YP[k] * alpha^k.
    vector<Fr> YP(PADD_Y);
    for (size_t k = 0; k < PADD_Y; ++k)
    {
        Fr acc = 0;
        for (size_t d = 0; d < static_cast<size_t>(C_out); ++d)
        {
            acc += Y_fr[d * PADD_Y + k] * eq_D[d];
        }
        YP[k] = acc;
    }

    vector<Fr> alpha_pow_Y(alpha_pow.begin(), alpha_pow.begin() + PADD_Y);

    SumcheckFGResult sumcheck_res_YP = run_sumcheck_fg(S1, alpha_pow_Y, YP, "FG_YP");
    assert(sumcheck_res_YP.claim_F ==
           eval_alpha_table_claim_from_sumcheck_r(sumcheck_res_YP.sumcheck_r, alpha));

    // PCS Open for Y
    vector<Fr> r_d_lsb(static_cast<size_t>(lg_c_out));
    for (int t = 0; t < lg_c_out; ++t)
        r_d_lsb[static_cast<size_t>(t)] = r[static_cast<size_t>(lg_c_out - 1 - t)];
    vector<Fr> r_d_full(static_cast<size_t>(l_d));
    for (int t = 0; t < l_d; ++t)
        r_d_full[static_cast<size_t>(t)] = 0;
    for (int t = 0; t < min(l_d, lg_c_out); ++t)
        r_d_full[static_cast<size_t>(t)] = r_d_lsb[static_cast<size_t>(t)];

    vector<Fr> u_Y(static_cast<size_t>(l_k + l_d));
    int uy_pos = 0;
    for (int t = 0; t < l_k; ++t)
        u_Y[uy_pos++] = sumcheck_res_YP.sumcheck_r[static_cast<size_t>(t)];
    for (int t = 0; t < l_d; ++t)
        u_Y[uy_pos++] = r_d_full[static_cast<size_t>(t)];
    open(reinterpret_cast<ll*>(Y.data()), u_Y.data(), sumcheck_res_YP.claim_G, G_shared, g_shared, comm_Y, l_Y);

    vector<Fr> F(PADD_C);
    vector<Fr> G(PADD_C);

    for (size_t c = 0; c < PADD_C; ++c)
    {
        Fr eval_x = 0;
        for (size_t i = 0; i < PADD_X; ++i)
        {
            eval_x += X_fr[c * PADD_X + i] * alpha_pow[i];
        }
        F[c] = eval_x;
    }

    for (size_t c = 0; c < PADD_C; ++c)
    {
        Fr gc = 0;
        for (size_t d = 0; d < static_cast<size_t>(C_out); ++d)
        {
            Fr eval_w_dc = 0;
            for (size_t t = 0; t < active_w_j.size(); ++t)
            {
                const size_t j = active_w_j[t];
                eval_w_dc += W1_fr[(d * PADD_C + c) * PADD_W + j] * alpha_pow[j];
            }
            gc += eval_w_dc * eq_D[d];
        }
        G[c] = gc;
    }
    SumcheckFGResult sumcheck_res = run_sumcheck_fg(S1, F, G, "FG_main");
    vector<Fr> sumcheck_r_rev = sumcheck_res.sumcheck_r;
    reverse(sumcheck_r_rev.begin(), sumcheck_r_rev.end());
    vector<Fr> eq_C = build_tilde_eq_vector(static_cast<size_t>(PADD_C), sumcheck_r_rev);
    vector<Fr> XP(PADD_X);
    for (size_t i = 0; i < PADD_X; ++i)
    {
        Fr acc = 0;
        for (size_t c = 0; c < PADD_C; ++c)
        {
            acc += eq_C[c] * X_fr[c * PADD_X + i];
        }
        XP[i] = acc;
    }

    vector<Fr> WP(PADD_W);
    for (size_t j = 0; j < PADD_W; ++j)
    {
        WP[j] = 0;
    }
    for (size_t t = 0; t < active_w_j.size(); ++t)
    {
        const size_t j = active_w_j[t];
        Fr acc = 0;
        for (size_t d = 0; d < static_cast<size_t>(C_out); ++d)
        {
            for (size_t c = 0; c < PADD_C; ++c)
            {
                acc += eq_D[d] * eq_C[c] * W1_fr[(d * PADD_C + c) * PADD_W + j];
            }
        }
        WP[j] = acc;
    }
    //use twisted eval to prove that f_table[0]==\sum_i alpha_pow[i] * XP[i], and g_table[0]=\sum_j alpha_pow[j] * WP[j]
    vector<Fr> alpha_pow_X(alpha_pow.begin(), alpha_pow.begin() + PADD_X);
    vector<Fr> alpha_pow_W(alpha_pow.begin(), alpha_pow.begin() + PADD_W);
    SumcheckFGResult sumcheck_res_F = run_sumcheck_fg(sumcheck_res.claim_F, alpha_pow_X, XP, "FG_F");
    SumcheckFGResult sumcheck_res_G = run_sumcheck_fg(sumcheck_res.claim_G, alpha_pow_W, WP, "FG_G");

    vector<Fr> rF_for_i = sumcheck_res_F.sumcheck_r;
    vector<Fr> rG_for_j = sumcheck_res_G.sumcheck_r;

    
    //PCS Open for X
    vector<Fr> u_X(static_cast<size_t>(l_i + l_c));
    int ux_pos = 0;
    for (int t = 0; t < l_i; ++t) 
        u_X[ux_pos++] = rF_for_i[static_cast<size_t>(t)];
    for (int t = 0; t < l_c; ++t) 
        u_X[ux_pos++] = sumcheck_res.sumcheck_r[static_cast<size_t>(t)];
    open(reinterpret_cast<ll*>(X.data()), u_X.data(), sumcheck_res_F.claim_G, G_shared, g_shared, comm_X, l_X); 

    // PCS Open for W.
    vector<Fr> u_W(static_cast<size_t>(l_j + l_c + l_d));
    int uw_pos = 0;
    for (int t = 0; t < l_j; ++t) 
        u_W[uw_pos++] = rG_for_j[static_cast<size_t>(t)];
    for (int t = 0; t < l_c; ++t) 
        u_W[uw_pos++] = sumcheck_res.sumcheck_r[static_cast<size_t>(t)];
    for (int t = 0; t < l_d; ++t) 
        u_W[uw_pos++] = r_d_full[static_cast<size_t>(t)];
    open_w_sparse(reinterpret_cast<ll*>(W1.data()), u_W.data(), sumcheck_res_G.claim_G, G_shared, g_shared, comm_W, l_W, w_sparse_idx);

    assert(sumcheck_res_F.claim_F==eval_alpha_table_claim_from_sumcheck_r(sumcheck_res_F.sumcheck_r, alpha));

    assert(sumcheck_res_G.claim_F==eval_alpha_table_claim_from_sumcheck_r(sumcheck_res_G.sumcheck_r, alpha));
    

    // Permutation check with K = Y \ OUTPUT (by Y-index partition):
    // prod_{all Y idx}(beta*Y[idx]+idx)
    // = prod_{mapped OUTPUT}(beta*OUTPUT[out_idx]+mapped_y_idx) * prod_{idx in K}(beta*Y[idx]+idx).
    

    Fr beta;
    beta.setByCSPRNG();
    Fr output_prod = 1;
    for (size_t out_idx = 0; out_idx < mapped_y_idx.size(); ++out_idx)
        output_prod *= output[out_idx] * beta + mapped_y_idx[out_idx];
    

    Fr K_prod = 1;
    for (size_t t = 0; t < K.size(); ++t)
        K_prod *= K[t] * beta + idx_K[t];
    

    Fr Y_prod = 1;
    for (size_t y_lin = 0; y_lin < y_domain; ++y_lin)
        Y_prod *= Y[y_lin] * beta + y_lin;

    //next prove that output_prod is correct, K_prod, Y_prod, using GKR-based grand-product check
    assert(K_prod*output_prod == Y_prod);
    ProductReduceResult output_reduce = prove_product_beta_linear_gkr(output, mapped_y_idx.data(), mapped_y_idx.size(), beta, output_prod);
    ProductReduceResult y_reduce = prove_product_beta_linear_gkr(Y.data(), idx_Y.data(), idx_Y.size(), beta, Y_prod);
    ProductReduceResult k_reduce = prove_product_beta_linear_gkr(K.data(), idx_K.data(), K.size(), beta, K_prod);
    open(const_cast<ll*>(output), output_reduce.r_cur.data(), output_reduce.claim_a, G_shared, g_shared, conv_output_commit, l_out);
    
    open(Y.data(), y_reduce.r_cur.data(), y_reduce.claim_a, G_shared, g_shared, comm_Y, l_Y);
    
    open(K.data(), k_reduce.r_cur.data(), k_reduce.claim_a, G_shared, g_shared, comm_kvals, l_kvals);

    assert(output_reduce.claim_b == eval_public_int_vector_claim(mapped_y_idx, output_reduce.r_cur));
    assert(y_reduce.claim_b == eval_public_int_vector_claim(idx_Y, y_reduce.r_cur));
    assert(k_reduce.claim_b == eval_public_int_vector_claim(idx_K, k_reduce.r_cur));

    auto conv_t1 = std::chrono::high_resolution_clock::now();
    total_conv_time_sec += std::chrono::duration<double>(conv_t1 - conv_t0).count();
}

void pooling2d(int *x, int *y, int N, int C_out, int H, int W, int H_new, int W_new,
    int k_size_h, int k_size_w,  int stride_h, int stride_w)
{
    assert(k_size_h == 2 && k_size_w == 2);
    assert(stride_h == 2 && stride_w == 2);
    assert(H == H_new * 2 && W == W_new * 2);
    assert((H & (H - 1)) == 0 && (W & (W - 1)) == 0);
    assert((H_new & (H_new - 1)) == 0 && (W_new & (W_new - 1)) == 0);
    assert((C_out & (C_out - 1)) == 0);

    int c_out_j;
    int n, m;
    int i, j;
    const size_t x_len = static_cast<size_t>(C_out) * static_cast<size_t>(H) * static_cast<size_t>(W);
    const size_t y_len = static_cast<size_t>(C_out) * static_cast<size_t>(H_new) * static_cast<size_t>(W_new);
    vector<ll> Y(x_len, 0);

    for (c_out_j = 0; c_out_j < C_out; c_out_j++)
    {
        for (i = 0; i < H_new; i++)
        {
            for (j = 0; j < W_new; j++)
            {
                int mx = 0;
                for (n = 0; n < k_size_w; n++)
                {
                    for (m = 0; m < k_size_h; m++)
                        mx = max(mx, x[c_out_j * H * W + (i * 2 + n) * W + j * 2 + m]);
                }
                y[c_out_j * H_new * W_new + i * W_new + j] = mx;
                for (n = 0; n < k_size_w; n++)
                {
                    for (m = 0; m < k_size_h; m++)
                    {
                        Y[static_cast<size_t>(c_out_j) * static_cast<size_t>(H) * static_cast<size_t>(W)
                          + static_cast<size_t>(i * stride_h + n) * static_cast<size_t>(W)
                          + static_cast<size_t>(j * stride_w + m)] = mx;
                    }
                }
            }
        }
    }

    const int l_x = log2_(x_len);
    const int l_y = log2_(y_len);
    const int l_c = log2_(static_cast<size_t>(C_out));
    const int l_h_new = log2_(static_cast<size_t>(H_new));
    const int l_w_new = log2_(static_cast<size_t>(W_new));
    const int l_h = log2_(static_cast<size_t>(H));
    const int l_w = log2_(static_cast<size_t>(W));
    assert(l_h == l_h_new + 1);
    assert(l_w == l_w_new + 1);

    vector<ll> x_ll(x_len, 0), y_ll(y_len, 0), Y_minus_x_ll(x_len, 0);
    vector<Fr> Y_minus_x_fr(x_len);
    for (size_t idx = 0; idx < x_len; ++idx)
    {
        x_ll[idx] = x[idx];
        Y_minus_x_ll[idx] = Y[idx] - x_ll[idx];
        Y_minus_x_fr[idx] = Y_minus_x_ll[idx];
    }
    for (size_t idx = 0; idx < y_len; ++idx)
    {
        y_ll[idx] = y[idx];
    }
    auto pooling_t0 = std::chrono::high_resolution_clock::now();
    G1* x_commit = prover_commit(x_ll.data(), g_shared, l_x, 1);
    G1* y_commit = prover_commit(y_ll.data(), g_shared, l_y, 1);
    G1* Y_commit = prover_commit(Y.data(), g_shared, l_x, 1);

    const int x_group_num = 1 << (l_x / 2);
    G1* x_minus_Y_commit = new G1[x_group_num];
    for (int k = 0; k < x_group_num; ++k)
        x_minus_Y_commit[k] = Y_commit[k] - x_commit[k];

    static bool range_table_ready = false;
    static vector<Fr> range_t_fr;
    static G1* range_t_commit = nullptr;
    if (!range_table_ready)
    {
        vector<ll> range_t_ll(1024);
        range_t_fr.resize(1024);
        for (int v = 0; v < 1024; ++v)
        {
            range_t_ll[static_cast<size_t>(v)] = v;
            range_t_fr[static_cast<size_t>(v)] = v;
        }
        range_t_commit = prover_commit(range_t_ll.data(), g_shared, 10, 1);
        range_table_ready = true;
    }
    //check that Y-x >=0
    logup(Y_minus_x_fr.data(), range_t_fr.data(), x_minus_Y_commit, range_t_commit, static_cast<int>(x_len), 1024, 1, G_shared, g_shared);

    // check that Y and y is consistent
    vector<Fr> rc(static_cast<size_t>(l_c));
    vector<Fr> ri(static_cast<size_t>(l_h_new));
    vector<Fr> rj(static_cast<size_t>(l_w_new));
    Fr r1, r2;
    for (int t = 0; t < l_c; ++t) 
        rc[static_cast<size_t>(t)].setByCSPRNG();
    for (int t = 0; t < l_h_new; ++t) 
        ri[static_cast<size_t>(t)].setByCSPRNG();
    for (int t = 0; t < l_w_new; ++t)   
        rj[static_cast<size_t>(t)].setByCSPRNG();
    r1.setByCSPRNG();
    r2.setByCSPRNG();

    vector<Fr> r_y(static_cast<size_t>(l_y));
    int pos = 0;
    for (int t = 0; t < l_w_new; ++t) r_y[static_cast<size_t>(pos++)] = rj[static_cast<size_t>(t)];
    for (int t = 0; t < l_h_new; ++t) r_y[static_cast<size_t>(pos++)] = ri[static_cast<size_t>(t)];
    for (int t = 0; t < l_c; ++t) r_y[static_cast<size_t>(pos++)] = rc[static_cast<size_t>(t)];
    assert(pos == l_y);

    vector<Fr> r_Y(static_cast<size_t>(l_x));
    pos = 0;
    r_Y[static_cast<size_t>(pos++)] = r2;
    for (int t = 0; t < l_w_new; ++t) 
        r_Y[static_cast<size_t>(pos++)] = rj[static_cast<size_t>(t)];
    r_Y[static_cast<size_t>(pos++)] = r1;
    for (int t = 0; t < l_h_new; ++t) 
        r_Y[static_cast<size_t>(pos++)] = ri[static_cast<size_t>(t)];
    for (int t = 0; t < l_c; ++t)   
        r_Y[static_cast<size_t>(pos++)] = rc[static_cast<size_t>(t)];
    assert(pos == l_x);

    Fr y_claim = prover_evaluate(y_ll.data(), r_y.data(), l_y);
    Fr Y_claim = prover_evaluate(Y.data(), r_Y.data(), l_x);
    open(y_ll.data(), r_y.data(), y_claim, G_shared, g_shared, y_commit, l_y);
    open(Y.data(), r_Y.data(), Y_claim, G_shared, g_shared, Y_commit, l_x);
    assert(Y_claim == y_claim);

    delete[] x_commit;
    delete[] y_commit;
    delete[] Y_commit;
    delete[] x_minus_Y_commit;

    auto pooling_t1 = std::chrono::high_resolution_clock::now();
    const double pooling_elapsed = std::chrono::duration<double>(pooling_t1 - pooling_t0).count();
    total_pooling_time_sec += pooling_elapsed;
}

void linear_layer(const int *x, const int *w, int *output, const int x_scale_factor,const int x_scale_factor_inv,
                  const int w_scale_factor_inv, 
                  const unsigned int  N, const unsigned int  K, const unsigned int  M,
                  const unsigned int  hidden_layer)
{
    fprintf(file,"matrix mult weight %d %d, output:%d  \n",M,K,M);
    int* ww=new int[K*M];
    fprintf(file,"matrix weight:");
    for(int i=0;i<K*M;i++)
    {
        ww[i]=w[i]*w_scale_factor_inv;   //dump this I think
        fprintf(file,"%d ",ww[i]);
    }
    fprintf(file,"\n");
    fprintf(file,"matrix output:");
    mat_mult(x, ww, output, K, M);
    for(int i=0;i<M;i++)
    {
        fprintf(file,"%d ",output[i]);
    }
    fprintf(file,"\n");
    fclose(file);
}


static vector<int> relu_input;
static vector<int> relu_output;

void conv2d_layer(const int *x, const int *w,int *output, const int x_scale_factor, const int x_scale_factor_inv, const int w_scale_factor_inv, 
                  const unsigned int N, const unsigned int C_in, const unsigned int C_out, const int H, const int W,
                  const int H_conv, const int W_conv, const int k_size_h, const int k_size_w,  const int stride_h, const int stride_w,int pad_h,int pad_w)
{
    
    
    static int SUM1=0;
    static int SUM2=0;
    SUM1+=C_in*H*W;
    SUM2+=C_out*C_in*k_size_h*k_size_w;
    int* ww=new int[C_in*C_out*k_size_h*k_size_w];
    for(int i=0;i<C_in*C_out*k_size_h*k_size_w;i++)
        ww[i]=w[i]*w_scale_factor_inv;   //dump this I think

    
    
    file=fopen("/tmp/dat.txt","w");        
    if (file == NULL)
    {
        file = stderr;        
    }
    

    fprintf(file,"conv input %d %d %d\n",C_in,H,W);
    for(int i=0;i<C_in*H*W;i++)
    {
        fprintf(file,"%d ",x[i]);
    }
    fprintf(file,"\n");
    
    fprintf(file,"conv weight %d %d %d %d\n",C_in,C_out,k_size_h,k_size_w);
    for(int i=0;i<C_in*C_out*k_size_h*k_size_w;i++)
    {
        fprintf(file,"%d ",ww[i]);
    }
    fprintf(file,"\n");
    const size_t conv_out_len = static_cast<size_t>(C_out) * static_cast<size_t>(H_conv) * static_cast<size_t>(W_conv);
    ll* output_ll = new ll[conv_out_len];
    conv2d(x, ww, output_ll, N, C_in, C_out, H, W, H_conv, W_conv,
            k_size_h, k_size_w,  stride_h, stride_w,pad_h,pad_w);
    conv2d_prover(x, ww, output_ll, C_in, C_out, H, W, H_conv, W_conv,
            k_size_h, k_size_w, stride_h, stride_w, pad_h, pad_w);
    for (size_t idx = 0; idx < conv_out_len; ++idx)
    {
        assert(output_ll[idx] < (1LL << 30) && output_ll[idx] > -(1LL << 30));
        output[idx] = static_cast<int>(output_ll[idx]);
    }
    delete[] output_ll;

    static int mx=-1e9,mn=1e9;
    int oldmx=mx,oldmn=mn;
    
    fprintf(file,"conv output %d %d %d\n",C_out,H_conv,W_conv);

    for (int c = 0; c < C_out; c++)
        for (int k =0; k < H_conv*W_conv; k++)
        {
            assert(output[c* H_conv*W_conv + k]<(1<<30));
            fprintf(file,"%d ",output[c* H_conv*W_conv + k]);  // I think no need to consider batching
        }
    fprintf(file,"\n");
    fprintf(file,"relu input(y1) %d %d %d\n",C_out,H_conv,W_conv);

        for (int c = 0; c < C_out; c++)
            for (int k =0; k < H_conv*W_conv; k++)
            {
                fprintf(file,"%d ",output[c* H_conv*W_conv + k]);  // I think no need to consider batching
            }
    fprintf(file,"\n relu output(y2)");
    for (int c = 0; c < C_out; c++)
    for (int k =0; k < H_conv*W_conv; k++)
    {
        relu_input.push_back(output[c* H_conv*W_conv + k]);
        relu_output.push_back(max(output[c* H_conv*W_conv + k]/(1<<FXP_VALUE), 0));
        output[c* H_conv*W_conv + k] =relu_output.back();
        fprintf(file,"%d ",output[c* H_conv*W_conv + k]); 
    }
    fprintf(file,"\n");
    
}


void argmax_over_cols(const int *mat_in, unsigned int *indices, const unsigned int N, const unsigned int M)
{

    // calculate max of each row
    unsigned int n, m, max_idx;
    int row_max, value;
    for (n = 0; n < N; n++)
    {
        row_max = mat_in[n*M];
        max_idx = 0;
        for (m = 0; m < M; m++)
        {
            value = mat_in[n*M + m];
            if (value > row_max)
            {
                row_max = value;
                max_idx = m; // return column
            }
        }
        indices[n] = max_idx;
    }
}


void run_convnet(const int *x, unsigned int *class_indices)
{
    static bool mcl_inited = false;
    if (!mcl_inited)
    {
        initPairing(mcl::BN254);
        mcl_inited = true;
        g_shared=new G1[1<<(MAXL/2)];
        G_shared= gen_gi(g_shared, 1<<(MAXL/2));
    }
    

    const int LAYER_NUM=19;
    string model[100]={"C64","C64","M","C128","C128","M","C256","C256","C256","M","C512","C512","C512","M","C512","C512","C512","M","L10"};
    const int* layer_w[100]={layer_1_weight,layer_2_weight,layer_3_weight,layer_4_weight,layer_5_weight,layer_6_weight,layer_7_weight,layer_8_weight,layer_9_weight,layer_10_weight,layer_11_weight,layer_12_weight,layer_13_weight,layer_14_weight};
    const int layer_sx[100]={layer_1_s_x,layer_2_s_x,layer_3_s_x,layer_4_s_x,layer_5_s_x,layer_6_s_x,layer_7_s_x,layer_8_s_x,layer_9_s_x,layer_10_s_x,layer_11_s_x,layer_12_s_x,layer_13_s_x,layer_14_s_x};
    const int layer_sx_inv[100]={layer_1_s_x_inv,layer_2_s_x_inv,layer_3_s_x_inv,layer_4_s_x_inv,layer_5_s_x_inv,layer_6_s_x_inv,layer_7_s_x_inv,layer_8_s_x_inv,layer_9_s_x_inv,layer_10_s_x_inv,layer_11_s_x_inv,layer_12_s_x_inv,layer_13_s_x_inv,layer_14_s_x_inv};
    const int layer_sw_inv[100]={layer_1_s_w_inv,layer_2_s_w_inv,layer_3_s_w_inv,layer_4_s_w_inv,layer_5_s_w_inv,layer_6_s_w_inv,layer_7_s_w_inv,layer_8_s_w_inv,layer_9_s_w_inv,layer_10_s_w_inv,layer_11_s_w_inv,layer_12_s_w_inv,layer_13_s_w_inv,layer_14_s_w_inv};
    int* inbuf[100];
    int* outbuf[100];
    int now_in_c=C0;
    
    int now_in=32;
    int *in=(int*)x;
    int pc=0;
    for(int i=0;i<LAYER_NUM;i++)
    {
        
        int now_out;
        int now_out_c;
        int* out;
        //cout<<i<<" start"<<" "<<model[i]<<" "<<now_in<<" "<<now_in_c<<endl;
        if (model[i][0]=='C')
        {
            sscanf(model[i].c_str(),"C%d",&now_out_c);
            now_out=get_output_dim(now_in,3,1,1);
            
            out=new int[now_out_c*BATCH_SIZE*now_out*now_out];
             
            //cout<<i<<" start"<<" "<<model[i]<<" "<<now_out<<" "<<now_out_c<<endl;
            conv2d_layer(in, layer_w[pc], out, layer_sx[pc], layer_sx_inv[pc],layer_sw_inv[pc], 
                 BATCH_SIZE, now_in_c, now_out_c, now_in, now_in, now_out, now_out,
                 3, 3,  1, 1,1,1);
            ++pc;
        }
        else if(model[i][0]=='M')
        {
            now_out=get_output_dim(now_in,2,2,0);
            now_out_c=now_in_c;
            out=new int[now_out_c*BATCH_SIZE*now_out*now_out];
            
            pooling2d(in, out, BATCH_SIZE, now_in_c, now_in, now_in, now_out, now_out, 2, 2,  2, 2);
        }
        else if(model[i][0]=='L')
        {
            sscanf(model[i].c_str(),"L%d",&now_out_c);
            out=new int[now_out_c*BATCH_SIZE*now_in*now_in];
            
            linear_layer(in, layer_w[pc], out, layer_sx[pc], layer_sx_inv[pc],layer_sw_inv[pc],
                  BATCH_SIZE, now_in_c*now_in*now_in, OUTPUT_DIM, i!=7);  // not general
            ++pc;
        }
        inbuf[i]=in;
        outbuf[i]=out;
        now_in=now_out;
        now_in_c=now_out_c;
        in=out;
        if(i==LAYER_NUM-1)
            argmax_over_cols(out, class_indices, BATCH_SIZE, OUTPUT_DIM);
    }
    auto relu_t0 = std::chrono::high_resolution_clock::now();
    int len=next_pow2(relu_input.size());
    const int l_relu = log2_(static_cast<size_t>(len));
    const int l_relu_t = 18;

    vector<ll> relu_input_ll(len, 0);
    vector<ll> relu_output_ll(len, 0);
    for (size_t i = 0; i < relu_input.size(); ++i)
    {
        relu_input_ll[i] = relu_input[i];
        relu_output_ll[i] = relu_output[i];
    }

    relu_input_commit = prover_commit(relu_input_ll.data(), g_shared, l_relu, 1);
    relu_output_commit = prover_commit(relu_output_ll.data(), g_shared, l_relu, 1);

    Fr gamma;
    gamma.setByCSPRNG();

    Fr* relu_f=new Fr[len];
    for(int i=0;i<len;i++)
    {
        if(i<relu_input.size())
            relu_f[i]=gamma*relu_input[i]+relu_output[i];
        else
            relu_f[i]=0;
    }

    const int relu_group_num = 1 << (l_relu / 2);
    relu_f_commit = new G1[relu_group_num];
    for (int i = 0; i < relu_group_num; ++i)
        relu_f_commit[i] = relu_input_commit[i] * gamma + relu_output_commit[i];

    

    Fr* relu_t=new Fr[1<<18];
    vector<ll> relu_t_now_ll(1<<18, 0);
    vector<ll> relu_t_out_ll(1<<18, 0);
    for(int i=0;i<(1<<18);i++)
    {
        int now=i-(1<<17);
        int out=max(now/(1<<FXP_VALUE), 0);
        relu_t_now_ll[i] = now;
        relu_t_out_ll[i] = out;
        relu_t[i]=gamma*now+out;
    }
    G1* relu_t_now_commit = prover_commit(relu_t_now_ll.data(), g_shared, l_relu_t, 1);
    G1* relu_t_out_commit = prover_commit(relu_t_out_ll.data(), g_shared, l_relu_t, 1);
    const int relu_t_group_num = 1 << (l_relu_t / 2);
    relu_t_commit = new G1[relu_t_group_num];
    for (int i = 0; i < relu_t_group_num; ++i)
        relu_t_commit[i] = relu_t_now_commit[i] * gamma + relu_t_out_commit[i];
    
    logup(relu_f,relu_t,relu_f_commit,relu_t_commit,len,(1<<18),1,G_shared,g_shared);

    delete[] relu_input_commit;
    delete[] relu_output_commit;
    delete[] relu_f_commit;
    delete[] relu_t_now_commit;
    delete[] relu_t_out_commit;
    delete[] relu_t_commit;
    delete[] relu_f;
    delete[] relu_t;
    auto relu_t1 = std::chrono::high_resolution_clock::now();
    total_relu_time_sec += std::chrono::duration<double>(relu_t1 - relu_t0).count();

    cout << "prover conv time: " << total_conv_time_sec << " seconds" << endl;
    cout << "prover relu time: " << total_relu_time_sec << " seconds" << endl;
    cout << "prover pooling time: " << total_pooling_time_sec << " seconds" << endl;
}





