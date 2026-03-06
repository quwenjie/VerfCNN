#include "logup.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <unordered_map>

using namespace std;
using namespace mcl::bn;
double total_logup_time_sec = 0.0;

namespace {

int log2_pow2(size_t x)
{
    assert(x > 0 && (x & (x - 1)) == 0);
    int lg = 0;
    while (x > 1)
    {
        x >>= 1;
        ++lg;
    }
    return lg;
}

} // namespace

Fr poly_eval(Fr x0, Fr x1, Fr x2, Fr x3, Fr u)
{
    Fr y = 1 / Fr(6) * ((-x0) * (u - 1) * (u - 2) * (u - 3)
        + 3 * x1 * u * (u - 2) * (u - 3)
        - 3 * x2 * u * (u - 1) * (u - 3)
        + x3 * u * (u - 1) * (u - 2));
    return y;
}

SC_Return sumcheck_deg1(int l, Fr* f, Fr S)
{
    Fr* ran = new Fr[l];
    for (int i = l; i >= 1; i--)
    {
        Fr sum0 = 0, sum1 = 0;
        for (int j = 0; j < (1 << i); j++)
        {
            if ((j & 1) == 0) sum0 += f[j];
            else sum1 += f[j];
        }
        assert(sum0 + sum1 == S);
        Fr new_chlg;
        new_chlg.setByCSPRNG();
        ran[l - i] = new_chlg;
        S = new_chlg * (sum1 - sum0) + sum0;
        Fr* new_f = new Fr[1 << (i - 1)];
        for (int j = 0; j < (1 << (i - 1)); j++)
            new_f[j] = (1 - new_chlg) * f[j * 2] + new_chlg * f[j * 2 + 1];
        f = new_f;
    }
    SC_Return s;
    s.random = ran;
    s.claim_f = f[0];
    s.claim_g = 0;
    return s;
}

SC_Return sumcheck_deg3(int l, Fr* r, Fr* f, Fr* g, Fr S)
{
    Fr* lag = get_eq(r, l);
    Fr* ran = new Fr[l];
    Fr* S0 = new Fr[1 << l];
    Fr* S1 = new Fr[1 << l];
    Fr* S2 = new Fr[1 << l];
    Fr* S3 = new Fr[1 << l];
    for (int i = l; i >= 1; i--)
    {
        memset(S0, 0, sizeof(Fr) * (1 << i));
        memset(S1, 0, sizeof(Fr) * (1 << i));
        memset(S2, 0, sizeof(Fr) * (1 << i));
        memset(S3, 0, sizeof(Fr) * (1 << i));

        Fr sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int j = 0; j < (1 << i); j += 2)
        {
            S0[j >> 1] = lag[j] * f[j] * g[j];
            S1[j >> 1] = lag[j + 1] * f[j + 1] * g[j + 1];
            S2[j >> 1] = (lag[j + 1] + lag[j + 1] - lag[j])
                * (f[j + 1] + f[j + 1] - f[j])
                * (g[j + 1] + g[j + 1] - g[j]);
            S3[j >> 1] = (lag[j + 1] + lag[j + 1] + lag[j + 1] - lag[j] - lag[j])
                * (f[j + 1] + f[j + 1] + f[j + 1] - f[j] - f[j])
                * (g[j + 1] + g[j + 1] + g[j + 1] - g[j] - g[j]);
        }

        if (i < 8)
        {
            for (int j = 0; j < (1 << (i - 1)); j++)
            {
                sum0 += S0[j];
                sum1 += S1[j];
                sum2 += S2[j];
                sum3 += S3[j];
            }
        }
        else
        {
            Fr s0[8], s1[8], s2[8], s3[8];
            memset(s0, 0, sizeof(Fr) * 8);
            memset(s1, 0, sizeof(Fr) * 8);
            memset(s2, 0, sizeof(Fr) * 8);
            memset(s3, 0, sizeof(Fr) * 8);
            for (int k = 0; k < (1 << 3); k++)
            {
                for (int j = 0; j < (1 << (i - 1 - 3)); j++)
                {
                    s0[k] += S0[(k << (i - 1 - 3)) + j];
                    s1[k] += S1[(k << (i - 1 - 3)) + j];
                    s2[k] += S2[(k << (i - 1 - 3)) + j];
                    s3[k] += S3[(k << (i - 1 - 3)) + j];
                }
            }
            sum0 = s0[0] + s0[1] + s0[2] + s0[3] + s0[4] + s0[5] + s0[6] + s0[7];
            sum1 = s1[0] + s1[1] + s1[2] + s1[3] + s1[4] + s1[5] + s1[6] + s1[7];
            sum2 = s2[0] + s2[1] + s2[2] + s2[3] + s2[4] + s2[5] + s2[6] + s2[7];
            sum3 = s3[0] + s3[1] + s3[2] + s3[3] + s3[4] + s3[5] + s3[6] + s3[7];
        }
        assert(sum0 + sum1 == S);
        Fr new_chlg;
        new_chlg.setByCSPRNG();
        ran[l - i] = new_chlg;
        S = poly_eval(sum0, sum1, sum2, sum3, new_chlg);
        Fr* new_lag = new Fr[1 << (i - 1)];
        Fr* new_f = new Fr[1 << (i - 1)];
        Fr* new_g = new Fr[1 << (i - 1)];
        for (int j = 0; j < (1 << (i - 1)); j++)
        {
            new_lag[j] = lag[j * 2] + new_chlg * (lag[j * 2 + 1] - lag[j * 2]);
            new_f[j] = f[j * 2] + new_chlg * (f[j * 2 + 1] - f[j * 2]);
            new_g[j] = g[j * 2] + new_chlg * (g[j * 2 + 1] - g[j * 2]);
        }
        f = new_f;
        g = new_g;
        lag = new_lag;
    }
    SC_Return s;
    s.random = ran;
    s.claim_f = f[0];
    s.claim_g = g[0];
    return s;
}

void logup(Fr* f, Fr* t, G1* f_comm, G1* t_comm, int m, int n, int thread, G1& G_shared, G1* g_shared)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    Fr* G = new Fr[m];
    Fr* F = new Fr[m];
    Fr* Hp = new Fr[n];
    Fr* H = new Fr[n];
    ll* c = new ll[n];
    memset(c, 0, sizeof(ll) * n);

    unordered_map<string, int> t_index;
    t_index.reserve(static_cast<size_t>(n) * 2);
    for (int j = 0; j < n; ++j)
    {
        const string key = t[j].getStr();
        auto it = t_index.find(key);
        assert(it == t_index.end());
        t_index[key] = j;
    }
    for (int i = 0; i < m; ++i)
    {
        const string key = f[i].getStr();
        auto it = t_index.find(key);
        assert(it != t_index.end());
        c[it->second]++;
    }

    Fr r;
    r.setByCSPRNG();
    for (int i = 0; i < m; i++) F[i] = r + f[i];
    for (int i = 0; i < n; i++) Hp[i] = r + t[i];

    invVec(G, F, m);
    invVec(H, Hp, n);
    Fr sum = 0;
    for (int i = 0; i < m; i++) sum += G[i];
    for (int i = 0; i < n; i++) H[i] = H[i] * Fr(c[i]);

    G1* c_comm = prover_commit(c, g_shared, log2_pow2(static_cast<size_t>(n)), thread);
    G1* g_comm = prover_commit_fr_general(G, g_shared, log2_pow2(static_cast<size_t>(m)), thread);
    G1* h_comm = prover_commit_fr_general(H, g_shared, log2_pow2(static_cast<size_t>(n)), thread);

    Fr* rp1 = new Fr[log2_pow2(static_cast<size_t>(n))];
    for (int i = 0; i < log2_pow2(static_cast<size_t>(n)); i++) rp1[i].setByCSPRNG();
    Fr c_eva = prover_evaluate(c, rp1, log2_pow2(static_cast<size_t>(n)));
    SC_Return ret1 = sumcheck_deg3(log2_pow2(static_cast<size_t>(n)), rp1, H, Hp, c_eva);

    Fr* rp2 = new Fr[log2_pow2(static_cast<size_t>(m))];
    for (int i = 0; i < log2_pow2(static_cast<size_t>(m)); i++) rp2[i].setByCSPRNG();
    SC_Return ret2 = sumcheck_deg3(log2_pow2(static_cast<size_t>(m)), rp2, G, F, 1);
    SC_Return ret3 = sumcheck_deg1(log2_pow2(static_cast<size_t>(m)), G, sum);
    SC_Return ret4 = sumcheck_deg1(log2_pow2(static_cast<size_t>(n)), H, sum);

    open(c,rp1,c_eva,G_shared,g_shared,c_comm,log2_(n));
    open(H,ret1.random,ret1.claim_f,G_shared,g_shared,h_comm,log2_(n));
    open(t,ret1.random,ret1.claim_g-r,G_shared,g_shared,t_comm,log2_(n));
    open(G,ret2.random,ret2.claim_f,G_shared,g_shared,g_comm,log2_(m));
    open(f,ret2.random,ret2.claim_g-r,G_shared,g_shared,f_comm,log2_(m));
    open(G,ret3.random,ret3.claim_f,G_shared,g_shared,g_comm,log2_(m));
    open(H,ret4.random,ret4.claim_f,G_shared,g_shared,h_comm,log2_(n));

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    total_logup_time_sec += elapsed.count();

    delete[] G;
    delete[] F;
    delete[] Hp;
    delete[] H;
    delete[] c;
}
