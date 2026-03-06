#include "tools.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

using namespace mcl::bn;
using namespace std;

int next_pow2(int v)
{
    if (v <= 1) return 1;
    size_t p = 1;
    while (p < static_cast<size_t>(v)) p <<= 1;
    return static_cast<int>(p);
}

int log2_(size_t x)
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

vector<Fr> build_tilde_eq_vector(size_t C_out, const vector<Fr>& r)
{
    const int lg = log2_(C_out);
    assert(static_cast<int>(r.size()) == lg);

    Fr one = 1;
    vector<Fr> eq(1);
    eq[0] = 1;
    for (int i = 0; i < lg; ++i)
    {
        vector<Fr> nxt(eq.size() * 2);
        const Fr one_minus_ri = one - r[i];
        for (size_t j = 0; j < eq.size(); ++j)
        {
            nxt[2 * j] = eq[j] * one_minus_ri;
            nxt[2 * j + 1] = eq[j] * r[i];
        }
        eq.swap(nxt);
    }
    assert(eq.size() == C_out);
    return eq;
}

Fr eval_alpha_table_claim_from_sumcheck_r(const vector<Fr>& sumcheck_r, const Fr& alpha)
{
    Fr one = 1;
    Fr claim = 1;
    Fr alpha_pow_2t = alpha;
    for (size_t t = 0; t < sumcheck_r.size(); ++t)
    {
        const Fr rt = sumcheck_r[t];
        claim *= ((one - rt) + alpha_pow_2t * rt);
        alpha_pow_2t *= alpha_pow_2t;
    }
    return claim;
}

SumcheckFGResult run_sumcheck_fg(const Fr& sum, const vector<Fr>& F, const vector<Fr>& G, const char* tag)
{
    assert(F.size() == G.size());
    assert(!F.empty());
    assert((F.size() & (F.size() - 1)) == 0);

    const int lg = log2_(F.size());
    vector<Fr> f_table = F;
    vector<Fr> g_table = G;
    Fr claim = sum;
    vector<Fr> sumcheck_r(static_cast<size_t>(lg));

    Fr direct_sum0 = 0;
    for (size_t i = 0; i < F.size(); ++i) direct_sum0 += F[i] * G[i];
    if (direct_sum0 != claim)
        cerr << "[sumcheck:" << tag << "] initial claim mismatch, size=" << F.size() << endl;

    for (int round = 0; round < lg; ++round)
    {
        const size_t half = f_table.size() / 2;
        Fr a0 = 0, a1 = 0, a2 = 0;
        for (size_t i = 0; i < half; ++i)
        {
            const Fr f0 = f_table[2 * i], f1 = f_table[2 * i + 1];
            const Fr g0 = g_table[2 * i], g1 = g_table[2 * i + 1];
            const Fr df = f1 - f0, dg = g1 - g0;
            a0 += f0 * g0;
            a1 += f0 * dg + g0 * df;
            a2 += df * dg;
        }

        auto eval_quad = [&](const Fr& x) -> Fr { return a0 + a1 * x + a2 * x * x; };
        Fr zero = 0, one = 1;
        Fr lhs = eval_quad(zero) + eval_quad(one);
        if (lhs != claim)
            cerr << "[sumcheck:" << tag << "] round " << round << " failed, table_size=" << f_table.size() << endl;
        assert(lhs == claim);

        Fr ri;
        ri.setByCSPRNG();
        sumcheck_r[static_cast<size_t>(round)] = ri;
        claim = eval_quad(ri);

        vector<Fr> next_f(half), next_g(half);
        for (size_t i = 0; i < half; ++i)
        {
            next_f[i] = f_table[2 * i] + (f_table[2 * i + 1] - f_table[2 * i]) * ri;
            next_g[i] = g_table[2 * i] + (g_table[2 * i + 1] - g_table[2 * i]) * ri;
        }
        f_table.swap(next_f);
        g_table.swap(next_g);
    }

    assert(f_table.size() == 1 && g_table.size() == 1);
    assert(claim == f_table[0] * g_table[0]);
    return SumcheckFGResult{sumcheck_r, f_table[0], g_table[0]};
}

SumcheckFGHResult run_sumcheck_fgh(const Fr& sum, const vector<Fr>& F, const vector<Fr>& G, const vector<Fr>& H, const char* tag)
{
    assert(F.size() == G.size() && G.size() == H.size());
    assert(!F.empty());
    assert((F.size() & (F.size() - 1)) == 0);

    const int lg = log2_(F.size());
    vector<Fr> f_table = F, g_table = G, h_table = H;
    Fr claim = sum;
    vector<Fr> sumcheck_r(static_cast<size_t>(lg));

    Fr direct_sum0 = 0;
    for (size_t i = 0; i < F.size(); ++i) direct_sum0 += F[i] * G[i] * H[i];
    if (direct_sum0 != claim)
        cerr << "[sumcheck3:" << tag << "] initial claim mismatch, size=" << F.size() << endl;

    for (int round = 0; round < lg; ++round)
    {
        const size_t half = f_table.size() / 2;
        Fr a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (size_t i = 0; i < half; ++i)
        {
            const Fr f0 = f_table[2 * i], f1 = f_table[2 * i + 1];
            const Fr g0 = g_table[2 * i], g1 = g_table[2 * i + 1];
            const Fr h0 = h_table[2 * i], h1 = h_table[2 * i + 1];
            const Fr df = f1 - f0, dg = g1 - g0, dh = h1 - h0;
            a0 += f0 * g0 * h0;
            a1 += df * g0 * h0 + f0 * dg * h0 + f0 * g0 * dh;
            a2 += df * dg * h0 + df * g0 * dh + f0 * dg * dh;
            a3 += df * dg * dh;
        }

        auto eval_cubic = [&](const Fr& x) -> Fr { return a0 + a1 * x + a2 * x * x + a3 * x * x * x; };
        Fr zero = 0, one = 1;
        Fr lhs = eval_cubic(zero) + eval_cubic(one);
        if (lhs != claim)
            cerr << "[sumcheck3:" << tag << "] round " << round << " failed, table_size=" << f_table.size() << endl;
        assert(lhs == claim);

        Fr ri;
        ri.setByCSPRNG();
        sumcheck_r[static_cast<size_t>(round)] = ri;
        claim = eval_cubic(ri);

        vector<Fr> next_f(half), next_g(half), next_h(half);
        for (size_t i = 0; i < half; ++i)
        {
            next_f[i] = f_table[2 * i] + (f_table[2 * i + 1] - f_table[2 * i]) * ri;
            next_g[i] = g_table[2 * i] + (g_table[2 * i + 1] - g_table[2 * i]) * ri;
            next_h[i] = h_table[2 * i] + (h_table[2 * i + 1] - h_table[2 * i]) * ri;
        }
        f_table.swap(next_f);
        g_table.swap(next_g);
        h_table.swap(next_h);
    }

    assert(f_table.size() == 1 && g_table.size() == 1 && h_table.size() == 1);
    assert(claim == f_table[0] * g_table[0] * h_table[0]);
    return SumcheckFGHResult{sumcheck_r, f_table[0], g_table[0], h_table[0], claim};
}

Fr eval_mle_lsb(const vector<Fr>& table, const vector<Fr>& r_lsb)
{
    assert(!table.empty());
    assert((table.size() & (table.size() - 1)) == 0);
    assert(static_cast<int>(r_lsb.size()) == log2_(table.size()));

    vector<Fr> cur = table;
    for (size_t t = 0; t < r_lsb.size(); ++t)
    {
        const size_t half = cur.size() / 2;
        vector<Fr> nxt(half);
        for (size_t i = 0; i < half; ++i)
            nxt[i] = cur[2 * i] + (cur[2 * i + 1] - cur[2 * i]) * r_lsb[t];
        cur.swap(nxt);
    }
    assert(cur.size() == 1);
    return cur[0];
}

vector<Fr> prepend_bit_lsb(const vector<Fr>& tail_r_lsb, int bit01)
{
    Fr b = 0;
    if (bit01 != 0) b = 1;
    vector<Fr> out;
    out.reserve(tail_r_lsb.size() + 1);
    out.push_back(b);
    out.insert(out.end(), tail_r_lsb.begin(), tail_r_lsb.end());
    return out;
}

vector<Fr> prepend_scalar_lsb(const vector<Fr>& tail_r_lsb, const Fr& head)
{
    vector<Fr> out;
    out.reserve(tail_r_lsb.size() + 1);
    out.push_back(head);
    out.insert(out.end(), tail_r_lsb.begin(), tail_r_lsb.end());
    return out;
}

ProductReduceResult prove_product_beta_linear_gkr(const ll* a, const int* b, size_t n, const Fr& beta, const Fr& K_claim)
{
    assert(a != NULL && b != NULL);
    assert(n > 0 && (n & (n - 1)) == 0);
    const int L = log2_(n);

    vector<vector<Fr>> V(static_cast<size_t>(L + 1));
    V[0].resize(n);
    for (size_t i = 0; i < n; ++i) V[0][i] = beta * a[i] + b[i];
    for (int lev = 1; lev <= L; ++lev)
    {
        const size_t m = V[lev - 1].size() / 2;
        V[lev].resize(m);
        for (size_t i = 0; i < m; ++i) V[lev][i] = V[lev - 1][2 * i] * V[lev - 1][2 * i + 1];
    }
    assert(V[L].size() == 1 && V[L][0] == K_claim);

    vector<Fr> r_cur;
    Fr claim_cur = K_claim;
    for (int lev = L; lev >= 1; --lev)
    {
        const size_t m = V[lev].size();
        vector<Fr> eq_tbl;
        if (m == 1)
        {
            Fr one = 1;
            eq_tbl.assign(1, one);
        }
        else
        {
            vector<Fr> r_msb = r_cur;
            reverse(r_msb.begin(), r_msb.end());
            eq_tbl = build_tilde_eq_vector(m, r_msb);
        }

        vector<Fr> F(m), G(m), H(m);
        Fr sum_check = 0;
        for (size_t i = 0; i < m; ++i)
        {
            F[i] = eq_tbl[i];
            G[i] = V[lev - 1][2 * i];
            H[i] = V[lev - 1][2 * i + 1];
            sum_check += F[i] * G[i] * H[i];
        }
        assert(sum_check == claim_cur);

        SumcheckFGHResult sc = run_sumcheck_fgh(claim_cur, F, G, H, "GKR_prod");
        const Fr eq_at_q = eval_mle_lsb(eq_tbl, sc.sumcheck_r);
        assert(sc.claim_F == eq_at_q);
        assert(sc.final_claim == sc.claim_F * sc.claim_G * sc.claim_H);

        Fr r_pp;
        r_pp.setByCSPRNG();
        claim_cur = (1 - r_pp) * sc.claim_G + r_pp * sc.claim_H;
        r_cur = prepend_scalar_lsb(sc.sumcheck_r, r_pp);
    }

    assert(static_cast<int>(r_cur.size()) == L);
    vector<Fr> a_fr(n), b_fr(n);
    for (size_t i = 0; i < n; ++i) a_fr[i] = a[i];
    for (size_t i = 0; i < n; ++i) b_fr[i] = b[i];
    const Fr claim_a = eval_mle_lsb(a_fr, r_cur);
    const Fr claim_b = eval_mle_lsb(b_fr, r_cur);
    assert(beta * claim_a + claim_b == claim_cur);
    return ProductReduceResult{claim_a, claim_b, r_cur};
}

Fr eval_public_int_vector_claim(const vector<int>& b, const vector<Fr>& r_cur)
{
    vector<Fr> b_fr(b.size());
    for (size_t i = 0; i < b.size(); ++i) b_fr[i] = b[i];
    return eval_mle_lsb(b_fr, r_cur);
}
