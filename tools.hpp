#ifndef TOOLS_HPP
#define TOOLS_HPP

#include "hyrax.hpp"
#include <vector>

struct SumcheckFGResult
{
    std::vector<Fr> sumcheck_r;
    Fr claim_F;
    Fr claim_G;
};

struct SumcheckFGHResult
{
    std::vector<Fr> sumcheck_r;
    Fr claim_F;
    Fr claim_G;
    Fr claim_H;
    Fr final_claim;
};

struct ProductReduceResult
{
    Fr claim_a;
    Fr claim_b;
    std::vector<Fr> r_cur;
};

int next_pow2(int v);
int log2_(size_t x);
std::vector<Fr> build_tilde_eq_vector(size_t C_out, const std::vector<Fr>& r);
Fr eval_alpha_table_claim_from_sumcheck_r(const std::vector<Fr>& sumcheck_r, const Fr& alpha);
SumcheckFGResult run_sumcheck_fg(const Fr& sum, const std::vector<Fr>& F, const std::vector<Fr>& G, const char* tag = "");
SumcheckFGHResult run_sumcheck_fgh(const Fr& sum, const std::vector<Fr>& F, const std::vector<Fr>& G, const std::vector<Fr>& H, const char* tag = "");
Fr eval_mle_lsb(const std::vector<Fr>& table, const std::vector<Fr>& r_lsb);
std::vector<Fr> prepend_bit_lsb(const std::vector<Fr>& tail_r_lsb, int bit01);
std::vector<Fr> prepend_scalar_lsb(const std::vector<Fr>& tail_r_lsb, const Fr& head);
ProductReduceResult prove_product_beta_linear_gkr(const ll* a, const int* b, size_t n, const Fr& beta, const Fr& K_claim);
Fr eval_public_int_vector_claim(const std::vector<int>& b, const std::vector<Fr>& r_cur);

#endif
