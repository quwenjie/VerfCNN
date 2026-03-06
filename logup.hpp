#ifndef LOGUP_HPP
#define LOGUP_HPP

#include "hyrax.hpp"
#include "tools.hpp"
struct SC_Return
{
    Fr* random;
    Fr claim_f;
    Fr claim_g;
};

Fr poly_eval(Fr x0, Fr x1, Fr x2, Fr x3, Fr u);
SC_Return sumcheck_deg1(int l, Fr* f, Fr S);
SC_Return sumcheck_deg3(int l, Fr* r, Fr* f, Fr* g, Fr S);
void logup(Fr* f, Fr* t, G1* f_comm, G1* t_comm, int m, int n, int thread, G1& G_shared, G1* g_shared);
extern double total_logup_time_sec;

#endif
