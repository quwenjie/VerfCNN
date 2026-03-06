#ifndef HYRAX_DEFINE
#define HYRAX_DEFINE
#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>         // std::thread
#include <mcl/bn.hpp>


using namespace std;
using namespace mcl::bn;

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;


typedef long long i64;
typedef int i32;
typedef char i8;
typedef long long ll;

struct Pack 
{
    G1 gamma;
    Fr a;
    G1 g;
    Fr x;
    Fr y;
    
    Pack(G1 gamm,Fr fa, G1 gg,Fr xx,Fr yy)
    {
        gamma=gamm;
        a=fa;
        g=gg;
        x=xx;
        y=yy;
    }
};
const int MAXL = 26;
G1 perdersen_commit(G1* g,ll* f,int n,G1* W=NULL); //support 2^80, optimized using pippenger
G1 perdersen_commit_classic(G1* g,Fr* f,int n);
G1 perdersen_commit_redundant(G1* g,Fr* f,ll* id,int n,int m); 
Fr lagrange(Fr *r,int l,int k);
G1* prover_commit_fr(ll* w, Fr* f,int m,G1* g, int l,int thread_n) ;
G1* prover_commit_fr_general(Fr* w, G1* g, int l,int thread_n) ;
Fr* get_eq(Fr*r, int l);

G1 gen_gi(G1* g,int n);

bool prove_dot_product(G1 comm_x, G1 comm_y, Fr* a, Fr* x,Fr y, G1*g ,G1& G,int n);

G1* prover_commit(ll* w, G1* g, int l,int thread_n=1);
Fr prover_evaluate(ll*w ,Fr*r,int l);  
void open(ll*w,Fr*r,Fr eval,G1&G,G1*g,G1*comm,int l);
void open(Fr*w,Fr*r,Fr eval,G1&G,G1*g,G1*comm,int l);

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() {}

    void Push(T value) {
        unique_lock<mutex> lock(mutex_);
        queue_.push(value);
        lock.unlock();
        condition_variable_.notify_one();
    }

    bool TryPop(T& value) {
        lock_guard<mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = queue_.front();
        queue_.pop();
        return true;
    }

    void WaitPop(T& value) {
        unique_lock<mutex> lock(mutex_);
        condition_variable_.wait(lock, [this] { return !queue_.empty(); });
        value = queue_.front();
        queue_.pop();
    }

    bool Empty() const {
        lock_guard<mutex> lock(mutex_);
        return queue_.empty();
    }
    int Size() const {
        lock_guard<mutex> lock(mutex_);
        return queue_.size();
    }
    void Clear()  
    {
        lock_guard<mutex> lock(mutex_);
        queue_={};
    }

private:
    mutable mutex mutex_;
    queue<T> queue_;
    condition_variable condition_variable_;
};

#endif