#include "../src/ds4/mmq_scratch_alloc.hpp"
#include <cassert>
#include <cstdlib>
#include <set>

int main() {
    using namespace cilla::mmq_scratch;
    std::set<void*> live;
    int calls = 0, fail_at = 0;
    auto alloc = [&](void** p, std::size_t n) { ++calls; if (calls == fail_at) return false; *p = std::malloc(n); live.insert(*p); return *p != nullptr; };
    auto free = [&](void* p) { assert(live.erase(p) == 1); std::free(p); };

    void *g=nullptr, *u=nullptr; std::size_t cap=0;
    for (int fail=1; fail<=2; ++fail) {
        calls=0; fail_at=fail;
        assert(!replace_pair(g,u,cap,8192,8192*sizeof(float),alloc,free));
        assert(!g && !u && cap==0 && live.empty());
    }
    calls=0; fail_at=0;
    assert(replace_pair(g,u,cap,8192,8192*sizeof(float),alloc,free));
    assert(cap==8192 && live.size()==2);
    auto oldg=g; auto oldu=u;
    calls=0; fail_at=1;
    assert(!replace_pair(g,u,cap,9000,9000*sizeof(float),alloc,free));
    assert(g==oldg && u==oldu && cap==8192 && live.size()==2);
    calls=0; fail_at=2;
    assert(!replace_pair(g,u,cap,9000,9000*sizeof(float),alloc,free));
    assert(g==oldg && u==oldu && cap==8192 && live.size()==2);
    calls=0; fail_at=0;
    assert(replace_pair(g,u,cap,8192,8192*sizeof(float),alloc,free));
    assert(cap==8192 && g==oldg && u==oldu);
    free(g); free(u); assert(live.empty());
}
