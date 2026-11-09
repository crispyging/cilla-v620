#pragma once

#include <cstddef>

namespace cilla::mmq_scratch {
// Allocate a complete replacement before publishing it; failure preserves the old allocation.
template<class Alloc, class Free>
bool replace(void*& current, std::size_t& capacity, std::size_t needed, Alloc alloc, Free free) {
    if (capacity >= needed) return true;
    void* fresh = nullptr;
    if (!alloc(&fresh, needed) || fresh == nullptr) {
        if (fresh) free(fresh);
        return false;
    }
    void* old = current;
    current = fresh;
    capacity = needed;
    if (old) free(old);
    return true;
}

// Both allocations form one capacity unit. Roll back every fresh pointer unless both succeed.
template<class T, class U, class Alloc, class Free>
bool replace_pair(T*& first, U*& second, std::size_t& capacity, std::size_t needed,
                  std::size_t bytes, Alloc alloc, Free free) {
    if (capacity >= needed) return true;
    void* fresh_first = nullptr;
    void* fresh_second = nullptr;
    if (!alloc(&fresh_first, bytes) || fresh_first == nullptr ||
        !alloc(&fresh_second, bytes) || fresh_second == nullptr) {
        if (fresh_first) free(fresh_first);
        if (fresh_second) free(fresh_second);
        return false;
    }
    void* old_first = first;
    void* old_second = second;
    first = static_cast<T*>(fresh_first);
    second = static_cast<U*>(fresh_second);
    capacity = needed;
    if (old_first) free(old_first);
    if (old_second) free(old_second);
    return true;
}
}  // namespace cilla::mmq_scratch
