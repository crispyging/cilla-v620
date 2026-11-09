// src/ds4/ds4_main.cpp - maya-ds4: DeepSeek-V4.1-Flash on Maya.
//
//   maya-ds4 --pack <model>/maya-ds4 --ids 0,128803,... --n 16          greedy tokens after the ids
//   maya-ds4 --pack ... --ids ... --dump-logits out.f32                every position's logits (the oracle's check)
//   maya-ds4 --pack ... --serve                                       Maya's engine protocol on stdin / stdout
//
// --serve speaks what serve/server.py's StrataEngine expects: INFO lines, then `READY <context> stop`; per request
// `GEN <max_new> [key=value ...] <id,id,...>` answered by `T <id>` lines and `DONE <generated> <prompt> <prompt ms>
// <decode ms> <stop|length|cancel> 0 0 <reused> 0 0`; `STOP` ends a request early, `QUIT` the engine.  A prompt that
// continues the state (the conversation so far) reuses it; one that only shares the last prompt's start goes back to
// that prompt's end; anything else starts over.  `GENI <max_new> [keys] <file> <ids>` carries the prompt's pictures
// (the file: see read_pics); the state is shared only where the same picture sits at the same place.
#include "ds4_model.hpp"
#include "strata/platform/env_alias.hpp"

#include <csignal>
#include <cstdlib>
#include <execinfo.h>

// ds4_fatal_backtrace: print the stack on a fatal signal before the default action runs
static void ds4_fatal_backtrace(int sig) {
    if (std::getenv("DS4_BT") != nullptr && std::atoi(std::getenv("DS4_BT")) == 0) {
        std::signal(sig, SIG_DFL);
        std::raise(sig);
        return;
    }
    void* bt[64];
    const int n = backtrace(bt, 64);
    std::fprintf(stderr, "ds4: FATAL signal %d (%s) - backtrace, %d frames:\n", sig,
                 sig == SIGSEGV ? "SIGSEGV" : sig == SIGABRT ? "SIGABRT" : sig == SIGBUS ? "SIGBUS" : "?", n);
    backtrace_symbols_fd(bt, n, 2);
    std::fflush(stderr);
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}


#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <set>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#include <cerrno>
#endif

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the bytes this process had the storage read for it (Linux's /proc/self/io; -1 where there is none): the Monitor's
// disk reads a token, the experts read in place from the model's files included
long long io_read_bytes() {
#ifdef __linux__
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (f == nullptr) return -1;
    char line[128];
    long long v = -1;
    while (std::fgets(line, sizeof line, f) != nullptr)
        if (std::sscanf(line, "read_bytes: %lld", &v) == 1) break;
    std::fclose(f);
    return v;
#else
    return -1;
#endif
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int) x);
        p = e;
        while (*p == ',' || *p == ' ') ++p;
    }
    return v;
}

void usage() {
    std::fprintf(stderr,
                 "maya-ds4 --pack <model>/maya-ds4 [--gpus 0,1,...] [--max-context N] [--reserve-mib N]\n"
                 "           [--vram-experts F] [--cpu-threads N] [--expert-profile FILE|none] [--no-expert-cache] [--verbose]\n"
                 "           [--tp N] [--threads on|off] [--prefill-chunk N] [--dspark on|off|none] [--dspark-conf F]\n"
                 "           [--vision on|off] [--mmap-experts on|off|auto]\n"
                 "           (--ids I,J,... [--images FILE] [--n N] [--dump-logits FILE] | --serve\n"
                 "           | --encode-image FILE.rgb --size WxH --dump-image OUT)\n");
}

// stdin lines on a thread, so a request can see STOP while it decodes.  read(2), not std::cin - glibc's exit()
// flushes stdio and waits on stdin's lock, which getline holds while it waits for input (the GLM serve's pattern)
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::atomic<bool> pause{false};   // (cilla: PAUSE - end the answer where the state holds what it wrote)
    std::set<long long> stop_rid, pause_rid;   // (cilla: STOP <rid> / PAUSE <rid>, the multi protocol)
    bool take(std::set<long long>& s, long long rid) {
        std::lock_guard<std::mutex> lk(mu);
        return s.erase(rid) > 0;
    }
    bool try_next(std::string& l) {
        std::lock_guard<std::mutex> lk(mu);
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
    bool ended() {
        std::lock_guard<std::mutex> lk(mu);
        return eof && q.empty();
    }
    void start() {
        std::thread([this] {
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
#if defined(_WIN32)
                    const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
#endif
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                std::lock_guard<std::mutex> lk(mu);
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    stop.store(true);
                    if (l.size() > 5) stop_rid.insert(std::atoll(l.c_str() + 5));
                    continue;
                }
                if (l == "PAUSE" || l.rfind("PAUSE ", 0) == 0) {   // (cilla: the server's fair slices)
                    pause.store(true);
                    if (l.size() > 6) pause_rid.insert(std::atoll(l.c_str() + 6));
                    continue;
                }
                if (l == "QUIT") stop.store(true);
                q.push_back(l);
                cv.notify_all();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            stop.store(true);
            cv.notify_all();
        }).detach();
    }
    bool next(std::string& l) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
};

// The drafts between two stats: each draft position (1st, 2nd, ..) - how often it was verified with every draft
// before it kept, and how often it was kept then
std::string per_position(const strata::ds4::Stats& s0, const strata::ds4::Stats& s1) {
    std::string pp = "per position (kept / reached):";
    for (int i = 0; i < 8; ++i) {
        const long long r = s1.draft_reached[i] - s0.draft_reached[i], k = s1.draft_kept[i] - s0.draft_kept[i];
        if (r <= 0) break;
        char b[64];
        std::snprintf(b, sizeof b, " %d: %lld/%lld %.0f%%", i + 1, k, r, 100.0 * (double) k / (double) r);
        pp += b;
    }
    return pp;
}

// The drafts' cut on the product of their confidences (DeepInfer's cumulative rule): a draft is verified while the
// chance that it and every draft before it are kept stays at least this - a long window whose later drafts are likely
// lost is not verified whole.  MAYA_DS4_DSPARK_CUM=F (STRATA_DS4_DSPARK_CUM), a request's dspark_cum key; 0: off.
// 0.5 by default (upstream measured +6.5% decode on a Radeon 8065S, the same tokens).
float dspark_cum_default() {
    static const float v = [] {
        const char* e = std::getenv("STRATA_DS4_DSPARK_CUM");
        // (cilla: upstream's 0.5 measured 54.6 vs 62.9 tok/s here - the drafter's 0.5 floor already cuts the
        // long windows on this box, and the product then rejects drafts that would have been kept)
        return e != nullptr ? (float) std::atof(e) : 0.0f;
    }();
    return v;
}

// One round of decoding after `tok` (the token at pos(), not fed yet): with the drafter, its drafts while their
// confidence holds (at most `room` - 1 of them), verified together; else a plain step.  `out` gets the tokens that follow
// tok (the accepted drafts, then the trunk's next), the last of them the next round's tok.
bool round(strata::ds4::Model& m, int tok, int room, bool spec, float conf_min, float temperature, float top_p, int top_k,
           float min_p, unsigned long long seed, unsigned long long& counter, std::vector<int>& out, int& offered,
           int& accepted, std::string& err, float cum_min = -1.0f) {
    out.clear();
    offered = accepted = 0;
    if (spec && room > 1 && m.draft_block() > 0) {
        std::vector<int> drafts;
        std::vector<float> conf;
        if (!m.draft(tok, drafts, conf, err, temperature, top_p, top_k, min_p, seed, counter)) return false;
        int keep = 0;
        if (cum_min < 0) cum_min = dspark_cum_default();
        float cum = 1.0f;
        while (keep < (int) drafts.size() && keep < room - 1 && conf[(size_t) keep] >= conf_min &&
               (cum *= conf[(size_t) keep]) >= cum_min)
            ++keep;
        if (keep > 0) {
            drafts.resize((size_t) keep);
            int acc = 0, next = -1;
            if (!m.verify(tok, drafts, temperature, top_p, top_k, min_p, seed, counter, acc, next, err)) return false;
            counter += (unsigned long long) acc + 1;
            for (int i = 0; i < acc; ++i) out.push_back(drafts[(size_t) i]);
            out.push_back(next);
            offered = keep;
            accepted = acc;
            return true;
        }
    }
    if (!m.step(tok, true, err)) return false;
    out.push_back(m.sample(temperature, top_p, top_k, min_p, seed, counter++));
    return true;
}

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int spec = -1;          // spec=0|1: this request drafts or not (-1: as the engine was started)
    float conf = -1.0f;     // dspark_conf=F: the drafts' confidence floor for this request (-1: the engine's)
    float cum = -1.0f;      // dspark_cum=F: the cut on the product of the confidences (-1: the engine's)
    // the penalties (penalty_last_n / _repeat / _freq / _present, llama.cpp's): over the last pen_n tokens of the state
    // (-1: the whole context, which the engine caps at its window; 0: none)
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    std::string file;   // GENI: the pictures
    std::vector<int> ids;
    long long rid = -1;   // (cilla: the multi protocol's request id)
};

// A prompt's picture: where its span starts, its pixels (the server's resize: RGB rows, both sides multiples of the
// patch size), a hash of them; once encoded, its span's rows
struct Pic {
    int64_t start = 0;
    int width = 0, height = 0;
    uint64_t hash = 0;
    size_t len = 0;   // the span (0: not known yet)
    std::vector<uint8_t> rgb;
};

// GENI's file: int32 'DSI1', int32 count, then per picture int64 start, int32 width, int32 height and its
// height x width x 3 RGB bytes, in prompt order
bool read_pics(const std::string& path, std::vector<Pic>& v, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open the pictures' file " + path;
        return false;
    }
    char magic[4] = {};
    int32_t n = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DSI1", 4) == 0 && std::fread(&n, 4, 1, f) == 1 &&
              n >= 0 && n <= 4096;
    for (int i = 0; ok && i < n; ++i) {
        Pic p;
        int32_t wh[2] = {};
        ok = std::fread(&p.start, 8, 1, f) == 1 && std::fread(wh, 4, 2, f) == 2 && wh[0] > 0 && wh[1] > 0 &&
             (int64_t) wh[0] * wh[1] <= (64 << 20);
        if (!ok) break;
        p.width = wh[0], p.height = wh[1];
        p.rgb.resize((size_t) p.width * p.height * 3);
        ok = std::fread(p.rgb.data(), 1, p.rgb.size(), f) == p.rgb.size();
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the size and the pixels
        for (int32_t x : wh)
            for (int b = 0; b < 4; ++b) h = (h ^ (uint8_t) (x >> (8 * b))) * 1099511628211ull;
        for (uint8_t b : p.rgb) h = (h ^ b) * 1099511628211ull;
        p.hash = h;
        v.push_back(std::move(p));
    }
    std::fclose(f);
    if (!ok) err = path + ": not a pictures' file (DSI1)";
    return ok;
}

// the encoded spans, the latest few (a conversation sends its pictures with every turn)
struct SpanCache {
    std::deque<std::pair<uint64_t, std::shared_ptr<const std::vector<uint16_t>>>> q;
    std::shared_ptr<const std::vector<uint16_t>> get(strata::ds4::Model& m, const Pic& p, std::string& err) {
        for (size_t i = 0; i < q.size(); ++i)
            if (q[i].first == p.hash) {
                auto r = q[i].second;
                q.erase(q.begin() + (long) i);
                q.emplace_front(p.hash, r);
                return r;
            }
        auto rows = std::make_shared<std::vector<uint16_t>>();
        if (!m.encode_image(p.rgb.data(), p.width, p.height, *rows, err)) return nullptr;
        q.emplace_front(p.hash, rows);
        if (q.size() > 8) q.pop_back();
        return rows;
    }
};

bool parse_gen(const std::string& line, Req& r, bool geni) {
    const char* p = line.c_str() + (geni ? 5 : 4);
    char* e = nullptr;
    r.max_new = std::strtoll(p, &e, 10);
    if (e == p) return false;
    p = e;
    for (;;) {
        while (*p == ' ') ++p;
        const char* s = p;
        while (*p && *p != ' ') ++p;
        if (p == s) break;
        const std::string tok(s, (size_t) (p - s));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos && geni && r.file.empty()) {
            r.file = tok;
            continue;
        }
        if (eq == std::string::npos) {
            r.ids = parse_ids(tok);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "spec") r.spec = std::atoi(v.c_str()) != 0;
        else if (k == "dspark_conf") r.conf = std::strtof(v.c_str(), nullptr);
        else if (k == "dspark_cum" || k == "spec_cum") r.cum = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "rid") r.rid = std::atoll(v.c_str());
    }
    return !r.ids.empty() && (!geni || !r.file.empty());
}

int serve(strata::ds4::Model& m, const std::string& eos_list, bool spec, float conf_min) {
    std::vector<int> eos = parse_ids(eos_list);
    for (const std::string& l : m.info_lines()) std::printf("%s\n", l.c_str());
    std::printf("READY %lld stop pause\n", (long long) m.max_context());
    std::fflush(stdout);
    std::fprintf(stderr, "maya ds4 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <pictures> <ids>; STOP, QUIT\n");
    Lines in;
    in.start();
    std::string line;
    unsigned long long counter = 0;
    std::vector<Pic> held;   // the pictures the state holds (start, hash, span)
    // cilla: per conversation slot - its pictures, and whether its last answer ended at a PAUSE (the next request that
    // continues it by exactly the paused token is that answer's next slice: it keeps the slot's checkpoints)
    std::vector<std::vector<Pic>> held_by((size_t) m.conversations());
    std::vector<char> paused_by((size_t) m.conversations(), 0);
    int conv = 0;
    // (DS4_SERVE_DUMP=<prefix>, tests: request i's logits - the prompt's last row, then every decode step's rows - into
    //  <prefix>.<i>.f32, to compare a request's logits byte for byte across processes and histories)
    const char* dump_prefix = std::getenv("DS4_SERVE_DUMP");
    int n_req = 0;
    std::vector<float> dump_lg;
    std::FILE* dump_f = nullptr;
    auto dump_rows = [&]() {
        if (dump_f != nullptr && m.logits(dump_lg)) std::fwrite(dump_lg.data(), 4, dump_lg.size(), dump_f);
    };
    SpanCache spans;
    const int img_tok = m.image_token();
    while (in.next(line)) {
        if (line == "QUIT") break;
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        in.pause.store(false);
        Req r;
        if (!parse_gen(line, r, geni)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        m.begin_request();   // (DISK_SKIP_AUTO: between requests only)
        const double t0 = now_ms();
        std::vector<Pic> pics;
        {
            std::string e;
            if (geni && img_tok < 0) e = "this engine has no vision tower, so it cannot read pictures";
            else if (geni && read_pics(r.file, pics, e))
                for (const Pic& q : pics)
                    if (q.start < 0 || (size_t) q.start >= r.ids.size() || r.ids[(size_t) q.start] != img_tok)
                        e = "a picture's span does not start at an image token of the prompt";
            if (!e.empty()) {
                std::printf("ERR %s\n", e.c_str());
                std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
                std::fflush(stdout);
                continue;
            }
        }
        {   // cilla: the slot this prompt continues best
            held_by[(size_t) conv] = std::move(held);
            conv = m.pick_conversation(r.ids);
            held = std::move(held_by[(size_t) conv]);
        }
        const auto& st = m.tokens();
        const size_t n_state = st.size();
        size_t c = 0;
        while (c < n_state && c < r.ids.size() && st[c] == r.ids[c]) ++c;
        // the same tokens with another picture (or none) do not share the state from that picture's span on
        auto same = [](const std::vector<Pic>& v, const Pic& q) {
            for (const Pic& x : v)
                if (x.start == q.start && x.hash == q.hash) return true;
            return false;
        };
        for (const Pic& q : pics)
            if ((size_t) q.start < c && !same(held, q)) c = (size_t) q.start;
        for (const Pic& q : held)
            if ((size_t) q.start < c && !same(pics, q)) c = (size_t) q.start;
        size_t from = 0;
        if (c == n_state && n_state < r.ids.size()) from = n_state;
        else if (m.restorable((int64_t) std::min(c, r.ids.size() - 1)) >= 0 &&
                 m.restore((int64_t) std::min(c, r.ids.size() - 1)))
            from = (size_t) m.pos();
        else m.reset();
        if (from == 0) held.clear();
        // (a paused answer's next slice: prompt + what it wrote, the state one token short of it)
        const bool resume = paused_by[(size_t) conv] && from == n_state && r.ids.size() == n_state + 1;
        paused_by[(size_t) conv] = 0;
        if ((int64_t) r.ids.size() + std::max(0LL, r.max_new) > m.max_context()) {
            std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", r.ids.size(),
                        r.max_new, (long long) m.max_context());
            std::fflush(stdout);
            continue;
        }
        std::string err;
        bool ok = true;
        // the pictures whose spans the prompt reads (the rest the state holds): encoded (or the cache's), checked
        if (!pics.empty()) {
            std::vector<strata::ds4::Image> imgs;
            const double te = now_ms();
            int encoded = 0;
            for (Pic& q : pics) {
                for (const Pic& x : held)
                    if (x.start == q.start && x.hash == q.hash) q.len = x.len;
                if (q.len > 0 && (size_t) q.start + q.len <= from) continue;
                std::shared_ptr<const std::vector<uint16_t>> rows = spans.get(m, q, err);
                if (!rows) {
                    ok = false;
                    break;
                }
                ++encoded;
                q.len = rows->size() / (size_t) m.dim();
                bool fits = (size_t) q.start + q.len <= r.ids.size() - 1;
                for (size_t i = 0; fits && i < q.len; ++i) fits = r.ids[(size_t) q.start + i] == img_tok;
                if (!fits) {
                    err = "a picture's span (" + std::to_string(q.len) + " tokens at " + std::to_string(q.start) +
                          ") is not a run of image tokens in the prompt";
                    ok = false;
                    break;
                }
                imgs.push_back({q.start, rows});
            }
            if (ok) {
                m.set_images(std::move(imgs));
                std::fprintf(stderr, "maya ds4 serve: %zu pictures, %d read (%.0f ms)\n", pics.size(), encoded,
                             now_ms() - te);
            }
            for (Pic& q : pics) q.rgb = {};   // (only the start, hash and span stay)
        }
        // the prompt in windows (PP <read> <total> <ms> <tok/s> <reuse> every 64 tokens), a checkpoint before its last
        // token and one after it: the next turn re-renders this one, and shares its start up to one of them
        const size_t total = r.ids.size();
        size_t done = from, next_pp = from + 64;
        m.set_progress([&](int64_t d) {   // (a run of chunks: each one through every layer)
            const double ms = now_ms() - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", done + (size_t) d, r.ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (done + (size_t) d - from) / ms : 0.0, from);
            std::fflush(stdout);
        });
        // (DS4_LAST_WINDOW=1: the last token in a window of its own after the checkpoint, as before - for A/B runs)
        static const bool last_window = std::getenv("DS4_LAST_WINDOW") != nullptr;
        while (ok && done < total) {
            const size_t n = std::min<size_t>((size_t) m.feed_unit(), total - done);
            // (the prompt's end: the last token in the same run, the checkpoint before it kept on the way)
            const bool fin = done + n == total;
            if (fin && resume) {   // (cilla: keeps the checkpoints at the answer's prompt for the chat's next turn)
                ok = m.feed(r.ids.data() + done, (int) n, 1, err);
            } else if (fin && last_window) {
                ok = (n < 2 || m.feed(r.ids.data() + done, (int) n - 1, 0, err));
                if (ok) m.checkpoint(1);
                ok = ok && m.feed(r.ids.data() + done + n - 1, 1, 1, err);
            } else
                ok = fin ? m.feed_mark(r.ids.data() + done, (int) n, 1, 1, err)
                         : m.feed(r.ids.data() + done, (int) n, 0, err);
            done += n;
            if (ok && (done >= next_pp || done == total)) {
                const double ms = now_ms() - t0;
                std::printf("PP %zu %zu %.1f %.1f %zu\n", done, r.ids.size(), ms,
                            ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
                std::fflush(stdout);
                next_pp = done + 64;
            }
        }
        m.set_images({});
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            std::printf("ERR %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            m.reset();
            held.clear();
            continue;
        }
        held = std::move(pics);
        if (!resume) m.checkpoint(0);
        const double t1 = now_ms();
        long long produced = 0;
        const char* finish = "length";
        const long long limit = r.max_new > 0 ? r.max_new : m.max_context() - (long long) r.ids.size();
        long long d_off = 0, d_acc = 0;
        bool cool = false;
        if (dump_prefix != nullptr) {
            if (dump_f != nullptr) std::fclose(dump_f);
            dump_f = std::fopen((std::string(dump_prefix) + "." + std::to_string(++n_req) + ".f32").c_str(), "wb");
            dump_rows();
        }
        // this request's penalties (a repeat penalty of 0 or below, or a value that is not a number: neutral) for every
        // pick of its answer, the drafts' verify included
        m.set_penalties(r.pen_n < 0 ? INT_MAX : r.pen_n,
                        std::isfinite(r.pen_repeat) && r.pen_repeat > 0.0f ? r.pen_repeat : 1.0f,
                        std::isfinite(r.pen_freq) ? r.pen_freq : 0.0f, std::isfinite(r.pen_present) ? r.pen_present : 0.0f);
        int tok = m.sample(r.temperature, r.top_p, r.top_k, r.min_p, r.seed, counter++);
        std::vector<int> got;
        // STAT for the Monitor every 8 tokens and at the end, over this answer so far: the GLM engine's keys (speed, the
        // VRAM hit, promotions a token) and DeepSeek's own - experts a token on the CPU lane, MB a token the storage
        // read (Linux), the n-gram memory's (Engram) read time a token, the drafts' acceptance
        const strata::ds4::Stats s_dec = m.stats();
        const long long io_dec = io_read_bytes();
        auto stat_line = [&]() {
            const strata::ds4::Stats& s = m.stats();
            const double ms = now_ms() - t1;
            const double fed = (double) std::max<int64_t>(1, s.tokens - s_dec.tokens);
            const double ex = (double) (s.experts - s_dec.experts), cx = (double) (s.cpu_experts - s_dec.cpu_experts),
                         sx = (double) (s.stream_experts - s_dec.stream_experts);
            std::string out = "STAT";
            char b[160];
            std::snprintf(b, sizeof b, " tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f ram=%.3f promo=%.3f engram_ms=%.3f",
                          ms > 0 ? 1000.0 * (double) produced / ms : 0.0, produced > 0 ? ms / (double) produced : 0.0,
                          ex > 0 ? 1.0 - (cx + sx) / ex : 1.0, cx / fed, (double) (s.ram_experts - s_dec.ram_experts) / fed,
                          (double) (s.swaps - s_dec.swaps) / fed, (s.engram_ms - s_dec.engram_ms) / fed);
            out += b;
            if (sx > 0) {   // (the PCIe lane: experts a token streamed to the GPU)
                std::snprintf(b, sizeof b, " pcie=%.3f", sx / fed);
                out += b;
            }
            const long long io = io_read_bytes();
            if (io >= 0 && io_dec >= 0) {
                std::snprintf(b, sizeof b, " disk_mb=%.3f", (double) (io - io_dec) / 1e6 / fed);
                out += b;
            }
            if (d_off > 0) {
                std::snprintf(b, sizeof b, " drafts=%.3f", (double) d_acc / (double) d_off);
                out += b;
            }
            if (s.skipped > s_dec.skipped) {   // (STRATA_DS4_DISK_SKIP_W: experts left out a token)
                std::snprintf(b, sizeof b, " skipped=%.3f", (double) (s.skipped - s_dec.skipped) / fed);
                out += b;
            }
            if (s.skip_on >= 0) {   // (... _AUTO: skipping now)
                std::snprintf(b, sizeof b, " skip_on=%d", s.skip_on);
                out += b;
            }
            std::printf("%s\n", out.c_str());
        };
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            std::printf("T %d\n", tok);
            ++produced;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            bool end = false;
            for (int e : eos) end = end || tok == e;
            if (end) {
                finish = "stop";
                break;
            }
            if (produced >= limit || m.pos() >= m.max_context()) break;
            if (got.empty() && in.pause.load()) {   // (cilla: the state holds all it wrote but this last token)
                finish = "cancel";
                paused_by[(size_t) conv] = 1;
                break;
            }
            if (got.empty()) {   // the next round: the drafts (if any) and the token after them
                const int room = (int) std::min<long long>(limit - produced, m.max_context() - m.pos());
                int off = 0, acc = 0;
                // a round whose drafts were all below the confidence floor: the next one does without the drafter
                // (a request's spec / dspark_conf keys over the engine's: the speed calibration measures both ways)
                const bool try_spec = (r.spec < 0 ? spec : r.spec != 0) && !cool;
                if (!round(m, tok, room, try_spec, r.conf >= 0 ? r.conf : conf_min, r.temperature, r.top_p, r.top_k,
                           r.min_p, r.seed, counter, got,
                           off, acc, err, r.cum)) {
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    finish = "cancel";
                    break;
                }
                d_off += off;
                d_acc += acc;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
                dump_rows();
            }
            tok = got.back();
            got.pop_back();
        }
        if (dump_f != nullptr) std::fflush(dump_f);
        // (a stop within accepted drafts leaves the state past it: the next prompt shares less with it, and goes back to
        //  the checkpoint at this prompt's end)
        stat_line();
        const double t2 = now_ms();
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", produced, r.ids.size(), t1 - t0, t2 - t1, finish,
                    d_acc, d_off, from);
        std::fflush(stdout);
        const auto& s = m.stats();
        std::fprintf(stderr,
                     "maya ds4 serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s); CPU lane %.1f%% of the experts\n",
                     r.ids.size(), from, r.ids.size() - from, t1 - t0,
                     t1 > t0 ? 1000.0 * (double) (r.ids.size() - from) / (t1 - t0) : 0.0, produced, t2 - t1,
                     t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0,
                     s.experts ? 100.0 * (double) s.cpu_experts / (double) s.experts : 0.0);
        {   // cilla: this answer's decode alone - the VRAM hit, CPU-lane / streamed experts a token, DS4_PROF's stages
            const double fed = (double) std::max<int64_t>(1, s.tokens - s_dec.tokens);
            const double ex = (double) (s.experts - s_dec.experts), cx = (double) (s.cpu_experts - s_dec.cpu_experts),
                         sx = (double) (s.stream_experts - s_dec.stream_experts);
            std::fprintf(stderr, "cilla decode: %lld tokens fed %.0f, experts %.1f a fed token, vram_hit %.4f, cpu %.2f, "
                                 "streamed %.2f, swaps %.2f a fed token\n", produced, fed, ex / fed,
                         ex > 0 ? 1.0 - (cx + sx) / ex : 1.0, cx / fed, sx / fed, (double) (s.swaps - s_dec.swaps) / fed);
            if (s.prof_tokens > s_dec.prof_tokens) {
                const double pt = (double) (s.prof_tokens - s_dec.prof_tokens);
                double sum = 0;
                for (int i = 0; i < 6; ++i) sum += (s.prof_ms[i] - s_dec.prof_ms[i]) / pt;
                std::fprintf(stderr, "cilla decode GPU per window (%.0f windows): %.2f ms - between layers %.2f, attention %.2f, "
                                     "routing + shared %.2f, experts gate/up %.2f, wait CPU lane %.2f, down + post %.2f\n",
                             pt, sum, (s.prof_ms[0] - s_dec.prof_ms[0]) / pt, (s.prof_ms[1] - s_dec.prof_ms[1]) / pt,
                             (s.prof_ms[2] - s_dec.prof_ms[2]) / pt, (s.prof_ms[3] - s_dec.prof_ms[3]) / pt,
                             (s.prof_ms[4] - s_dec.prof_ms[4]) / pt, (s.prof_ms[5] - s_dec.prof_ms[5]) / pt);
            }
        }
        if (d_off > 0) std::fprintf(stderr, "maya ds4 serve: drafts %lld of %lld kept; %s\n", d_acc, d_off,
                                    per_position(s_dec, s).c_str());
        m.save_profile();
    }
    m.save_profile();
    return 0;
}


// ---- cilla: several conversations at once (READY ... multi=<lanes>): requests tagged rid=<id>, their lines T / PP /
// DONE / ERR <rid> ...; a prompt is read alone (the chunked path takes every GPU), then the conversations decode
// together - each step one window per conversation, pipelined over the layer split (Model::step_multi). A conversation
// alone decodes as before, with the drafter. Each holds one of the engine's conversation slots while it is in flight.
int serve_multi(strata::ds4::Model& m, const std::string& eos_list, bool spec, float conf_min) {
    std::vector<int> eos = parse_ids(eos_list);
    const int N = m.lanes();
    for (const std::string& l : m.info_lines()) std::printf("%s\n", l.c_str());
    std::printf("READY %lld stop pause multi=%d\n", (long long) m.max_context(), N);
    std::fflush(stdout);
    std::fprintf(stderr, "maya ds4 serve: %d conversations at once (cilla lanes); GEN/GENI with rid=<id>, STOP/PAUSE <id>\n", N);
    Lines in;
    in.start();
    const int NC = m.conversations();
    std::vector<std::vector<Pic>> held_by((size_t) NC);
    std::vector<char> paused_by((size_t) NC, 0), busy((size_t) NC, 0);
    SpanCache spans;
    const int img_tok = m.image_token();
    struct Act {
        Req r;
        int conv = 0, tok = -1;
        long long produced = 0, limit = 0, d_off = 0, d_acc = 0;
        unsigned long long counter = 0;
        double t0 = 0, t1 = 0;
        size_t from = 0;
        bool cool = false, seeded = true, done = false;
        const char* finish = "length";
    };
    std::vector<Act> act;
    std::deque<std::string> pending;
    unsigned long long counter0 = 0;
    auto is_eos = [&](int t) {
        for (int e : eos)
            if (t == e) return true;
        return false;
    };
    auto pen = [&](const Req& r) {
        m.set_penalties(r.pen_n < 0 ? INT_MAX : r.pen_n,
                        std::isfinite(r.pen_repeat) && r.pen_repeat > 0.0f ? r.pen_repeat : 1.0f,
                        std::isfinite(r.pen_freq) ? r.pen_freq : 0.0f, std::isfinite(r.pen_present) ? r.pen_present : 0.0f);
    };
    auto end_act = [&](Act& a) {
        const double t2 = now_ms();
        std::printf("DONE %lld %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", a.r.rid, a.produced, a.r.ids.size(),
                    a.t1 - a.t0, t2 - a.t1, a.finish, a.d_acc, a.d_off, a.from);
        std::fflush(stdout);
        std::fprintf(stderr,
                     "maya ds4 serve: [rid %lld conv %d] prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), "
                     "%lld generated in %.0f ms (%.2f tok/s)\n",
                     a.r.rid, a.conv, a.r.ids.size(), a.from, a.r.ids.size() - a.from, a.t1 - a.t0,
                     a.t1 > a.t0 ? 1000.0 * (double) (a.r.ids.size() - a.from) / (a.t1 - a.t0) : 0.0, a.produced,
                     t2 - a.t1, t2 > a.t1 ? 1000.0 * (double) a.produced / (t2 - a.t1) : 0.0);
        busy[(size_t) a.conv] = 0;
        a.done = true;
    };
    // token t written by a (the state holds everything before it): T line, then whether a ends here
    auto emit = [&](Act& a, int t, bool last_of_round) {
        std::printf("T %lld %d\n", a.r.rid, t);
        ++a.produced;
        a.tok = t;
        if (in.take(in.stop_rid, a.r.rid)) {
            a.finish = "cancel";
            return true;
        }
        if (is_eos(t)) {
            a.finish = "stop";
            return true;
        }
        if (a.produced >= a.limit || m.conversation_pos(a.conv) + 1 >= m.max_context()) {
            a.finish = "length";
            return true;
        }
        if (last_of_round && in.take(in.pause_rid, a.r.rid)) {   // (the state holds all it wrote but this last token)
            a.finish = "cancel";
            paused_by[(size_t) a.conv] = 1;
            return true;
        }
        return false;
    };
    // a request's prompt, read alone; then its first token
    auto admit = [&](const std::string& line) {
        const bool geni = line.rfind("GENI ", 0) == 0;
        Act a;
        Req& r = a.r;
        if (!parse_gen(line, r, geni)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        const long long rid = r.rid;
        auto fail = [&](const std::string& e) {
            std::printf("ERR %lld %s\n", rid, e.c_str());
            std::printf("DONE %lld 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", rid, r.ids.size());
            std::fflush(stdout);
        };
        m.begin_request();
        a.t0 = now_ms();
        std::vector<Pic> pics;
        {
            std::string e;
            if (geni && img_tok < 0) e = "this engine has no vision tower, so it cannot read pictures";
            else if (geni && read_pics(r.file, pics, e))
                for (const Pic& q : pics)
                    if (q.start < 0 || (size_t) q.start >= r.ids.size() || r.ids[(size_t) q.start] != img_tok)
                        e = "a picture's span does not start at an image token of the prompt";
            if (!e.empty()) return fail(e);
        }
        if (r.ids.empty()) return fail("an empty prompt");
        if ((int64_t) r.ids.size() + std::max(0LL, r.max_new) > m.max_context())
            return fail("the prompt (" + std::to_string(r.ids.size()) + " tokens) and max_new " + std::to_string(r.max_new) +
                        " are more than the context " + std::to_string(m.max_context()));
        a.conv = m.pick_conversation_free(r.ids, busy);
        if (a.conv < 0) {
            std::string busy_n;
            for (size_t i = 0; i < busy.size(); ++i) busy_n += busy[i] ? "1" : "0";
            return fail("no free conversation slot (" + std::to_string(busy.size()) + " slots, busy " + busy_n + ")");
        }
        std::vector<Pic>& held = held_by[(size_t) a.conv];
        const auto& st = m.tokens();
        const size_t n_state = st.size();
        size_t c = 0;
        while (c < n_state && c < r.ids.size() && st[c] == r.ids[c]) ++c;
        auto same = [](const std::vector<Pic>& v, const Pic& q) {
            for (const Pic& x : v)
                if (x.start == q.start && x.hash == q.hash) return true;
            return false;
        };
        for (const Pic& q : pics)
            if ((size_t) q.start < c && !same(held, q)) c = (size_t) q.start;
        for (const Pic& q : held)
            if ((size_t) q.start < c && !same(pics, q)) c = (size_t) q.start;
        size_t from = 0;
        if (c == n_state && n_state < r.ids.size()) from = n_state;
        else if (m.restorable((int64_t) std::min(c, r.ids.size() - 1)) >= 0 &&
                 m.restore((int64_t) std::min(c, r.ids.size() - 1)))
            from = (size_t) m.pos();
        else m.reset();
        if (from == 0) held.clear();
        const bool resume = paused_by[(size_t) a.conv] && from == n_state && r.ids.size() == n_state + 1;
        paused_by[(size_t) a.conv] = 0;
        std::string err;
        bool ok = true;
        if (!pics.empty()) {
            std::vector<strata::ds4::Image> imgs;
            for (Pic& q : pics) {
                for (const Pic& x : held)
                    if (x.start == q.start && x.hash == q.hash) q.len = x.len;
                if (q.len > 0 && (size_t) q.start + q.len <= from) continue;
                std::shared_ptr<const std::vector<uint16_t>> rows = spans.get(m, q, err);
                if (!rows) {
                    ok = false;
                    break;
                }
                q.len = rows->size() / (size_t) m.dim();
                bool fits = (size_t) q.start + q.len <= r.ids.size() - 1;
                for (size_t i = 0; fits && i < q.len; ++i) fits = r.ids[(size_t) q.start + i] == img_tok;
                if (!fits) {
                    err = "a picture's span is not a run of image tokens in the prompt";
                    ok = false;
                    break;
                }
                imgs.push_back({q.start, rows});
            }
            if (ok) m.set_images(std::move(imgs));
            for (Pic& q : pics) q.rgb = {};
        }
        const size_t total = r.ids.size();
        size_t done = from, next_pp = from + 64;
        const double t0 = a.t0;
        m.set_progress([&](int64_t d) {
            const double ms = now_ms() - t0;
            std::printf("PP %lld %zu %zu %.1f %.1f %zu\n", rid, done + (size_t) d, total, ms,
                        ms > 0 ? 1000.0 * (double) (done + (size_t) d - from) / ms : 0.0, from);
            std::fflush(stdout);
        });
        while (ok && done < total) {
            const size_t n = std::min<size_t>((size_t) m.feed_unit(), total - done);
            const bool fin = done + n == total;
            if (fin && resume) ok = m.feed(r.ids.data() + done, (int) n, 1, err);
            else ok = fin ? m.feed_mark(r.ids.data() + done, (int) n, 1, 1, err) : m.feed(r.ids.data() + done, (int) n, 0, err);
            done += n;
            if (ok && (done >= next_pp || done == total)) {
                const double ms = now_ms() - t0;
                std::printf("PP %lld %zu %zu %.1f %.1f %zu\n", rid, done, total, ms,
                            ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
                std::fflush(stdout);
                next_pp = done + 64;
            }
        }
        m.set_progress(nullptr);
        m.set_images({});
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            m.reset();
            held.clear();
            return fail(err);
        }
        held = std::move(pics);
        if (!resume) m.checkpoint(0);
        a.t1 = now_ms();
        a.from = from;
        a.limit = r.max_new > 0 ? r.max_new : m.max_context() - (long long) r.ids.size();
        a.counter = counter0;
        counter0 += 1ull << 32;
        busy[(size_t) a.conv] = 1;
        pen(r);
        const int t = m.sample(r.temperature, r.top_p, r.top_k, r.min_p, r.seed, a.counter++);
        if (emit(a, t, true)) end_act(a);
        else act.push_back(std::move(a));
        std::fflush(stdout);
    };
    for (;;) {
        // the requests that came in (waiting for one when nothing is in flight)
        std::string line;
        bool quit = false;
        if (act.empty() && pending.empty()) {
            if (!in.next(line)) break;
            if (line == "QUIT") break;
            pending.push_back(line);
        }
        while (in.try_next(line)) {
            if (line == "QUIT") {
                quit = true;
                break;
            }
            if (line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) {
                std::printf("ERR unsupported command '%s'\n", line.substr(0, line.find(' ')).c_str());
                std::fflush(stdout);
                continue;
            }
            pending.push_back(line);
        }
        if (quit) break;
        if (!pending.empty()) {   // (the busy slots follow from the requests in flight, not from a counter that can drift)
            std::fill(busy.begin(), busy.end(), 0);
            for (const Act& a : act) busy[(size_t) a.conv] = 1;
        }
        while (!pending.empty() && (int) act.size() < N) {
            const std::string l = pending.front();
            pending.pop_front();
            admit(l);
            std::fill(busy.begin(), busy.end(), 0);
            for (const Act& a : act) busy[(size_t) a.conv] = 1;
        }
        // a STOP / PAUSE for a conversation in flight between its steps
        for (Act& a : act) {
            if (in.take(in.stop_rid, a.r.rid)) {
                a.finish = "cancel";
                end_act(a);
            } else if (in.take(in.pause_rid, a.r.rid)) {
                a.finish = "cancel";
                paused_by[(size_t) a.conv] = 1;
                end_act(a);
            }
        }
        act.erase(std::remove_if(act.begin(), act.end(), [](const Act& a) { return a.done; }), act.end());
        if (act.empty()) continue;
        std::string err;
        if (act.size() == 1) {   // alone: a round with the drafter, as the single-conversation serve
            Act& a = act[0];
            m.use_conversation(a.conv);
            pen(a.r);
            const Req& r = a.r;
            const int room = (int) std::min<long long>(a.limit - a.produced, m.max_context() - m.pos());
            int off = 0, acc = 0;
            const bool try_spec = (r.spec < 0 ? spec : r.spec != 0) && !a.cool && a.seeded;
            std::vector<int> got;
            if (!round(m, a.tok, room, try_spec, r.conf >= 0 ? r.conf : conf_min, r.temperature, r.top_p, r.top_k,
                       r.min_p, r.seed, a.counter, got, off, acc, err, r.cum)) {
                std::fprintf(stderr, "ds4: %s\n", err.c_str());
                a.finish = "cancel";
                end_act(a);
            } else {
                a.d_off += off;
                a.d_acc += acc;
                a.cool = try_spec && off == 0;
                a.seeded = true;   // (the round fed the trunk: the drafter is seeded from this conversation)
                for (size_t i = 0; i < got.size(); ++i)
                    if (emit(a, got[i], i + 1 == got.size())) {
                        end_act(a);
                        break;
                    }
            }
        } else {   // together: one window each, pipelined
            std::vector<int> cv, tk;
            for (const Act& a : act) {
                cv.push_back(a.conv);
                tk.push_back(a.tok);
            }
            if (!m.step_multi(cv, tk, err)) {
                std::fprintf(stderr, "ds4: step_multi: %s\n", err.c_str());
                for (Act& a : act) {
                    std::printf("ERR %lld %s\n", a.r.rid, err.c_str());
                    a.finish = "cancel";
                    end_act(a);
                }
            } else
                for (size_t i = 0; i < act.size(); ++i) {
                    Act& a = act[i];
                    a.seeded = false;   // (the drafter's taps are another conversation's now: a plain step first)
                    pen(a.r);
                    const int t = m.sample_lane((int) i, a.conv, a.r.temperature, a.r.top_p, a.r.top_k, a.r.min_p,
                                                a.r.seed, a.counter++);
                    if (emit(a, t, true)) end_act(a);
                }
        }
        std::fflush(stdout);
        act.erase(std::remove_if(act.begin(), act.end(), [](const Act& a) { return a.done; }), act.end());
    }
    m.save_profile();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGSEGV, ds4_fatal_backtrace);
    std::signal(SIGABRT, ds4_fatal_backtrace);
    std::signal(SIGBUS, ds4_fatal_backtrace);
    strata::platform::env_aliases();   // (MAYA_* settings under their older names too)
    strata::ds4::Options o;
    std::string ids_s, dump, eos = "1";
    int n = 0, window = 16;
    bool spec = true;
    float temp = 0.0f;   // the CLI's sampling (0: greedy)
    unsigned long long cli_seed = 1;
    bool serve_mode = false;
    std::string images, enc_rgb, enc_out;
    int enc_w = 0, enc_h = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--pack") o.pack = val("--pack");
        else if (a == "--gpus") {
            for (int g : parse_ids(val("--gpus"))) o.gpus.push_back(g);
        } else if (a == "--max-context") o.max_context = std::atoll(val("--max-context").c_str());
        // cilla: DS4_MMQ=1 allocates each device's MMQ scratch outside the expert pool, so keep that room free
        // (otherwise a pool filled to the reserve leaves too little and the engine's own scratch allocations fail)
        else if (a == "--reserve-mib") o.reserve_mib = std::atoll(val("--reserve-mib").c_str());
        else if (a == "--vram-experts") o.vram_experts = std::atof(val("--vram-experts").c_str());
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val("--cpu-threads").c_str());
        else if (a == "--expert-profile") o.profile = val("--expert-profile");
        else if (a == "--eos") eos = val("--eos");
        else if (a == "--no-expert-cache") o.expert_cache = false;
        else if (a == "--dspark") {
            const std::string v = val("--dspark");
            spec = v != "off";
            o.dspark = v != "none";   // none: not even loaded
        } else if (a == "--dspark-conf") o.dspark_conf = (float) std::atof(val("--dspark-conf").c_str());
        else if (a == "--tp") o.tp = std::max(1, std::atoi(val("--tp").c_str()));
        else if (a == "--prefill-chunk") o.prefill_chunk = std::max(0, std::atoi(val("--prefill-chunk").c_str()));
        else if (a == "--threads") o.threads = val("--threads") != "off";
        else if (a == "--vision") o.vision = val("--vision") != "off";
        else if (a == "--mmap-experts") {
            const std::string v = val("--mmap-experts");
            o.mmap_experts = v == "on" ? 1 : v == "off" ? 0 : -1;
        } else if (a == "--images") images = val("--images");
        else if (a == "--encode-image") enc_rgb = val("--encode-image");
        else if (a == "--dump-image") enc_out = val("--dump-image");
        else if (a == "--size") {
            const std::string v = val("--size");
            enc_w = std::atoi(v.c_str());
            enc_h = v.find('x') != std::string::npos ? std::atoi(v.c_str() + v.find('x') + 1) : 0;
        }
        else if (a == "--temperature") temp = (float) std::atof(val("--temperature").c_str());
        else if (a == "--seed") cli_seed = std::strtoull(val("--seed").c_str(), nullptr, 10);
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--ids") ids_s = val("--ids");
        else if (a == "--n") n = std::atoi(val("--n").c_str());
        else if (a == "--window") window = std::max(1, std::atoi(val("--window").c_str()));   // (> 16: the chunked prompt path)
        else if (a == "--dump-logits") dump = val("--dump-logits");
        else if (a == "--serve") serve_mode = true;
        else if (a == "--layer-split") (void) val("--layer-split");   // the server's (this engine balances its own split)
        else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (o.pack.empty() || (!serve_mode && ids_s.empty() && enc_rgb.empty())) {
        usage();
        return 2;
    }
    if (o.profile.empty()) o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none") o.profile.clear();
    o.vision_only = !enc_rgb.empty();
    if (std::getenv("DS4_MMQ") && std::atoi(std::getenv("DS4_MMQ")) != 0)
        o.reserve_mib += 320;   // (cilla: room for the MMQ path's per-device scratch - 8192 rows of gate/up/down
                                 // scratch + the q8 activations, ~200 MB - outside the expert pool)
    strata::ds4::Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
        return 1;
    }
    if (!o.vision_only) std::fprintf(stderr, "%s", m.describe().c_str());
    if (serve_mode) {   // (cilla: DS4_MULTI=0 keeps the one-at-a-time serve with lanes loaded)
        const char* mv = std::getenv("DS4_MULTI");
        if (m.lanes() > 1 && (mv == nullptr || std::atoi(mv) != 0)) return serve_multi(m, eos, spec, o.dspark_conf);
        return serve(m, eos, spec, o.dspark_conf);
    }

    if (!enc_rgb.empty()) {   // the vision tower alone: a picture's span rows (BF16 [rows][dim]) into --dump-image
        std::vector<uint8_t> px((size_t) std::max(0, enc_w) * std::max(0, enc_h) * 3);
        std::FILE* f = std::fopen(enc_rgb.c_str(), "rb");
        const bool got = f != nullptr && !px.empty() && std::fread(px.data(), 1, px.size(), f) == px.size();
        if (f != nullptr) std::fclose(f);
        if (!got) {
            std::fprintf(stderr, "maya-ds4: %s: expected %d x %d RGB bytes (--size WxH)\n", enc_rgb.c_str(), enc_w, enc_h);
            return 1;
        }
        std::vector<uint16_t> span;
        for (int k = 0; k < 2; ++k) {   // (the second: warm)
            const double te = now_ms();
            if (!m.encode_image(px.data(), enc_w, enc_h, span, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "vision: %d x %d pixels -> a span of %zu rows in %.0f ms\n", enc_w, enc_h,
                         span.size() / (size_t) m.dim(), now_ms() - te);
        }
        std::FILE* o2 = enc_out.empty() ? nullptr : std::fopen(enc_out.c_str(), "wb");
        if (o2 != nullptr) {
            std::fwrite(span.data(), 2, span.size(), o2);
            std::fclose(o2);
        }
        return 0;
    }
    const std::vector<int> ids = parse_ids(ids_s);
    if (!images.empty()) {   // --images: a GENI file's pictures, for the prompt's spans
        std::vector<Pic> pics;
        if (!read_pics(images, pics, err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        std::vector<strata::ds4::Image> imgs;
        for (const Pic& q : pics) {
            auto rows = std::make_shared<std::vector<uint16_t>>();
            const double te = now_ms();
            if (!m.encode_image(q.rgb.data(), q.width, q.height, *rows, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "picture at %lld: %d x %d pixels, %zu tokens, %.0f ms\n", (long long) q.start, q.width,
                         q.height, rows->size() / (size_t) m.dim(), now_ms() - te);
            imgs.push_back({q.start, rows});
        }
        m.set_images(std::move(imgs));
    }
    std::FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    std::vector<float> lg;
    m.begin_request();
    const double t0 = now_ms();
    for (size_t a = 0; a < ids.size();) {
        const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
        const bool last = a + (size_t) T == ids.size();
        if (!m.feed(ids.data() + a, T, df != nullptr ? T : (last ? 1 : 0), err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
        a += (size_t) T;
    }
    m.set_images({});
    const double t1 = now_ms();
    std::fprintf(stderr, "prompt: %zu tokens in %.0f ms (%.2f tok/s, windows of %d)\n", ids.size(), t1 - t0,
                 1000.0 * (double) ids.size() / (t1 - t0), window);
    const strata::ds4::Stats s0 = m.stats();
    if (s0.eg_rows > 0)
        std::fprintf(stderr, "prompt's Engram rows: %lld, %.1f%% without a read (repeats and the row cache), %lld read\n",
                     (long long) s0.eg_rows, 100.0 * (double) s0.eg_hits / (double) s0.eg_rows, (long long) s0.eg_reads);
    const long long io0 = io_read_bytes();
    std::vector<int> out;
    unsigned long long counter = 0;
    long long d_off = 0, d_acc = 0, rounds = 0;
    bool cool = false;
    if (n > 0) {
        int tok = m.sample(temp, 1.0f, temp > 0 ? 64 : 1, 0.0f, cli_seed, counter++);
        std::vector<int> got;
        for (;;) {
            out.push_back(tok);
            std::printf("%d,", tok);
            std::fflush(stdout);
            if (tok == 1 || (int) out.size() >= n) break;
            if (got.empty()) {
                int off = 0, acc = 0;
                const bool try_spec = spec && !cool;
                if (!round(m, tok, n - (int) out.size(), try_spec, o.dspark_conf, temp, 1.0f, temp > 0 ? 64 : 1, 0.0f,
                           cli_seed, counter, got, off, acc, err)) {
                    std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                    return 1;
                }
                d_off += off;
                d_acc += acc;
                ++rounds;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
            }
            tok = got.back();
            got.pop_back();
        }
        std::printf("\n");
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
    }
    if (spec && rounds > 0) {
        const auto& s1 = m.stats();
        const double nv = (double) std::max<int64_t>(1, s1.drafts_rounds - s0.drafts_rounds);
        std::fprintf(stderr, "dspark: %lld rounds, %lld drafts offered, %lld accepted (%.2f tokens a round); drafting "
                             "%.2f ms a round, verifying %.1f ms in all (%lld windows)\n",
                     rounds, d_off, d_acc, (double) out.size() / (double) rounds, (s1.draft_ms - s0.draft_ms) / nv,
                     s1.verify_ms - s0.verify_ms, (long long) (s1.drafts_offered > 0 ? 0 : 0));
        std::fprintf(stderr, "dspark %s\n", per_position(s0, s1).c_str());
    }
    const double t2 = now_ms();
    const auto& s = m.stats();
    const double nt = std::max<double>(1.0, (double) (s.tokens - s0.tokens));
    std::fprintf(stderr, "decode: %zu tokens in %.0f ms (%.2f tok/s); CPU lane: %lld of %lld experts, streamed to the GPU: "
                         "%lld; the cache moved %lld (%lld in all)\n", out.size(), t2 - t1,
                 out.size() > 1 ? 1000.0 * (double) (out.size() - 1) / (t2 - t1) : 0.0,
                 (long long) (s.cpu_experts - s0.cpu_experts), (long long) (s.experts - s0.experts),
                 (long long) (s.stream_experts - s0.stream_experts), (long long) (s.swaps - s0.swaps),
                 (long long) s.swaps);
    std::fprintf(stderr, "decode per token: layers %.2f ms (of it: routing waits %.2f, CPU lane %.2f, the rest host launches), "
                         "head wait %.2f, Engram reads %.2f\n",
                 (s.layer_ms - s0.layer_ms) / nt, (s.gpu_wait_ms - s0.gpu_wait_ms) / nt, (s.cpu_ms - s0.cpu_ms) / nt,
                 (s.head_ms - s0.head_ms) / nt, (s.engram_ms - s0.engram_ms) / nt);
    if (s.eg_rows > s0.eg_rows) {   // the Engram rows: how many the row cache served, the reads' time behind the layers
        const double rows = (double) (s.eg_rows - s0.eg_rows);
        std::fprintf(stderr, "Engram per token: %.0f rows, %.1f%% without a read (the row cache holds %lld of %lld), "
                             "%.1f read, the reads %.2f ms from asking to the last one (left to wait: %.2f)\n",
                     rows / nt, 100.0 * (double) (s.eg_hits - s0.eg_hits) / rows, (long long) s.eg_cached,
                     (long long) s.eg_cache_cap, (double) (s.eg_reads - s0.eg_reads) / nt, (s.eg_io_ms - s0.eg_io_ms) / nt,
                     (s.engram_ms - s0.engram_ms) / nt);
    }
    const long long io1 = io_read_bytes();
    if (io0 >= 0 && io1 >= 0)
        std::fprintf(stderr, "decode reads from the disk: %.1f MB a token\n", (double) (io1 - io0) / 1e6 / nt);
    if (s.skipped > s0.skipped)   // STRATA_DS4_DISK_SKIP_W
        std::fprintf(stderr, "decode left out %.1f experts a token not in RAM (%.1f%% of the routed)\n",
                     (double) (s.skipped - s0.skipped) / nt,
                     100.0 * (double) (s.skipped - s0.skipped) / std::max<double>(1.0, (double) (s.experts - s0.experts)));
    if (s.skip_on >= 0)
        std::fprintf(stderr, "DISK_SKIP auto: %s at the end, %lld switches\n", s.skip_on ? "on" : "off",
                     (long long) s.skip_switches);
    if (s.ahead_picks > s0.ahead_picks) {   // lookahead
        const double p = (double) (s.ahead_picks - s0.ahead_picks), c = (double) (s.ahead_cold - s0.ahead_cold);
        std::fprintf(stderr, "lookahead: the guesses a layer before held %.1f%% of the picks (their top k %.1f%%, two layers "
                             "before %.1f%%); file-read picks asked for early %.1f%% (%lld of %lld); %.1f experts asked "
                             "for early a token\n",
                     100.0 * (double) (s.ahead_hit - s0.ahead_hit) / p, 100.0 * (double) (s.ahead_hit_k - s0.ahead_hit_k) / p,
                     100.0 * (double) (s.ahead_hit2 - s0.ahead_hit2) / p,
                     c > 0 ? 100.0 * (double) (s.ahead_cold_hit - s0.ahead_cold_hit) / c : 0.0,
                     (long long) (s.ahead_cold_hit - s0.ahead_cold_hit), (long long) (s.ahead_cold - s0.ahead_cold),
                     (double) (s.ahead_pf - s0.ahead_pf) / nt);
        std::string rk = "lookahead: picked by the guess's rank (1 ..):";
        for (int i = 0; i < 16; ++i) {
            const long long n = (long long) (s.ahead_rank_n[i] - s0.ahead_rank_n[i]);
            if (n <= 0) break;
            char b[24];
            std::snprintf(b, sizeof b, " %.0f%%", 100.0 * (double) (s.ahead_rank_hit[i] - s0.ahead_rank_hit[i]) / (double) n);
            rk += b;
        }
        std::fprintf(stderr, "%s\n", rk.c_str());
    }
    if (s.prof_tokens > s0.prof_tokens) {   // DS4_PROF
        const double pt = (double) (s.prof_tokens - s0.prof_tokens);
        double sum = 0;
        for (int i = 0; i < 6; ++i) sum += (s.prof_ms[i] - s0.prof_ms[i]) / pt;
        std::fprintf(stderr, "GPU per token (DS4_PROF): %.2f ms - between layers %.2f, attention %.2f, routing + shared "
                             "expert %.2f, experts gate/up %.2f, waiting for the CPU lane %.2f, down + post %.2f\n", sum,
                     (s.prof_ms[0] - s0.prof_ms[0]) / pt, (s.prof_ms[1] - s0.prof_ms[1]) / pt,
                     (s.prof_ms[2] - s0.prof_ms[2]) / pt, (s.prof_ms[3] - s0.prof_ms[3]) / pt,
                     (s.prof_ms[4] - s0.prof_ms[4]) / pt, (s.prof_ms[5] - s0.prof_ms[5]) / pt);
    }
    if (df != nullptr) std::fclose(df);
    m.save_profile();
    return 0;
}
