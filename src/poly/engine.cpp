// src/poly/engine.cpp - see the header.  What every architecture shares: the files, the weights' way to VRAM, the
// routed experts (the cache, the CPU, a prompt batch), the sequence's state, and the calls that feed tokens.
// R56: windows.h comes in further down for the Stage B sidecar mapping; its min/max macros would break this file.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "engine.hpp"

#include "deepseek4.hpp"
#include "gemma4.hpp"
#include "glm5next.hpp"
#include "lfm2moe.hpp"
#include "mmq.hpp"
#include "nemotronhmoe.hpp"
#include "qwen35.hpp"
#include "runtime.hpp"
#include "threads.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <new>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

namespace strata::poly {
namespace {

// A launch's tiles are as wide as the most rows one of its matrices has, up to 128, and every matrix in it
// computes whole tiles.  So a prompt batch's cached experts go by how many rows they have (moe_rows()): a launch
// for the experts of up to 32 rows, one for those of up to 128, and one for the experts with more.  Narrower
// widths saved less work than their launches cost.
constexpr int kTile[] = {32, 128};
constexpr int kClasses = (int) (sizeof(kTile) / sizeof(kTile[0])) + 1;

// STRATA_SCRATCH_TILE: the most compact rows one batch of experts may hold.  A prompt batch's gu_/hx_/mq2_ keep
// only this many rows instead of rows_ * n_used_; the experts run a batch at a time, each batch a run of experts
// whose rows lie one interval [r0, r0 + rows).  8192 rows is a few hundred MiB at this model's expert widths.
constexpr size_t kScratchRows = 8192;

// STRATA_DRAFT_CHUNKED: the draft block's rows are primed a chunk of tokens at a time (three row sets - the normed
// embedding, the normed model output, and the block's residual - of at most kWayDraftChunk + kStageRows - 1 tokens)
// in a scratch of their own, instead of three whole-call row sets inside r_.  r_ then keeps the size a run without
// the draft block has, and the MTP's VRAM falls by the row sets it used to hold for it.  Off by default.
constexpr int kWayDraftChunk = 512;
constexpr int kDraftChunkMax = kWayDraftChunk + 31;   // a chunk's tokens: the remainder after a chunk joins the one before
bool draft_chunked() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DRAFT_CHUNKED");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

// STRATA_PROMPT_BORROW: the most of the expert cache's tail slots the prompt path may be lent, and the VRAM a run
// keeps free whatever it lends (the desktop's, as R21 keeps it): a lend beyond either is refused loudly.
constexpr int kBorrowMaxSlots = 2048;
constexpr int kBorrowMinFreeMib = 1024;

int tile_class(int rows) {
    int c = 0;
    while (c < kClasses - 1 && rows > kTile[c]) ++c;
    return c;
}

// A container's CPU quota is not in the affinity mask: a 25-CPU pod on a 208-thread host shows 208.
int usable_cpus() {
    long long n = (long long) std::thread::hardware_concurrency();
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) n = CPU_COUNT(&set);
    long long quota = -1, period = 0;
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {   // "max 100000" when there is no quota
        if (std::fscanf(f, "%lld %lld", &quota, &period) != 2) quota = -1;
        std::fclose(f);
    } else if (FILE* q = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
        if (std::fscanf(q, "%lld", &quota) != 1) quota = -1;
        std::fclose(q);
        if (FILE* p = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
            if (std::fscanf(p, "%lld", &period) != 1) period = 0;
            std::fclose(p);
        }
    }
    if (quota > 0 && period > 0) n = std::min(n, std::max(1LL, quota / period));
#endif
    return (int) std::max(n, 1LL);
}

// On a machine with several NUMA nodes the file cache's pages are on the node whose CPU read them first, and the
// CPU reads the experts at the speed of memory: from the other node that is slower (29.4 against 22.5 ms of CPU
// experts a token on a two-socket Xeon).  `spans` are the experts in the mapped files.  When most of their pages
// are on one node, and the process may use `threads` CPUs of it, this thread is kept on that node's CPUs, and so
// are the threads it starts from then on.  Returns the node, or -1 when nothing was changed.
int stay_near(const std::vector<std::pair<const uint8_t*, size_t>>& spans, const std::vector<std::string>& files,
              int threads, double& share) {
#ifdef __linux__
    if (access("/sys/devices/system/node/node1", F_OK) != 0) return -1;   // one node
    // A page shows in numa_maps once it is mapped: touch a few thousand, spread over the experts.
    size_t total = 0;
    for (const auto& s : spans) total += s.second;
    const size_t step = std::max<size_t>(total / 4096, 4096);
    volatile uint8_t seen = 0;
    for (const auto& s : spans)
        for (size_t at = 0; at < s.second; at += step) seen = (uint8_t) (seen + s.first[at]);

    std::vector<std::string> real;
    for (const std::string& f : files)
        if (char* r = realpath(f.c_str(), nullptr)) {
            real.emplace_back(r);
            std::free(r);
        }
    std::vector<long long> pages;   // per node, of the model's files
    if (FILE* f = std::fopen("/proc/self/numa_maps", "r")) {
        char* line = nullptr;
        size_t cap = 0;
        while (getline(&line, &cap, f) > 0) {
            const char* file = std::strstr(line, " file=");
            if (!file) continue;
            file += 6;
            const size_t len = std::strcspn(file, " \n");
            bool ours = false;
            for (const std::string& r : real) ours = ours || (r.size() == len && r.compare(0, len, file, len) == 0);
            for (const char* p = file + len; ours && (p = std::strstr(p, " N")) != nullptr; ++p) {
                int node = 0;
                long long n = 0;
                if (std::sscanf(p, " N%d=%lld", &node, &n) != 2 || node < 0 || node > 1023) continue;
                if ((size_t) node >= pages.size()) pages.resize((size_t) node + 1, 0);
                pages[(size_t) node] += n;
            }
        }
        std::free(line);
        std::fclose(f);
    }
    long long all = 0;
    for (long long n : pages) all += n;
    if (all == 0) return -1;
    const int best = (int) (std::max_element(pages.begin(), pages.end()) - pages.begin());
    share = (double) pages[(size_t) best] / (double) all;
    if (share < 0.6) return -1;   // split between the nodes: one is as good as another

    cpu_set_t may, node;
    CPU_ZERO(&node);
    const std::string list = "/sys/devices/system/node/node" + std::to_string(best) + "/cpulist";
    if (FILE* f = std::fopen(list.c_str(), "r")) {   // "52-103,156-207"
        int a = 0, b = 0;
        while (std::fscanf(f, "%d", &a) == 1) {
            b = a;
            int c = std::fgetc(f);
            if (c == '-' && std::fscanf(f, "%d", &b) == 1) c = std::fgetc(f);
            for (int i = a; i <= b && i < CPU_SETSIZE; ++i) CPU_SET(i, &node);
            if (c != ',') break;
        }
        std::fclose(f);
    }
    if (sched_getaffinity(0, sizeof(may), &may) != 0) return -1;
    CPU_AND(&node, &node, &may);
    if (CPU_COUNT(&node) < threads || sched_setaffinity(0, sizeof(node), &node) != 0) return -1;
    return best;
#else
    (void) spans, (void) files, (void) threads, (void) share;
    return -1;
#endif
}

// The mapped file's pages come into this process's page table as they are first read, a fault every few pages,
// and an expert read through those faults arrives at half the speed: 21.0 against 11.0 ms a token for GLM 5.3
// Flash's CPU experts on the two-socket Xeon, while every expert a token meets is new to the process.  So one
// thread reads a byte of every page of the experts, once, beside whatever the engine is doing: the 100 GiB of a
// model that the file cache holds take it 4 s.  What the cache does not hold is read from the disk here, not
// while a token waits.
void map_pages(const std::vector<std::pair<const uint8_t*, size_t>>& spans, const std::atomic<bool>& quit) {
    constexpr size_t kPage = 4096, kLook = size_t(1) << 24;   // `quit` is looked at every 16 MiB
    volatile uint8_t seen = 0;
    for (const auto& s : spans) {
        for (size_t at = 0; at < s.second; at += kPage) {
            if (at % kLook == 0 && quit.load(std::memory_order_relaxed)) return;
            seen = (uint8_t) (seen + s.first[at]);
        }
        if (s.second > 0) seen = (uint8_t) (seen + s.first[s.second - 1]);   // its last page, when it starts in one
    }
}

// Lines of "layer expert count".  A missing file is not an error: the first run writes it.
void read_profile(const std::string& path, int n_layer, int n_expert, std::vector<int64_t>& out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {   // the first run writes it; still say so, an unranked cache used to be silent
        std::fprintf(stderr, "strata-poly: no profile at %s, the expert cache is unranked\n", path.c_str());
        return;
    }
    out.assign((size_t) n_layer * n_expert, 0);
    int il = 0, e = 0;
    long long n = 0;
    long long loaded = 0;
    int bad = 0;
    while (std::fscanf(f, "%d %d %lld", &il, &e, &n) == 3)
        if (il >= 0 && il < n_layer && e >= 0 && e < n_expert && n > 0) {
            out[(size_t) il * n_expert + e] = n;
            ++loaded;
        } else {
            ++bad;
        }
    const bool malformed = std::feof(f) == 0;   // stopped on something that is not three ints
    std::fclose(f);
    // A damaged table is refused, never used half-read (loud, not silent); the run continues unranked.
    if (malformed || bad > 0) {
        std::fprintf(stderr, "strata-poly: profile: REFUSED: %s: %s; running without a table\n", path.c_str(),
                     malformed ? "not lines of 'layer expert count'" : "entry out of range");
        out.clear();
        return;
    }
    std::fprintf(stderr, "strata-poly: profile: loaded %lld entries from %s\n", loaded, path.c_str());
}

void cu_ck(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string(what) + ": " + (s ? s : "a CUDA driver error"));
}

size_t up_to(size_t n, size_t grain) { return (n + grain - 1) / grain * grain; }

// The largest of n values, as eight maxima side by side: one best-so-far is a chain every compare waits on, and
// a vocabulary is 150,000 values.
constexpr int kLanes = 8;
float largest(const float* x, int n) {
    float mx[kLanes];
    std::fill(mx, mx + kLanes, -INFINITY);
    int i = 0;
    for (; i + kLanes <= n; i += kLanes)
        for (int j = 0; j < kLanes; ++j) mx[j] = std::max(mx[j], x[i + j]);
    float best = -INFINITY;
    for (int j = 0; j < kLanes; ++j) best = std::max(best, mx[j]);
    for (; i < n; ++i) best = std::max(best, x[i]);
    return best;
}

// The CUDA calls of a card's engine go to its card: a split's engines share the caller's thread (but in read()),
// and the card is the thread's.  Back to the caller's card at the end.  One engine alone changes nothing.
class OnCard {
public:
    OnCard(int device, bool split) : split_(split) {
        if (!split_) return;
        cudaGetDevice(&was_);
        cudaSetDevice(device);   // a failure shows in the calls that follow
    }
    ~OnCard() {
        if (split_) cudaSetDevice(was_);
    }
    OnCard(const OnCard&) = delete;
    OnCard& operator=(const OnCard&) = delete;

private:
    bool split_;
    int was_ = 0;
};

// Work on a thread of its own, whose exception wait() throws.  Going out of scope it waits and drops the exception.
class Behind {
public:
    Behind() = default;
    Behind(const Behind&) = delete;
    Behind& operator=(const Behind&) = delete;
    ~Behind() {
        if (th_.joinable()) th_.join();
    }
    void go(std::function<void()> work) {
        th_ = std::thread([this, work = std::move(work)] {
            try {
                work();
            } catch (...) {
                err_ = std::current_exception();
            }
        });
    }
    void wait() {
        if (th_.joinable()) th_.join();
        if (err_) std::rethrow_exception(std::exchange(err_, nullptr));
    }

private:
    std::thread th_;
    std::exception_ptr err_;
};

// The first layer of each card's share of n_layer layers, for Options::layer_split on `cards` cards; empty: no
// split.  "auto" shares the layers' bytes out by each card's free VRAM, less what every card's engine takes beside
// its layers and, on the last card, the output and the draft block.  Then every card caches about the same share
// of its layers' experts, which is where a byte of VRAM on one is worth as much as on another.
std::vector<int> layer_starts(const std::string& split, const Shards& file, int n_layer, int cards) {
    if (split.empty() || cards < 2) return {};
    std::vector<int> starts{0};
    if (split != "auto") {
        for (size_t at = 0; at < split.size();) {
            const size_t comma = std::min(split.find(',', at), split.size());
            const std::string s = split.substr(at, comma - at);
            char* end = nullptr;
            const long v = std::strtol(s.c_str(), &end, 10);
            // the last card keeps two layers at least: Gemma 4's draft model reads the last two layers' caches
            if (s.empty() || *end || v <= starts.back() || v > n_layer - 2)
                throw std::runtime_error("--layer-split " + split + ": the first layer of each later card, rising, "
                                         "between 1 and " + std::to_string(n_layer - 2) + " (\"auto\" shares them "
                                         "out by each card's free VRAM)");
            starts.push_back((int) v);
            at = comma + 1;
        }
        if ((int) starts.size() > cards)
            throw std::runtime_error("--layer-split " + split + " is for " + std::to_string(starts.size()) +
                                     " cards; this process sees " + std::to_string(cards));
        return starts;
    }
    if (n_layer < 2 * cards) throw std::runtime_error("the model has too few layers to split them on " + std::to_string(cards) + " cards");
    std::vector<double> before((size_t) n_layer + 1, 0.0);   // the bytes of the layers before each one
    double head = 0;
    file.each([&](const TensorInfo& t) {
        int il = -1;
        if (std::sscanf(t.name.c_str(), "blk.%d.", &il) == 1 && il >= 0 && il < n_layer)
            before[(size_t) il + 1] += (double) file.payload(t);
        else if (il >= n_layer || t.name == "output.weight")   // the draft block, the output
            head += (double) file.payload(t);
    });
    if (!file.find("output.weight") && file.find("token_embd.weight"))   // the output is the token table
        head += (double) file.payload(*file.find("token_embd.weight"));
    for (int il = 0; il < n_layer; ++il) before[(size_t) il + 1] += before[(size_t) il];
    constexpr double kBeside = 1.5 * double(1ull << 30);   // scratch, the staging buffer and the reserve, about
    std::vector<double> room((size_t) cards);
    double all_room = 0;
    for (int d = 0; d < cards; ++d) {
        size_t free_b = 0, total_b = 0;
        ck(cudaSetDevice(d), "cudaSetDevice");
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        room[(size_t) d] = std::max((double) free_b - kBeside - (d == cards - 1 ? head : 0.0), double(1ull << 28));
        all_room += room[(size_t) d];
    }
    double share = 0;
    for (int d = 1; d < cards; ++d) {
        share += room[(size_t) d - 1] / all_room;
        int il = starts.back() + 1;   // card d starts where the layers before it pass the cards before it's share
        while (il < n_layer - 2 * (cards - d) && before[(size_t) il] < share * before[(size_t) n_layer]) ++il;
        starts.push_back(il);
    }
    return starts;
}

// The reserve when none is asked for: a thirty-second of the card, 256 to 768 MiB.  When the machine's RAM is
// tight for the model, a sixteenth, 512 to 1,024: on an 8 GB laptop with 32 GB, Ornith's APEX-I-Quality (23 GB)
// read a 24,000-token prompt at 1,392 and 1,138 tokens/s with 256 MiB kept free and at 1,532 and 1,451 with 512,
// in turns with the build that kept 512 (1,496 and 1,311); the smaller files read as fast either way.  Under
// Windows what the card is short of moves to the system's RAM, which the file then has less of.
int default_reserve_mib(size_t total_b, bool ram_tight) {
    if (ram_tight) return (int) std::clamp<size_t>(total_b >> 24, 512, 1024);
    return (int) std::clamp<size_t>(total_b >> 25, 256, 768);
}

// The RAM the machine has free for the model's file, or 0 when it cannot tell.
size_t ram_available() {
#ifdef _WIN32
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (size_t) st.ullAvailPhys : 0;
#else
    size_t kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[128];
        while (std::fgets(line, sizeof(line), f))
            if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
        std::fclose(f);
    }
    return kb << 10;
#endif
}

}  // namespace

// One expert on its way into the cache.  The engine's thread asks, the follower copies and sends, and the engine's
// thread gives it the slot once the upload has arrived.
struct Engine::Swap {
    enum { kFree, kAsked, kSent };
    std::atomic<int> at{kFree};
    int il = 0, e = 0;
    uint8_t* slot = nullptr;       // where it goes in VRAM: the slot of the expert it replaces
    uint8_t* pinned = nullptr;
    cudaEvent_t done = nullptr;    // marks its upload
};

Engine::Engine() = default;

// Stage D D2a: one expert's down, row-major, written into a device destination (slot down segment / staging down
// area / swap or borrow region).  The source is the pinned sidecar's T block: DMA it into the one-expert scratch,
// then the R58-verified T->row kernel.  MMQ, mmvq, moe_card and the slot layout are untouched.
void Engine::materialize_down(const Layer& L, int il, size_t e, uint8_t* dst, void* stream) {
    const uint8_t* src = r2d_base_ + r2d_layer_off_[(size_t) il] + e * r2d_expert_bytes_;
    uint8_t* scratch = (uint8_t*) r2d_scratch_for(stream);
    ck(cudaMemcpyAsync(scratch, src, r2d_expert_bytes_, cudaMemcpyHostToDevice, (cudaStream_t) stream), "down T scratch");
    kernels::native_relu2_t_to_row(scratch, dst, 2688, 1024, stream);
}

// One 1.48 MiB scratch for every stream that materializes down: fill/grow (null stream), refill (st_), ring2's
// ahead copy (up_), a batch's stage (cp_/st_) and the swap follower (swap_st_) may all be in flight at once.
void* Engine::r2d_scratch_for(void* stream) {
    std::lock_guard<std::mutex> l(r2d_scratch_m_);
    for (auto& e : r2d_scratch_)
        if (e.first == stream) return e.second;
    void* p = nullptr;
    ck(cudaMalloc(&p, r2d_expert_bytes_), "down T scratch");
    r2d_scratch_.emplace_back(stream, p);
    return p;
}

namespace {

// R56 (Stage B): the current process's working set, through psapi.dll dynamically (no link dependency on psapi.lib).
struct ProcMem {
    size_t ws = 0, peak = 0;
};
#ifdef _WIN32
ProcMem process_memory() {
    ProcMem r{};
    struct Counters {
        DWORD cb, PageFaultCount;
        size_t PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage,
            QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage, PrivateUsage;
    };
    using Fn = BOOL(WINAPI*)(HANDLE, void*, DWORD);
    static Fn fn = [] {
        HMODULE d = LoadLibraryA("psapi.dll");
        return d ? (Fn) (void*) GetProcAddress(d, "GetProcessMemoryInfo") : (Fn) nullptr;
    }();
    Counters c{};
    c.cb = sizeof c;
    if (fn && fn(GetCurrentProcess(), &c, sizeof c)) {
        r.ws = c.WorkingSetSize;
        r.peak = c.PeakWorkingSetSize;
    }
    return r;
}
#else
ProcMem process_memory() { return ProcMem{}; }
#endif

// R56 (Stage B): the inverse of Stage A's to_T for one expert's down block (the sidecar's Stage A code is
// repack_down.py).  values n*512+b: low nibble -> down row 2b, high -> row 2b+1; scales g*2048+r*2: the fp16 scale
// of IQ4_NL block g in row r.  out is the original d_bytes: n_out rows x (n_ff/32) blocks of 18 B.  On-load check
// only; nothing in the compute path calls it.
void relu2_from_T(const uint8_t* vals, const uint8_t* scales, int n_ff, size_t d_bytes, std::vector<uint8_t>& out) {
    const int nb = n_ff / 32;                                  // 84 IQ4_NL blocks a row
    const int n_out = (int) (d_bytes / ((size_t) nb * 18));    // 1024 output rows
    out.assign(d_bytes, 0);
    const int vals_per_neuron = n_out / 2;                     // 512 B: rows 2b (low) and 2b+1 (high)
    for (int n = 0; n < n_ff; ++n) {
        const int k = n / 32, m = n % 32;
        const int j = m < 16 ? m : m - 16;
        const int sh = m < 16 ? 0 : 4;
        const uint8_t* v = vals + (size_t) n * (size_t) vals_per_neuron;
        for (int b = 0; b < vals_per_neuron; ++b) {
            const uint8_t x = v[b];
            out[(size_t) (2 * b) * nb * 18 + (size_t) k * 18 + 2 + j] |= (uint8_t) ((x & 15) << sh);
            out[(size_t) (2 * b + 1) * nb * 18 + (size_t) k * 18 + 2 + j] |= (uint8_t) ((x >> 4) << sh);
        }
    }
    for (int g = 0; g < nb; ++g)
        for (int r = 0; r < n_out; ++r)
            for (int bb = 0; bb < 2; ++bb)
                out[(size_t) r * nb * 18 + (size_t) g * 18 + bb] =
                    scales[(size_t) g * n_out * 2 + (size_t) r * 2 + bb];
}

}  // namespace

// R56 (Stage B): STRATA_RELU2_DOWN=1 maps the neuron-major down sidecar and checks it against the GGUF.  The
// geometry must be the one Stage A packed (2688 neurons, 1024 rows, 512 experts, 40 main MoE layers).  The sidecar
// is NOT pinned and not touched whole: only the four check experts' blocks are read (R53 搂0.3: it is a second
// 31.7 GB copy, and pinning it would put this box past its RAM).  Every failure is loud and turns the switch off:
// the run then takes exactly the default path, which never reads this mapping.
void Engine::relu2_down_open() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_RELU2_DOWN");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (!on) return;
    auto unmap = [&] {
#ifdef _WIN32
        if (r2d_base_) UnmapViewOfFile((const void*) r2d_base_);
        if (r2d_map_) CloseHandle((HANDLE) r2d_map_);
        if (r2d_file_) CloseHandle((HANDLE) r2d_file_);
#else
        if (r2d_base_) ::munmap((void*) r2d_base_, r2d_bytes_);
        if (r2d_file_) ::close((int) (intptr_t) r2d_file_);
#endif
        r2d_base_ = nullptr;
        r2d_map_ = r2d_file_ = nullptr;
    };
    const auto refuse = [&](const std::string& why) {
        std::fprintf(stderr, "strata-poly: STRATA_RELU2_DOWN=1 REFUSED: %s -- the default path runs (bit-identical)\n",
                     why.c_str());
        unmap();
    };
    if (n_expert_ != 512 || n_ff_exp_ != 2688) {
        refuse("the model's experts are " + std::to_string(n_expert_) + " x " + std::to_string(n_ff_exp_) +
               ", the sidecar's are 512 x 2688");
        return;
    }
    // the main model's MoE layers, all on this engine (one card), in load order: only they are in the sidecar
    int n_main_moe = 0;
    size_t expert_bytes = 0;
    for (int il = 0; il < (int) layers_.size() && il < n_layer_; ++il) {
        const Layer& L = layers_[(size_t) il];
        if (!L.moe) continue;
        if (L.d_bytes == 0) { refuse("layer " + std::to_string(il) + " has no down"); return; }
        if (expert_bytes == 0) expert_bytes = L.d_bytes;
        if (L.d_bytes != expert_bytes) { refuse("layers disagree on an expert's down bytes"); return; }
        ++n_main_moe;
    }
    if (n_main_moe != 40) {
        refuse(std::to_string(n_main_moe) + " main MoE layers on this engine, the sidecar has 40 (one card only)");
        return;
    }
    const size_t row_bytes = (size_t) (n_ff_exp_ / 32) * 18;   // 1512 B: one down row
    if (expert_bytes != row_bytes * 1024) {
        refuse("down's bytes per expert are " + std::to_string(expert_bytes) + ", expected 1024 x " +
               std::to_string(row_bytes));
        return;
    }
    const std::string path = [] {
        const char* e = std::getenv("STRATA_RELU2_DOWN_PATH");
        return std::string(e && e[0] ? e : "E:\\sp3-sidecar\\downT.bin");
    }();
    const size_t expect = (size_t) n_main_moe * (size_t) n_expert_ * expert_bytes;
    const ProcMem mem0 = process_memory();
    const size_t ram0 = ram_available();
    const Clock::time_point t0 = Clock::now();
    void* view = nullptr;
#ifdef _WIN32
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        refuse(path + ": CreateFile failed (" + std::to_string((long long) GetLastError()) + ")");
        return;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(fh, &sz) || (size_t) sz.QuadPart != expect) {
        const long long got = (long long) sz.QuadPart;
        CloseHandle(fh);
        refuse(path + ": " + std::to_string(got) + " B, expected " + std::to_string(expect));
        return;
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    view = mh ? MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (!view) {
        if (mh) CloseHandle(mh);
        CloseHandle(fh);
        refuse(path + ": MapViewOfFile failed (" + std::to_string((long long) GetLastError()) + ")");
        return;
    }
    r2d_file_ = fh;
    r2d_map_ = mh;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    struct stat st{};
    if (fd < 0 || ::fstat(fd, &st) != 0 || (size_t) st.st_size != expect) {
        const long long got = fd >= 0 ? (long long) st.st_size : -1;
        if (fd >= 0) ::close(fd);
        refuse(path + ": " + std::to_string(got) + " B, expected " + std::to_string(expect));
        return;
    }
    view = ::mmap(nullptr, expect, PROT_READ, MAP_PRIVATE, fd, 0);
    if (view == MAP_FAILED) {
        ::close(fd);
        refuse(path + ": mmap failed");
        return;
    }
    r2d_file_ = (void*) (intptr_t) fd;
#endif
    r2d_base_ = (const uint8_t*) view;
    r2d_bytes_ = expect;
    r2d_expert_bytes_ = expert_bytes;
    r2d_layer_off_.assign(layers_.size(), (size_t) -1);
    size_t at = 0;
    for (int il = 0; il < (int) layers_.size() && il < n_layer_; ++il) {
        if (!layers_[(size_t) il].moe) continue;
        r2d_layer_off_[(size_t) il] = at;
        at += (size_t) n_expert_ * expert_bytes;
    }
    // the four check experts: the first and the last main MoE layer, each one's expert 0 and 511
    int first_moe = -1, last_moe = -1;
    for (int il = 0; il < n_layer_; ++il)
        if (layers_[(size_t) il].moe) {
            if (first_moe < 0) first_moe = il;
            last_moe = il;
        }
    const int n_out = (int) (expert_bytes / row_bytes);              // 1024
    const size_t vals_b = (size_t) n_ff_exp_ * (size_t) (n_out / 2);
    std::vector<uint8_t> rebuilt;
    int okv = 0, badv = 0;
    size_t touched = 0;
    for (const int il : {first_moe, last_moe}) {
        const Layer& L = layers_[(size_t) il];
        for (const int e : {0, n_expert_ - 1}) {
            const uint8_t* base = r2d_base_ + r2d_layer_off_[(size_t) il] + (size_t) e * expert_bytes;
            relu2_from_T(base, base + vals_b, n_ff_exp_, expert_bytes, rebuilt);
            touched += expert_bytes;
            if (std::memcmp(rebuilt.data(), L.exp_down + (size_t) e * L.d_bytes, L.d_bytes) == 0) ++okv;
            else ++badv;
        }
    }
    const ProcMem mem1 = process_memory();
    if (badv != 0) {
        refuse("the sidecar does not match the GGUF's down bytes");
        return;
    }
    // ---- Stage D D3/D4/D8: the pin budget and the slot layout change at load, once ----
    // Anything that would still read the GGUF row-major down is refused loudly (R63 reader table #14/#15/#16/#18).
    if (block_layer_) {
        refuse("--draft: the sidecar has no blk.88 (MTP) down; run with --no-draft");
        return;
    }
    if (opt_.pcie_frac > 0.0) {
        refuse("--pcie-frac>0 reads the GGUF down; not with STRATA_RELU2_DOWN=1");
        return;
    }
    if (std::getenv("STRATA_STAGE_RING2_CHECK") || std::getenv("STRATA_PROMPT_BORROW_CHECK")) {
        refuse("STRATA_STAGE_RING2_CHECK / STRATA_PROMPT_BORROW_CHECK memcmp GGUF down; not with the sidecar");
        return;
    }
    // The sidecar takes its pin here: fault every page in first (cudaHostRegister does NOT prefault a file mapping -
    // R57 搂8), then pin.  Hard gate (lead + builder4): WS +>=27 GiB, 1000 random 4 KiB reads all <200 us, <=40 s;
    // any miss = exit loudly (R57-E1: a half-resident sidecar ran decode at 1.59 tok/s).
    {
        const size_t ram_a = ram_available();
        const ProcMem wsa = process_memory();
        const Clock::time_point tf = Clock::now();
        const int nt = std::max(1, (int) std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        const size_t step = (r2d_bytes_ + (size_t) nt - 1) / (size_t) nt;
        const auto touch = [&](size_t lo, size_t hi) {
            volatile uint8_t seen = 0;
            for (size_t at = lo; at < hi; at += 4096) seen = (uint8_t) (seen + r2d_base_[at]);
            if (hi > lo) seen = (uint8_t) (seen + r2d_base_[hi - 1]);
        };
        for (int i = 1; i < nt; ++i)
            th.emplace_back(touch, std::min(r2d_bytes_, (size_t) i * step),
                            std::min(r2d_bytes_, (size_t) i * step + step));
        touch(0, std::min(r2d_bytes_, step));
        for (auto& t2 : th) t2.join();
        const double secs = ms_since(tf) / 1000.0;
        const ProcMem wsb = process_memory();
        const size_t ram_b = ram_available();
        const double ws_gib = ((double) wsb.ws - (double) wsa.ws) / (double) (size_t(1) << 30);
        std::fprintf(stderr,
                     "strata-poly: staged: faulted-in sidecar %.3f GiB in %.1f s (one-time), WS +%.1f GiB, avail %.1f "
                     "-> %.1f GiB\n",
                     (double) r2d_bytes_ / (double) (size_t(1) << 30), secs, ws_gib,
                     (double) ram_a / (double) (size_t(1) << 30), (double) ram_b / (double) (size_t(1) << 30));
        // random-read probe: 1000 x 4 KiB, each must come from RAM (a page fault would be far above 200 us)
        uint8_t probe[4096];
        double worst_us = 0.0;
        std::mt19937 rng(2688u);
        std::uniform_int_distribution<size_t> pick(0, (size_t) r2d_bytes_ - 4096);
        for (int i = 0; i < 1000; ++i) {
            const size_t at = pick(rng) & ~(size_t) 0xFFF;
            const Clock::time_point t1 = Clock::now();
            std::memcpy(probe, r2d_base_ + at, sizeof probe);
            const double us = ms_since(t1) * 1000.0;
            worst_us = std::max(worst_us, us);
            probe[0] = (uint8_t) (probe[0] + 1);
        }
        std::fprintf(stderr, "strata-poly: staged: 1000 random 4 KiB reads, worst %.1f us (must be <200)\n", worst_us);
        if (ws_gib < 27.0 || worst_us >= 200.0 || secs > 40.0) {
            std::fprintf(stderr,
                         "strata-poly: staged: REFUSED: fault-in gate failed (WS +%.1f GiB <27 or worst read %.1f us "
                         ">=200 or %.1f s >40); the sidecar is not resident. RAM is not enough.\n",
                         ws_gib, worst_us, secs);
            std::exit(3);
        }
        const cudaError_t pr = cudaHostRegister((void*) r2d_base_, r2d_bytes_, cudaHostRegisterReadOnly);
        if (pr != cudaSuccess) {
            cudaGetLastError();
            std::fprintf(stderr, "strata-poly: staged: REFUSED: sidecar cudaHostRegisterReadOnly: %s\n",
                         cudaGetErrorString(pr));
            std::exit(3);
        }
        r2d_pinned_ = true;
    }
    const size_t ram_ok = ram_available();
    if (ram_ok < ((size_t) 8 << 30)) {
        std::fprintf(stderr, "strata-poly: staged: REFUSED: avail RAM %.1f GiB < 8 GiB after pinning; exiting\n",
                     (double) ram_ok / (double) (size_t(1) << 30));
        std::exit(3);
    }
    std::fprintf(stderr,
                 "strata-poly: staged: sidecar PINNED at load (%.2f GiB), up pinned (%.2f GiB), avail %.1f GiB; "
                 "T->row materialization ON (fill/swap/stage/borrow keep row-major); ON-LOAD CHECK %d/%d experts == "
                 "GGUF (%.1f MB, load-time only), WS +%.1f MB\n",
                 (double) r2d_bytes_ / (double) (size_t(1) << 30), (double) pin_bytes_ / (double) (size_t(1) << 30),
                 (double) ram_ok / (double) (size_t(1) << 30), okv, okv + badv, (double) touched / 1e6,
                 (double) (mem1.ws - mem0.ws) / 1e6);
}

Engine::~Engine() {
    if (route_dump_) {
        std::fclose(route_dump_);
        route_dump_ = nullptr;
    }
    // R56/R57: the Stage B sidecar mapping (and its option B pin); nothing else points into it.
    if (r2d_base_) {
        if (r2d_pinned_) {
            cudaHostUnregister((void*) r2d_base_);
            r2d_pinned_ = false;
        }
#ifdef _WIN32
        UnmapViewOfFile((const void*) r2d_base_);
        if (r2d_map_) CloseHandle((HANDLE) r2d_map_);
        if (r2d_file_) CloseHandle((HANDLE) r2d_file_);
#else
        ::munmap((void*) r2d_base_, r2d_bytes_);
        if (r2d_file_) ::close((int) (intptr_t) r2d_file_);
#endif
        r2d_base_ = nullptr;
        r2d_map_ = r2d_file_ = nullptr;
    }
    next_.reset();
    OnCard here(device_, split_);
    map_quit_.store(true);
    if (mapper_.joinable()) mapper_.join();
    if (follower_.joinable()) {
        {
            std::lock_guard<std::mutex> l(swap_m_);
            swap_quit_ = true;
        }
        swap_cv_.notify_all();
        follower_.join();
    }
    if (swap_st_) cudaStreamSynchronize(cs(swap_st_));
    if (up_) cudaStreamSynchronize(cs(up_));
    if (cp_) cudaStreamSynchronize(cs(cp_));
    if (rs_) cudaStreamSynchronize(cs(rs_));
    if (pcie_st_) cudaStreamSynchronize(cs(pcie_st_));   // --pcie-frac: the copies into the staging slots
    if (st_) cudaStreamSynchronize(cs(st_));
    if (sh_) cudaStreamSynchronize(cs(sh_));   // STRATA_SHARED_OVERLAP: its last products before the stream goes
    finish_async_refill();   // STRATA_BORROW_ASYNC_REFILL: no dropped work; its slots become resident, then they go
    workers_.reset();
    mmq_.reset();
    for (void* p : dev_allocs_) cudaFree(p);
    for (Kv& k : kv_) vmm_free(k);
    vmm_free(arena_);
    vmm_free(borrow_);   // STRATA_PROMPT_BORROW: the region the prompt's staging buffers were lent
    for (void* p : host_allocs_) cudaFreeHost(p);
    for (void* e : {ev_stage_[0], ev_stage_[1], ev_ahead_, ev_free_, ev_in_[0], ev_in_[1], ev_out_[0], ev_out_[1],
                    ev_hr_, ev_krn_, ev_copy_, ev_pcie_, ev_refill_, ev_sh_, ev_st_})
        if (e) cudaEventDestroy((cudaEvent_t) e);
    for (int s = 0; s < 2; ++s)
        for (void* e : {ev_ring_ahead_[s], ev_ring_free_[s]})
            if (e) cudaEventDestroy((cudaEvent_t) e);
    for (void* e : ev_row_)
        if (e) cudaEventDestroy((cudaEvent_t) e);
    for (int j = 0; j < n_swap_; ++j)
        if (swap_[(size_t) j].done) cudaEventDestroy(swap_[(size_t) j].done);
    for (void* g : graphs_) cudaGraphExecDestroy((cudaGraphExec_t) g);
    blas_destroy(blas_);
    if (swap_st_) cudaStreamDestroy(cs(swap_st_));
    if (up_) cudaStreamDestroy(cs(up_));
    if (cp_) cudaStreamDestroy(cs(cp_));
    if (rs_) cudaStreamDestroy(cs(rs_));
    if (pcie_st_) cudaStreamDestroy(cs(pcie_st_));
    if (st_) cudaStreamDestroy(cs(st_));
    if (as_) cudaStreamDestroy(cs(as_));   // STRATA_BORROW_ASYNC_REFILL: the refill's own stream
    if (sh_) cudaStreamDestroy(cs(sh_));   // STRATA_SHARED_OVERLAP: the shared expert's stream
}

void* Engine::dmalloc(size_t bytes, const char* what) {
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, std::max<size_t>(bytes, 256));
    if (e != cudaSuccess) {
        cudaGetLastError();   // as ck() does
        throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(bytes >> 20) +
                                 " MiB): " + cudaGetErrorString(e));
    }
    dev_allocs_.push_back(p);
    return p;
}

void* Engine::hmalloc(size_t bytes, const char* what) {
    void* p = nullptr;
    const std::string why = std::string(what) + " (" + std::to_string(bytes) + " bytes)";
    // A container at its memory limit refused pinned memory with "invalid argument" five times in five minutes
    // while a download filled its page cache; a moment later the kernel has reclaimed some.
    cudaError_t e = cudaSuccess;
    for (int tries = 0; tries < 4; ++tries) {
        e = cudaMallocHost(&p, std::max<size_t>(bytes, 256));
        if (e == cudaSuccess) break;
        cudaGetLastError();
        std::fprintf(stderr, "strata-poly: %s: %s, trying again\n", why.c_str(), cudaGetErrorString(e));
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    ck(e, why.c_str());
    host_allocs_.push_back(p);
    return p;
}

// A container whose memory is not much more than the model's files.  The file cache cannot hold them beside
// the engine, so an answer reads experts from the disk: on a pod with 80 GiB and two RTX 3080s, GLM REAP50's
// 79 GB file wrote at 6 tokens/s where 20.  And the pod ended the engine with SIGKILL in half of its long reads,
// within 2% of the limit, with no OOM kill counted by the kernel.  Said once at load; the remedy is RAM.
void Engine::mind_memory() {
#ifdef __linux__
    if (split_ && lo_ > 0) return;   // the first card's engine says it for all of them
    static const char* const files[] = {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
    unsigned long long limit = 0;
    for (const char* path : files) {
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        if (std::fscanf(f, "%llu", &limit) != 1) limit = 0;   // "max" when there is none
        std::fclose(f);
        break;
    }
    if (limit == 0 || limit >= (1ULL << 50)) return;   // none, or one beyond any machine's RAM
    const uint64_t file = gg_->bytes() + (dg_ ? dg_->bytes() : 0);
    if (file + (8ULL << 30) <= limit) return;   // the engine and the rest of the container take a few GiB
    std::fprintf(stderr, "strata-poly: the model's files are %.1f GiB and the container has %.1f GiB: the file "
                         "cache cannot hold them beside the engine, so answers will read experts from the disk, "
                         "and the container may end the engine\n",
                 (double) file / (1 << 30), (double) limit / (1 << 30));
#endif
}

void Engine::vmm_reserve(Mapped& m, size_t bytes) {
    CUdeviceptr p = 0;
    m.reserved = up_to(std::max<size_t>(bytes, 1), grain_);
    cu_ck(cuMemAddressReserve(&p, m.reserved, 0, 0, 0), "reserving addresses for VRAM");
    m.base = p;
}

// VRAM for the first `bytes` of m, in pieces of at most `piece` bytes.  Throws std::runtime_error.
void Engine::vmm_map(Mapped& m, size_t bytes, size_t piece, const char* what) {
    bytes = std::min(up_to(bytes, grain_), m.reserved);
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;
    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    piece = std::max(grain_, piece / grain_ * grain_);
    while (m.mapped < bytes) {
        const size_t n = std::min(bytes - m.mapped, piece);
        CUmemGenericAllocationHandle h = 0;
        if (cuMemCreate(&h, n, &prop, 0) != CUDA_SUCCESS)
            throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(n >> 20) +
                                     " MiB more)");
        const CUdeviceptr at = m.base + m.mapped;
        const CUresult mapped = cuMemMap(at, n, 0, h, 0);
        if (mapped != CUDA_SUCCESS) cuMemRelease(h);
        cu_ck(mapped, "mapping VRAM");
        m.pieces.emplace_back(h, n);
        m.mapped += n;
        const CUresult acc = cuMemSetAccess(at, n, &access, 1);
        if (acc != CUDA_SUCCESS) {   // HIP hands out the physical pages here, not at create: the card may not have
            cuMemUnmap(at, n);       // them even though the addresses do.  Leave m as it was and say no.
            cuMemRelease(h);
            m.pieces.pop_back();
            m.mapped -= n;
            (void) cudaGetLastError();   // HIP's failed SetAccess leaves a sticky error: do not let it escape with the throw
            throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(n >> 20) +
                                     " MiB more)");
        }
    }
}

// Gives back the pieces of m that lie past its first `bytes`.  The card must be done with them.
void Engine::vmm_unmap(Mapped& m, size_t bytes) {
    while (!m.pieces.empty() && m.mapped - m.pieces.back().second >= bytes) {
        const auto [h, n] = m.pieces.back();
        m.mapped -= n;
        m.pieces.pop_back();
        cu_ck(cuMemUnmap(m.base + m.mapped, n), "unmapping VRAM");
        cu_ck(cuMemRelease(h), "unmapping VRAM");
    }
}

// For the destructor: what is left of m, and its addresses, without throwing.
void Engine::vmm_free(Mapped& m) {
    for (; !m.pieces.empty(); m.pieces.pop_back()) {
        m.mapped -= m.pieces.back().second;
        cuMemUnmap(m.base + m.mapped, m.pieces.back().second);
        cuMemRelease(m.pieces.back().first);
    }
    if (m.base) cuMemAddressFree(m.base, m.reserved);
    m.base = 0;
}

void* Engine::kv_alloc(size_t bytes, int per) {
    const size_t all = (size_t) ((opt_.max_context + per - 1) / per) * bytes;
    if (!vmm_) return dmalloc(all, "the attention cache");
    Kv k;
    k.bytes = bytes;
    k.per = per;
    vmm_reserve(k, all);
    kv_.push_back(std::move(k));
    return (void*) kv_.back().base;
}

// The attention caches' VRAM for the rows below `end`: rows kKvGrow at a time, kKvAhead more for the answer to go
// on.  The expert cache gives it up (shrink_cache()), as if the rows had been there when it was filled: the
// reserve stays for what is recorded later.  A quarter of the reserve stays free whatever else took VRAM since
// (mind_vram()).
void Engine::cover(int64_t end) {
    if (kv_.empty() || end <= kv_rows_) return;
    const int64_t rows = std::min<int64_t>(opt_.max_context, (end + kKvAhead + kKvGrow - 1) / kKvGrow * kKvGrow);
    const auto size = [rows](const Kv& k) { return (size_t) ((rows + k.per - 1) / k.per) * k.bytes; };
    size_t more = 0, held = arena_.mapped;
    for (const Kv& k : kv_) {
        more += std::min(up_to(size(k), grain_), k.reserved) - k.mapped;
        held += k.mapped;
    }
    if (exp_) {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t quarter = (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
        size_t cut = held + more > budget_ ? held + more - budget_ : 0;
        if (free_b < more + quarter) cut = std::max(cut, more + quarter - free_b);
        if (cut) shrink_cache(cut);
    }
    for (Kv& k : kv_) vmm_map(k, size(k), SIZE_MAX, "the attention cache");
    kv_rows_ = rows;
}

// Every expert on its way into the cache arrives and takes its slot.
void Engine::settle_swaps() {
    if (n_swap_ == 0) return;
    const auto failed = [&] {
        std::lock_guard<std::mutex> l(swap_m_);
        return !swap_error_.empty();
    };
    for (int j = 0; j < n_swap_; ++j)
        while (swap_[(size_t) j].at.load() == Swap::kAsked && !failed()) std::this_thread::yield();
    if (failed()) throw std::runtime_error("expert upload: " + swap_error_);
    ck(cudaStreamSynchronize(cs(swap_st_)), "expert upload");
    follow(0);
}

// The expert cache gives up `bytes` or a little more from its end, and loads nothing again: the experts whose
// slots lie there and that were used more than cached ones before the cut take those ones' slots, as follow()
// ranks them, and the others leave the cache.
void Engine::shrink_cache(size_t bytes) {
    size_t cut = arena_.mapped;
    for (size_t i = arena_.pieces.size(); i > 0 && arena_.mapped - cut < bytes; --i) cut -= arena_.pieces[i - 1].second;
    settle_swaps();
    ck(cudaStreamSynchronize(cs(st_)), "the expert cache");
    const size_t ne = (size_t) n_expert_, end = cut > kTail ? cut - kTail : 0;   // kTail stays past the last slot
    struct Held {
        float recent;
        int64_t prior;
        int il, e;
    };
    const auto held = [&](int il, int e) {
        const size_t i = (size_t) il * ne + (size_t) e;
        return Held{recent_.empty() ? 0.f : recent_[i], (prior_.empty() ? 0 : prior_[i]) + use_[i], il, e};
    };
    const auto more = [](const Held& a, const Held& b) {
        return a.recent != b.recent ? a.recent > b.recent : a.prior > b.prior;
    };
    std::vector<Held> past;
    std::vector<std::vector<Held>> low((size_t) n_trade_);   // the cached ones before the cut, by slot size
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        for (int e = 0; e < n_expert_; ++e) {
            const uint8_t* s = L.slot[(size_t) e];
            if (!s) continue;
            // STRATA_PROMPT_BORROW: an expert refilled into a borrowed slot is not in the arena, and the cache's
            // cut is the arena's: leave it be (its slot stays where the refill put it).
            if (s < exp_ || s >= exp_ + arena_.mapped) continue;
            if ((size_t) (s - exp_) + L.slot_bytes > end) past.push_back(held(il, e));
            else low[(size_t) L.trade].push_back(held(il, e));
        }
    }
    std::sort(past.begin(), past.end(), more);
    for (std::vector<Held>& l : low) std::sort(l.begin(), l.end(), [&](const Held& a, const Held& b) { return more(b, a); });
    std::vector<size_t> next((size_t) n_trade_, 0);
    for (const Held& x : past) {
        const Layer& L = layers_[(size_t) x.il];
        uint8_t*& from = layers_[(size_t) x.il].slot[(size_t) x.e];
        const std::vector<Held>& l = low[(size_t) L.trade];
        size_t& k = next[(size_t) L.trade];
        if (k < l.size() && more(x, l[k])) {
            uint8_t*& to = layers_[(size_t) l[k].il].slot[(size_t) l[k].e];
            ck(cudaMemcpyAsync(to, from, L.slot_bytes, cudaMemcpyDeviceToDevice, cs(st_)), "the expert cache");
            from = to;
            to = nullptr;
            ++k;
        } else {
            from = nullptr;
        }
        --cached_;
    }
    ck(cudaStreamSynchronize(cs(st_)), "the expert cache");
    vmm_unmap(arena_, cut);
    if (cut > kTail) ck(cudaMemset(exp_ + end, 0, kTail), "the expert cache");
    exp_at_ = std::min(exp_at_, end);
    staged_il_ = -1;
}

float* Engine::fbuf(size_t n) {
    float* p = scratch_ ? scratch_ + scratch_at_ : nullptr;
    scratch_at_ += (n + 63) & ~size_t(63);
    return p;
}

int64_t Engine::meta_u(const char* key, int64_t fallback) const {
    const MetaValue* v = gg_ ? gg_->get(key) : nullptr;
    return v && v->is_num() ? (int64_t) v->u : fallback;
}

// A layer's tensor, under any of its names.  The draft block's are in the file it comes from.
const TensorInfo* Engine::tensor(int il, const char* name) const {
    const Shards& g = block_ && il >= n_layer_ ? *block_ : *gg_;
    if (const TensorInfo* t = g.find(blk(il, name))) return t;
    if (const Names* n = tensor_alias(name))
        for (const char* other : n->other)
            if (const TensorInfo* t = other ? g.find(blk(il, other)) : nullptr) return t;
    return nullptr;
}

// A value of the metadata of the model that `g` is a file of, under any of the names converters give it.
const MetaValue* Engine::meta_in(const Shards& g, const char* key) const {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return nullptr;
    const std::string pre = arch->s + ".";
    if (const MetaValue* v = g.get(pre + key)) return v;
    if (const Names* n = meta_alias(key))
        for (const char* other : n->other)
            if (const MetaValue* v = other ? g.get(pre + other) : nullptr) return v;
    return nullptr;
}

int Engine::shape_i(const char* key) const {
    const MetaValue* v = meta_in(*gg_, key);
    if (!v) throw std::runtime_error("the GGUF has no " + gg_->get("general.architecture")->s + "." + key);
    return (int) v->u;
}

float Engine::shape_f(const char* key) const {
    const MetaValue* v = meta_in(*gg_, key);
    if (!v) throw std::runtime_error("the GGUF has no " + gg_->get("general.architecture")->s + "." + key);
    return (float) v->f;
}

// What every architecture's file says of the model: its layers, its width, its vocabulary and its experts.
//
// The block that guesses a second token after the model's own is one more block after the model's.  It is the
// model's and not a file's: a file that leaves it out can take it from another file of the model (draft_model).
// One converter gives the block a token table or an output matrix of its own, which this engine does not run.
void Engine::load_shape() {
    const MetaValue* nextn = meta_in(*gg_, "nextn_predict_layers");
    n_layer_ = shape_i("block_count") - (nextn ? (int) nextn->u : 0);
    if (opt_.draft && opt_.window != 1) {
        const std::string probe = blk(n_layer_, "nextn.eh_proj.weight");
        if (nextn && nextn->u == 1 && gg_->find(probe)) block_ = gg_.get();
        else if (dg_ && dg_->find(probe)) block_ = dg_.get();
        else if (dg_)
            throw std::runtime_error(opt_.draft_model + " has no block that guesses ahead for this model: no tensor " +
                                     probe);
    }
    draft_ = block_ && !tensor(n_layer_, "nextn.embed_tokens.weight") &&
             !tensor(n_layer_, "nextn.shared_head_head.weight");
    if (!draft_) block_ = nullptr;
    block_layer_ = draft_;
    n_embd_ = shape_i("embedding_length");
    // a file without the key has as many tokens as its token table
    const MetaValue* vocab = meta_in(*gg_, "vocab_size");
    const TensorInfo* te = gg_->find("token_embd.weight");
    n_vocab_ = vocab ? (int) vocab->u : te && te->shape.size() == 2 ? (int) te->shape[1] : 0;
    rms_eps_ = shape_f("attention.layer_norm_rms_epsilon");
    n_expert_ = shape_i("expert_count");
    n_used_ = shape_i("expert_used_count");
    n_ff_exp_ = shape_i("expert_feed_forward_length");
    const MetaValue* shexp = meta_in(*gg_, "expert_shared_feed_forward_length");
    n_ff_shexp_ = shexp ? (int) shexp->u : n_ff_exp_;

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    need(n_layer_ > 0 && n_embd_ > 0 && n_embd_ % 32 == 0 && n_vocab_ > 0, "bad block_count or embedding_length");
    need(n_used_ >= 1 && n_used_ <= kMaxUsed && n_used_ <= n_expert_, "bad expert_used_count");

    // a block from another file must be this model's: a pruned model's has fewer experts to choose among
    if (block_ && block_ != gg_.get()) {
        const TensorInfo* gate = tensor(n_layer_, "ffn_gate_inp.weight");
        if (!gate || gate->shape.size() != 2 || (int64_t) gate->shape[0] != n_embd_ ||
            (int64_t) gate->shape[1] != n_expert_)
            throw std::runtime_error(opt_.draft_model + " is not a file of this model: its block's router is " +
                                     (gate ? shape_of(*gate) : std::string("missing")) + ", the model's layers have [" +
                                     std::to_string(n_embd_) + ", " + std::to_string(n_expert_) + "]");
    }
}

// A MoE layer: its router and shared expert to VRAM, and its routed experts' places in the file, where they
// stay; fill_expert_cache() copies the ones VRAM has room for.  A file may keep an expert's gate and up in one
// tensor, the gate's rows first.
void Engine::load_moe(int il, bool shared, bool gated, int exp_in, int exp_out) {
    Layer& L = layers_[(size_t) il];
    const int in_w = exp_in > 0 ? exp_in : n_embd_;
    const int out_w = exp_out > 0 ? exp_out : n_embd_;
    L.gated = gated;
    L.relu2 = !gated;
    L.exp_in = in_w;
    L.exp_out = out_w;
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    L.moe = true;
    L.gate_inp = mat("ffn_gate_inp.weight", n_embd_, n_expert_);
    if (shared) {
        // a non-gated layer's shared expert has no gate (Nemotron)
        if (gated) L.sh_gate = mat("ffn_gate_shexp.weight", n_embd_, n_ff_shexp_);
        L.sh_up = mat("ffn_up_shexp.weight", n_embd_, n_ff_shexp_);
        L.sh_down = mat("ffn_down_shexp.weight", n_ff_shexp_, n_embd_);
    }

    auto exps = [&](const char* n, int64_t n_in, int64_t rows) -> const TensorInfo& {
        const TensorInfo* t = tensor(il, n);
        if (!t) throw std::runtime_error("the GGUF has no tensor " + blk(il, n));
        if (t->shape.size() != 3 || (int64_t) t->shape[0] != n_in || (int64_t) t->shape[1] != rows ||
            (int64_t) t->shape[2] != n_expert_)
            throw std::runtime_error(t->name + " is " + shape_of(*t) + ", expected [" + std::to_string(n_in) + ", " +
                                     std::to_string(rows) + ", " + std::to_string(n_expert_) + "]");
        return *t;
    };
    const TensorInfo& td = exps("ffn_down_exps.weight", n_ff_exp_, out_w);
    L.d_type = td.type;
    L.d_bytes = home(td).payload(td) / (size_t) n_expert_;
    L.exp_down = home(td).data(td);
    if (L.d_type == GGML_TYPE_MXFP4) mxfp4_to_q5_0(il);
    if (!gated) {
        // no gate and no fused gate_up: one up per expert, activated by relu2 (llama.cpp nemotron-h.cpp:284)
        const TensorInfo& tu = exps("ffn_up_exps.weight", in_w, n_ff_exp_);
        L.gu_type = tu.type;
        L.gu_stride = L.gu_bytes = home(tu).payload(tu) / (size_t) n_expert_;
        L.exp_gate = nullptr;
        L.exp_up = home(tu).data(tu);
    } else if (tensor(il, "ffn_gate_up_exps.weight")) {
        const TensorInfo& gu = exps("ffn_gate_up_exps.weight", n_embd_, 2 * n_ff_exp_);
        L.gu_type = gu.type;
        L.gu_stride = home(gu).payload(gu) / (size_t) n_expert_;
        L.gu_bytes = L.gu_stride / 2;
        L.exp_gate = home(gu).data(gu);
        L.exp_up = L.exp_gate + L.gu_bytes;
    } else {
        const TensorInfo& tg = exps("ffn_gate_exps.weight", n_embd_, n_ff_exp_);
        const TensorInfo& tu = exps("ffn_up_exps.weight", n_embd_, n_ff_exp_);
        if (tg.type != tu.type)
            throw std::runtime_error(tg.name + " and " + tu.name +
                                     " have different types, which this engine does not run");
        L.gu_type = tg.type;
        L.gu_stride = L.gu_bytes = home(tg).payload(tg) / (size_t) n_expert_;
        home(tu).payload(tu);
        L.exp_gate = home(tg).data(tg);
        L.exp_up = home(tu).data(tu);
    }
    std::string err;
    if (!cpu_expert_types_ok((int) L.gu_type, (int) L.d_type, in_w, n_ff_exp_, err))
        throw std::runtime_error(blk(il, "ffn_*_exps.weight: ") + err);
    L.slot.assign((size_t) n_expert_, nullptr);
    ++n_moe_;
}

// No kernel here multiplies MXFP4 on the card, by matvec or MMQ, so a layer whose down experts are MXFP4 ran them on
// the CPU, a prompt batch's too: DeepSeek V4 Flash UD-IQ2_XXS has two such layers, and they took 10.7 s of a
// 10,398-token read whose other 41 layers' experts took 4.9.  An MXFP4 value is 0, 1, 2, 3, 4, 6, 8 or 12, with a
// sign, times the block's power of two halved, which Q5_0 holds exactly while that power fits a half.  So the experts
// are converted once, into RAM of the engine's own, 22 bytes a block of 32 instead of 17.  On the CPU both types meet
// the activations as Q8_0, and the dot products differ only in the order of their sums.  A layer with a power out of
// range stays MXFP4.
void Engine::mxfp4_to_q5_0(int il) {
    Layer& L = layers_[(size_t) il];
    constexpr size_t kIn = 17, kOut = 22;   // block_mxfp4, block_q5_0
    const size_t blocks = L.d_bytes * (size_t) n_expert_ / kIn;
    std::unique_ptr<uint8_t[]> out;
    try {
        out.reset(new uint8_t[blocks * kOut]);
    } catch (const std::bad_alloc&) {
        return;
    }
    std::atomic<bool> bad{false};
    auto convert = [&](size_t lo, size_t hi) {
        for (size_t b = lo; b < hi && !bad.load(std::memory_order_relaxed); ++b)
            if (!mxfp4_block_to_q5_0(L.exp_down + b * kIn, out.get() + b * kOut)) bad.store(true);
    };
    const size_t n = (size_t) std::clamp(usable_cpus(), 1, 32), step = (blocks + n - 1) / n;
    std::vector<std::thread> th;
    for (size_t i = 1; i < n; ++i) th.emplace_back(convert, std::min(blocks, i * step), std::min(blocks, (i + 1) * step));
    convert(0, std::min(blocks, step));
    for (std::thread& t : th) t.join();
    if (bad) return;
    L.exp_down = out.get();
    L.d_type = GGML_TYPE_Q5_0;
    L.d_bytes = blocks / (size_t) n_expert_ * kOut;
    converted_.push_back(std::move(out));
    std::fprintf(stderr, "strata-poly: layer %d's down experts converted from MXFP4 to Q5_0, %.2f GiB of RAM\n", il,
                 (double) (blocks * kOut) / (1 << 30));
}

// The engine for a model file: its general.architecture says which.
std::unique_ptr<Engine> Engine::open(const Options& o) {
    const Shards file(o.model);
    const MetaValue* arch = file.get("general.architecture");
    const std::string name = arch ? arch->s : std::string();
    const auto make = [&]() -> std::unique_ptr<Engine> {
        if (name == "glm5-next" || name == "glm5next") return std::make_unique<Glm5Next>();   // Unsloth: no hyphen
        if (name == "qwen35moe") return std::make_unique<Qwen35>();
        if (name == "gemma4") return std::make_unique<Gemma4>();
        if (name == "deepseek4") return std::make_unique<DeepSeek4>();
        if (name == "lfm2moe") return std::make_unique<Lfm2Moe>();
        if (name == "nemotron_h_moe") return std::make_unique<NemotronHMoe>();   // Nemotron 3.5 Super VL 120B A12B
        throw std::runtime_error(o.model + " is a '" + name + "' model, which this engine does not run");
    };
    int cards = 1;
    if (!o.layer_split.empty()) ck(cudaGetDeviceCount(&cards), "cudaGetDeviceCount");
    std::vector<int> starts;
    if (cards > 1) {
        const MetaValue* blocks = file.get(name + ".block_count");
        const MetaValue* nextn = file.get(name + ".nextn_predict_layers");
        if (!blocks) throw std::runtime_error("the GGUF has no " + name + ".block_count");
        int was = 0;
        ck(cudaGetDevice(&was), "cudaGetDevice");
        starts = layer_starts(o.layer_split, file, (int) blocks->u - (nextn ? (int) nextn->u : 0), cards);
        ck(cudaSetDevice(was), "cudaSetDevice");
    }
    if (starts.empty()) {
        std::unique_ptr<Engine> engine = make();
        engine->load(o);
        return engine;
    }

    // a card after another, each with the next run of layers; the last one has the output and the draft block
    std::unique_ptr<Engine> first;
    Engine* prev = nullptr;
    for (size_t k = 0; k < starts.size(); ++k) {
        std::unique_ptr<Engine> e = make();
        const bool last = k + 1 == starts.size();
        Options so = o;
        if (!last) {
            so.draft = false;
            so.draft_model.clear();
        }
        if (first) {
            so.window = first->opt_.window;   // every card's windows are the first one's
            e->workers_ = first->workers_;
        }
        e->lo_ = starts[k];
        e->end_ = last ? -1 : starts[k + 1];
        e->split_ = true;
        ck(cudaSetDevice((int) k), "cudaSetDevice");
        try {
            e->load(so);
        } catch (const std::runtime_error& err) {
            throw std::runtime_error("card " + std::to_string(k) + ": " + err.what());
        }
        Engine* at = e.get();
        if (prev) prev->next_ = std::move(e);
        else first = std::move(e);
        prev = at;
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    int k = 0;
    for (const Engine* e = first.get(); e; e = e->next_.get(), ++k)
        std::fprintf(stderr, "strata-poly: layers %d to %d on card %d: %d of %d routed experts in VRAM (%.1f GiB)\n",
                     e->lo_, e->hi_ - 1, k, e->cached_, e->n_moe_ * e->n_expert_, double(e->exp_at_) / double(1ull << 30));
    return first;
}

// The file a tensor is in.
const Shards& Engine::home(const TensorInfo& t) const {
    return block_ && gg_->find(t.name) != &t ? *block_ : *gg_;
}

// A tensor to VRAM.  It goes as the file has it when the kernels compute with that type.  Otherwise it is
// converted on the way: to F32 where a kernel reads floats, to BF16 for a matrix.  `fix` is applied to every
// value, which also takes the float path.  While dense_ is null the call only adds up the arena (load() runs the
// model's tensors through twice).
Dev Engine::upload(const TensorInfo* t, const std::string& name, int64_t n_in, int64_t n_out, Want want,
                   float (*fix)(float)) {
    if (!t) throw std::runtime_error("the GGUF has no tensor " + name);
    const uint64_t rest = t->elements() / std::max<uint64_t>(t->shape[0], 1);
    if ((int64_t) t->shape[0] != n_in || (int64_t) rest != n_out)
        throw std::runtime_error(t->name + " is " + shape_of(*t) + ", expected [" + std::to_string(n_in) + ", " +
                                 std::to_string(n_out) + "]");
    const size_t bytes = home(*t).payload(*t), count = (size_t) t->elements();
    if (want == kGuessMat && !fix && n_in % ggml_blck_size(GGML_TYPE_Q4_0) == 0 &&
        ggml_get_type_traits((ggml_type) t->type)->to_float) {
        const size_t row = ggml_row_size(GGML_TYPE_Q4_0, n_in), out = row * (size_t) n_out;
        const Dev d{dense_ ? dense_ + dense_at_ : nullptr, (uint32_t) GGML_TYPE_Q4_0, (int) n_in, (int) n_out};
        dense_at_ += pad(out) + kTail;
        if (!dense_) return d;
        if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + t->name);
        // a few thousand rows at a time, to floats and to Q4_0, on threads of their own: the engine's are not
        // started yet
        const size_t in_row = bytes / (size_t) n_out, rows = 4096;
        std::vector<uint8_t> q(out);
        std::vector<float> f(rows * (size_t) n_in);
        const ggml_to_float_t to_float = ggml_get_type_traits((ggml_type) t->type)->to_float;
        const uint8_t* src = (const uint8_t*) home(*t).data(*t);
        Workers pool(std::max(1u, std::thread::hardware_concurrency() / 2));
        for (size_t r0 = 0; r0 < (size_t) n_out; r0 += rows) {
            const size_t m = std::min(rows, (size_t) n_out - r0);
            pool.run((int) ((m + 255) / 256), [&](int k) {
                const size_t a = (size_t) k * 256, b = std::min(m, a + 256);
                to_float(src + (r0 + a) * in_row, f.data() + a * n_in, (int64_t) ((b - a) * n_in));
                ggml_quantize_chunk(GGML_TYPE_Q4_0, f.data() + a * n_in, q.data() + (r0 + a) * row, 0,
                                    (int64_t) (b - a), n_in, nullptr);
            });
        }
        ck(cudaMemcpy(d.p, q.data(), out, cudaMemcpyHostToDevice), "weight upload");
        return d;
    }
    const bool as_is = !fix && (t->type == kTypeF32 || (want != kF32 && t->type == kTypeBF16) ||
                                (want == kAnyMat && kernels::native_mmvq_supported((int) t->type)));
    const uint32_t type = as_is ? t->type : want == kF32 ? kTypeF32 : kTypeBF16;
    const size_t out = as_is ? bytes : count * (type == kTypeF32 ? sizeof(float) : sizeof(uint16_t));
    if (type == kTypeBF16) bf16_max_ = std::max(bf16_max_, count);
    const Dev d{dense_ ? dense_ + dense_at_ : nullptr, type, (int) n_in, (int) n_out};
    dense_at_ += pad(out) + kTail;
    if (!dense_) return d;
    if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + t->name);
    if (as_is) {
        ck(cudaMemcpy(d.p, home(*t).data(*t), bytes, cudaMemcpyHostToDevice), "weight upload");
        return d;
    }

    std::vector<float> f(count);
    const ggml_type_traits* traits = ggml_get_type_traits((ggml_type) t->type);
    if (t->type == kTypeF32) std::memcpy(f.data(), home(*t).data(*t), bytes);
    else if (traits && traits->to_float) traits->to_float(home(*t).data(*t), f.data(), (int64_t) count);
    else throw std::runtime_error(t->name + " is " + shape_of(*t) + ", a type this engine cannot read");
    if (fix)
        for (float& v : f) v = fix(v);
    if (type == kTypeF32) {
        ck(cudaMemcpy(d.p, f.data(), out, cudaMemcpyHostToDevice), "weight upload");
        return d;
    }
    std::vector<ggml_bf16_t> h(count);
    ggml_fp32_to_bf16_row(f.data(), h.data(), (int64_t) count);
    ck(cudaMemcpy(d.p, h.data(), out, cudaMemcpyHostToDevice), "weight upload");
    return d;
}

int64_t Engine::vram_free_mib() const {
    OnCard here(device_, split_);
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const int64_t mib = (int64_t) (free_b >> 20);
    return next_ ? std::min(mib, next_->vram_free_mib()) : mib;
}

// Experts go to VRAM by use per byte when there are counts, the profile's and this run's, and otherwise the same
// number from every layer.
//
// The cache takes what is free but for a reserve.  What the engine allocates after it comes out of the reserve:
// the recorded launches and a library's work space took 180 MiB of an 8 GB card in the first request.  Unless the
// caller names it, the reserve is a thirty-second of the card, between 256 and 768 MiB (mind_vram() keeps what is
// free near half of it later), or a sixteenth when the machine's RAM is within 4 GiB of the model's files
// (default_reserve_mib()).  Under Windows a card that is asked for more than it has does not refuse: memory
// moves to the system's, and an answer came at 13 tokens/s where it comes at 30.
// STRATA_PIN_ARENA_GIB=<n>: cudaHostRegisterReadOnly the first n GiB of the MoE layers' expert sources, so stage()
// DMAs straight from the mapping instead of gathering it on the host.  WHOLE LAYERS: routing inside one layer is broad
// (a prefill token set hits most of its experts), and a per-expert register would be thousands of calls.  The switch
// is off by default and every failure is loud; the engine then runs exactly as before.
void Engine::pin_expert_arena() {
    const char* env = std::getenv("STRATA_PIN_ARENA_GIB");
    const double gib = env ? std::atof(env) : 0.0;
    if (gib <= 0) return;
    // Stage D D4: with the sidecar on, this function pins only the GGUF up half (29.53 GiB); the neuron-major
    // sidecar takes the other 29.53 GiB at load.  The RAM floor must be the up half's, not the full 59.1: the
    // budget's "keep every expert in memory" is the two halves taken together, and they are pinned at different
    // points (sidecar first: relu2_down_open runs before this when the switch is on).
    static const bool r2d_mode = [] {
        const char* e = std::getenv("STRATA_RELU2_DOWN");
        return e != nullptr && std::atoi(e) != 0;
    }();
    // builder4's precise P1: floor = budget - the segment already pinned elsewhere + 8 GiB.  With the sidecar on,
    // relu2_down_open() has already pinned r2d_bytes_ (it runs before alloc()), so this pins the up half against the
    // rest of the budget.  When the sidecar is off / not mapped, r2d_base_ is null and the FULL 59.1 + 8 stays.
    const size_t budget = (size_t) (gib * (double) (size_t(1) << 30));
    // builder4 P1: floor = budget - the segment already pinned elsewhere + 8 GiB.  relu2_down_open() runs before
    // alloc() in r2d mode and has already pinned r2d_bytes_; r2d_base_ null keeps the FULL 59.1 + 8 floor.
    const size_t already_pinned = r2d_base_ ? r2d_bytes_ : 0;
    const size_t floor_bytes = budget - already_pinned + (size_t(8) << 30);
    const size_t ram_before = ram_available();
    // The user's floor: keep every expert in memory.  A full 40-layer pin is 59.06 GiB (40 x 1.4766); with this
    // box's 76 GiB free, 8 GiB headroom makes it fit where 16 GiB refused it whenever free RAM drifted down.
    if (ram_before < floor_bytes) {
        std::fprintf(stderr,
                     "strata-poly: STRATA_PIN_ARENA_GIB=%.1f REFUSED: only %.1f GiB RAM free, needs %.1f + 8 GiB\n",
                     gib, (double) ram_before / (double) (size_t(1) << 30),
                     (double) (budget - already_pinned) / (double) (size_t(1) << 30));
        return;
    }
    // Stage D D4: with the sidecar on, this function pins the GGUF up half only, and the neuron-major sidecar takes
    // the other 29.53 GiB at load (relu2_down_open).  With the switch off: both halves, exactly as before.
    pin_layer_.assign(layers_.size(), 0);
    size_t used = 0;
    const Clock::time_point t0 = Clock::now();
    int ok = 0, bad = 0;
    for (size_t il = 0; il < layers_.size() && used < budget; ++il) {
        Layer& L = layers_[il];
        if (!L.moe || !L.exp_up || !L.exp_down) continue;
        const size_t up_len = (size_t) n_expert_ * L.gu_stride;
        const size_t dn_len = r2d_mode ? 0 : (size_t) n_expert_ * L.d_bytes;
        const size_t gt_len = L.exp_gate ? (size_t) n_expert_ * L.gu_stride : 0;
        if (used + up_len + dn_len + gt_len > budget) break;   // whole layers only
        bool good = true;
        // --pcie-mode direct: the kernels read these bytes themselves, so the registration must also map them into
        // the device's address space (cudaHostRegisterMapped) - a registration for the staging DMA need not.
        const bool for_direct = opt_.pcie_mode == "direct" || opt_.pcie_mode == "auto";
        const unsigned reg_flags = cudaHostRegisterReadOnly | (for_direct ? cudaHostRegisterMapped : 0u);
        const auto reg = [&](const uint8_t* p, size_t n, const char* what) {
            const cudaError_t r = cudaHostRegister((void*) p, n, reg_flags);
            std::fprintf(stderr, "strata-poly: pin %s layer %zu: %.1f MB -> %s (%d)\n", what, il,
                         (double) n / (double) (size_t(1) << 20), cudaGetErrorString(r), (int) r);
            if (r != cudaSuccess) { cudaGetLastError(); good = false; }
        };
        reg(L.exp_up, up_len, "up");
        if (good && !r2d_mode) reg(L.exp_down, dn_len, "down");
        if (good && L.exp_gate) reg(L.exp_gate, gt_len, "gate");
        if (!good) { ++bad; break; }   // the failed register was printed above; keep what succeeded
        pin_layer_[il] = 1;
        used += up_len + dn_len + gt_len;
        ++ok;
    }
    pin_bytes_ = used;
    std::fprintf(stderr,
                 "strata-poly: STRATA_PIN_ARENA_GIB=%.1f: pinned %d layers (%.2f GiB) of %zu, %d refused; "
                 "RAM %.1f -> %.1f GiB free; took %.0f ms\n",
                 gib, ok, (double) used / (double) (size_t(1) << 30), layers_.size(), bad,
                 (double) ram_before / (double) (size_t(1) << 30),
                 (double) ram_available() / (double) (size_t(1) << 30), ms_since(t0));
}

// R57c (D0): the pin budget changes hands at the decode boundary.  Prefill runs exactly as the delivery always has:
// the GGUF's up AND down halves are page-locked (pin_expert_arena()) and staged directly, so prefill is unchanged.
// Before the answer's decode starts, the down half's 29.53 GiB of pin is given up and the same amount pins the
// neuron-major sidecar, which the CPU down kernel then reads in place (relu2_down_t.hpp).  The up half's pin never
// moves.  With STRATA_RELU2_DOWN off nothing here ever runs; a failure here is loud and leaves the decode on the
// GGUF down path (bit-identical), only slower.
bool Engine::relu2_pin_switch() {
    if (r2d_base_ == nullptr || r2d_pinned_) return false;
    if (!exp_down_dev_.empty()) {
        std::fprintf(stderr,
                     "strata-poly: pin switch: REFUSED with --pcie-mode direct (its device aliases point into the "
                     "down half); decode keeps the GGUF down path\n");
        return false;
    }
    if (pin_layer_.empty() || pin_bytes_ == 0) {
        std::fprintf(stderr, "strata-poly: pin switch: the GGUF experts are not pinned; the sidecar stays unpinned "
                             "(decode reads will page fault)\n");
        return false;
    }
    const Clock::time_point t0 = Clock::now();
    int n_un = 0, bad = 0;
    size_t freed = 0;
    for (size_t il = 0; il < layers_.size() && il < pin_layer_.size(); ++il) {
        if (!pin_layer_[il]) continue;
        const Layer& L = layers_[il];
        const cudaError_t r = cudaHostUnregister((void*) L.exp_down);
        if (r != cudaSuccess) {
            cudaGetLastError();
            ++bad;
        } else {
            ++n_un;
            freed += (size_t) n_expert_ * L.d_bytes;
        }
    }
    if (bad != 0) std::fprintf(stderr, "strata-poly: pin switch: %d down ranges refused to unregister\n", bad);
    // R57c-fix: cudaHostRegisterReadOnly does NOT fault a file mapping's pages in (R57 搂8: the first switch took
    // 567 ms and avail did not move, so the decode still page-faulted every sidecar read).  Touch one byte of every
    // 4 KiB page of the sidecar before registering it: that faults the whole 29.53 GiB in (the pages we just let go
    // of, the GGUF down half, are what Windows reclaims for them), and the registration then pins resident pages.
    // One-time per process, done at the decode boundary; the workers split it so the NVMe is not read single-file.
    const size_t ram_before_fault = ram_available();
    const Clock::time_point t_pre = Clock::now();
    const auto touch = [&](size_t lo, size_t hi) {
        volatile uint8_t seen = 0;
        for (size_t at = lo; at < hi; at += 4096) seen = (uint8_t) (seen + r2d_base_[at]);
        if (hi > lo) seen = (uint8_t) (seen + r2d_base_[hi - 1]);
    };
    const int nsl = workers_ ? std::max(1, workers_->threads()) : 1;
    const size_t step = (r2d_bytes_ + (size_t) nsl - 1) / (size_t) nsl;
    if (workers_ && nsl > 1) {
        workers_->run(nsl, [&](int i) {
            const size_t lo = std::min(r2d_bytes_, (size_t) i * step);
            touch(lo, std::min(r2d_bytes_, lo + step));
        });
    } else {
        touch(0, r2d_bytes_);
    }
    const double pre_ms = ms_since(t_pre);
    const size_t ram_after_fault = ram_available();
    const double drop_gib =
        ((double) ram_before_fault - (double) ram_after_fault) / (double) (size_t(1) << 30);
    std::fprintf(stderr,
                 "strata-poly: pin switch: faulted-in %.3f GiB sidecar in %.0f ms, avail before/after = %.1f/%.1f GiB "
                 "(drop %.1f GiB)\n",
                 (double) r2d_bytes_ / (double) (size_t(1) << 30), pre_ms,
                 (double) ram_before_fault / (double) (size_t(1) << 30),
                 (double) ram_after_fault / (double) (size_t(1) << 30), drop_gib);
    if (drop_gib < 20.0) {
        std::fprintf(stderr,
                     "strata-poly: pin switch: FAULT-IN DID NOT RESIDE (avail drop %.1f GiB < 20) -- the sidecar pages "
                     "are not in RAM; decode keeps the GGUF down path (bit-identical)\n",
                     drop_gib);
        return false;
    }
    const cudaError_t pr = cudaHostRegister((void*) r2d_base_, r2d_bytes_, cudaHostRegisterReadOnly);
    if (pr != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stderr,
                     "strata-poly: pin switch: sidecar cudaHostRegisterReadOnly FAILED: %s -- decode keeps the GGUF "
                     "down path (bit-identical), the sidecar stays unpinned\n",
                     cudaGetErrorString(pr));
        return false;
    }
    r2d_pinned_ = true;
    const size_t ram_post = ram_available();
    std::fprintf(stderr,
                 "strata-poly: pin switch: gguf-down unpinned (%d ranges, %.2f GiB), sidecar PINNED (%.2f GiB), "
                 "total %.1f ms, avail now %.1f GiB\n",
                 n_un, (double) freed / (double) (size_t(1) << 30), (double) r2d_bytes_ / (double) (size_t(1) << 30),
                 ms_since(t0), (double) ram_post / (double) (size_t(1) << 30));
    if (ram_post < ((size_t) 8 << 30)) {
        std::fprintf(stderr,
                     "strata-poly: pin switch: REFUSED (only %.1f GiB RAM free after the switch, the floor is 8 GiB); "
                     "the sidecar goes unpinned and the decode reads page-fault\n",
                     (double) ram_post / (double) (size_t(1) << 30));
        cudaHostUnregister((void*) r2d_base_);
        r2d_pinned_ = false;
        return false;
    }
    return true;
}

// A slot is a whole number of blocks of both of the experts' types, so the slots of a layer that lie one behind
// the other are the matrices of one launch (moe_rows()), and of 8 bytes, which keeps every block where its type
// may be read.  The cache's slots and --pcie-frac's staging slots are both this size.
size_t Engine::expert_slot_bytes(const Layer& L) const {
    const size_t unit = std::lcm(std::lcm(ggml_type_size((ggml_type) L.gu_type),
                                          ggml_type_size((ggml_type) L.d_type)),
                                 size_t(8));
    return (L.slot_down() + L.d_bytes + kTail + unit - 1) / unit * unit;
}

void Engine::fill_expert_cache() {
    if (!opt_.gpu_experts || n_moe_ == 0) return;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const size_t ram = ram_available(), files = gg_->bytes() + (dg_ ? dg_->bytes() : 0);
    ram_tight_ = ram > 0 && files + (4ULL << 30) > ram;
    // 256 MiB at least, whatever was asked: the launches recorded later take theirs out of the reserve (mind_vram())
    reserve_mib_ = opt_.vram_reserve_mib >= 0 ? std::max(opt_.vram_reserve_mib, 256)
                                             : default_reserve_mib(total_b, ram_tight_);
    const size_t reserve = size_t(reserve_mib_) << 20;
    // STRATA_PROMPT_BORROW: the cache is sized as it is when the prompt's staging buffers hold their VRAM (a run
    // without the switch), so it holds the same experts in the same slots: take that VRAM for a moment, measure
    // what one buffer costs, and give it back - the buffers are lent the cache's tail instead of holding their own.
    if (borrow_lend_) {
        if (stage_alloc_bytes_ == 0) {
            borrow_lend_ = false;
            std::fprintf(stderr, "strata-poly: STRATA_PROMPT_BORROW: no staging buffer to lend: nothing is lent\n");
        } else {
            const size_t before = free_b;
            void* probe = nullptr;
            if (cudaMalloc(&probe, stage_alloc_bytes_) != cudaSuccess) {
                cudaGetLastError();
                throw std::runtime_error("prompt borrow: the prompt's staging buffer (" +
                                         std::to_string(stage_alloc_bytes_ >> 20) + " MiB) do not fit: " +
                                         std::to_string(free_b >> 20) + " MiB of VRAM free, " +
                                         std::to_string(reserve_mib_) + " MiB kept for what comes later");
            }
            ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
            cudaFree(probe);
            borrow_one_buffer_ = before > free_b ? before - free_b : stage_alloc_bytes_;
        }
    }
    if (free_b <= reserve) {
        if (borrow_lend_)
            throw std::runtime_error("prompt borrow: the prompt's staging buffer do not fit: " +
                                     std::to_string(free_b >> 20) + " MiB of VRAM free and " +
                                     std::to_string(reserve_mib_) + " MiB kept for what comes later");
        return;
    }

    struct Cand {
        int il, e;
        size_t bytes;
        double value;
    };
    std::vector<Cand> cands;
    std::vector<size_t> sizes;   // the slot sizes there are; a layer's place among them is its trade number
    size_t all = 0;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        Layer& L = layers_[(size_t) il];
        if (!L.moe || !kernels::native_mmvq_supported((int) L.gu_type) ||
            !kernels::native_mmvq_supported((int) L.d_type))
            continue;
        const size_t bytes = expert_slot_bytes(L);
        L.slot_bytes = bytes;
        if (std::find(sizes.begin(), sizes.end(), bytes) == sizes.end()) sizes.push_back(bytes);
        L.trade = (int) (std::find(sizes.begin(), sizes.end(), bytes) - sizes.begin());
        slot_max_ = std::max(slot_max_, bytes);
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            const double used = (double) ((prior_.empty() ? 0 : prior_[i]) + use_[i]);
            cands.push_back({il, e, bytes, used / (double) bytes});
            all += bytes;
        }
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.value != b.value ? a.value > b.value : a.e < b.e; });

    n_trade_ = (int) sizes.size();
    // sized again (resize_cache()), it is no larger than load() made it, or than its addresses with VMM
    const size_t most = vmm_ ? std::min(all, total_b) : full_bytes_ > 0 ? full_bytes_ : all;
    size_t arena = std::min({all, free_b - reserve, arena_.base ? arena_.reserved - 256 : most});
    if (arena == 0) return;
    if (vmm_) {   // in pieces, so that it can give some up to the attention caches (shrink_cache()) and grow
        if (!arena_.base) vmm_reserve(arena_, most + 256);   // addresses only: grow_cache() maps more of them
        try {
            vmm_map(arena_, arena + 256, kArenaPiece, "the expert cache");
        } catch (const std::runtime_error&) {   // the card may not have it all (grow_cache() is just as tolerant)
            arena = arena_.mapped > 256 ? arena_.mapped - 256 : 0;
            if (arena == 0) return;
        }
        exp_ = (uint8_t*) arena_.base;
        budget_ = free_b - reserve;   // when it holds every expert, what is left over goes to the rows first
        for (const Kv& k : kv_) budget_ += k.mapped;
    } else {
        exp_ = (uint8_t*) dmalloc(arena + 256, "the expert cache");
    }
    ck(cudaMemset(exp_, 0, arena + 256), "the expert cache");   // the slots' tails stay zero (kTail)
    // STRATA_PROMPT_BORROW: the region the staging buffers are lent is mapped here, before the experts fill the
    // cache: it is the tail slots the cache is too small for, and a lend that does not fit fails loudly (C1).
    if (borrow_lend_) map_prompt_borrow();
    // Which experts is by value; where they go is by layer, a layer's side by side in the experts' order.
    std::vector<Cand> taken;
    std::vector<char> is_taken(cands.size(), 0);
    size_t room = arena;
    for (size_t i = 0; i < cands.size(); ++i) {
        const Cand& c = cands[i];
        if (c.bytes > room) continue;   // a smaller expert of another layer may still fit
        room -= c.bytes;
        taken.push_back(c);
        is_taken[i] = 1;
    }
    std::sort(taken.begin(), taken.end(),
              [](const Cand& a, const Cand& b) { return a.il != b.il ? a.il < b.il : a.e < b.e; });
    for (const Cand& c : taken) {
        Layer& L = layers_[(size_t) c.il];
        uint8_t* s = exp_ + exp_at_;
        exp_at_ += c.bytes;
        const size_t e = (size_t) c.e;
        if (L.gated)
            ck(cudaMemcpy(s, L.exp_gate + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice), "expert upload");
        ck(cudaMemcpy(s + L.slot_up(), L.exp_up + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        if (r2d_base_)
            materialize_down(L, (int) (&L - layers_.data()), e, s + L.slot_down(), nullptr);
        else ck(cudaMemcpy(s + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        ++fill_dmas_;
        fill_dma_bytes_ += (L.gated ? L.gu_bytes : 0) + L.gu_bytes + L.d_bytes;
        L.slot[e] = s;
        ++cached_;
    }
    if (full_bytes_ == 0) {
        full_bytes_ = arena;
        full_cached_ = cached_;
    }
    if (borrow_lend_ && borrow_bytes_ > 0) {
        // The experts the lent slots stand in for: the ones the cache would hold next, lowest rank first (a lent
        // slot's expert streams during the prompt and is refilled after it).  Kept so the refill needs no second
        // sizing pass and cannot pick a different set than the fill would have.
        borrow_cands_.clear();
        size_t hold = borrow_bytes_;
        for (size_t i = 0; i < cands.size(); ++i) {
            if (is_taken[i]) continue;
            if (cands[i].bytes > hold) continue;
            hold -= cands[i].bytes;
            borrow_cands_.push_back({cands[i].il, cands[i].e, cands[i].bytes});
        }
        if (borrow_cands_.empty())
            throw std::runtime_error("prompt borrow: the lent slots have no expert to stand in for (the cache is not "
                                     "full)");
    }
    if (std::getenv("STRATA_PROMPT_BORROW_DEBUG"))
        std::fprintf(stderr, "cache-debug: free_b=%zu MiB reserve=%zu MiB arena=%zu MiB arena_mapped=%zu MiB "
                             "cached=%d exp_at=%zu MiB borrow=%zu MiB\n",
                     free_b >> 20, reserve_mib_, arena >> 20, arena_.mapped >> 20, cached_, exp_at_ >> 20,
                     borrow_bytes_ >> 20);
    if (r2d_base_) ck(cudaStreamSynchronize(nullptr), "down T sync");   // P2: default stream done before use
}

// STRATA_PROMPT_BORROW: the region the prompt's staging buffers are lent from the expert cache's tail.  It is a
// mapping of its own - cover() and shrink_cache() only ever cut arena_ - and the cache was sized as if the buffers
// held their own VRAM (fill_expert_cache()), so the region is exactly the slots the cache has no room for.  A lend
// that does not fit is loud (C1): never a smaller one, never a cache built on top of the buffers.
void Engine::map_prompt_borrow() {
    const int buffers = ring2_ ? 2 : 1;
    const size_t need = (size_t) buffers * borrow_one_buffer_;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const size_t keep = size_t(kBorrowMinFreeMib) << 20;
    const size_t cap_bytes = free_b > keep ? free_b - keep : 0;
    const size_t cap_slots = std::min<size_t>((size_t) kBorrowMaxSlots, cap_bytes / std::max<size_t>(slot_max_, 1));
    size_t want = need;
    if (borrow_slots_req_ > 0) {   // STRATA_PROMPT_BORROW_SLOTS: force the lend (an A/B and the conflict check)
        want = (size_t) borrow_slots_req_ * slot_max_;
        if (want < need)
            throw std::runtime_error("prompt borrow: " + std::to_string(borrow_slots_req_) + " slots (" +
                                     std::to_string(want >> 20) + " MiB) do not fit the prompt's " +
                                     std::to_string(buffers) + " staging buffer(s) (" + std::to_string(need >> 20) +
                                     " MiB)");
        if (want > cap_bytes)
            throw std::runtime_error("prompt borrow: " + std::to_string(borrow_slots_req_) + " slots (" +
                                     std::to_string(want >> 20) + " MiB) do not fit: the cache can lend " +
                                     std::to_string(cap_slots) + " slots (" + std::to_string(cap_bytes >> 20) +
                                     " MiB) and " + std::to_string(kBorrowMinFreeMib) + " MiB stay free");
    }
    if (want > cap_bytes) {   // auto: what the buffers need; a second buffer that does not fit holds its own
        if (buffers > 1 && borrow_one_buffer_ <= cap_bytes) {
            std::fprintf(stderr, "strata-poly: STRATA_PROMPT_BORROW: %.0f MiB can be lent: one staging buffer is lent, "
                                 "the second holds its own VRAM\n", (double) (cap_bytes >> 20));
            want = borrow_one_buffer_;
        } else {
            throw std::runtime_error("prompt borrow: the prompt's staging buffer(s) (" + std::to_string(need >> 20) +
                                     " MiB) do not fit: " + std::to_string(cap_bytes >> 20) + " MiB can be lent and " +
                                     std::to_string(kBorrowMinFreeMib) + " MiB stay free");
        }
    }
    vmm_reserve(borrow_, want + 256);
    vmm_map(borrow_, want, kArenaPiece, "the borrowed prompt buffers");
    ck(cudaMemset((void*) borrow_.base, 0, want), "the borrowed prompt buffers");
    borrow_bytes_ = want;
    borrow_buffers_ = (int) std::min<size_t>((size_t) buffers, want / std::max<size_t>(borrow_one_buffer_, 1));
    borrow_off_[0] = 0;
    borrow_off_[1] = borrow_one_buffer_;
    if (std::getenv("STRATA_PROMPT_BORROW_DEBUG"))
        std::fprintf(stderr,
                     "borrow-debug: free=%zu MiB keep=%zu MiB cap=%zu MiB cap_slots=%zu need=%zu MiB want=%zu MiB "
                     "buffers=%d one_buffer=%.2f MiB region_slots=%zu arena=%zu MiB cached=%d\n",
                     free_b >> 20, keep >> 20, cap_bytes >> 20, cap_slots, need >> 20, want >> 20, borrow_buffers_,
                     (double) borrow_one_buffer_ / (1024.0 * 1024.0), want / std::max<size_t>(slot_max_, 1),
                     arena_.mapped >> 20, cached_);
}

// STRATA_PROMPT_BORROW: a prompt is about to be read and its buffers are written into the lent region.  The experts
// refilled there by the answer before go out again first: their slots are the buffers' bytes now.
void Engine::prompt_borrow_begin() {
    if (!borrow_lend_ || borrow_bytes_ == 0 || prompt_lent_) return;
    finish_async_refill();   // a refill still pending must be in: the region is about to be the staging buffers
    if (borrow_lent_ > 0) {
        for (const Lent& c : borrow_cands_) {
            Layer& L = layers_[(size_t) c.il];
            if (L.slot[(size_t) c.e] != nullptr) {
                L.slot[(size_t) c.e] = nullptr;
                --cached_;
            }
        }
        borrow_lent_ = 0;
    }
    prompt_lent_ = true;
}

// STRATA_PROMPT_BORROW: the prompt is read and done with its buffers: put the experts they stand in for back, so the
// answer is written from the whole cache.  Its cost is prompt time (the caller times the prompt around it).
void Engine::prompt_borrow_end() {
    if (!prompt_lent_) return;
    prompt_lent_ = false;
    if (borrow_bytes_ == 0) return;
    // the prompt's last copies and kernels must be done with the region before it holds experts again
    ck(cudaStreamSynchronize(cs(st_)), "prompt borrow");
    if (up_) ck(cudaStreamSynchronize(cs(up_)), "prompt borrow");
    if (cp_) ck(cudaStreamSynchronize(cs(cp_)), "prompt borrow");
    if (!borrow_refill_) {
        std::fprintf(stderr, "borrow: lent %d slots (%.1f MB), refill 0.0 ms (STRATA_PROMPT_BORROW_REFILL=0: nothing "
                             "put back, the cache is the one without the switch)\n",
                     (int) (borrow_bytes_ / std::max<size_t>(slot_max_, 1)), (double) (borrow_bytes_ >> 20));
        return;
    }
    refill_prompt_borrow();
}

// STRATA_PROMPT_BORROW: the experts the lent slots stand in for, back into the region side by side by layer as
// fill_expert_cache() laid the cached ones.  The region is zeroed first (the kernels read up to kTail past a matrix);
// the time is measured and printed with the lend, and a slot that does not hold its expert is loud (verify).
void Engine::refill_prompt_borrow() {
    const Clock::time_point t0 = Clock::now();
    ck(cudaMemsetAsync((void*) borrow_.base, 0, borrow_bytes_, cs(st_)), "the borrowed prompt buffers");
    std::vector<Lent> took;
    took.reserve(borrow_cands_.size());
    size_t room = borrow_bytes_;
    for (const Lent& c : borrow_cands_) {
        if (c.bytes > room) continue;
        room -= c.bytes;
        took.push_back(c);
    }
    std::sort(took.begin(), took.end(),
              [](const Lent& a, const Lent& b) { return a.il != b.il ? a.il < b.il : a.e < b.e; });
    if (borrow_async_) {   // STRATA_BORROW_ASYNC_REFILL: the copies go on their own stream, the answer starts now
        start_async_refill(std::move(took));
        return;
    }
    size_t at = 0, bytes = 0;
    for (const Lent& c : took) {
        Layer& L = layers_[(size_t) c.il];
        uint8_t* const s = (uint8_t*) borrow_.base + at;
        at += c.bytes;
        bytes += c.bytes;
        const size_t e = (size_t) c.e;
        if (L.gated)
            ck(cudaMemcpyAsync(s, L.exp_gate + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice, cs(st_)),
               "expert refill");
        ck(cudaMemcpyAsync(s + L.slot_up(), L.exp_up + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice, cs(st_)),
           "expert refill");
        if (r2d_base_)
            materialize_down(L, (int) (&L - layers_.data()), e, s + L.slot_down(), cs(st_));
        else ck(cudaMemcpyAsync(s + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice,
                                cs(st_)),
           "expert refill");
        L.slot[e] = s;
        ++cached_;
    }
    ck(cudaStreamSynchronize(cs(st_)), "prompt borrow refill");
    borrow_lent_ = (int) took.size();
    borrow_refill_ms_ = ms_since(t0);
    borrow_refill_bytes_ = bytes;
    std::fprintf(stderr, "borrow: lent %d slots (%.1f MB), refill %.1f ms (%zu experts, %.0f MB)\n",
                 (int) (borrow_bytes_ / std::max<size_t>(slot_max_, 1)), (double) (borrow_bytes_ >> 20),
                 borrow_refill_ms_, took.size(), (double) (bytes >> 20));
    if (std::getenv("STRATA_PROMPT_BORROW_DEBUG"))
        std::fprintf(stderr, "borrow-debug: refilled %zu experts (%zu MB) into %zu MiB, cached=%d, exp_at=%zu MB\n",
                     took.size(), bytes >> 20, borrow_bytes_ >> 20, cached_, exp_at_ >> 20);
    if (borrow_verify_ > 0) verify_prompt_borrow(took);
}

// STRATA_PROMPT_BORROW_CHECK: every refilled slot holds its expert byte for byte, and slots that were not lent are
// untouched (the control).  A mismatch is loud: an answer written from that cache would be wrong.
void Engine::verify_prompt_borrow(const std::vector<Lent>& took) {
    const auto same = [](const Layer& L, int e, const uint8_t* slot) -> bool {
        std::vector<uint8_t> buf(L.slot_bytes);
        if (cudaMemcpy(buf.data(), slot, L.slot_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        const size_t es = (size_t) e;
        if (L.gated && std::memcmp(buf.data(), L.exp_gate + es * L.gu_stride, L.gu_bytes) != 0) return false;
        if (std::memcmp(buf.data() + L.slot_up(), L.exp_up + es * L.gu_stride, L.gu_bytes) != 0) return false;
        if (std::memcmp(buf.data() + L.slot_down(), L.exp_down + es * L.d_bytes, L.d_bytes) != 0) return false;
        return true;
    };
    int bad = 0, refilled_ok = 0, control_ok = 0, control = 0;
    const size_t n = std::min<size_t>((size_t) borrow_verify_, took.size());
    for (size_t i = 0; i < n; ++i) {
        const Layer& L = layers_[(size_t) took[i].il];
        if (same(L, took[i].e, L.slot[(size_t) took[i].e])) ++refilled_ok; else ++bad;
    }
    // the control: resident experts that were not lent (their slots lie in arena_, not in the borrowed region)
    for (int il = 0; il < (int) layers_.size() && control < borrow_verify_; ++il) {
        const Layer& L = layers_[(size_t) il];
        if (!L.moe) continue;   // a layer without routed experts has no slots
        for (int e = 0; e < n_expert_ && control < borrow_verify_; ++e) {
            const uint8_t* const s = L.slot[(size_t) e];
            if (s == nullptr || s < exp_ || s >= exp_ + arena_.mapped) continue;
            if (same(L, e, s)) ++control_ok; else ++bad;
            ++control;
        }
    }
    std::fprintf(stderr, "borrow verify: refilled %d/%zu slots hold their expert, control %d/%d not-lent slots\n",
                 refilled_ok, n, control_ok, control);
    if (bad > 0)
        throw std::runtime_error("prompt borrow: a slot does not hold its expert after the refill: the cache is wrong");
}

// STRATA_BORROW_ASYNC_REFILL: the refill's copies go on a stream of their own.  The slots stay not resident until
// they land (install_async_refill()), so the answer's first rounds take those experts' CPU path: correct, just
// slower, and nothing reads a half expert.  coming_ keeps follow() from bringing the same expert in a second time.
void Engine::start_async_refill(std::vector<Lent> took) {
    if (as_ == nullptr) {
        cudaStream_t as = nullptr;
        ck(cudaStreamCreateWithFlags(&as, cudaStreamNonBlocking), "cudaStreamCreate");
        as_ = as;
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        ev_refill_ = ev;
    }
    ck(cudaMemsetAsync((void*) borrow_.base, 0, borrow_bytes_, cs(as_)), "the borrowed prompt buffers");
    const size_t ne = (size_t) n_expert_;
    size_t at = 0;
    for (const Lent& c : took) {
        Layer& L = layers_[(size_t) c.il];
        uint8_t* const s = (uint8_t*) borrow_.base + at;
        at += c.bytes;
        const size_t e = (size_t) c.e;
        if (L.gated)
            ck(cudaMemcpyAsync(s, L.exp_gate + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice, cs(as_)),
               "expert refill");
        ck(cudaMemcpyAsync(s + L.slot_up(), L.exp_up + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice, cs(as_)),
           "expert refill");
        if (r2d_base_)
            materialize_down(L, (int) (&L - layers_.data()), e, s + L.slot_down(), cs(as_));
        else
            ck(cudaMemcpyAsync(s + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice,
                               cs(as_)),
               "expert refill");
        if (!coming_.empty()) coming_[(size_t) c.il * ne + e] = 1;   // follow() must not bring it in too
    }
    ck(cudaEventRecord((cudaEvent_t) ev_refill_, cs(as_)), "expert refill");
    async_took_ = std::move(took);
    async_refill_ = true;
    async_pos0_ = pos_;
    async_miss0_ = stats_.cpu_experts;   // the pending window's misses: its experts take the CPU path until they land
    async_tok0_ = stats_.tokens;
    async_t0_s_ = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    std::fprintf(stderr, "borrow: async refill started, %zu slots pending (their experts take the CPU path until "
                         "they land)\n", async_took_.size());
}

// The copies have landed: the experts become resident (their slots, the cache's count) so the rounds after this one
// see them.  The time and the tokens that ran while they were in flight are the self-check line.
void Engine::install_async_refill() {
    const size_t ne = (size_t) n_expert_;
    size_t at = 0, bytes = 0;
    for (const Lent& c : async_took_) {
        Layer& L = layers_[(size_t) c.il];
        const size_t e = (size_t) c.e;
        L.slot[e] = (uint8_t*) borrow_.base + at;
        at += c.bytes;
        bytes += c.bytes;
        if (!coming_.empty()) coming_[(size_t) c.il * ne + e] = 0;
        ++cached_;
    }
    borrow_refill_bytes_ = bytes;
    borrow_lent_ = (int) async_took_.size();
    borrow_refill_ms_ =
        (std::chrono::duration<double>(Clock::now().time_since_epoch()).count() - async_t0_s_) * 1000.0;
    // The self-check must let a reader take this window's decode apart: the pending slots' experts go the CPU path,
    // so their window has more misses than the answer's later one - the count is printed, not left to be guessed.
    const int64_t d_miss = stats_.cpu_experts - async_miss0_, d_tok = stats_.tokens - async_tok0_;
    std::fprintf(stderr,
                 "borrow: async refill done in %.1f ms (overlapped with %lld tokens; %zu slots pending, +%lld CPU "
                 "misses = %.2f/token over the window)\n",
                 borrow_refill_ms_, (long long) (pos_ - async_pos0_), async_took_.size(), (long long) d_miss,
                 d_tok > 0 ? (double) d_miss / (double) d_tok : 0.0);
    if (borrow_verify_ > 0) verify_prompt_borrow(async_took_);
    async_took_.clear();
    async_refill_ = false;
}

// run()'s start: when the refill's event says its copies are in, the slots become resident before this call's layers
// look at the cache.  A failure is loud and leaves those experts on the CPU path for good (never half data).
void Engine::poll_async_refill() {
    if (!async_refill_) return;
    const cudaError_t st = cudaEventQuery((cudaEvent_t) ev_refill_);
    if (st == cudaErrorNotReady) return;
    if (st != cudaSuccess) {
        cudaGetLastError();
        const size_t lost = async_took_.size();
        for (const Lent& c : async_took_)
            if (!coming_.empty()) coming_[(size_t) c.il * (size_t) n_expert_ + (size_t) c.e] = 0;
        async_took_.clear();
        async_refill_ = false;
        std::fprintf(stderr, "borrow: async refill FAILED (%s): %zu slots stay not resident, their experts take the "
                             "CPU path\n", cudaGetErrorString(st), lost);
        return;
    }
    install_async_refill();
}

// The answer is over (or another prompt is about to use the region): the refill must be done and its slots installed
// before the region is touched again - its work is never dropped.
void Engine::finish_async_refill() {
    if (!async_refill_) return;
    const cudaError_t st = cudaStreamSynchronize(cs(as_));
    if (st != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stderr, "borrow: async refill did not finish (%s); its %zu slots stay not resident\n",
                     cudaGetErrorString(st), async_took_.size());
        for (const Lent& c : async_took_)
            if (!coming_.empty()) coming_[(size_t) c.il * (size_t) n_expert_ + (size_t) c.e] = 0;
        async_took_.clear();
        async_refill_ = false;
        return;
    }
    std::fprintf(stderr, "borrow: async refill finished at the answer's end (%zu slots pending installed)\n",
                 async_took_.size());
    install_async_refill();
}

// The thread and the stream that bring experts into the cache while an answer is written (follow()).
void Engine::start_swaps() {
    if (swap_ || cached_ == 0 || opt_.expert_swaps == 0) return;
    swap_ = std::make_unique<Swap[]>((size_t) opt_.expert_swaps);
    cudaStream_t up = nullptr;
    ck(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking), "cudaStreamCreate");
    swap_st_ = up;
    for (; n_swap_ < opt_.expert_swaps; ++n_swap_) {
        Swap& s = swap_[(size_t) n_swap_];
        s.pinned = (uint8_t*) hmalloc(slot_max_, "pinned memory");
        ck(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "cudaEventCreate");
    }
    recent_.assign(use_.size(), 0.f);
    coming_.assign(use_.size(), 0);
    follower_ = std::thread([this] { follow_loop(); });
}

// The cache gives up what it has and is filled again for another reserve.  The experts on their way into it
// arrive first: their slots are about to go.
void Engine::resize_cache(int mib) {
    for (Engine* e = this; e; e = e->next_.get()) e->resize_card_cache(mib);
}

void Engine::resize_card_cache(int mib) {
    if (!opt_.gpu_experts || n_moe_ == 0) return;
    // STRATA_PROMPT_BORROW: the staging buffers are the cache's lent tail, so a cache sized again would have to
    // take them back: it is refused loudly rather than leave the prompt path writing into a cache's slots.
    if (borrow_bytes_ > 0)
        throw std::runtime_error("resize_cache is not available with STRATA_PROMPT_BORROW: the prompt's staging "
                                 "buffers are lent the expert cache's tail; start the engine without the switch");
    OnCard here(device_, split_);
    ck(cudaStreamSynchronize(cs(st_)), "resize");
    const auto failed = [&] {
        std::lock_guard<std::mutex> l(swap_m_);
        return !swap_error_.empty();
    };
    for (int j = 0; j < n_swap_; ++j)   // one that is asked for is the follower's, now or in a moment
        while (swap_[(size_t) j].at.load() == Swap::kAsked && !failed()) std::this_thread::yield();
    if (failed()) throw std::runtime_error("expert upload: " + swap_error_);
    if (swap_st_) ck(cudaStreamSynchronize(cs(swap_st_)), "resize");
    for (int j = 0; j < n_swap_; ++j) swap_[(size_t) j].at.store(Swap::kFree);
    std::fill(coming_.begin(), coming_.end(), (uint8_t) 0);
    for (Layer& L : layers_) std::fill(L.slot.begin(), L.slot.end(), nullptr);
    staged_il_ = -1;
    cached_ = 0;
    exp_at_ = 0;
    if (exp_ && vmm_) {
        vmm_unmap(arena_, 0);
        exp_ = nullptr;
    } else if (exp_) {
        dev_allocs_.erase(std::find(dev_allocs_.begin(), dev_allocs_.end(), (void*) exp_));
        ck(cudaFree(exp_), "resize");
        exp_ = nullptr;
    }
    opt_.vram_reserve_mib = mib < 0 ? asked_reserve_mib_ : mib;
    fill_expert_cache();
    start_swaps();
}

// The launches the engine records as it runs take VRAM after the cache was sized: 180 to 200 MiB of an 8 GB card,
// most of it in the first request.  That is what the reserve is for.  From the second request on the cache keeps
// `keep` free: the reserve the caller named, else half the one it was filled for and 128 MiB at least, or the
// whole of it when the PC's RAM is tight for the model (default_reserve_mib(): with the reserve named, the laptop
// read as the release did; with half of it kept, 7% to 14% slower).  It gives back what another program took, its
// least used experts, and takes VRAM that came free again for the most used experts it lacks.  Filled again
// whole it went down with every program that held VRAM a moment, and never came back up.
bool Engine::mind_vram() {
    bool changed = false;
    for (Engine* e = this; e; e = e->next_.get()) changed = e->mind_card_vram() || changed;
    return changed;
}

bool Engine::mind_card_vram() {
    if (!exp_ || reserve_mib_ <= 0) return false;
    OnCard here(device_, split_);
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const int64_t free = (int64_t) (free_b >> 20);   // this card's
    if (!vmm_) {   // a cache in one allocation is filled again
        if (free >= reserve_mib_ / 4) return false;
        resize_card_cache(opt_.vram_reserve_mib);
        return true;
    }
    if (minds_++ == 0) return false;   // nothing is recorded before the first request
    const int64_t keep = opt_.vram_reserve_mib >= 0 || ram_tight_ ? reserve_mib_ : std::max(reserve_mib_ / 2, 128);
    if (free < keep / 2) {
        shrink_cache(size_t(keep - free) << 20);
        held_is_budget();
        return true;
    }
    return free >= keep + (int64_t) (kArenaPiece >> 20) && grow_cache(size_t(free - keep) << 20) > 0;
}

// The cache's VRAM and the attention caches' as they are now become the budget cover() keeps them to.
void Engine::held_is_budget() {
    budget_ = arena_.mapped;
    for (const Kv& k : kv_) budget_ += k.mapped;
}

// `bytes` more VRAM at the cache's end, for the experts it lacks that are used most (use per byte, as
// fill_expert_cache() chooses), side by side by layer.  VRAM the card does not give is not an error: the cache
// keeps what was mapped.  Returns the experts it took.
int Engine::grow_cache(size_t bytes) {
    if (!vmm_ || !exp_) return 0;
    settle_swaps();
    ck(cudaStreamSynchronize(cs(st_)), "the expert cache");
    if (swap_st_) ck(cudaStreamSynchronize(cs(swap_st_)), "the expert cache");
    struct Cand {
        int il, e;
        double value;
    };
    std::vector<Cand> cands;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            if (L.slot[(size_t) e] || (!coming_.empty() && coming_[i])) continue;
            const double used = (double) ((prior_.empty() ? 0 : prior_[i]) + use_[i]);
            cands.push_back({il, e, used / (double) L.slot_bytes});
        }
    }
    if (cands.empty()) return 0;
    try {
        vmm_map(arena_, std::min(arena_.mapped + bytes, arena_.reserved), kArenaPiece, "the expert cache");
    } catch (const std::runtime_error&) {
    }
    if (arena_.mapped < exp_at_ + kTail) return 0;
    ck(cudaMemset(exp_ + exp_at_, 0, arena_.mapped - exp_at_), "the expert cache");   // the new slots' tails
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.value != b.value ? a.value > b.value : a.e < b.e; });
    std::vector<Cand> taken;
    size_t room = arena_.mapped - kTail - exp_at_;
    for (const Cand& c : cands) {
        const size_t n = layers_[(size_t) c.il].slot_bytes;
        if (n > room) continue;
        room -= n;
        taken.push_back(c);
    }
    std::sort(taken.begin(), taken.end(),
              [](const Cand& a, const Cand& b) { return a.il != b.il ? a.il < b.il : a.e < b.e; });
    for (const Cand& c : taken) {
        Layer& L = layers_[(size_t) c.il];
        uint8_t* s = exp_ + exp_at_;
        exp_at_ += L.slot_bytes;
        const size_t e = (size_t) c.e;
        if (L.gated)
            ck(cudaMemcpy(s, L.exp_gate + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice), "expert upload");
        ck(cudaMemcpy(s + L.slot_up(), L.exp_up + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        if (r2d_base_)
            materialize_down(L, (int) (&L - layers_.data()), e, s + L.slot_down(), nullptr);
        else ck(cudaMemcpy(s + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        ++fill_dmas_;
        fill_dma_bytes_ += (L.gated ? L.gu_bytes : 0) + L.gu_bytes + L.d_bytes;
        L.slot[e] = s;
        ++cached_;
    }
    staged_il_ = -1;
    if (r2d_base_) ck(cudaStreamSynchronize(0), "down T sync");   // D2a on the default stream: finish before use
    held_is_budget();
    return (int) taken.size();
}

// What a layer copies to the staging buffer ahead of its routing: the experts outside the cache, the most used
// first, as many as the buffer holds.  Worked out when the layer runs: the cache changes (follow()).
void Engine::plan_ahead(Layer& L, int il) {
    L.ahead.clear();
    for (int e = 0; e < n_expert_; ++e)
        if (!L.slot[(size_t) e]) L.ahead.push_back(e);
    if (!prior_.empty()) {
        const int64_t* used = prior_.data() + (size_t) il * n_expert_;
        std::stable_sort(L.ahead.begin(), L.ahead.end(), [used](int32_t a, int32_t b) { return used[a] > used[b]; });
    }
    // STRATA_AHEAD_EXACT=1: the buffer holds what stage() writes for an expert - the expert's own slot layout,
    // slot_down() + d_bytes (for a gated expert that is gate+up+down; for Nemotron's non-gated one, up+down) -
    // not the gated worst case 2*gu_bytes + d_bytes. On this model that is 341 -> 512 experts ahead in the same
    // 1512 MiB buffer: no extra VRAM, only the list the stage() DMA fills under the layer's attention grows.
    // Off (the default): the release path's 2*gu_bytes + d_bytes, byte for byte the list it was.
    static const bool ahead_exact = [] {
        const char* e = std::getenv("STRATA_AHEAD_EXACT");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    const size_t per_expert = ahead_exact ? L.slot_down() + L.d_bytes : 2 * (size_t) L.gu_bytes + L.d_bytes;
    const size_t cap = std::min<size_t>((size_t) n_expert_, stage_bytes_ / std::max<size_t>(per_expert, 1));
    L.ahead.resize(std::min(L.ahead.size(), cap));
    if (ahead_exact) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            std::fprintf(stderr,
                         "plan_ahead: %zu experts ahead (footprint %.2f MB/expert, %s layout; %.0f MiB staging)\n", cap,
                         (double) per_expert / (1024.0 * 1024.0), L.gated ? "gated" : "actual (up+down)",
                         (double) (stage_bytes_ >> 20));
    }
}

// The follower: an asked-for expert goes from the mapped file to its pinned buffer, and from there to its slot.
// The file may have to be read for it, which is why this is not the engine's thread.
namespace {
// STRATA_SWAP_DIRECT_FROM_PIN (default 0): a swap of a pinned layer's expert copies the three slot parts straight
// from the page-locked mapping instead of memcpy -> pinned -> one DMA.  The DMA sources are already resident and
// registered; this only stops the follower from taking host bandwidth while the decode reads its misses.
bool swap_direct_from_pin() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SWAP_DIRECT_FROM_PIN");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}
}  // namespace

void Engine::follow_loop() {
    cudaSetDevice(device_);   // a thread starts on card 0
    affinity::pin_aux_thread();   // --aux-cpus: no-op unless the switch named a processor for this thread
    std::unique_lock<std::mutex> lk(swap_m_);
    for (;;) {
        Swap* s = nullptr;
        swap_cv_.wait(lk, [&] {
            for (int j = 0; j < n_swap_ && !s; ++j)
                if (swap_[(size_t) j].at.load() == Swap::kAsked) s = &swap_[(size_t) j];
            return swap_quit_ || s;
        });
        if (swap_quit_) return;
        lk.unlock();
        const Layer& L = layers_[(size_t) s->il];
        const size_t e = (size_t) s->e, gu = L.gu_bytes;
        if (L.gated) std::memcpy(s->pinned, L.exp_gate + e * L.gu_stride, gu);
        std::memcpy(s->pinned + L.slot_up(), L.exp_up + e * L.gu_stride, gu);
        cudaError_t err = cudaSuccess;
        if (r2d_base_) {   // gate|up first, then the down materialized T->row straight into the slot
            err = cudaMemcpyAsync(s->slot, s->pinned, L.slot_down(), cudaMemcpyHostToDevice, cs(swap_st_));
            if (err == cudaSuccess) materialize_down(L, s->il, e, s->slot + L.slot_down(), cs(swap_st_));
        } else if (swap_direct_from_pin() && s->il >= 0 && (size_t) s->il < pin_layer_.size() &&
                   pin_layer_[(size_t) s->il]) {
            // #3': the layer's experts live in the pinned GGUF mapping; copy the slot straight from it.  No host
            // memcpy -> pinned on the follower, so a swap takes no CPU bandwidth and no lock while a miss reads.
            static std::atomic<bool> said{false};
            if (!said.exchange(true))
                std::fprintf(stderr, "strata-poly: swap-direct: from the pinned mapping "
                                     "(STRATA_SWAP_DIRECT_FROM_PIN=1)\n");
            if (L.gated)
                err = cudaMemcpyAsync(s->slot, L.exp_gate + e * L.gu_stride, gu, cudaMemcpyHostToDevice,
                                      cs(swap_st_));
            if (err == cudaSuccess)
                err = cudaMemcpyAsync(s->slot + L.slot_up(), L.exp_up + e * L.gu_stride, gu,
                                      cudaMemcpyHostToDevice, cs(swap_st_));
            if (err == cudaSuccess)
                err = cudaMemcpyAsync(s->slot + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes,
                                      cudaMemcpyHostToDevice, cs(swap_st_));
        } else {
            std::memcpy(s->pinned + L.slot_down(), L.exp_down + e * L.d_bytes, L.d_bytes);
            err = cudaMemcpyAsync(s->slot, s->pinned, L.slot_down() + L.d_bytes, cudaMemcpyHostToDevice,
                                  cs(swap_st_));
        }
        if (err == cudaSuccess) {
            ++swap_dmas_;
            swap_dma_bytes_ += (size_t) (L.slot_down() + L.d_bytes);
        }
        if (err == cudaSuccess) err = cudaEventRecord(s->done, cs(swap_st_));
        lk.lock();
        if (err != cudaSuccess) {
            swap_error_ = cudaGetErrorString(err);
            return;
        }
        s->at.store(Swap::kSent);
    }
}

// The cache follows what is written.  Called at the start of an eval, where nothing queued reads the cache, with
// the number of tokens the eval writes (0 for a prompt batch).  The experts whose upload has arrived get their
// slots.  Then, for written tokens, the experts used lately that the cache lacks start on their way, each to the
// slot of the cached expert of its size that was used least (among equals, the one the profile ranks last).
//
// A prompt batch neither counts nor moves anything.  What a prompt routes to says no more about the answer than
// the profile does, and moving the cache while the prompt was read cost 3% of the reading speed (measured).
void Engine::follow(int written) {
    {
        std::lock_guard<std::mutex> l(swap_m_);
        if (!swap_error_.empty()) throw std::runtime_error("expert upload: " + swap_error_);
    }
    const size_t ne = (size_t) n_expert_;
    int n_free = 0;
    for (int j = 0; j < n_swap_; ++j) {
        Swap& s = swap_[(size_t) j];
        if (s.at.load() == Swap::kSent) {
            const cudaError_t arrived = cudaEventQuery(s.done);
            if (arrived == cudaErrorNotReady) continue;
            ck(arrived, "expert upload");
            layers_[(size_t) s.il].slot[(size_t) s.e] = s.slot;
            coming_[(size_t) s.il * ne + (size_t) s.e] = 0;
            s.at.store(Swap::kFree);
            ++stats_.swapped_experts;
            staged_il_ = -1;   // the staging buffer holds what was not cached
        }
        n_free += s.at.load() == Swap::kFree;
    }
    if (written <= 0) return;
    const float keep = std::exp2(-(float) written / (float) opt_.swap_half);
    for (float& r : recent_) r *= keep;
    if (n_free == 0) return;

    // One pass over the experts lists, for each slot size, the ones that may come in, the most used first, and
    // the cached ones that may go, the least used first: as many of each as uploads are free, which is all the
    // pairs below can take.  A pass for every upload was 0.6 ms of a window of GLM 5.3 Flash's.
    const size_t most = (size_t) n_free;
    in_.assign((size_t) n_trade_ * most, Pick{});
    out_.assign((size_t) n_trade_ * most, Pick{});
    // p to its place in a list, whose last one drops out; equals stay in the order they came
    const auto put = [most](Pick* list, const Pick& p, bool (*before)(const Pick&, const Pick&)) {
        if (list[most - 1].il >= 0 && !before(p, list[most - 1])) return;
        size_t at = most - 1;
        for (; at > 0 && (list[at - 1].il < 0 || before(p, list[at - 1])); --at) list[at] = list[at - 1];
        list[at] = p;
    };
    const auto more = [](const Pick& a, const Pick& b) {
        return a.recent != b.recent ? a.recent > b.recent : a.prior > b.prior;
    };
    const auto less = [](const Pick& a, const Pick& b) {
        return a.recent != b.recent ? a.recent < b.recent : a.prior < b.prior;
    };
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        Pick* in = in_.data() + (size_t) L.trade * most;
        Pick* out = out_.data() + (size_t) L.trade * most;
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * ne + (size_t) e;
            const float r = recent_[i];
            const int64_t p = prior_.empty() ? 0 : prior_[i];
            if (L.slot[(size_t) e]) put(out, {il, e, r, p}, less);
            else if (r >= opt_.adapt_min_gain && !coming_[i]) put(in, {il, e, r, p}, more);
        }
    }

    // An upload goes to the size whose next pair gains most.  A cached expert that has given its slot is no
    // candidate to come in: it was used no more than the cached ones left.
    trade_at_.assign((size_t) n_trade_, 0);
    bool asked = false;
    for (; n_free > 0; --n_free) {
        int best = -1;
        float gain = 0.f;
        for (int c = 0; c < n_trade_; ++c) {
            const size_t k = (size_t) c * most + trade_at_[(size_t) c];
            const Pick& in = in_[k];
            const Pick& out = out_[k];
            if (in.il < 0 || out.il < 0 || in.recent < opt_.swap_ratio * out.recent + opt_.adapt_min_gain) continue;
            if (best < 0 || in.recent - out.recent > gain) {
                best = c;
                gain = in.recent - out.recent;
            }
        }
        if (best < 0) break;

        const size_t k = (size_t) best * most + trade_at_[(size_t) best]++;
        const Pick in = in_[k], out = out_[k];
        Swap* s = nullptr;
        for (int j = 0; j < n_swap_ && !s; ++j)
            if (swap_[(size_t) j].at.load() == Swap::kFree) s = &swap_[(size_t) j];
        uint8_t*& given = layers_[(size_t) out.il].slot[(size_t) out.e];
        s->il = in.il;
        s->e = in.e;
        s->slot = given;
        given = nullptr;
        staged_il_ = -1;
        coming_[(size_t) in.il * ne + (size_t) in.e] = 1;
        s->at.store(Swap::kAsked);
        asked = true;
    }
    if (asked) {
        { std::lock_guard<std::mutex> l(swap_m_); }
        swap_cv_.notify_one();
    }
}

void Engine::load(const Options& o) {
    opt_ = o;
    asked_reserve_mib_ = o.vram_reserve_mib;
    if (opt_.max_context < 16) throw std::runtime_error("the context must be at least 16 tokens");
    // The prompt-batch cap: 4096 as shipped.  A whole prompt in ONE chunk amortizes the expert staging over every
    // token (the upstream program line does this at 6,912), so STRATA_MAX_PROMPT_BATCH may widen it.  Unset (or a
    // value below 4096) leaves the shipped cap, so the old behaviour is one env away.
    int max_prompt_batch = 4096;
    if (const char* e = std::getenv("STRATA_MAX_PROMPT_BATCH")) {
        const int v = std::atoi(e);
        if (v > max_prompt_batch) max_prompt_batch = v;
    }
    if (opt_.prompt_batch < 1 || opt_.prompt_batch > max_prompt_batch)
        throw std::runtime_error("a prompt batch is 1 to " + std::to_string(max_prompt_batch) + " tokens");
    if (opt_.stage_mib < 1) throw std::runtime_error("the staging buffer must be at least 1 MiB");
    if (opt_.stage_ahead < 0) throw std::runtime_error("the batch size to stage ahead from cannot be negative");
    if (opt_.expert_swaps < 0 || opt_.expert_swaps > 64)
        throw std::runtime_error("0 to 64 experts can be on their way into the cache at a time");
    if (opt_.window < 0 || opt_.window > 8)
        throw std::runtime_error("a window is 1 to 8 tokens, or 0 for the model's own");
    if (opt_.tail_skip < 0) throw std::runtime_error("--tail-skip (STRATA_ROUTE_TAIL_SKIP) cannot be negative");
    if (opt_.adapt_min_gain < 0.f) throw std::runtime_error("--adapt-min-gain cannot be negative");
    if (opt_.swap_half < 1) throw std::runtime_error("--swap-half must be at least 1 token");
    if (opt_.swap_ratio < 0.f) throw std::runtime_error("--swap-ratio cannot be negative");
    if (opt_.pcie_frac < 0.0 || opt_.pcie_frac > 1.0)
        throw std::runtime_error("--pcie-frac is a share of the misses, 0 to 1");
    if (opt_.pcie_slots < 0 || opt_.pcie_slots > 256) throw std::runtime_error("--pcie-slots is 0 to 256");
    if (opt_.pcie_frac > 0.0 && opt_.pcie_slots < 1)
        throw std::runtime_error("--pcie-frac above 0 needs at least one --pcie-slots slot");
    if (opt_.pcie_mode != "dma" && opt_.pcie_mode != "direct" && opt_.pcie_mode != "auto")
        throw std::runtime_error("--pcie-mode is dma, direct or auto");
    if (opt_.adapt_min_gain != kSwapMinDefault || opt_.swap_half != kSwapHalfDefault ||
        opt_.swap_ratio != kSwapRatioDefault)
        std::fprintf(stderr, "strata-poly: adaptive tier: min gain %.3f, half %d tokens, ratio %.3f\n",
                     (double) opt_.adapt_min_gain, opt_.swap_half, (double) opt_.swap_ratio);
    if (opt_.pcie_auto && opt_.pcie_frac <= 0.0)
        std::fprintf(stderr, "strata-poly: --pcie-auto: the share fitting is not in this build; only --pcie-frac is "
                             "read\n");
    if (opt_.dump || std::min(opt_.prompt_batch, opt_.max_context) == 1) opt_.window = 1;
    // STRATA_DRAFT_LOAD_ONLY=1: for telling a draft block's VRAM cost from what it computes.  The block is still
    // detected, loaded and wired (its weights, its cache and the state_/r_ buffers keep their VRAM), but nothing is
    // computed from it and no guess is proposed: the run's numbers are the ones without a draft block, its VRAM the
    // ones with it.  Off unless the variable says 1, so default runs are unchanged.
    draft_load_only_ = [] {
        const char* e = std::getenv("STRATA_DRAFT_LOAD_ONLY");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    // STRATA_PROMPT_BORROW: the prompt path's staging buffers come out of slots lent from the expert cache's tail
    // instead of holding VRAM of their own all the time: the cache is sized as if they did hold it, the slots they
    // stand in for stream during the prompt, and their experts are refilled after it - so the cache the answer is
    // written from is the larger one.  Off unless the switch says 1 (then the past behaviour, byte for byte).
    // STRATA_PROMPT_BORROW_SLOTS: the slots to lend (0 = none, an A/B; the default is what the buffers take).
    // STRATA_PROMPT_BORROW_REFILL=0: lend, but put nothing back (an A/B of the refill's own cost).
    // STRATA_PROMPT_BORROW_CHECK=n: check n refilled slots against the experts they stand for, and n that were
    // not lent (an A/B of the refill's content; a mismatch is loud).
    prompt_borrow_ = [] {
        const char* e = std::getenv("STRATA_PROMPT_BORROW");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    if (const char* e = std::getenv("STRATA_PROMPT_BORROW_SLOTS")) {
        const int v = std::atoi(e);
        if (v < 0) throw std::runtime_error("STRATA_PROMPT_BORROW_SLOTS cannot be negative");
        borrow_slots_req_ = v;
    }
    if (const char* e = std::getenv("STRATA_PROMPT_BORROW_REFILL")) borrow_refill_ = std::atoi(e) != 0;
    if (const char* e = std::getenv("STRATA_PROMPT_BORROW_CHECK")) borrow_verify_ = std::atoi(e);
    borrow_lend_ = prompt_borrow_ && borrow_slots_req_ != 0;
    if (prompt_borrow_)
        std::fprintf(stderr,
                     "strata-poly: STRATA_PROMPT_BORROW=1: the prompt's staging buffers are lent the expert cache's "
                     "tail%s\n",
                     !borrow_lend_ ? "; STRATA_PROMPT_BORROW_SLOTS=0, none lent: they hold their own VRAM"
                     : borrow_refill_ ? "" : "; STRATA_PROMPT_BORROW_REFILL=0, nothing is put back");
    // STRATA_BORROW_ASYNC_REFILL: the refill runs on a stream of its own while the answer's first rounds run; off
    // (the refill blocks the prompt's end), the default.  The switch says what it did.
    borrow_async_ = [] {
        const char* e = std::getenv("STRATA_BORROW_ASYNC_REFILL");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    if (borrow_async_ && (!borrow_lend_ || !borrow_refill_)) {
        borrow_async_ = false;
        std::fprintf(stderr, "strata-poly: STRATA_BORROW_ASYNC_REFILL=1 needs STRATA_PROMPT_BORROW=1 with a lend and "
                             "the refill on: the switch stays off\n");
    } else if (borrow_async_) {
        std::fprintf(stderr, "strata-poly: STRATA_BORROW_ASYNC_REFILL=1: the refill runs on a stream of its own while "
                             "the answer's first rounds run; its experts take the CPU path until they land\n");
    }
    // STRATA_CUT_HANDOFF: the per-layer handoff cuts (C1: a layer's routed-experts sum and latent projection are
    // recorded once and replayed with one call; C3: a few rows' activation conversion runs on the calling thread
    // instead of a worker pass).  Off by default - the delivery default is the old path byte for byte; =1 turns
    // them on for the A/B.
    if (const char* e = std::getenv("STRATA_CUT_HANDOFF"))
        std::fprintf(stderr, "strata-poly: STRATA_CUT_HANDOFF=%s: %s\n", e,
                     e[0] == '1' && e[1] == '\0'
                         ? "on (the add+latent pair recorded a layer; the activation pass inline for a few rows)"
                         : "off (the old launches)");
    gg_ = std::make_unique<Shards>(opt_.model);
    if (!opt_.draft_model.empty() && opt_.draft && opt_.window != 1) dg_ = std::make_unique<Shards>(opt_.draft_model);
    load_geometry();
    if (block_ != dg_.get()) dg_.reset();
    // the switch says what it did, so a run can be told from one that did not see it at all
    if (draft_load_only_)
        std::fprintf(stderr, "strata-poly: STRATA_DRAFT_LOAD_ONLY=1: %s\n",
                     drafting() ? "the block that guesses ahead is loaded and its VRAM kept, but nothing is computed "
                                  "or guessed from it"
                                : "this model has no block that guesses ahead; nothing changes");
    hi_ = end_ < 0 ? n_layer_ : end_;

    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
    st_ = st;
    for (void*& row : ev_row_) {
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        row = ev;
    }
    ev_ = ev_row_[0];
    blas_ = blas_create(st_);
    // The card maps VRAM onto reserved addresses a piece at a time: the attention caches grow with the sequence,
    // and the expert cache has what a short one leaves (cover()).
    ck(cudaGetDevice(&device_), "cudaGetDevice");
    CUdevice dev = 0;
    int vmm = 0;
    if (opt_.kv_grow && cuDeviceGet(&dev, device_) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) == CUDA_SUCCESS && vmm) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_;
        vmm_ = cuMemGetAllocationGranularity(&grain_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS &&
               grain_ > 0;
    }
    if (borrow_lend_ && !vmm_) {   // the region is a mapping of its own: a card without VMM has none to give
        borrow_lend_ = false;
        std::fprintf(stderr, "strata-poly: STRATA_PROMPT_BORROW: this card does not map VRAM a piece at a time: the "
                             "switch stays off, the staging buffers hold their own VRAM\n");
    }

    // the token table stays in the file, a row a token, and embed() reads a row in whatever type it has
    const TensorInfo* te = gg_->find("token_embd.weight");
    const ggml_type_traits* tt = te ? ggml_get_type_traits((ggml_type) te->type) : nullptr;
    if (!te || te->shape.size() != 2 || (int64_t) te->shape[0] != n_embd_ || (int64_t) te->shape[1] != n_vocab_ ||
        !tt || tt->blck_size <= 0 || n_embd_ % tt->blck_size != 0 || (te->type != kTypeF32 && !tt->to_float))
        throw std::runtime_error("token_embd.weight must be [" + std::to_string(n_embd_) + ", " +
                                 std::to_string(n_vocab_) + "] in a type ggml converts to floats, it is " +
                                 (te ? shape_of(*te) : "missing"));
    tok_row_ = gg_->payload(*te) / (size_t) n_vocab_;
    tok_type_ = te->type;
    tok_embd_ = gg_->data(*te);

    // twice: the first pass adds up the arena, the second fills it
    for (int pass = 0; pass < 2; ++pass) {
        const int n_all = n_layer_ + (block_layer_ ? 1 : 0);
        layers_.assign((size_t) n_all, Layer{});
        n_moe_ = 0;
        dense_at_ = 0;
        begin_pass();
        for (int il = 0; il < n_all; ++il)
            if (runs(il)) load_layer(il);
        if (hi_ == n_layer_) {
            const TensorInfo* norm = gg_->find("output_norm.weight");
            if (const Names* a = norm ? nullptr : tensor_alias("output_norm.weight"))
                for (const char* other : a->other)
                    if (other && !norm) norm = gg_->find(other);
            out_norm_ = upload(norm, "output_norm.weight", n_embd_, 1, kF32);
            // a file without an output matrix ties the output to the token table
            const TensorInfo* head = gg_->find("output.weight");
            output_ = upload(head ? head : gg_->find("token_embd.weight"), "output.weight", n_embd_, n_vocab_,
                             kAnyMat);
        }
        if (draft_) load_drafter();
        if (pass == 0) {
            dense_bytes_ = dense_at_;
            dense_ = (uint8_t*) dmalloc(dense_bytes_ + 256, "the dense weights");
            ck(cudaMemset(dense_, 0, dense_bytes_ + 256), "the dense weights");   // between the tensors: kTail
        }
    }

    // A guess needs the recurrent layers' state as it was before it.  Where that is a copy in VRAM for every
    // guess of a window, a window is four tokens: the copies come out of what the experts would have.  A model
    // without such layers pays nothing for a longer one, and one whose layers leave the state after a few of a
    // window's tokens (snaps_spread()) no more than for four.  A repeat of the conversation can fill a window
    // of eight.  The layers are counted by now.
    if (opt_.window == 0) opt_.window = recurrent_floats() == 0 || snaps_spread() ? 8 : 4;
    opt_.window = std::min({opt_.window, opt_.prompt_batch, opt_.max_context});
    // Stage D: the sidecar is pinned and faulted in BEFORE alloc()/pin_expert_arena(), so that pin's floor subtracts it.
    relu2_down_open();
    alloc();
    // STRATA_PROFILE_AUTO_LEARN=1: no table yet -> learn this run's answers and write it back on exit / after
    // each answer (--serve).  The first run is the slow one; the next start loads the same path and is faster.
    // Default 0 keeps every release behaviour: with it unset this block only reads a table if one is given.
    {
        static const bool auto_learn = [] {
            const char* e = std::getenv("STRATA_PROFILE_AUTO_LEARN");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        static const bool decode_only_env = [] {
            const char* e = std::getenv("STRATA_PROFILE_DECODE_ONLY");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        bool empty = true;
        if (!opt_.profile.empty()) {
            std::error_code ec;
            const auto sz = std::filesystem::exists(opt_.profile, ec) ? std::filesystem::file_size(opt_.profile, ec) : 0;
            empty = ec || sz == 0;
        }
        if (auto_learn && !opt_.profile.empty()) {
            // A seeded table continues to learn too: without this a shipped starter table would never grow.
            opt_.profile_write = true;
            profile_auto_learn_ = true;
            // The counts auto-learn writes/accumulates are the ANSWERS' routing in both cases: a seeded table is a
            // decode-only table (the shipped one is), and mixing prompt counts back in is what made p_deliver drag
            // the decode ranking (R76b: hit 51.2% seeded vs 76.4% decode-only).
            profile_decode_only_ = true;
            if (empty)
                std::fprintf(stderr,
                             "strata-poly: profile: auto-learn on (no table at %s; learning from this run; the "
                             "first run is ~10%% slower, the next ones gain 15-40%% on template-like loads)\n",
                             opt_.profile.c_str());
            else
                std::fprintf(stderr,
                             "strata-poly: profile: auto-learn on (seeded table at %s; continuing to learn)\n",
                             opt_.profile.c_str());
        } else {
            profile_decode_only_ = decode_only_env;   // the switch that already existed, unchanged
        }
    }
    if (!opt_.profile.empty()) read_profile(opt_.profile, (int) layers_.size(), n_expert_, prior_);
    fill_expert_cache();
    // STRATA_PROMPT_BORROW: the staging buffers the fill lent a region of the cache's tail are pointed at it now:
    // stage_ at the region's first buffer, stage2_ at the second when the region holds both.  The switch says what
    // it did, so a run can be told from one that did not see it.
    if (borrow_lend_ && borrow_bytes_ == 0)
        throw std::runtime_error("prompt borrow: the expert cache had no room to lend the prompt's staging buffers "
                                 "(a smaller --prompt-batch or a smaller reserve is what changes that)");
    if (borrow_lend_ && borrow_bytes_ > 0) {
        stage_ = (uint8_t*) borrow_.base;
        if (ring2_) slot_[0] = stage_;
        if (ring2_ && borrow_buffers_ > 1) {
            stage2_ = stage_ + borrow_off_[1];
            slot_[1] = stage2_;
        }
        std::fprintf(stderr,
                     "strata-poly: STRATA_PROMPT_BORROW: the prompt path's %d staging buffer(s) (%.0f MiB) are lent "
                     "a region of %.0f MiB = %zu slots of the expert cache's tail; their experts are refilled after "
                     "the prompt\n",
                     borrow_buffers_, (double) (borrow_one_buffer_ >> 20), (double) (borrow_bytes_ >> 20),
                     borrow_cands_.size());
    }
    // What a prompt batch costs in buffer bytes and tokens a call: the figures the borrow and the staging A/Bs are
    // read against (STRATA_PROMPT_BORROW needs the workspace size it is meant to stand in for).
    if (mmq_ && slot_max_ > 0)
        std::fprintf(stderr, "strata-poly: prompt batch: %d tokens a call (%.0f MiB of staging = %d expert slots)%s\n",
                     rows_, (double) (stage_bytes_ >> 20), (int) (stage_bytes_ / slot_max_),
                     borrow_lend_ ? "; the staging buffers are borrowed from the expert cache" : "");
    // STRATA_STAGE_RING2: the second round buffer is made here, after the cache has been filled, so that the cache
    // is byte for byte the one the switch-off run gets (the buffer lives in the reserve's headroom).  A failure to
    // make it is loud and leaves the engine on the one-buffer path; it is never a silent short buffer.
    if (ring2_ && stage2_ == nullptr) {
        try {
            stage2_ = (uint8_t*) dmalloc(stage_alloc_bytes_, "the second expert staging buffer");
            slot_[1] = stage2_;
            std::fprintf(stderr, "strata-poly: STRATA_STAGE_RING2: the second round buffer is %.0f MiB, after the "
                                 "expert cache (%.0f MiB of it used for %d experts)\n",
                         (double) (stage_bytes_ >> 20), (double) (cached_gib() * 1024.0), cached_);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-poly: STRATA_STAGE_RING2: %s; the switch stays off, one buffer\n", e.what());
            stage2_ = nullptr;
            slot_[1] = nullptr;
        }
    }
    // A card of a split that has less free beside its layers than the reserve it keeps by default runs out of VRAM
    // in a later request, where one card alone can do nothing else.  It may be that another program held VRAM when
    // the split was made.  A larger reserve asked for (--vram-reserve-mib) only leaves the card without experts.
    if (split_ && !exp_) {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const int need = default_reserve_mib(total_b, ram_tight_);
        if (free_b <= (size_t(need) << 20))
            throw std::runtime_error("layers " + std::to_string(lo_) + " to " + std::to_string(hi_ - 1) + " leave " +
                                     std::to_string(free_b >> 20) + " MiB of VRAM free, less than the " +
                                     std::to_string(need) + " MiB the card needs beside them: give it fewer layers "
                                     "(--layer-split), or free VRAM another program holds");
    }
    // A card that holds every expert has VRAM to spare, and the attention caches take the whole context's rows
    // there now when they fit: VRAM left free is what another program takes, and the caches would then grow out of
    // the expert cache (cover()), which does not get it back when that program ends.  A piece of the expert cache
    // (kArenaPiece) is the margin, so that cover() cuts nothing.
    if (!kv_.empty() && all_cached()) {
        size_t more = 0;
        for (const Kv& k : kv_) more += k.reserved - k.mapped;
        if ((int64_t) ((more + kArenaPiece) >> 20) + reserve_mib_ <= vram_free_mib()) cover(opt_.max_context);
    }
    const int threads = opt_.threads > 0 ? opt_.threads : usable_cpus();
    mind_memory();
    std::vector<std::pair<const uint8_t*, size_t>> spans;   // the routed experts in the mapped files
    // Stage D D4: with the sidecar on, the GGUF down pages are not needed at all (every fill/swap/stage source is
    // the pinned sidecar), so do not pre-touch their 29.53 GiB (R63 #13; the RAM headroom comes from here).  The
    // up/gate spans stay: the CPU's gate/up and the staging DMA still read them.
    static const bool r2d_no_down_touch = [] {
        const char* e = std::getenv("STRATA_RELU2_DOWN");
        return e != nullptr && std::atoi(e) != 0;
    }();
    for (const Layer& L : layers_) {
        if (!L.moe) continue;
        if (L.exp_gate) spans.emplace_back(L.exp_gate, (size_t) n_expert_ * L.gu_stride);
        if (L.gu_stride == L.gu_bytes) spans.emplace_back(L.exp_up, (size_t) n_expert_ * L.gu_bytes);
        if (!r2d_no_down_touch) spans.emplace_back(L.exp_down, (size_t) n_expert_ * L.d_bytes);
    }
    if (!workers_) {   // a later card's engine has the first one's
        // --host-core/--aux-cpus: where the host thread, the CPU experts' workers and the auxiliary threads run.
        // The plan is made before any of those threads exists, and the thread that got here (the host) is pinned
        // first; Workers::loop() and the follower/mapper take their processors from the same plan.  Moving a
        // thread changes no arithmetic: the answers of a pinned run are the ones of an unpinned one.
        const std::string where = affinity::plan_threads(opt_.host_core, opt_.aux_cpus, threads - 1);
        std::fprintf(stderr, "strata-poly: threads: %s\n", where.c_str());
        affinity::pin_this_thread(affinity::plan().host);
        if (opt_.stay_near) {   // before any thread that reads the experts is started
            std::vector<std::string> files = gg_->paths();
            if (dg_) files.insert(files.end(), dg_->paths().begin(), dg_->paths().end());
            node_ = stay_near(spans, files, threads, node_share_);
        }
        workers_ = std::make_shared<Workers>(threads);
    }
    route_tmp_.assign((size_t) workers_->threads() * 2 * n_expert_, 0.f);
    start_swaps();
    reset();
    // nothing is read from the files when every expert is in VRAM
    if (!all_cached())
        mapper_ = std::thread([this, spans = std::move(spans)] {
            affinity::pin_aux_thread();   // --aux-cpus: the page mapper is an auxiliary thread too
            map_pages(spans, map_quit_);
        });
}

void Engine::load_drafter() {
    check_draft_block();
    auto part = [&](const char* n, int64_t a, int64_t b, Want w) {
        return upload(tensor(n_layer_, n), blk(n_layer_, n), a, b, w);
    };
    d_proj_ = part("nextn.eh_proj.weight", 2 * n_embd_, n_embd_, kAnyMat);
    d_enorm_ = part("nextn.enorm.weight", n_embd_, 1, kF32);
    d_hnorm_ = part("nextn.hnorm.weight", n_embd_, 1, kF32);
    // without a norm of its own, the block's output goes through the model's
    d_head_ = tensor(n_layer_, "nextn.shared_head_norm.weight")
                  ? part("nextn.shared_head_norm.weight", n_embd_, 1, kF32)
                  : out_norm_;
}

// Per-token buffers hold rows_ tokens, row after row.
void Engine::shared_scratch() {
    const size_t n = (size_t) rows_, e = n * n_embd_;
    const size_t ff = n * std::max({n_ff_, n_ff_shexp_, 2 * n_ff_exp_});
    // the draft block works in it too (draft_input()); read() keeps two calls' rows in it.  STRATA_DRAFT_CHUNKED:
    // the block's three row sets move to draft_work_, so r_ is the size a run without the draft block has.
    r_ = fbuf(std::max<size_t>((size_t) streams(), (size_t) ((draft_ && !draft_chunked()) ? 3 : 1)) *
              (size_t) (read_rows_ > rows_ ? 2 : 1) * rows_ * n_embd_);
    cur_ = fbuf(e);
    xn_ = fbuf(e);
    ao_ = fbuf(e);
    fo_ = fbuf(e);
    g_ = fbuf(ff);   // a routed expert's gate and up rows come out of one launch, back to back
    u_ = fbuf(ff);
    h_ = fbuf(ff);
    // A latent MoE's experts read and write exp_in/exp_out values a token, not the model's width.
    const size_t in_w = (size_t) std::max(moe_in_max_, n_embd_), out_w = (size_t) std::max(moe_out_max_, n_embd_);
    moe_in_ = fbuf(n * in_w);
    moe_out_ = fbuf(n * out_w);
    moe_up_ = fbuf(e);
    d_ = fbuf((size_t) kSumRows * out_w);   // the routed experts of a few tokens, until they are added up
    rl_ = fbuf(n * n_expert_);
    logits_ = fbuf((size_t) n_vocab_);
    if (rows_ >= kStageRows && bf16_max_ > 0) wf_ = fbuf(bf16_max_);   // a BF16 matrix as floats, for lin()
    if (mmq_) {
        const size_t rr = n * n_used_;
        // STRATA_SCRATCH_TILE=1: a batch of the call's experts at a time, not every row of it.  Default 0 keeps rr.
        static const bool tile = []() {
            const char* e = std::getenv("STRATA_SCRATCH_TILE");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        scratch_rows_ = tile ? std::min<size_t>(rr, kScratchRows) : rr;
        gu_ = fbuf(scratch_rows_ * 2 * n_ff_exp_);
        hx_ = fbuf(scratch_rows_ * n_ff_exp_);
        // dx_ holds a routed row of the expert's OUTPUT, so its width is the experts' own exp_out (a latent MoE:
        // 1024), NOT moe_out_max_ (which is max(exp_out, n_embd_) = 4096 because moe_out_ must hold the model's
        // width) and not n_embd_ (4096).  moe_rows() writes and moe_sum() reads dx_ at out_w (= L.exp_out).
        size_t exp_out_w = 1;
        for (const Layer& L : layers_)
            if (L.moe) exp_out_w = std::max(exp_out_w, (size_t) L.exp_out);
        dx_ = fbuf(rr * exp_out_w);
    }
}

// The sequence's state, the caches and every buffer, once the weights are known.
void Engine::alloc() {
    // the widest expert input/output any layer has: a latent MoE's is moe_latent_size, the others' n_embd_
    moe_in_max_ = moe_out_max_ = n_embd_;
    for (const Layer& L : layers_) {
        if (!L.moe) continue;
        moe_in_max_ = std::max(moe_in_max_, L.exp_in > 0 ? L.exp_in : n_embd_);
        moe_out_max_ = std::max(moe_out_max_, L.exp_out > 0 ? L.exp_out : n_embd_);
    }
    // --dump-layers prints one token's stages
    rows_ = opt_.dump ? 1 : std::min(opt_.prompt_batch, opt_.max_context);
    const size_t most = std::min<size_t>(kReadBytes / ((size_t) streams() * n_embd_ * sizeof(float)), opt_.max_context);
    read_rows_ = rows_ < kStageRows ? rows_ : std::max(rows_, (int) most / rows_ * rows_);
    rec_n_ = recurrent_floats();
    state_n_ = rec_n_ + (draft_ ? (size_t) n_embd_ : 0);
    state_ = (float*) dmalloc((state_n_ + (draft_ ? (size_t) rows_ * n_embd_ : 0)) * sizeof(float),
                              "the recurrent state");
    if (draft_) {
        hn_ = state_ + rec_n_;
        h_guess_ = (Likeliest*) hmalloc(sizeof(Likeliest), "pinned memory");   // likeliest() writes it
    }
    // window(): the state after tokens of it, for keep(), and every token's logits
    snap_places_ = snaps_spread() ? std::min(opt_.window - 1, kSnapPlaces) : opt_.window - 1;
    // STRATA_SNAP_PLACES=1|2|3: how many of the window's states are kept.  With fewer, keep() has the state after
    // the last accepted token that left one only, and runs the accepted tokens after it through the layers again
    // (that path already exists and its results are the ones the state would have had - see keep("the right tokens
    // after that one go through the layers again")).  Default: the places above, byte for byte the old allocation.
    if (const char* e = std::getenv("STRATA_SNAP_PLACES")) {
        const int asked = std::atoi(e);
        if (asked < 1) throw std::runtime_error("STRATA_SNAP_PLACES must be at least 1");
        snap_places_ = std::min(snap_places_, asked);
        std::fprintf(stderr, "snap: %d places (%.1f MiB)\n", snap_places_,
                     (double) (size_t(snap_places_) * rec_n_ * sizeof(float)) / (1024.0 * 1024.0));
    }
    if (snap_places_ > 0)
        snap_ = (float*) dmalloc(size_t(snap_places_) * rec_n_ * sizeof(float), "the recurrent state");
    wlogits_ = (float*) dmalloc(size_t(opt_.window) * n_vocab_ * sizeof(float), "scratch");
    h_wlogits_ = (float*) hmalloc(size_t(opt_.window) * n_vocab_ * sizeof(float), "pinned memory");
    alloc_caches();
    pin_expert_arena();   // STRATA_PIN_ARENA_GIB: the expert sources, registered read-only for stage()'s direct DMA
    // --pcie-mode direct (Strata's zero-copy half of the split): the kernels read the pinned host mapping where it
    // is, so the card must map host memory into its address space (cudaHostRegisterReadOnly asks for that) and the
    // pointer must be a device address as it is (UVA).  Probed once here; "auto" falls back to dma, "direct" refuses.
    if (opt_.pcie_frac > 0.0 && (opt_.pcie_mode == "direct" || opt_.pcie_mode == "auto")) {
        int uva = 0;
        ck(cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device_), "cudaDeviceGetAttribute");
        bool ok = uva == 1;
        const Layer* probe = nullptr;
        for (const Layer& L : layers_)
            if (L.moe && L.exp_up) {
                probe = &L;
                break;
            }
        if (ok && probe) {
            void* dp = nullptr;
            const cudaError_t r = cudaHostGetDevicePointer(&dp, (void*) probe->exp_up, 0);
            cudaPointerAttributes pa{};
            const cudaError_t r2 = cudaPointerGetAttributes(&pa, (void*) probe->exp_up);
            // A mapped registration gives a device address that may differ from the host one (legacy registration
            // maps it into a separate window): what has to hold is that the card can read it there.
            ok = (r == cudaSuccess && dp != nullptr) || (r2 == cudaSuccess && pa.devicePointer != nullptr);
            if (dp != (void*) probe->exp_up)
                std::fprintf(stderr, "strata-poly: --pcie-mode %s: the pinned mapping is device-visible at %p "
                                     "(host %p)\n", opt_.pcie_mode.c_str(), dp, (void*) probe->exp_up);
            if (!ok) {
                std::fprintf(stderr,
                             "strata-poly: --pcie-mode %s probe: uva=%d hostGetDevicePointer=%s dp=%p host=%p "
                             "pointerAttributes=%s type=%d dev=%p\n",
                             opt_.pcie_mode.c_str(), uva, cudaGetErrorString(r), dp, (void*) probe->exp_up,
                             cudaGetErrorString(r2), (int) pa.type, pa.devicePointer);
            }
            cudaGetLastError();
        }
        // the device aliases of every pinned layer's up and down regions, one lookup each (see the probe above)
        if (ok) {
            exp_up_dev_.assign(layers_.size(), nullptr);
            exp_down_dev_.assign(layers_.size(), nullptr);
            for (size_t il = 0; il < layers_.size() && ok; ++il) {
                const Layer& L = layers_[il];
                if (!L.moe || !L.exp_up || !L.exp_down || il >= pin_layer_.size() || !pin_layer_[il]) continue;
                void* up = nullptr;
                void* dn = nullptr;
                if (cudaHostGetDevicePointer(&up, (void*) L.exp_up, 0) != cudaSuccess || up == nullptr ||
                    cudaHostGetDevicePointer(&dn, (void*) L.exp_down, 0) != cudaSuccess || dn == nullptr) {
                    ok = false;
                    cudaGetLastError();
                    break;
                }
                exp_up_dev_[il] = (const uint8_t*) up;
                exp_down_dev_[il] = (const uint8_t*) dn;
            }
        }
        pcie_direct_ = ok;
        if (ok) {
            std::fprintf(stderr, "strata-poly: --pcie-mode %s: the split's kernels will read the pinned experts in "
                                 "place (zero copy, no staging slots or copies)\n", opt_.pcie_mode.c_str());
        } else if (opt_.pcie_mode == "direct") {
            throw std::runtime_error("--pcie-mode direct: this card cannot read the pinned host mapping in place "
                                     "(no UVA or the mapping is not device-readable); use --pcie-mode auto or dma");
        } else {
            std::fprintf(stderr, "strata-poly: --pcie-mode auto: this card cannot read the pinned mapping in place; "
                                 "falling back to dma\n");
        }
    }
    cover(1);   // a short conversation's rows, before the expert cache is sized

    // A prompt batch runs a layer's routed experts through MMQ when the card has a tile for each of their types
    // (batched()).  The model's layers: the draft block's experts never run a batch at a time.
    const size_t n = (size_t) rows_;
    bool tiles = false;
    size_t one_expert = 0;
    for (int il = 0; il < n_layer_; ++il) {
        Layer& L = layers_[(size_t) il];
        if (!L.moe) continue;
        // the batched MMQ path: moe_rows()/stage() now take the widths from the expert's own layout (slot_up()/
        // slot_down(), a relu2 for the non-gated), so a non-gated (latent) layer takes it too.  B3, B3-閰嶆柟鐩樼偣-20261016.md.
        const int gu_w = L.gated ? 2 * n_ff_exp_ : n_ff_exp_;
        L.mmq = opt_.gpu_experts && rows_ >= kStageRows &&
                Mmq::fits((int) L.gu_type, gu_w, L.exp_in > 0 ? L.exp_in : n_embd_) &&
                Mmq::fits((int) L.d_type, L.exp_out > 0 ? L.exp_out : n_embd_, n_ff_exp_);
        tiles = tiles || L.mmq;
        one_expert = std::max(one_expert, L.slot_down() + L.d_bytes);
    }
    if (tiles) mmq_ = std::make_unique<Mmq>();
    // STRATA_PROMPT_BORROW: a model whose prompt batches do not go through the staging buffer has nothing to lend
    // (the buffers are only ever read by the batched prompt path); the switch says so rather than lend nothing.
    if (borrow_lend_ && (!mmq_ || rows_ < kStageRows)) {
        borrow_lend_ = false;
        std::fprintf(stderr, "strata-poly: STRATA_PROMPT_BORROW: this model runs no batched prompt (no MMQ tile or a "
                             "prompt batch under %d tokens): nothing is lent\n", kStageRows);
    }
    if (opt_.pcie_frac > 0.0 && !mmq_)
        std::fprintf(stderr, "strata-poly: --pcie-frac %.3f: this card has no batched expert path (no MMQ tile for "
                             "the experts' types), so no staging slots are made\n", opt_.pcie_frac);

    shared_scratch();   // the first pass only counts
    assign_scratch();
    scratch_ = (float*) dmalloc(scratch_at_ * sizeof(float), "scratch");
    scratch_at_ = 0;
    shared_scratch();
    assign_scratch();

    const int q_max = std::max({n_embd_, n_ff_, n_ff_exp_, n_ff_shexp_, widest(), 256});
    xq_ = dmalloc(n * kernels::native_q8_1_bytes(n_embd_), "scratch");
    q8_ = dmalloc(n * kernels::native_q8_1_bytes((q_max + 255) / 256 * 256), "scratch");
    // STRATA_SHARED_OVERLAP=1: the shared expert's two products run on a stream of its own in decode (n <= 8
    // tokens), beside the routed experts' work; moe_latent() waits for its event before the add into fo_.  Its
    // rows and quantized inputs are its own (the routed path uses g_/h_/q8_ at the same time).  Off: nothing is
    // made and every call runs as it always did, byte for byte.
    static const bool shared_overlap_env = [] {
        const char* e = std::getenv("STRATA_SHARED_OVERLAP");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    if (shared_overlap_env && n_ff_shexp_ > 0 && !all_cached()) {
        cudaStream_t sh = nullptr;
        ck(cudaStreamCreateWithFlags(&sh, cudaStreamNonBlocking), "cudaStreamCreate");
        sh_ = sh;
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        ev_sh_ = ev;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        ev_st_ = ev;
        sh_g_ = (float*) dmalloc((size_t) 8 * n_ff_shexp_ * sizeof(float), "the shared expert's rows");
        sh_h_ = (float*) dmalloc((size_t) 8 * n_ff_shexp_ * sizeof(float), "the shared expert's rows");
        q8_sh_ = dmalloc((size_t) 8 * kernels::native_q8_1_bytes(std::max(n_embd_, n_ff_shexp_)),
                         "the shared expert's scratch");
        if (bf16_max_ > 0)
            wf_sh_ = (float*) dmalloc(bf16_max_ * sizeof(float), "the shared expert's scratch");   // lin's BF16 path
        shared_overlap_ = true;
        std::fprintf(stderr, "shared-overlap: moe_shared on stream sh_ (event before the routed add)\n");
    } else if (shared_overlap_env) {
        std::fprintf(stderr, "strata-poly: STRATA_SHARED_OVERLAP=1 needs a shared expert and experts outside the "
                             "cache (the recorded-graph path keeps the old call): the switch stays off\n");
    }
    // STRATA_DRAFT_CHUNKED=1: the draft block's priming rows (three row sets of at most kDraftChunkMax tokens) get a
    // scratch of their own, so r_ keeps the size a run without the draft block has and the MTP's VRAM falls by the
    // row sets it used to hold for the block.
    if (draft_ && draft_chunked()) {
        draft_work_ = (float*) dmalloc(3 * (size_t) kDraftChunkMax * (size_t) n_embd_ * sizeof(float),
                                       "the draft block's rows");
        std::fprintf(stderr, "draft-priming: chunked (%zu MiB scratch; r_ back to non-draft size)\n",
                     (3 * (size_t) kDraftChunkMax * (size_t) n_embd_ * sizeof(float)) >> 20);
    }

    h_r_ = (float*) hmalloc(n * streams() * n_embd_ * sizeof(float), "pinned memory");
    at_ = (long long*) dmalloc((size_t) opt_.window * 2 * sizeof(long long), "scratch");
    h_at_ = (long long*) hmalloc((size_t) opt_.window * 2 * sizeof(long long), "pinned memory");
    h_rl_ = (float*) hmalloc(n * n_expert_ * sizeof(float), "pinned memory");
    h_x_ = (float*) hmalloc(n * std::max(moe_in_max_, n_embd_) * sizeof(float), "pinned memory");
    h_cpu_ = (float*) hmalloc(n * std::max(moe_out_max_, n_embd_) * sizeof(float), "pinned memory");
    h_logits_ = (float*) hmalloc((size_t) n_vocab_ * sizeof(float), "pinned memory");

    if (mmq_) {
        const size_t rr = n * n_used_,
                     tab = (3 * rr + 3 * (size_t) n_expert_ + 3 + kClasses * ((size_t) n_expert_ + 1)) * sizeof(int32_t);
        mq_ = dmalloc(Mmq::q8_bytes((int64_t) rr, n_embd_), "scratch");
        mq2_ = dmalloc(Mmq::q8_bytes((int64_t) scratch_rows_, n_ff_exp_), "scratch");   // rr without STRATA_SCRATCH_TILE
        if (scratch_rows_ < rr)   // the tiled up's input; q8's layout is [K block][row], so a batch is quantized whole
            mq_tile_ = dmalloc(Mmq::q8_bytes((int64_t) scratch_rows_, moe_in_max_), "scratch");
        // One bounds table for the whole call: the cached batches' and every staged sub-batch's entries, each
        // launch with one entry per expert and one behind it.  The worst case is 2 per expert over the call's
        // rows plus the experts that have none (2 * n_expert_ covers both sides), so it is one table per call.
        const size_t rel_cap = 2 * ((size_t) rows_ * n_used_ + (size_t) n_expert_) + 4;
        d_rel_local_ = (int32_t*) dmalloc(rel_cap * sizeof(int32_t), "scratch");
        h_rel_local_ = (int32_t*) hmalloc(rel_cap * sizeof(int32_t), "pinned memory");
        if (scratch_rows_ < rr)
            std::fprintf(stderr, "strata-poly: STRATA_SCRATCH_TILE: %zu rows a batch (%.2f GiB gu_+hx_+mq2_); "
                                 "bounds table %.2f MiB\n",
                         scratch_rows_,
                         (double) (scratch_rows_ * (2 * (size_t) n_ff_exp_ * sizeof(float) +
                                                    (size_t) n_ff_exp_ * sizeof(float)) +
                                   Mmq::q8_bytes((int64_t) scratch_rows_, n_ff_exp_)) /
                             (1024.0 * 1024.0 * 1024.0),
                         (double) (rel_cap * sizeof(int32_t)) / (1024.0 * 1024.0));
        mqx_ = dmalloc(Mmq::q8_bytes(rows_, q_max), "scratch");
        d_ids_ = (int32_t*) dmalloc(rr * sizeof(int32_t), "scratch");
        d_wts_ = (float*) dmalloc(rr * sizeof(float), "scratch");
        h_ids_ = (int32_t*) hmalloc(rr * sizeof(int32_t), "pinned memory");
        d_slots_ = (void**) dmalloc((size_t) n_layer_ * n_expert_ * sizeof(void*), "scratch");
        tab_ = (int32_t*) dmalloc(tab, "scratch");
        h_tab_ = (int32_t*) hmalloc(tab, "pinned memory");
        iota_ = (int32_t*) dmalloc(rr * sizeof(int32_t), "scratch");
        std::vector<int32_t> ramp(rr);
        std::iota(ramp.begin(), ramp.end(), 0);
        ck(cudaMemcpy(iota_, ramp.data(), rr * sizeof(int32_t), cudaMemcpyHostToDevice), "scratch");

        // One layer's experts pass through it at a time, so it never holds more than a layer has.  A model of
        // small experts has a few hundred MiB a layer, and the rest of the default would be VRAM that the
        // expert cache of a small card does without.
        stage_bytes_ = std::max(std::min((size_t) opt_.stage_mib << 20, (size_t) n_expert_ * one_expert), one_expert);
        chunk_bytes_ = std::max(kChunkBytes, one_expert);
        stage_alloc_bytes_ = stage_bytes_ + 256 + kTail;   // what one staging buffer takes, the slack included
        // STRATA_PROMPT_BORROW: the staging buffers hold no VRAM of their own; fill_expert_cache() maps them a
        // region of the cache's tail and load() points them at it (after the cache is filled).
        if (!borrow_lend_) stage_ = (uint8_t*) dmalloc(stage_alloc_bytes_, "the expert staging buffer");
        // --pcie-frac: the staging slots a share of an answer's misses is copied into, and the stream and event that
        // take them there.  Made here, before fill_expert_cache() reads the free VRAM, so they come out of the expert
        // cache's arena the way the split's copies will; with the switch at its default nothing is made and the run's
        // VRAM, cache and arithmetic are exactly what they were.  The slots are zeroed like the cache's (the kernels
        // read up to kTail past a matrix).
        if (opt_.pcie_frac > 0.0 && pcie_direct_) {
            // --pcie-mode direct: no slot and no copy - the kernels read the pinned mapping itself, so the split
            // costs no VRAM at all (only the per-layer cap of --pcie-slots experts stays).
            pcie_slots_ = opt_.pcie_slots;
            for (const Layer& L : layers_)
                if (L.moe) pcie_slot_bytes_ = std::max(pcie_slot_bytes_, expert_slot_bytes(L));
            std::fprintf(stderr, "strata-poly: --pcie-frac %.3f (--pcie-mode %s): up to %d experts a layer read in "
                                 "place from the pinned mapping; no staging slots, no copies\n",
                         opt_.pcie_frac, opt_.pcie_mode.c_str(), pcie_slots_);
        } else if (opt_.pcie_frac > 0.0) {
            for (const Layer& L : layers_)
                if (L.moe) pcie_slot_bytes_ = std::max(pcie_slot_bytes_, expert_slot_bytes(L));
            pcie_slots_ = opt_.pcie_slots;
            pcie_stage_ = (uint8_t*) dmalloc((size_t) pcie_slots_ * pcie_slot_bytes_ + kTail,
                                             "the PCIe staging slots");
            ck(cudaMemset(pcie_stage_, 0, (size_t) pcie_slots_ * pcie_slot_bytes_ + kTail), "the PCIe staging slots");
            cudaStream_t pc = nullptr;
            ck(cudaStreamCreateWithFlags(&pc, cudaStreamNonBlocking), "cudaStreamCreate");
            pcie_st_ = pc;
            cudaEvent_t pev = nullptr;
            ck(cudaEventCreateWithFlags(&pev, cudaEventDisableTiming), "cudaEventCreate");
            ev_pcie_ = pev;
            std::fprintf(stderr, "strata-poly: --pcie-frac %.3f (%d/256 of an answer's misses): %d staging slots of "
                                 "%.2f MiB (%.1f MiB in all, out of the expert cache's arena)\n",
                         opt_.pcie_frac, std::min(256, (int) std::lround(std::min(opt_.pcie_frac, 1.0) * 256.0)),
                         pcie_slots_, (double) pcie_slot_bytes_ / (1024.0 * 1024.0),
                         (double) ((size_t) pcie_slots_ * pcie_slot_bytes_) / (1024.0 * 1024.0));
        }
        for (int b = 0; b < 2; ++b) h_stage_[b] = (uint8_t*) hmalloc(chunk_bytes_, "pinned memory");
        for (void** e : {&ev_stage_[0], &ev_stage_[1], &ev_ahead_, &ev_free_}) {
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            *e = ev;
        }
        cudaStream_t up = nullptr;
        ck(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking), "cudaStreamCreate");
        up_ = up;
        // STRATA_MOE_ROWS_CP_STREAM=1: a stream for the staging copies inside a prompt batch's rounds, so that a
        // round's copy can overlap the kernels of the round before.  Off (the default): cp_ stays null, no stream
        // or event is made, and moe_rows() runs the code it ran before this switch existed (the negative control).
        static const bool cp_stream = []() {
            const char* e = std::getenv("STRATA_MOE_ROWS_CP_STREAM");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        // The switch says which way it goes, so a run can be told from one that did not see it at all.
        if (const char* e = std::getenv("STRATA_MOE_ROWS_CP_STREAM"))
            std::fprintf(stderr, "strata-poly: STRATA_MOE_ROWS_CP_STREAM=%s: %s\n", e,
                         cp_stream ? "a prompt batch's round copies run on a stream of their own" : "off");
        if (cp_stream) {
            cudaStream_t cp = nullptr;
            ck(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking), "cudaStreamCreate");
            cp_ = cp;
            for (void** e : {&ev_krn_, &ev_copy_}) {
                cudaEvent_t ev = nullptr;
                ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
                *e = ev;
            }
        }
        // STRATA_STAGE_RING2=1: a second round buffer of the same size, so the ahead copy of the next layer runs on
        // up_ under this layer's kernels.  One event pair a slot: a copy waits that slot's last reader, a round
        // waits that slot's copy.  Off (default): nothing is made and the one-buffer path runs exactly as before.
        static const bool ring2 = []() {
            const char* e = std::getenv("STRATA_STAGE_RING2");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        if (const char* e = std::getenv("STRATA_STAGE_RING2"))
            std::fprintf(stderr, "strata-poly: STRATA_STAGE_RING2=%s: %s\n", e,
                         ring2 ? "two round buffers, the next layer's ahead copy runs on its own stream" : "off");
        if (ring2) {
            // The second buffer itself is made in load(), after fill_expert_cache(): it comes out of the reserve's
            // headroom, so the expert cache's size and its choice of experts stay exactly as they are without the
            // switch (that is what keeps the staging rounds' batches, and so the arithmetic, the same).
            ring2_ = true;
            slot_[0] = stage_;
            for (int s = 0; s < 2; ++s)
                for (void** e : {&ev_ring_ahead_[s], &ev_ring_free_[s]}) {
                    cudaEvent_t ev = nullptr;
                    ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
                    *e = ev;
                }
        }
        ids_.resize(rr);
        first_.resize((size_t) n_expert_);

        for (void** e : {&ev_in_[0], &ev_in_[1], &ev_out_[0], &ev_out_[1]}) {   // read()
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            *e = ev;
        }
        cudaStream_t rs = nullptr;
        ck(cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking), "cudaStreamCreate");
        rs_ = rs;
    }
    use_.assign(layers_.size() * n_expert_, 0);
    use_decode_.assign(use_.size(), 0);    // STRATA_PROFILE_DECODE_ONLY: the answer's and the prompt's, apart
    use_prefill_.assign(use_.size(), 0);
    if (const char* pdo = std::getenv("STRATA_PROFILE_DECODE_ONLY"); pdo && pdo[0] == '1' && pdo[1] == '\0')
        std::fprintf(stderr, "strata-poly: STRATA_PROFILE_DECODE_ONLY=1: the expert profile is built from the "
                            "answer's routing only\n");
    if (const char* dr = std::getenv("STRATA_DUMP_ROUTE"); dr && dr[0] == '1' && dr[1] == '\0') {
        route_dump_ = std::fopen("C:\\sp3\\route_dump.txt", "w");
        if (!route_dump_) throw std::runtime_error("STRATA_DUMP_ROUTE: cannot open C:\\sp3\\route_dump.txt");
        std::fprintf(stderr, "strata-poly: STRATA_DUMP_ROUTE=1: routing dumped to C:\\sp3\\route_dump.txt\n");
    }
}

void Engine::save_profile() const {
    if (opt_.profile.empty()) return;
    if (!opt_.profile_write) return;   // DEFAULT READ-ONLY: only --expert-profile-write writes it back
    // STRATA_PROFILE_DECODE_ONLY=1: the table is the answer's counts alone (the prompt's were 98.26% of every count
    // the table was built from and did not match the decode distribution).  Off (the default): the counts are the
    // ones it always wrote - the two sides added - so the release path is byte for byte the one it was.
    const bool decode_only = profile_decode_only_ || [] {
        const char* e = std::getenv("STRATA_PROFILE_DECODE_ONLY");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    const std::string tmp = opt_.profile + ".tmp";
    // every card's engine counts its own layers' experts, and the last one's profile has the draft block's too
    const Engine* last = this;
    while (last->next_) last = last->next_.get();
    std::vector<int64_t> all = last->prior_;
    unsigned long long dec = 0, pre = 0;
    for (const Engine* e = this; e; e = e->next_.get()) {
        const std::vector<int64_t>& counts = decode_only ? e->use_decode_ : e->use_;
        if (all.size() < counts.size()) all.resize(counts.size(), 0);
        for (size_t i = 0; i < counts.size(); ++i) all[i] += counts[i];
        for (const int64_t c : e->use_decode_) dec += (unsigned long long) c;
        for (const int64_t c : e->use_prefill_) pre += (unsigned long long) c;
    }
    if (decode_only)   // the two totals: a run shows the split the table was built from
        std::fprintf(stderr, "strata-poly: profile: decode-only counts (decode=%llu prefill=%llu)\n", dec, pre);
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    long long saved = 0;
    for (size_t i = 0; i < all.size(); ++i) {
        const int64_t n = all[i];
        if (n > 0) {
            std::fprintf(f, "%d %d %lld\n", (int) (i / (size_t) n_expert_), (int) (i % (size_t) n_expert_),
                         (long long) n);
            ++saved;
        }
    }
    const bool written = std::fclose(f) == 0;
    std::error_code ec;
    if (written) std::filesystem::rename(tmp, opt_.profile, ec);   // std::rename replaces no file on Windows
    if (!written || ec) throw std::runtime_error("cannot write " + opt_.profile);
    std::fprintf(stderr, "strata-poly: profile: saved %lld entries to %s\n", saved, opt_.profile.c_str());
}

void Engine::reset() {
    OnCard here(device_, split_);
    ck(cudaMemsetAsync(state_, 0, state_n_ * sizeof(float), cs(st_)), "reset");
    pos_ = 0;
    ahead_pos_ = -1;
    if (next_) next_->reset();
}

void Engine::save_state(uint8_t* dst) {
    OnCard here(device_, split_);
    ck(cudaStreamSynchronize(cs(st_)), "save state");
    for (int il = 0; il < (int) layers_.size(); ++il)
        if (runs(il)) save_layer(il, pos_, dst);
    save_last(dst);
    if (next_) next_->save_state(dst + card_state_bytes());
}

// What the state has that is no layer's: the model's output for the last token, which the draft block pairs the
// next one with.  It lies behind the layers' state.
void Engine::save_last(uint8_t* dst) {
    if (!draft_) return;
    ck(cudaMemcpy(dst + rec_n_ * sizeof(float), hn_, (size_t) n_embd_ * sizeof(float), cudaMemcpyDeviceToHost),
       "save state");
}

void Engine::load_state(const uint8_t* src, int64_t pos) {
    if (pos < 0 || pos > opt_.max_context) throw std::logic_error("load_state: the position is outside the context");
    OnCard here(device_, split_);
    ck(cudaStreamSynchronize(cs(st_)), "load state");
    ck(cudaMemcpy(state_, src, state_n_ * sizeof(float), cudaMemcpyHostToDevice), "load state");
    load_extra(src + state_n_ * sizeof(float), pos);
    pos_ = pos;
    ahead_pos_ = -1;
    if (next_) next_->load_state(src + card_state_bytes(), pos);
}

void Engine::save_rows(int64_t first, int64_t n, uint8_t* dst) {
    OnCard here(device_, split_);
    save_card_rows(first, n, dst);
    if (next_) next_->save_rows(first, n, dst + (size_t) n * card_row_bytes());
}

void Engine::load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    OnCard here(device_, split_);
    load_card_rows(src, from, count, first, n);
    if (next_) next_->load_rows(src + (size_t) count * card_row_bytes(), from, count, first, n);
}

const Stats& Engine::stats() const {
    if (!next_) return stats_;
    all_stats_ = stats_;   // the tokens are counted once
    for (const Engine* e = next_.get(); e; e = e->next_.get()) {
        const Stats& s = e->stats_;
        all_stats_.gpu_experts += s.gpu_experts;
        all_stats_.cpu_experts += s.cpu_experts;
        all_stats_.batch_cached += s.batch_cached;
        all_stats_.batch_staged += s.batch_staged;
        all_stats_.swapped_experts += s.swapped_experts;
        all_stats_.tail_experts += s.tail_experts;
        all_stats_.cpu_ms += s.cpu_ms;
        all_stats_.route_wait_ms += s.route_wait_ms;
    }
    return all_stats_;
}

void Engine::embed(int32_t token, float* dst) const {
    const uint8_t* row = tok_embd_ + (size_t) token * tok_row_;
    if (tok_type_ == kTypeF32) std::memcpy(dst, row, (size_t) n_embd_ * sizeof(float));
    else ggml_get_type_traits((ggml_type) tok_type_)->to_float(row, dst, n_embd_);
}

void Engine::images(const Image* im, int n) {
    for (int i = 0; i < n; ++i) {
        const Image& a = im[i];
        if (a.n < 1 || a.nx < 1 || a.n % a.nx != 0 || !a.rows || a.at < (i ? im[i - 1].at + im[i - 1].n : 0))
            throw std::logic_error("images: a picture's rows are not a grid, or the pictures are out of order");
    }
    images_.assign(im, im + n);
    ++images_set_;
    if (next_) next_->images(im, n);   // positions and masks depend on where the pictures are
}

const Image* Engine::image_at(int64_t row) const {
    for (const Image& im : images_)
        if (row >= im.at && row < im.at + im.n) return &im;
    return nullptr;
}

// What the model reads at a row of the sequence: a picture's row there, else the token's embedding.  True for a
// picture's row, which goes in as the encoder made it (llama.cpp scales token embeddings only).
bool Engine::input(int32_t token, int64_t row, float* dst) const {
    if (const Image* im = image_at(row)) {
        std::memcpy(dst, im->rows + (size_t) (row - im->at) * n_embd_, (size_t) n_embd_ * sizeof(float));
        return true;
    }
    if (token < 0 || token >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    embed(token, dst);
    return false;
}

void Engine::dbg(const char* name, int il, const float* dev, int n) {
    if (!opt_.dump) return;
    std::vector<float> h((size_t) n);
    ck(cudaStreamSynchronize(cs(st_)), "dump");
    ck(cudaMemcpy(h.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "dump");
    double sum = 0, abs = 0;
    for (float v : h) {
        sum += v;
        abs += std::fabs(v);
    }
    std::fprintf(stderr, "dump pos=%lld blk=%d %-8s n=%d sum=%.6f abs=%.6f first=%.6f %.6f %.6f\n", (long long) pos_,
                 il, name, n, sum, abs, h[0], h[1], h[2]);
}

// y = w x for n tokens.  xq is x already in Q8_1, when the caller has it.  rows_per_x below n_out gives each group
// of rows its own input (see gemv_f32); only the float types take it.
void Engine::lin(const Dev& w, const float* x, float* y, int n, const void* xq, int rows_per_x, void* s) {
    // The stream the products go on: st_ unless a caller runs them beside it (STRATA_SHARED_OVERLAP's shared expert).
    cudaStream_t st = s != nullptr ? (cudaStream_t) s : cs(st_);
    // The scratch a product quantizes into (and the BF16 matrix's floats): the members' when the product runs on
    // st_, the shared expert's own when a caller runs it beside st_ (STRATA_SHARED_OVERLAP) - two streams must not
    // write one scratch.  With s == nullptr this is the old code, byte for byte.
    void* q8 = s ? q8_sh_ : q8_;
    float* wf = s ? wf_sh_ : wf_;
    if (w.type == kTypeF32 || w.type == kTypeBF16) {
        const int per = rows_per_x ? rows_per_x : w.n_out;
        const size_t count = (size_t) w.n_in * w.n_out;
        // A prompt batch: one matrix product.  The kernel below has a block for every output of every token,
        // and GLM 5.3 Flash's per-head projections are 32,768 outputs a token.
        if (n >= kStageRows && (w.type == kTypeF32 || (wf && count <= bf16_max_))) {
            const float* wf2 = fp(w);
            if (w.type == kTypeBF16) {
                bf16_to_f32((const uint16_t*) w.p, wf, count, st);
                wf2 = wf;
            }
            return gemm_f32(blas_, wf2, x, y, w.n_in, w.n_out, per, n);
        }
        if (w.type == kTypeF32) return gemv_f32(fp(w), x, y, w.n_in, w.n_out, per, n, st);
        return gemv_bf16((const uint16_t*) w.p, x, y, w.n_in, w.n_out, per, n, st);
    }
    // a prompt batch: llama.cpp's matrix-matrix kernels read the weights once for all the tokens
    if (mmq_ && n >= kStageRows && Mmq::fits((int) w.type, w.n_out, w.n_in)) {
        Mmq::quantize(x, nullptr, mqx_, (int) w.type, w.n_in, w.n_in, n, st);
        return mmq_->mul(w.p, (int) w.type, w.n_out, w.n_in, 0, 1, mqx_, nullptr, nullptr, n, n, y, w.n_out, st);
    }
    if (!xq) {
        // the tokens' blocks are back to back, so they quantize as one long vector
        kernels::native_quantize_q8_1(x, q8, w.n_in * n, 1, st);
        xq = q8;
    }
    // a launch takes up to 8 tokens, and each comes out as it would alone
    const size_t token = kernels::native_q8_1_bytes(w.n_in);
    for (int at = 0; at < n; at += 8)
        kernels::native_mmvq((int) w.type, w.p, (const uint8_t*) xq + (size_t) at * token,
                             y + (size_t) at * w.n_out, w.n_in, w.n_out, std::min(8, n - at), st);
}

// STRATA_DUMP_ROUTE=1 only (default 0: never called).
void Engine::dump_route(const int* ids, int il, int32_t token) {
    // One line, one write.  The old code did three fprintf + fputc with no lock, and the prefill path routes from 16
    // worker threads, so lines interleaved: the R74 dump had 1 to 168 fields a line.  Build the line in a local
    // buffer, then take a process-wide mutex for the single fwrite.
    static std::mutex m;
    char line[(2 + 22) * 12 + 4];
    int at = std::snprintf(line, sizeof line, "%d %d", il, (int) token);
    for (int k = 0; k < n_used_ && at > 0 && (size_t) at + 12 < sizeof line; ++k)
        at += std::snprintf(line + at, sizeof line - (size_t) at, " %d", ids[k]);
    if (at < 0 || (size_t) at >= sizeof line - 1) return;
    line[at++] = '\n';
    {
        std::lock_guard<std::mutex> lk(m);
        std::fwrite(line, 1, (size_t) at, route_dump_);
    }
}

void Engine::router_scores(const Layer& L, int n) { lin(L.gate_inp, xn_, rl_, n, xq_); }

bool Engine::batched(const Layer& L, int n) const { return mmq_ && L.mmq && n >= kStageRows; }

// The router's scores, and the copies of them and of the experts' input to the host.
void Engine::moe_router(const Layer& L, int n) {
    router_scores(L, n);   // the router eats the normed input (n_embd_); the experts may eat a narrower one
    const int in_w = L.exp_in > 0 ? L.exp_in : n_embd_;
    const float* in = xn_;
    if (L.latent_down.p) {
        // a latent MoE: the experts read the projection of the normed input, so xq_ is refilled from it
        lin(L.latent_down, xn_, moe_in_, n, xq_);
        // the tokens' rows back to back as one column, as lin() does: in_w is a multiple of the 32-element block, so
        // no block crosses a token and this is bitwise the same as n columns.  n as ncols overflowed native MMVQ's
        // MAX_NCOLS for a prompt over 8 tokens (t13, P3-20261016\鎻愰€熸柦宸ュ崟-20261016.md 搂1).
        kernels::native_quantize_q8_1(moe_in_, xq_, in_w * n, 1, st_);
        in = moe_in_;
    }
    copy2(rl_, h_rl_, n * n_expert_, in, h_x_, n * in_w, st_);
}

// The routed experts of tokens that come a few at a time.  moe_router() is on its way, with ev_ behind it.  True
// when experts were computed on the CPU: their sum waits in h_cpu_ for moe_add().  `first` is run once, by the
// thread that hands the cached experts to the card and ahead of them.
bool Engine::moe(Layer& L, int il, int n, const std::function<void()>* first) {
    const int in_w = L.exp_in > 0 ? L.exp_in : n_embd_;       // the experts' input width (a latent MoE: 1024)
    const int out_w = L.exp_out > 0 ? L.exp_out : n_embd_;    // and their output width
    const size_t e_row = (size_t) out_w;
    // a latent MoE's routed sum lands in moe_out_; latent_up maps it into fo_ after (layer()).  Otherwise, fo_.
    float* acc = L.latent_up.p ? moe_out_ : fo_;
    // The routed sum accumulates (add_scaled/axpy are read-modify-write).  fo_ is already written by moe_shared()
    // before this; moe_out_ is a fresh buffer nothing else writes, so it is cleared here.  Verifier caught this.
    if (L.latent_up.p)
        ck(cudaMemsetAsync(moe_out_, 0, (size_t) n * e_row * sizeof(float), cs(st_)), "experts");
    Clock::time_point t0 = Clock::now();
    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");
    stats_.route_wait_ms += ms_since(t0);

    const float limit = clamp_exp_[(size_t) il];
    const size_t xq_row = kernels::native_q8_1_bytes(in_w);
    bool any_host = false;
    // A window's tokens together: the CPU reads an expert that several of them chose once.
    for (int at = 0; at < n; at += kCpuRows) {
        const int m = std::min(kCpuRows, n - at);
        struct Cached {
            const uint8_t* slot;
            float w;
            int row;
            const uint8_t* down = nullptr;   // --pcie-mode direct: the down matrix where it is in the host mapping
        };
        CpuExpert host[kCpuRows * kMaxUsed];
        Cached cached[kCpuRows * kMaxUsed];
        int32_t host_e[kCpuRows * kMaxUsed], host_k[kCpuRows * kMaxUsed];   // a host entry's expert and its place
        int n_host = 0, n_cached = 0;
        for (int t = 0; t < m; ++t) {
            int ids[kMaxUsed];
            float w[kMaxUsed];
            route(il, h_rl_ + (size_t) (at + t) * n_expert_, tok_ ? tok_[at + t] : -1, ids, w, route_tmp_.data());
            if (route_dump_) dump_route(ids, il, (int32_t) (pos_ + at + t));
            for (int k = 0; k < n_used_; ++k) {
                const size_t e = (size_t) ids[k];
                ++use_[(size_t) il * n_expert_ + e];
                ++use_decode_[(size_t) il * n_expert_ + e];   // STRATA_PROFILE_DECODE_ONLY: the answer's counts
                if (n_swap_ > 0) recent_[(size_t) il * n_expert_ + e] += 1.f;
                if (const uint8_t* s = L.slot[e]) {
                    cached[n_cached++] = {s, w[k], at + t};
                    continue;
                }
                host_e[n_host] = (int32_t) e;
                host_k[n_host] = k;
                const uint8_t* downT =
                    r2d_base_ ? r2d_base_ + r2d_layer_off_[(size_t) il] + (size_t) e * r2d_expert_bytes_ : nullptr;
                host[n_host++] = {L.gated ? L.exp_gate + e * L.gu_stride : nullptr, L.exp_up + e * L.gu_stride,
                                  L.exp_down + e * L.d_bytes, w[k], t, downT};
            }
        }
        // Strata's STRATA_ROUTE_TAIL_SKIP: an expert outside VRAM that every token here ranks at tail_skip or later
        // adds little to any of them, so it is not computed.  The others keep the weights the router gave them.
        if (opt_.tail_skip > 0) {
            bool keep[kCpuRows * kMaxUsed];
            for (int i = 0; i < n_host; ++i) {
                bool low = true;
                for (int j = 0; j < n_host && low; ++j) low = host_e[j] != host_e[i] || host_k[j] >= opt_.tail_skip;
                keep[i] = !low;
                if (low) ++stats_.tail_experts;
            }
            int kept = 0;
            for (int i = 0; i < n_host; ++i)
                if (keep[i]) {
                    host[kept] = host[i];
                    host_e[kept] = host_e[i];   // --pcie-frac reads the expert id of a kept use
                    host_k[kept] = host_k[i];
                    ++kept;
                }
            n_host = kept;
        }
        // --pcie-frac: the last m = (this block's misses * frac) of them, in routing order, go over PCIe instead of to
        // the CPU: their experts are DMA'd out of the pinned mapping into the staging slots by a stream of their own,
        // and the compute stream reads the slots as cache slots.  The rest stay on the CPU.  Nothing here runs at the
        // switch's default (0), so that run is the one the engine gave before the switch existed.
        constexpr int kMaxPcie = 256;   // Options::pcie_slots is 0 to 256
        int n_take = 0;                 // distinct experts of this layer/block going over PCIe
        Cached pcie[kCpuRows * kMaxUsed];
        int n_pcie = 0;                 // the uses they serve
        uint8_t* pcie_dst[kMaxPcie];
        const uint8_t* pcie_src_gate[kMaxPcie];
        const uint8_t* pcie_src_up[kMaxPcie];
        const uint8_t* pcie_src_down[kMaxPcie];
        const bool direct = opt_.pcie_frac > 0.0 && pcie_direct_ && !L.gated && pcie_slots_ > 0 &&
                            (size_t) il < pin_layer_.size() && pin_layer_[(size_t) il] != 0 &&
                            (size_t) il < exp_up_dev_.size() && exp_up_dev_[(size_t) il] != nullptr &&
                            exp_down_dev_[(size_t) il] != nullptr;
        const bool pcie_ok = opt_.pcie_frac > 0.0 && pcie_slots_ > 0 && (size_t) il < pin_layer_.size() &&
                             pin_layer_[(size_t) il] != 0 &&
                             (direct || (pcie_stage_ != nullptr && pcie_st_ != nullptr && ev_pcie_ != nullptr &&
                                         pcie_slot_bytes_ > 0));
        if (pcie_ok && n_host > 0) {
            const int pcie_num = std::min(256, (int) std::lround(std::min(opt_.pcie_frac, 1.0) * 256.0));
            const int m_uses = (int) (((long long) n_host * pcie_num) >> 8);
            // their distinct experts, in routing order; the lowest ranked are the last of them
            int32_t take[kMaxPcie];
            for (int i = n_host - m_uses; i < n_host; ++i) {
                bool seen = false;
                for (int j = 0; j < n_take && !seen; ++j) seen = take[j] == host_e[i];
                if (!seen) take[n_take++] = host_e[i];
            }
            if (n_take > pcie_slots_) {   // keep the last slots: they are the experts ranked lowest
                std::memmove(take, take + (n_take - pcie_slots_), (size_t) pcie_slots_ * sizeof(take[0]));
                n_take = pcie_slots_;
            }
            for (int j = 0; j < n_take; ++j) {
                const size_t e = (size_t) take[j];
                if (direct) {
                    // read in place: no slot and no copy; the kernels take the device aliases of the pinned mapping
                    pcie_dst[j] = nullptr;
                    pcie_src_gate[j] = nullptr;
                    pcie_src_up[j] = exp_up_dev_[(size_t) il] + e * L.gu_stride;
                    pcie_src_down[j] = exp_down_dev_[(size_t) il] + e * L.d_bytes;
                    continue;
                }
                pcie_dst[j] = pcie_stage_ + (size_t) j * pcie_slot_bytes_;
                pcie_src_gate[j] = L.gated ? L.exp_gate + e * L.gu_stride : nullptr;
                pcie_src_up[j] = L.exp_up + e * L.gu_stride;
                pcie_src_down[j] = L.exp_down + e * L.d_bytes;
                stats_.pcie_dmas += L.gated ? 3 : 2;
                stats_.pcie_dma_bytes += (L.gated ? 2 : 1) * L.gu_bytes + L.d_bytes;
            }
            int kept = 0;   // the uses of those experts move off the CPU's list
            for (int i = 0; i < n_host; ++i) {
                int q = -1;
                for (int j = 0; j < n_take && q < 0; ++j) q = take[j] == host_e[i] ? j : -1;
                if (q < 0) host[kept++] = host[i];
                else if (direct) {
                    pcie[n_pcie++] = {pcie_src_up[q], host[i].weight, at + host[i].row, pcie_src_down[q]};
                    ++stats_.pcie_direct;   // one use read in place; the DMAs stay 0 for it
                } else pcie[n_pcie++] = {pcie_dst[q], host[i].weight, at + host[i].row};
            }
            n_host = kept;
            stats_.pcie_experts += n_pcie;
        }
        stats_.gpu_experts += n_cached;
        stats_.cpu_experts += n_host;
        // The launches cost this thread more than they cost the card, so each one less counts.  The cached experts
        // go kSlots at a time, a launch a stage for all of them: gate and up, the activation, its quantization,
        // down.  Then one launch adds up to kSumRows of them into their tokens' rows, in the order they came.  A
        // format without a kernel for several matrices gets the four launches an expert.
        // Nothing else goes to the stream until the CPU experts are done, so one of their threads hands these to
        // the card meanwhile: on this thread they held the CPU experts up.  launch_ms is how long that hand-over
        // took: cpu_busy_ms leaves it out, so a split's copies and launches do not read as CPU time.
        double launch_ms = 0;
        const std::function<void()> launch = [&] {
            const Clock::time_point lt0 = Clock::now();
            if (split_) cudaSetDevice(device_);   // one of the experts' threads, which every card's engine shares
            if (first && at == 0) (*first)();
            // --pcie-frac: this block's copies go on their own stream now, so they fly beside the cached experts and
            // the CPU's work; the wait for them goes in just before the split's experts are computed at the end.
            if (n_take > 0 && !direct) {
                for (int j = 0; j < n_take; ++j) {
                    uint8_t* const dst = pcie_dst[j];
                    if (L.gated)
                        ck(cudaMemcpyAsync(dst, pcie_src_gate[j], L.gu_bytes, cudaMemcpyHostToDevice, cs(pcie_st_)),
                           "PCIe staging");
                    ck(cudaMemcpyAsync(dst + L.slot_up(), pcie_src_up[j], L.gu_bytes, cudaMemcpyHostToDevice,
                                       cs(pcie_st_)), "PCIe staging");
                    ck(cudaMemcpyAsync(dst + L.slot_down(), pcie_src_down[j], L.d_bytes, cudaMemcpyHostToDevice,
                                       cs(pcie_st_)), "PCIe staging");
                }
                ck(cudaEventRecord((cudaEvent_t) ev_pcie_, cs(pcie_st_)), "PCIe staging");
            }
            const size_t ff = (size_t) n_ff_exp_, q8_row = kernels::native_q8_1_bytes(n_ff_exp_);
            const int per = rows_ >= kSlots ? kSlots : 1;   // the scratch holds rows_ experts' values
            ScaledRows sum{};
            for (int i0 = 0; i0 < n_cached; i0 += per) {
                const int m = std::min(per, n_cached - i0);
                kernels::MmvqSlots gate_up{}, down{};
                for (int j = 0; j < m; ++j) {
                    const Cached& c = cached[i0 + j];
                    gate_up.w[j] = c.slot;
                    gate_up.x[j] = (const uint8_t*) xq_ + (size_t) c.row * xq_row;
                    gate_up.y[j] = g_ + (size_t) j * (L.gated ? 2 * ff : ff);
                    down.w[j] = c.slot + L.slot_down();
                    down.x[j] = (const uint8_t*) q8_ + (size_t) j * q8_row;
                    down.y[j] = d_ + (size_t) ((i0 + j) % kSumRows) * e_row;
                }
                // STRATA_PROBE_XQ=<path>: dump each (pos, row, j)'s xq_row input bytes before the up mmvq, so two
                // runs' files can be diffed - the mmvq itself is reproducible (PROBE_REPEAT), so a difference here
                // is the source.  A synced observation (瑙傛祴鎺╃洊寰?: read it against a no-hook run in the same batch.
                if (static const char* xqf = std::getenv("STRATA_PROBE_XQ"); xqf && il == 1) {
                    static FILE* fp = std::fopen(xqf, "ab");
                    if (fp) {
                        std::vector<uint8_t> xb(xq_row);
                        for (int j = 0; j < m; ++j) {
                            ck(cudaStreamSynchronize(cs(st_)), "xq");
                            ck(cudaMemcpy(xb.data(), gate_up.x[j], xq_row, cudaMemcpyDeviceToHost), "xq");
                            const long long hdr[3] = {(long long) pos_, (long long) (i0 + j), (long long) j};
                            std::fwrite(hdr, sizeof(hdr), 1, fp);
                            std::fwrite(xb.data(), 1, xq_row, fp);
                        }
                        std::fflush(fp);
                    }
                }
                const int gu_w = L.gated ? 2 * (int) n_ff_exp_ : (int) n_ff_exp_;
                const bool one = m > 1 && kernels::native_mmvq_slots((int) L.gu_type, gate_up, m, in_w, gu_w, st_);
                for (int j = 0; j < m && !one; ++j)
                    kernels::native_mmvq((int) L.gu_type, gate_up.w[j], gate_up.x[j], gate_up.y[j], in_w, gu_w, 1, st_);
                // STRATA_PROBE_REPEAT: the same up mmvq run twice on the same input, compared bitwise - a kernel-
                // internal race shows as two different results.  control_diff is the comparator's own control (a
                // perturbed copy must read != 0).  Temporary; revert with the other probes.
                if (std::getenv("STRATA_PROBE_REPEAT") && il == 1) {
                    const size_t pn = (size_t) m * gu_w;
                    std::vector<float> p1(pn), p2(pn), p3(pn);
                    // st_ is a NonBlocking stream, so a legacy memcpy does not order after the launches: sync first.
                    const auto up = [&] {
                        const bool o = m > 1 && kernels::native_mmvq_slots((int) L.gu_type, gate_up, m, in_w, gu_w, st_);
                        for (int j = 0; j < m && !o; ++j)
                            kernels::native_mmvq((int) L.gu_type, gate_up.w[j], gate_up.x[j], gate_up.y[j], in_w, gu_w, 1, st_);
                    };
                    up();
                    ck(cudaStreamSynchronize(cs(st_)), "rep");
                    ck(cudaMemcpy(p1.data(), g_, pn * sizeof(float), cudaMemcpyDeviceToHost), "rep");   // call 1
                    up();
                    ck(cudaStreamSynchronize(cs(st_)), "rep");
                    ck(cudaMemcpy(p2.data(), g_, pn * sizeof(float), cudaMemcpyDeviceToHost), "rep");   // call 2, same input
                    // positive control: call 3 on a DIFFERENT input (expert 0 reads another token's row) - ctrl_diff
                    // must be non-zero, or the probe cannot tell inputs apart.
                    auto xs = gate_up.x[0];
                    gate_up.x[0] = (const uint8_t*) xq_ + (size_t) (cached[i0].row == 0 ? 1 : 0) * xq_row;
                    up();
                    ck(cudaStreamSynchronize(cs(st_)), "rep");
                    ck(cudaMemcpy(p3.data(), g_, pn * sizeof(float), cudaMemcpyDeviceToHost), "rep");
                    gate_up.x[0] = xs;
                    up();                                                // restore g_ for the activation that follows
                    ck(cudaStreamSynchronize(cs(st_)), "rep");
                    int sd = 0, cd = 0;
                    for (size_t q = 0; q < pn; ++q) sd += (p1[q] != p2[q]);
                    for (size_t q = 0; q < pn; ++q) cd += (p2[q] != p3[q]);
                    std::fprintf(stderr, "ndrep pos=%lld blk=%d n=%zu same_diff=%d ctrl_diff=%d\n",
                                 (long long) pos_, il, pn, sd, cd);
                }
                if (!L.gated) relu2(g_, h_, (size_t) m * n_ff_exp_, st_);   // no gate: relu2 of up
                else if (gelu_) geglu_rows(g_, h_, n_ff_exp_, m, st_);
                else swiglu_clamp_rows(g_, h_, n_ff_exp_, m, limit, st_);
                kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_, m, st_);
                if (r2d_slots_t_) {   // Stage D D3: the slot's down segment holds the neuron-major T layout
                    if (m > 1)
                        kernels::native_mmvq_slots_T((int) L.d_type, down, m, n_ff_exp_, out_w, st_);
                    else
                        for (int j = 0; j < m; ++j)
                            kernels::native_iq4_nl_mmvq_T(down.w[j], down.x[j], down.y[j], n_ff_exp_, out_w, 1, st_);
                } else if (!(m > 1 && kernels::native_mmvq_slots((int) L.d_type, down, m, n_ff_exp_, out_w, st_)))
                    for (int j = 0; j < m; ++j)
                        kernels::native_mmvq((int) L.d_type, down.w[j], down.x[j], down.y[j], n_ff_exp_, out_w, 1,
                                             st_);
                // STRATA_PROBE_NONDET=<layer>: temporary.  The block's GPU expert stages, so two runs' dumps differ
                // at the first stage a kernel made non-reproducible (g_ up, h_ activation, d_ down).  Revert.
                if (static const char* pn = std::getenv("STRATA_PROBE_NONDET"); pn && il == std::atoi(pn)) {
                    dbg("nd_g", il, g_, m * gu_w);
                    dbg("nd_h", il, h_, m * (int) n_ff_exp_);
                    dbg("nd_d", il, d_, m * (int) e_row);
                    // %.9g (dbg prints %.6f, too coarse to see a per-token drift before it crosses): the same
                    // g_ values read twice, so a2/b must match within a run - the hook's own control.
                    float hp[3] = {}, hq[3] = {};
                    ck(cudaMemcpy(hp, g_, sizeof(hp), cudaMemcpyDeviceToHost), "nd");
                    ck(cudaMemcpy(hq, g_, sizeof(hq), cudaMemcpyDeviceToHost), "nd");
                    std::fprintf(stderr, "ndhp pos=%lld blk=%d a=%.9g %.9g %.9g b=%.9g %.9g %.9g\n", (long long) pos_,
                                 il, hp[0], hp[1], hp[2], hq[0], hq[1], hq[2]);
                }
                for (int j = 0; j < m; ++j) {
                    const Cached& c = cached[i0 + j];
                    const int k = (i0 + j) % kSumRows;
                    sum.x[k] = down.y[j];
                    sum.acc[k] = acc + (size_t) c.row * e_row;
                    sum.w[k] = c.w;
                    if (k == kSumRows - 1 || i0 + j == n_cached - 1) add_scaled(sum, k + 1, out_w, st_);
                }
            }
            if (static const char* pn = std::getenv("STRATA_PROBE_NONDET"); pn && il == std::atoi(pn))
                dbg("nd_acc", il, acc, n * (int) e_row);   // temporary, with the other nd_ probes
            // --pcie-frac: the split's experts, computed from the staging slots as if they were cache slots (no new
            // kernel: the same native_mmvq* calls).  The wait is here, not before the cached loop, so the cached work
            // and the CPU's overlapped the copies.
            if (n_pcie > 0) {
                if (!direct) {
                    const Clock::time_point w0 = Clock::now();
                    ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_pcie_, 0), "PCIe staging");
                    stats_.pcie_wait_ms += ms_since(w0);   // R49: the ledger's real pcie_us/miss (0 at frac 0)
                }
                ScaledRows psum{};
                for (int i0 = 0; i0 < n_pcie; i0 += per) {
                    const int m = std::min(per, n_pcie - i0);
                    kernels::MmvqSlots gate_up{}, down{};
                    for (int j = 0; j < m; ++j) {
                        const Cached& c = pcie[i0 + j];
                        gate_up.w[j] = c.slot;
                        gate_up.x[j] = (const uint8_t*) xq_ + (size_t) c.row * xq_row;
                        gate_up.y[j] = g_ + (size_t) j * (L.gated ? 2 * ff : ff);
                        down.w[j] = c.down ? c.down : c.slot + L.slot_down();
                        down.x[j] = (const uint8_t*) q8_ + (size_t) j * q8_row;
                        down.y[j] = d_ + (size_t) ((i0 + j) % kSumRows) * e_row;
                    }
                    const int gu_w = L.gated ? 2 * (int) n_ff_exp_ : (int) n_ff_exp_;
                    const bool one = m > 1 && kernels::native_mmvq_slots((int) L.gu_type, gate_up, m, in_w, gu_w, st_);
                    for (int j = 0; j < m && !one; ++j)
                        kernels::native_mmvq((int) L.gu_type, gate_up.w[j], gate_up.x[j], gate_up.y[j], in_w, gu_w, 1,
                                             st_);
                    if (!L.gated) relu2(g_, h_, (size_t) m * n_ff_exp_, st_);   // no gate: relu2 of up
                    else if (gelu_) geglu_rows(g_, h_, n_ff_exp_, m, st_);
                    else swiglu_clamp_rows(g_, h_, n_ff_exp_, m, limit, st_);
                    kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_, m, st_);
                    if (r2d_slots_t_) {   // Stage D D3: the slot's down segment holds the neuron-major T layout
                        if (m > 1)
                            kernels::native_mmvq_slots_T((int) L.d_type, down, m, n_ff_exp_, out_w, st_);
                        else
                            for (int j = 0; j < m; ++j)
                                kernels::native_iq4_nl_mmvq_T(down.w[j], down.x[j], down.y[j], n_ff_exp_, out_w, 1,
                                                              st_);
                    } else if (!(m > 1 && kernels::native_mmvq_slots((int) L.d_type, down, m, n_ff_exp_, out_w, st_)))
                        for (int j = 0; j < m; ++j)
                            kernels::native_mmvq((int) L.d_type, down.w[j], down.x[j], down.y[j], n_ff_exp_, out_w, 1,
                                                 st_);
                    for (int j = 0; j < m; ++j) {
                        const int k = (i0 + j) % kSumRows;
                        psum.x[k] = down.y[j];
                        psum.acc[k] = acc + (size_t) pcie[i0 + j].row * e_row;
                        psum.w[k] = pcie[i0 + j].w;
                        if (k == kSumRows - 1 || i0 + j == n_pcie - 1) add_scaled(psum, k + 1, out_w, st_);
                    }
                }
            }
            launch_ms += ms_since(lt0);
        };

        // The next layer's router wait is what makes h_cpu_ free to write again.
        float* cpu = h_cpu_ + (size_t) at * e_row;
        if (n_host == 0) {
            launch();
            std::fill(cpu, cpu + (size_t) m * e_row, 0.f);
            continue;
        }
        any_host = true;
        t0 = Clock::now();
        const bool hand_over = n_cached > 0 || n_pcie > 0 || (first && at == 0);
        // R51: the ReLU^2 sparsity probe is read only; the flag is asked once per process.
        static const bool probe_relu2 = [] {
            const char* e = std::getenv("STRATA_PROBE_RELU2");
            return e != nullptr && std::atoi(e) != 0;
        }();
        Relu2Probe probe;
        Relu2Active relu2;   // R56 Stage B: read only; built only when the sidecar is on
        relu2.sidecar = r2d_base_ != nullptr;
        cpu_experts(*workers_, host, n_host, m, (int) L.gu_type, (int) L.d_type, in_w, n_ff_exp_,
                    h_x_ + (size_t) at * e_row, limit, gelu_, cpu, out_w, hand_over ? &launch : nullptr,
                    &stats_.pool_gate_up_ms, &stats_.pool_quant_ms, &stats_.pool_down_ms,
                    probe_relu2 ? &probe : nullptr, r2d_base_ ? &relu2 : nullptr);
        if (probe_relu2) {
            stats_.probe_values += probe.values;
            stats_.probe_zeros += probe.zeros;
            stats_.probe_blocks += probe.blocks;
            stats_.probe_blocks_zero += probe.blocks_zero;
            stats_.probe_lines += probe.lines;
            stats_.probe_lines_zero += probe.lines_zero;
        }
        if (r2d_base_) {
            stats_.r2d_uses += relu2.uses;
            stats_.r2d_active += relu2.active;
            stats_.r2d_active_q8 += relu2.active_q8;
            stats_.r2d_groups += relu2.groups;
            stats_.r2d_bytes += relu2.bytes;
        }
        const double wall = ms_since(t0);
        stats_.launch_ms += launch_ms;   // R51: the launch segment measured where it runs, not derived from cpu_ms
        stats_.cpu_ms += wall;
        stats_.cpu_busy_ms += std::max(0.0, wall - launch_ms);
    }
    return any_host;
}

// The CPU's experts of n tokens into the routed experts' sum.
void Engine::moe_add(Layer& L, int n) {
    const int out_w = L.exp_out > 0 ? L.exp_out : n_embd_;
    axpy(h_cpu_, 1.f, L.latent_up.p ? moe_out_ : fo_, n * out_w, true, st_);
}

// A latent MoE's routed sum is in moe_out_ (exp_out wide); latent_up maps it into fo_, where the shared expert is.
// STRATA_SHARED_TIMING=1: the same call, timed.  The decode steps are the n == 1 calls; the ledger prints
// shared_n1_ms / shared_n1 as "ms/token".  Default 0: one getenv at first use, then the plain call.
void Engine::moe_shared_timed(const Layer& L, int il, int n) {
    // STRATA_SKIP_SHARED=1: a TIMING PROBE only - the shared expert is not run at all, so fo_ loses its
    // contribution and the ids differ (class-E, never for a numeric gate).  Default 0: byte for byte the old path.
    static const bool skip_shared = [] {
        const char* e = std::getenv("STRATA_SKIP_SHARED");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    // Self-cert for the probe: one line a run, with fo_'s first/last four floats, so the =0 and =1 arms can be
    // compared on the output itself and not only on the ids (ids equal by chance is the failure mode we check for).
    if (std::getenv("STRATA_SKIP_SHARED") != nullptr) {
        static bool said = false;
        if (!said) {
            said = true;
            float b[8] = {};
            ck(cudaStreamSynchronize(cs(st_)), "shared probe");
            ck(cudaMemcpy(b, fo_, 4 * sizeof(float), cudaMemcpyDeviceToHost), "shared probe");
            ck(cudaMemcpy(b + 4, fo_ + (size_t) n * n_embd_ - 4, 4 * sizeof(float), cudaMemcpyDeviceToHost),
               "shared probe");
            std::fprintf(stderr,
                         "shared-skip: moe_shared %s (n=%d) fo[0..3]=%.6g %.6g %.6g %.6g fo[last4]=%.6g %.6g %.6g "
                         "%.6g\n",
                         skip_shared ? "SKIPPED (probe, output invalid)" : "RUN (control)", n, b[0], b[1], b[2], b[3],
                         b[4], b[5], b[6], b[7]);
        }
    }
    if (skip_shared) return;
    static const bool on = [] {
        const char* e = std::getenv("STRATA_SHARED_TIMING");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    if (!on) {
        moe_shared(L, il, n);
        return;
    }
    // Wall time of the function is launch time (the kernels are async): measure with events and wait, which is fine
    // for the timing run only.  Decode steps are the calls with n <= 8 (a window); prefill passes thousands.
    static cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (!ev0) {
        ck(cudaEventCreate(&ev0), "shared timing");
        ck(cudaEventCreate(&ev1), "shared timing");
    }
    ck(cudaEventRecord(ev0, cs(st_)), "shared timing");
    moe_shared(L, il, n);
    ck(cudaEventRecord(ev1, cs(st_)), "shared timing");
    ck(cudaEventSynchronize(ev1), "shared timing");
    float ms = 0.f;
    ck(cudaEventElapsedTime(&ms, ev0, ev1), "shared timing");
    stats_.shared_ms += (double) ms;
    if (n <= 8) {
        stats_.shared_n1_ms += (double) ms;
        ++stats_.shared_n1;
    }
}

void Engine::moe_latent(Layer& L, int n) {
    // STRATA_SHARED_OVERLAP: the shared expert wrote fo_ on sh_ beside the routed experts' work; that write must
    // land before the routed sum is added into fo_ (the only other writer).  Off: sh_pending_ is never set.
    if (sh_pending_) {
        ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_sh_), "shared expert");
        sh_pending_ = false;
    }
    if (!L.latent_up.p) return;
    lin(L.latent_up, moe_out_, moe_up_, n);
    axpy(moe_up_, 1.f, fo_, (size_t) n * n_embd_, true, st_);
}

// With every routed expert in VRAM, a few tokens' experts are routed on the card and found there by its picks
// (moe_card()), so a layer never waits for the host.  The cache's slots go to the card again when they changed.
bool Engine::card_moe(int n) {
    if (!all_cached() || !d_slots_ || n * n_used_ > kSumRows || opt_.dump) return false;
    bool same = !h_slots_.empty();
    for (int il = lo_; il < hi_ && same; ++il) {
        const Layer& L = layers_[(size_t) il];
        for (int e = 0; e < n_expert_ && same && L.moe; ++e)
            same = h_slots_[(size_t) il * n_expert_ + e] == L.slot[(size_t) e];
    }
    if (!same) {
        h_slots_.assign((size_t) n_layer_ * n_expert_, nullptr);
        for (int il = lo_; il < hi_; ++il)
            if (layers_[(size_t) il].moe)
                std::copy(layers_[(size_t) il].slot.begin(), layers_[(size_t) il].slot.end(),
                          h_slots_.begin() + (ptrdiff_t) il * n_expert_);
        ck(cudaMemcpy(d_slots_, h_slots_.data(), h_slots_.size() * sizeof(void*), cudaMemcpyHostToDevice),
           "expert slots");
    }
    return true;
}

// moe()'s launches for a few tokens whose experts are all in VRAM, with the card's picks for the host's: routed on
// the card, an expert's slot read from d_slots_ by its pick.  False, with nothing launched: this layer is routed
// on the host, or its formats have no kernel for picks on the card.
bool Engine::moe_card(Layer& L, int il, int n) {
    if (r2d_slots_t_) {   // Stage D: the card's routed down kernel reads the row-major slot; the T layout is D3+
        L.card = 0;
        return false;
    }
    if (!L.gated) return false;   // the card's routed launches are wired for gated experts (moe_rows() takes it next)
    if (L.card == 0) return false;
    if (!kernels::native_mmvq_ids_supported((int) L.gu_type) || !kernels::native_mmvq_ids_supported((int) L.d_type) ||
        !route_card(il, n, d_ids_, d_wts_)) {
        L.card = 0;
        return false;
    }
    L.card = 1;
    const int rr = n * n_used_;
    const void* const* table = d_slots_ + (size_t) il * n_expert_;
    const kernels::MmvqIds gate_up{table, d_ids_, 0, xq_, kernels::native_q8_1_bytes(n_embd_), n_used_, g_,
                                   2 * (size_t) n_ff_exp_};
    kernels::native_mmvq_ids((int) L.gu_type, gate_up, rr, n_embd_, 2 * n_ff_exp_, st_);
    if (gelu_) geglu_rows(g_, h_, n_ff_exp_, rr, st_);
    else swiglu_clamp_rows(g_, h_, n_ff_exp_, rr, clamp_exp_[(size_t) il], st_);
    kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_ * rr, 1, st_);   // the rows back to back, as one vector
    const kernels::MmvqIds down{table, d_ids_, 2 * L.gu_bytes, q8_, kernels::native_q8_1_bytes(n_ff_exp_), 1, d_,
                                (size_t) n_embd_};
    kernels::native_mmvq_ids((int) L.d_type, down, rr, n_ff_exp_, n_embd_, st_);
    moe_sum(d_, d_wts_, fo_, n_embd_, n_used_, n, st_);
    return true;
}

// m experts from the mapped file into the staging buffer: gate and up of each back to back from the start, the
// downs after them.  The two pinned buffers take turns, so the workers fill one while the other uploads.  `alone`:
// stagers_ fill them, while the workers compute experts.
void Engine::stage(const Layer& L, const int32_t* experts, int m, void* stream, bool alone, uint8_t* base) {
    staged_il_ = -1;   // whatever the buffer held
    // T8/P7: every byte this call copies to VRAM, whoever asked for it (early copy, a round, a prefetch).
    // One expert is slot_up()+gu_bytes+d_bytes = slot_down()+d_bytes (engine.hpp:139-140) - what actually moves;
    // expert_slot_bytes() (engine.cpp:1575) prices that plus kTail and the block alignment.
    ++stats_.stage_calls;
    stats_.stage_bytes += (size_t) m * (L.slot_down() + L.d_bytes);
    // STRATA_STAGE_RING2: the round buffer this copy fills (null: the one staging buffer, as always).
    uint8_t* const dst = base ? base : stage_;
    // the expert's own slot layout (slot_up()/slot_down()), not a gate-and-up width: a non-gated expert (Nemotron)
    // has no gate and its up is one matrix.  B3, P3-20261016\B3-閰嶆柟鐩樼偣-20261016.md.
    const size_t gu2 = L.slot_down(), down_at = pad((size_t) m * gu2);
    const int per = (int) (chunk_bytes_ / (gu2 + L.d_bytes));
    const size_t up_at = L.slot_up();          // gated: the up follows the gate; non-gated: 0
    const int pieces = L.gated ? 3 : 2;        // gate, up, down / up, down
    // STRATA_PIN_ARENA_GIB: this layer's sources are registered read-only, so DMA each expert straight from the
    // mapping and skip the host gather.  Whole layers only (see pin_expert_arena()), so no chunk is ever mixed.
    const int il = (int) (&L - layers_.data());
    const bool pin = il >= 0 && (size_t) il < pin_layer_.size() && pin_layer_[(size_t) il];
    for (int j0 = 0; j0 < m; j0 += per) {
        const size_t c = (size_t) std::min(per, m - j0);
        if (pin) {
            ++pin_chunks_;
            for (size_t j = 0; j < c; ++j) {
                const size_t e = (size_t) experts[(size_t) j0 + (int) j];
                const size_t at = (size_t) (j0 + (int) j) * gu2;
                if (L.gated) {
                    ck(cudaMemcpyAsync(dst + at, L.exp_gate + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice,
                                       cs(stream)),
                       "expert staging");
                    pin_dma_bytes_ += L.gu_bytes;
                    ++pin_dmas_;
                }
                ck(cudaMemcpyAsync(dst + at + up_at, L.exp_up + e * L.gu_stride, L.gu_bytes,
                                   cudaMemcpyHostToDevice, cs(stream)),
                   "expert staging");
                if (r2d_base_)
                    materialize_down(L, il, e, dst + down_at + (size_t) (j0 + (int) j) * L.d_bytes, cs(stream));
                else
                    ck(cudaMemcpyAsync(dst + down_at + (size_t) (j0 + (int) j) * L.d_bytes,
                                       L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice, cs(stream)),
                       "expert staging");
                pin_dma_bytes_ += L.gu_bytes + L.d_bytes;
                pin_dmas_ += 2;
            }
            continue;
        }
        if (!pin_layer_.empty()) ++pin_fallback_chunks_;
        stage_turn_ ^= 1;
        uint8_t* h = h_stage_[stage_turn_];
        const cudaEvent_t left = (cudaEvent_t) ev_stage_[stage_turn_];
        ck(cudaEventSynchronize(left), "expert staging");
        const auto copy = [&](int i) {
            const size_t j = (size_t) i / pieces, e = (size_t) experts[(size_t) j0 + j];
            const int k = i % pieces;
            if (L.gated && k == 0) std::memcpy(h + j * gu2, L.exp_gate + e * L.gu_stride, L.gu_bytes);
            else if (k == (L.gated ? 1 : 0)) std::memcpy(h + j * gu2 + up_at, L.exp_up + e * L.gu_stride, L.gu_bytes);
            else if (!r2d_base_) std::memcpy(h + c * gu2 + j * L.d_bytes, L.exp_down + e * L.d_bytes, L.d_bytes);
        };
        (alone ? *stagers_ : *workers_).run((int) (pieces * c), copy);
        ck(cudaMemcpyAsync(dst + (size_t) j0 * gu2, h, c * gu2, cudaMemcpyHostToDevice, cs(stream)),
           "expert staging");
        if (r2d_base_) {
            for (size_t j = 0; j < c; ++j) {
                const size_t e = (size_t) experts[(size_t) j0 + (int) j];
                materialize_down(L, il, e, dst + down_at + ((size_t) j0 + j) * L.d_bytes, cs(stream));
            }
        } else
            ck(cudaMemcpyAsync(dst + down_at + (size_t) j0 * L.d_bytes, h + c * gu2, c * L.d_bytes,
                               cudaMemcpyHostToDevice, cs(stream)),
               "expert staging");
        ck(cudaEventRecord(left, cs(stream)), "expert staging");
    }
    ck(cudaMemsetAsync(dst + down_at + (size_t) m * L.d_bytes, 0, kTail, cs(stream)), "expert staging");
}

// A MoE layer for a prompt batch: the experts on the GPU, by llama.cpp's MMQ.  A row is a token and one
// of the experts it routes to.  The rows are sorted by expert, the experts of the VRAM cache first; the other
// experts pass through the staging buffer, as many at a time as it holds.  A full batch starts the copy of
// L.ahead before it knows the routing, on a stream of its own, so the upload runs under the layer's attention.
// A call that copies only the experts it routes to leaves a share of them to the CPU, which computes them while
// the others are copied: the share that had the two take about as long (Options::cpu_share).
void Engine::moe_rows(Layer& L, int il, int n) {
    const int rr = n * n_used_;
    const size_t cap = (size_t) rows_ * n_used_;
    const bool ahead = opt_.stage_ahead > 0 && (n >= opt_.stage_ahead || shared_ahead_);
    const bool spare = !ahead && opt_.cpu_share != 0 && !all_cached();   // the CPU may take some (below)
    // The layer's tables.  tok and place: a row's token, and its place among that token's experts.  bounds: each
    // expert's first row.  rel: the same for the experts of a staging round, counted from the round's first row.
    // w: the weight of each place.
    int32_t* tok = h_tab_;
    int32_t* place = tok + cap;
    int32_t* bounds = place + cap;
    int32_t* rel = bounds + n_expert_ + 1;
    float* w = (float*) (rel + 2 * n_expert_ + 2);
    int32_t* cb = (int32_t*) (w + cap);   // per class: the cached experts' first rows, in the order of the slots
    router_scores(L, n);
    // The experts' own geometry.  A latent MoE's experts are L.exp_in/L.exp_out wide (1024 here) while the model is
    // n_embd_ (4096); moe() has always used those widths, but moe_rows() ran every product at n_embd_ and read past
    // the expert's matrices (NaN once a call reaches kStageRows tokens, #R).  STRATA_MOE_ROWS_LATENT=0 restores it.
    static const bool latent_rows = []() {
        const char* e = std::getenv("STRATA_MOE_ROWS_LATENT");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    const int in_w = latent_rows ? (L.exp_in > 0 ? L.exp_in : n_embd_) : n_embd_;
    const int out_w = latent_rows ? (L.exp_out > 0 ? L.exp_out : n_embd_) : n_embd_;
    const float* in = xn_;
    if (latent_rows && L.latent_down.p) {
        lin(L.latent_down, xn_, moe_in_, n, xq_);   // the experts read the latent projection, as in moe_router()
        in = moe_in_;
    }
    if (latent_rows && L.latent_up.p)   // the routed sum lands in moe_out_ (out_w wide) for moe_latent()
        ck(cudaMemsetAsync(moe_out_, 0, (size_t) n * out_w * sizeof(float), cs(st_)), "experts");
    // Routed on the card, a batch hands the host its picks: the host took longer to route a batch of 1,024 tokens
    // than the card took for its experts.
    const bool on_card = route_card(il, n, d_ids_, d_wts_);
    if (on_card) {
        copy2(d_ids_, h_ids_, rr, d_wts_, w, rr, st_);
        if (spare) copy2(in, h_x_, n * in_w, nullptr, nullptr, 0, st_);
    } else if (spare) {
        copy2(rl_, h_rl_, n * n_expert_, in, h_x_, n * in_w, st_);
    } else {
        ck(cudaMemcpyAsync(h_rl_, rl_, (size_t) n * n_expert_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
           "router");
    }
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");

    moe_shared_timed(L, il, n);

    // read() takes a layer's calls one after the other, so the buffer may hold this layer's experts from the
    // call before (staged_).  The ones copied ahead do not depend on the routing and are this call's too.  A
    // later round is this call's only if it routes to the same experts, and waiting to see means nothing is
    // copied ahead meanwhile.  That pays when the card gets to the routing in under half the time the copy
    // ahead takes: most of that copy then holds the card up.
    //
    // A small call on its own copies only the experts it routes to.  With another launch's experts the batched
    // kernels would split its sums another way, so it cannot take what a call before it left.  In a read of
    // several calls it copies ahead as the others do, and the copy is theirs too.
    if (ahead) plan_ahead(L, il);
    const int n_ahead = ahead ? (int) L.ahead.size() : 0;
    // STRATA_STAGE_RING2: two round buffers and the next layer's ahead copy running on up_ under this layer's
    // kernels (the prefetch at the end of this function).  Only in a single-call read: the several-call read keeps
    // its own staging reuse.  Off (default) every line below takes the old path, with one buffer and one event pair.
    static const bool ring_check = []() {
        const char* e = std::getenv("STRATA_STAGE_RING2_CHECK");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    // T8/P2: read()'s several calls included. The hold below (P3/P4) keeps one copy a layer, so opening this
    // here does not double the copy; with STRATA_STAGE_RING2=0 stage2_ is null and every line below is the old one.
    const bool ring = stage2_ != nullptr && ahead;
    // T8/P3: only a prefetch for a layer BEFORE this one is stale. A prefetch for il+1 is exactly what this
    // layer's later calls must leave alone (`!= il` here would clear it and re-issue it, 3 copies a layer).
    if (prefetched_il_ >= 0 && prefetched_il_ < il) prefetched_il_ = -1;
    bool prefetched = ring && prefetched_il_ == il;
    // T8/RING2-fix: only a prefetch this call consumes is checked against prefetched_n_; the hold branch reuses
    // THIS layer's list, whose count prefetched_n_ no longer describes (the next layer's prefetch overwrote it).
    const bool from_prefetch = prefetched;
    int rs = prefetched ? prefetched_slot_ : ring_slot_;   // the slot this call's round 0 reads
    if (prefetched)
        prefetched_il_ = -1;
    else if (ring && ring_hold_il_ == il) {   // a later call of this layer: the slot still holds its list
        prefetched = true;
        rs = ring_hold_slot_;
        ++stats_.ring_reuse;
    }
    uint8_t* const rbase = ring ? slot_[rs] : nullptr;
    if (from_prefetch && n_ahead > 0 && n_ahead != prefetched_n_)
        throw std::runtime_error("STRATA_STAGE_RING2: layer " + std::to_string(il) + " has " +
                                 std::to_string(n_ahead) + " ahead experts, " + std::to_string(prefetched_n_) +
                                 " were staged");
    const bool mine = !ring && ahead && staged_il_ == il && (staged_ == L.ahead || 2 * route_ms_ < ahead_ms_);
    const bool early = n_ahead > 0 && !mine && !prefetched;
    if (early) {
        const Clock::time_point t = Clock::now();
        // the buffer is free once the layer before has read it
        if (ring)
            ck(cudaStreamWaitEvent(cs(up_), (cudaEvent_t) ev_ring_free_[rs], 0), "expert staging");
        else
            ck(cudaStreamWaitEvent(cs(up_), (cudaEvent_t) ev_free_, 0), "expert staging");
        stage(L, L.ahead.data(), n_ahead, up_, false, rbase);
        if (ring)
            ck(cudaEventRecord((cudaEvent_t) ev_ring_ahead_[rs], cs(up_)), "expert staging");
        else
            ck(cudaEventRecord((cudaEvent_t) ev_ahead_, cs(up_)), "expert staging");
        ahead_ms_ = ms_since(t);
        // out already: how long before is not known until a call waits for it
        route_ms_ = cudaEventQuery((cudaEvent_t) ev_) == cudaSuccess ? 0 : ahead_ms_;
    }
    if (ring) {
        ring_slot_ = 1 - rs;                      // the slot the next prefetch fills: the one this call does not read
        ring_hold_il_ = n_ahead > 0 ? il : -1;    // later calls of this layer read rs, not a new copy
        ring_hold_slot_ = rs;
        prefetched_n_ = n_ahead;                  // the count the consumption check is against
    }

    const Clock::time_point t0 = Clock::now();
    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");
    if (ahead && !early) route_ms_ = ms_since(t0);
    stats_.route_wait_ms += ms_since(t0);

    const size_t cb_row = (size_t) n_expert_ + 1;
    const int32_t* d_tok = tab_;
    const int32_t* d_place = d_tok + cap;
    const int32_t* d_bounds = d_place + cap;
    const int32_t* d_rel = d_bounds + n_expert_ + 1;
    const float* d_w = (const float*) (d_rel + 2 * n_expert_ + 2);
    const int32_t* d_cb = (const int32_t*) (d_w + cap);

    // A token's routing is a softmax over every expert and a search for the largest few.  On one thread it was
    // 0.9 s of the 2.4 s a 7,709-token prompt took with 40 layers of 256 experts, so the host's threads take a
    // stretch of tokens each.
    const int parts = on_card ? 0 : std::min(workers_->threads(), (n + kStageRows - 1) / kStageRows);
    if (on_card) std::copy(h_ids_, h_ids_ + rr, ids_.begin());
    if (parts > 0) workers_->run(parts, [&](int p) {
        float* tmp = route_tmp_.data() + (size_t) p * 2 * n_expert_;
        const int end = (int) ((int64_t) n * (p + 1) / parts);
        for (int t = (int) ((int64_t) n * p / parts); t < end; ++t) {
            route(il, h_rl_ + (size_t) t * n_expert_, tok_ ? tok_[t] : -1, ids_.data() + (size_t) t * n_used_,
                  w + (size_t) t * n_used_, tmp);
            if (route_dump_) dump_route(ids_.data() + (size_t) t * n_used_, il, (int32_t) (pos_ + t));
        }
    });
    count_.assign((size_t) n_expert_, 0);
    for (int i = 0; i < rr; ++i) {
        ++count_[(size_t) ids_[(size_t) i]];
        ++use_[(size_t) il * n_expert_ + (size_t) ids_[(size_t) i]];
        ++use_prefill_[(size_t) il * n_expert_ + (size_t) ids_[(size_t) i]];   // and the prompt's, apart
    }
    // The CPU computes some of the routed experts outside VRAM while the others are copied: the ones the fewest
    // tokens chose, whose copy costs as much as any other's and which cost the CPU about as much as their tokens.
    int n_spare = 0, n_cpu = 0;
    const int size = std::clamp((int) std::bit_width((unsigned) n) - 6, 0, 4);   // which of share_
    on_cpu_.assign((size_t) n_expert_, 0);
    if (spare) {
        spare_.clear();
        for (int e = 0; e < n_expert_; ++e)
            if (count_[(size_t) e] > 0 && !L.slot[(size_t) e]) spare_.push_back(e);
        std::stable_sort(spare_.begin(), spare_.end(),
                         [&](int32_t a, int32_t b) { return count_[(size_t) a] < count_[(size_t) b]; });
        // Stage D: CPU-share stays ON.  Its experts run the already-verified CPU T kernel (R57) on the pinned sidecar
    // via CpuExpert.downT (set where host_ is built), so this path reads no GGUF down; the live-neuron index costs
    // ~0.85 KB a use and a few ms a layer.
    n_spare = (int) spare_.size();
        const double s = opt_.cpu_share > 0 ? std::min(opt_.cpu_share, 1.f) : share_[size];
        n_cpu = (int) std::lround(s * n_spare);
        for (int j = 0; j < n_cpu; ++j) on_cpu_[(size_t) spare_[(size_t) j]] = 1;
    }
    // The cached experts that have rows, class after class and a class's in the order of the slots (chan_ has
    // every cached expert in that order); the experts already on their way; then the others that have rows.
    chan_.clear();
    for (int e = 0; e < n_expert_; ++e)
        if (L.slot[(size_t) e]) chan_.push_back(e);
    std::sort(chan_.begin(), chan_.end(),
              [&](int32_t a, int32_t b) { return L.slot[(size_t) a] < L.slot[(size_t) b]; });
    const int n_chan = (int) chan_.size();
    group_.clear();
    for (int c = 0; c < kClasses; ++c)
        for (const int32_t e : chan_)
            if (count_[(size_t) e] > 0 && tile_class(count_[(size_t) e]) == c) group_.push_back(e);
    const int n_cached = (int) group_.size();
    group_.insert(group_.end(), L.ahead.begin(), L.ahead.begin() + n_ahead);
    sent_.assign((size_t) n_expert_, 0);
    for (int j = 0; j < n_ahead; ++j) sent_[(size_t) L.ahead[(size_t) j]] = 1;
    for (int e = 0; e < n_expert_; ++e)
        if (count_[(size_t) e] > 0 && !L.slot[(size_t) e] && !sent_[(size_t) e] && !on_cpu_[(size_t) e])
            group_.push_back(e);
    const int n_group = (int) group_.size();
    auto rows_of = [&](int g) { return count_[(size_t) group_[(size_t) g]]; };

    int at = 0;
    for (int g = 0; g < n_group; ++g) {
        bounds[g] = at;
        first_[(size_t) group_[(size_t) g]] = at;
        at += rows_of(g);
    }
    bounds[n_group] = at;
    const int rg = at;   // the card's rows
    host_.clear();
    for (int i = 0; i < rr; ++i) {
        const size_t e = (size_t) ids_[(size_t) i];
        if (on_cpu_[e]) {   // by token, as cpu_experts() takes them, and nothing of it in the card's sum
            // the gate as moe() builds it: a non-gated expert (Nemotron) has none, and exp_gate + offset there
            // would be null + offset (moe_rows() reaches this only for a gated layer today; B3 opens it wider).
            const uint8_t* downT =
                r2d_base_ ? r2d_base_ + r2d_layer_off_[(size_t) il] + e * r2d_expert_bytes_ : nullptr;
            host_.push_back({L.gated ? L.exp_gate + e * L.gu_stride : nullptr, L.exp_up + e * L.gu_stride,
                             L.exp_down + e * L.d_bytes, w[i], i / n_used_, downT});
            w[i] = 0.f;
            continue;
        }
        const int row = first_[e]++;
        tok[row] = i / n_used_;
        place[row] = i;
    }
    const size_t gu2 = L.slot_down();   // the expert's own slot layout (a non-gated expert: up + down only)
    // a round is what the staging buffer holds at a time; round_ has the first group of each, and the end
    const int per_round = (int) (stage_bytes_ / (gu2 + L.d_bytes));
    round_.assign(1, n_cached);
    if (n_ahead > 0) round_.push_back(n_cached + n_ahead);
    while (round_.back() < n_group) round_.push_back(std::min(round_.back() + per_round, n_group));
    const int n_round = (int) round_.size() - 1;
    for (int r = 0; r < n_round; ++r)
        for (int g = round_[(size_t) r]; g <= round_[(size_t) r + 1]; ++g)
            rel[g - n_cached + r] = bounds[g] - bounds[round_[(size_t) r]];
    // A class's launch runs over every cached expert of a run of slots, and the others must have no rows in it:
    // cb has, for each class, where an expert's rows start when only that class's experts have any.
    for (int c = 0, at = 0; c < kClasses; ++c) {
        int32_t* b = cb + (size_t) c * cb_row;
        for (int j = 0; j < n_chan; ++j) {
            b[j] = at;
            const int rows = count_[(size_t) chan_[(size_t) j]];
            if (rows > 0 && tile_class(rows) == c) at += rows;
        }
        b[n_chan] = at;
    }
    ck(cudaMemcpyAsync(tab_, h_tab_,
                       (3 * cap + 3 * (size_t) n_expert_ + 3 + kClasses * cb_row) * sizeof(int32_t),
                       cudaMemcpyHostToDevice, cs(st_)),
       "expert rows");

    const int rc = bounds[n_cached];   // the rows of the cached experts
    stats_.batch_cached += rc;
    stats_.batch_staged += rg - rc;

    const float limit = clamp_exp_[(size_t) il];
    const int gt = (int) L.gu_type, dt = (int) L.d_type;
    const int64_t ff = n_ff_exp_, ff2 = L.gated ? 2 * ff : ff;   // gated: gate|up; non-gated: up
    // STRATA_SCRATCH_TILE: the call runs a batch of its experts at a time.  A lone expert with more rows than the
    // tile cannot be split (mmq takes it whole): that call goes back to the full-rr buffers, never silently.
    const bool tiled = scratch_rows_ < (size_t) rr;
    int maxc = 0;
    for (const int32_t c : count_) maxc = std::max(maxc, (int) c);
    const bool fallback = tiled && (size_t) maxc > scratch_rows_;
    if (fallback)
        std::fprintf(stderr, "WARNING: STRATA_SCRATCH_TILE: expert rows %d > tile %zu; this call uses the full-rr path\n",
                     maxc, scratch_rows_);
    int holds = -1;   // the round in the buffer when this call is done
    // The card's part.  With the CPU's beside it, the rows of its experts' places are left at nothing.
    const auto card = [&] {
        if (fallback) {   // the first fallback makes the full-rr buffers; later ones reuse them
            if (!fb_gu_) {
                const size_t full = (size_t) rows_ * n_used_;
                fb_gu_ = (float*) dmalloc(full * 2 * n_ff_exp_ * sizeof(float), "scratch");
                fb_hx_ = (float*) dmalloc(full * n_ff_exp_ * sizeof(float), "scratch");
                fb_mq2_ = dmalloc(Mmq::q8_bytes((int64_t) full, n_ff_exp_), "scratch");
            }
            std::swap(gu_, fb_gu_);
            std::swap(hx_, fb_hx_);
            std::swap(mq2_, fb_mq2_);
        }
        if (n_cpu > 0) ck(cudaMemsetAsync(dx_, 0, (size_t) rr * out_w * sizeof(float), cs(st_)), "expert rows");
        // The old path's up reads all of mq_.  A tiled call quantizes each batch into mq_tile_ instead: mmq's q8
        // layout is [K block][row], so a batch's rows cannot be sliced out of mq_ (nor is ncols_y still rg).
        if (rg > 0 && (!tiled || fallback)) Mmq::quantize(in, d_tok, mq_, gt, in_w, in_w, rg, st_);

        // The cached experts whose slots lie one behind the other are the matrices of one launch, as the experts of
        // a staging round are; a launch for each class that has rows among them.  The cache is filled a layer's
        // experts side by side, so that is the whole layer until experts trade slots (follow()).  A launch for each
        // expert was a sixth of a long prompt's time in a model with 256 experts a layer.
        auto slot_of = [&](int j) { return L.slot[(size_t) chan_[(size_t) j]]; };
        // A launch carries ONE tile class's row table (cb[c]); a run that mixed classes would feed each class's
        // launch another class's row numbers (out of bounds, garbage floats, NaN once a call reaches 32 tokens).
        // So a contiguous slot segment is split again by class: a run holds only one class's experts (empty slots are
        // class 0 and carry no rows).  With every slot class 0 (a call below kTile[0] tokens) the split is the
        // segment itself, so that path is unchanged.  STRATA_CARD_SPLIT_CLASS=0 restores the old, class-mixing runs.
        static const bool split_class = []() {
            const char* e = std::getenv("STRATA_CARD_SPLIT_CLASS");
            return !(e && e[0] == '0' && e[1] == '\0');
        }();
        runs_.clear();
        for (int j = 0; j < n_chan;) {
            int m = 1;
            while (j + m < n_chan && slot_of(j + m) == slot_of(j + m - 1) + L.slot_bytes) ++m;
            const int end = j + m;
            if (!split_class) {
                int most[kClasses] = {};
                for (int i = j; i < end; ++i) {
                    const int rows = count_[(size_t) chan_[(size_t) i]];
                    if (rows > 0) most[tile_class(rows)] = std::max(most[tile_class(rows)], rows);
                }
                for (int c = 0; c < kClasses; ++c)
                    if (most[c] > 0) runs_.push_back({c, j, m, most[c]});
            } else {
                for (int i = j, e0; i < end; i = e0) {
                    const int c = tile_class(count_[(size_t) chan_[(size_t) i]]);
                    int most = count_[(size_t) chan_[(size_t) i]];
                    for (e0 = i + 1; e0 < end && tile_class(count_[(size_t) chan_[(size_t) e0]]) == c; ++e0)
                        most = std::max(most, count_[(size_t) chan_[(size_t) e0]]);
                    if (most > 0) runs_.push_back({c, i, e0 - i, most});
                }
            }
            j = end;
        }
        // STRATA_SCRATCH_TILE's launches: a segment is m experts of one run (one mmq launch); a batch is a run of
        // segments whose compact rows are the one interval [r0, r0 + rows), so the tile holds local rows 0..rows-1
        // and mq_ is read from row r0.  Each launch's bounds live in one table for the whole call.
        struct Seg { int c, j, m, most; size_t rel; };
        struct Batch { int r0, rows; std::vector<Seg> segs; };
        struct Sub { int gs, ge; size_t rel; };   // a staged round's sub-batch, groups [gs, ge)
        std::vector<Batch> batches;
        std::vector<std::vector<Sub>> round_subs((size_t) n_round);
        if (!tiled || fallback) {
            for (const Run& r : runs_)
                mmq_->mul(slot_of(r.j), gt, ff2, in_w, L.slot_bytes, r.m, mq_, d_cb + (size_t) r.c * cb_row + r.j,
                          iota_, rg, r.most, gu_, ff2, st_);
            if (rc > 0) {
                if (!L.gated) relu2(gu_, hx_, (size_t) rc * n_ff_exp_, st_);   // no gate: relu2 of up
                else if (gelu_) geglu_rows(gu_, hx_, n_ff_exp_, rc, st_);
                else swiglu_clamp_rows(gu_, hx_, n_ff_exp_, rc, limit, st_);
                Mmq::quantize(hx_, nullptr, mq2_, dt, ff, ff, rc, st_);
            }
            for (const Run& r : runs_)
                mmq_->mul(slot_of(r.j) + gu2, dt, out_w, ff, L.slot_bytes, r.m, mq2_,
                          d_cb + (size_t) r.c * cb_row + r.j, d_place, rc, r.most, dx_, out_w, st_);
        } else {
            // runs_ is in slot order and classes interleave, so the batches are formed in the order of cb's
            // absolute rows: only then is a batch one interval (a batch in runs_ order would leave holes in the
            // local row numbers).
            std::vector<int> order(runs_.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                const Run& x = runs_[(size_t) a];
                const Run& y = runs_[(size_t) b];
                return cb[(size_t) x.c * cb_row + x.j] < cb[(size_t) y.c * cb_row + y.j];
            });
            for (const int oi : order) {
                const Run& r = runs_[(size_t) oi];
                for (int pos = 0; pos < r.m; ++pos) {
                    const int j = r.j + pos;
                    const int abs = cb[(size_t) r.c * cb_row + j];
                    const int erows = cb[(size_t) r.c * cb_row + j + 1] - abs;
                    const bool same = !batches.empty() && batches.back().r0 + batches.back().rows == abs &&
                                      batches.back().rows + erows <= (int) scratch_rows_;
                    if (!same) batches.push_back({abs, 0, {}});
                    Batch& b = batches.back();
                    if (!b.segs.empty() && b.segs.back().c == r.c && b.segs.back().j + b.segs.back().m == j)
                        ++b.segs.back().m;
                    else
                        b.segs.push_back({r.c, j, 1, r.most, 0});
                    b.rows += erows;
                }
            }
            // The whole call's bounds table, built once: the cached batches, then every staged round's sub-batches.
            // The first entry of a batch (and of a staged sub-batch) is 0 by construction, written as 0.
            size_t rel_n = 0;
            const size_t rel_cap = 2 * ((size_t) rows_ * n_used_ + (size_t) n_expert_) + 4;
            for (Batch& b : batches) {
                const size_t b_start = rel_n;
                for (Seg& s : b.segs) {
                    s.rel = rel_n;
                    const size_t off = (size_t) s.c * cb_row + s.j;
                    h_rel_local_[rel_n] = cb[off] - b.r0;
                    for (int i = 1; i <= s.m; ++i) h_rel_local_[rel_n + i] = cb[off + (size_t) i] - b.r0;
                    rel_n += (size_t) s.m + 1;
                }
                h_rel_local_[b_start] = 0;   // the batch's first row: bounds[first] - bounds[first], written as 0
            }
            for (int r = 0; r < n_round; ++r) {
                const int g0 = round_[(size_t) r], m = round_[(size_t) r + 1] - g0;
                if (bounds[g0 + m] == bounds[g0]) continue;   // nothing routed to this round's experts
                for (int gs = g0; gs < g0 + m;) {
                    int ge = gs, sub = 0;
                    while (ge < g0 + m) {
                        const int er = bounds[ge + 1] - bounds[ge];
                        if (ge > gs && sub + er > (int) scratch_rows_) break;
                        sub += er;
                        ++ge;
                    }
                    Sub sb{gs, ge, rel_n};
                    h_rel_local_[rel_n] = 0;   // bounds[gs] - bounds[gs], written as 0
                    for (int i = 1; i <= ge - gs; ++i) h_rel_local_[rel_n + (size_t) i] = bounds[gs + i] - bounds[gs];
                    rel_n += (size_t) (ge - gs) + 1;
                    round_subs[(size_t) r].push_back(sb);
                    gs = ge;
                }
            }
            if (rel_n > rel_cap)
                throw std::runtime_error("STRATA_SCRATCH_TILE: bounds table " + std::to_string(rel_n) + " > " +
                                         std::to_string(rel_cap) + " entries");
            // from pinned host, one copy: it is complete when the call returns, so the host may reuse the buffer
            // in the next moe_rows() call; no batch rewrites it meanwhile.
            ck(cudaMemcpy(d_rel_local_, h_rel_local_, rel_n * sizeof(int32_t), cudaMemcpyHostToDevice),
               "expert rows");
            for (const Batch& b : batches) {
                // the batch's rows quantized whole: row i is the token of compact row b.r0 + i
                Mmq::quantize(in, d_tok + b.r0, mq_tile_, gt, in_w, in_w, b.rows, st_);
                for (const Seg& s : b.segs) {
                    mmq_->mul(slot_of(s.j), gt, ff2, in_w, L.slot_bytes, s.m, mq_tile_, d_rel_local_ + s.rel,
                              iota_, b.rows, s.most, gu_, ff2, st_);
                }
                if (!L.gated) relu2(gu_, hx_, (size_t) b.rows * n_ff_exp_, st_);
                else if (gelu_) geglu_rows(gu_, hx_, n_ff_exp_, b.rows, st_);
                else swiglu_clamp_rows(gu_, hx_, n_ff_exp_, b.rows, limit, st_);
                Mmq::quantize(hx_, nullptr, mq2_, dt, ff, ff, b.rows, st_);
                for (const Seg& s : b.segs) {
                    // the down writes dx_ at the global compact row: d_place + b.r0 with the local ids
                    mmq_->mul(slot_of(s.j) + gu2, dt, out_w, ff, L.slot_bytes, s.m, mq2_, d_rel_local_ + s.rel,
                              d_place + b.r0, b.rows, s.most, dx_, out_w, st_);
                }
            }
        }

        // The round the buffer holds goes first and is not copied.  If it holds none of this call's, the experts
        // that go ahead are copied last, so the next call finds them.  The order does not change a round's launches.
        int kept = -1;
        for (int r = 0; r < n_round && mine && kept < 0; ++r) {
            const auto first = group_.begin() + round_[(size_t) r], last = group_.begin() + round_[(size_t) r + 1];
            if ((size_t) (last - first) == staged_.size() && std::equal(first, last, staged_.begin())) kept = r;
        }
        const int from = mine && kept < 0 ? 1 : 0;
        holds = (early || prefetched) ? 0 : kept;
        // STRATA_MOE_ROWS_CP_STREAM: a round's copy runs on cp_ and starts once the kernels that read the staging
        // buffer are done (ev_krn_); the kernels of that round wait for the copy (ev_copy_).  The buffer is still
        // one, so this overlaps a copy with the round before it and nothing more.  An event never recorded waits
        // for nothing, so the call before this one needs no special case.
        bool stage_dirty = cp_ != nullptr && !ring;
        uint8_t* const sbuf = ring ? rbase : stage_;   // STRATA_STAGE_RING2: this call's round buffer
        for (int turn = 0; turn < n_round; ++turn) {
            const int r = kept < 0 ? (turn + from) % n_round : turn == 0 ? kept : turn - (turn <= kept ? 1 : 0);
            const int g0 = round_[(size_t) r], m = round_[(size_t) r + 1] - g0;
            const size_t down_at = pad((size_t) m * gu2);
            const int r0 = bounds[g0], rows = bounds[g0 + m] - r0;
            bool copied = false;
            if (r == kept) {
                // the buffer has it
            } else if (r == 0 && (early || prefetched)) {
                if (ring) {
                    ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_ring_ahead_[rs], 0), "expert staging");
                    // STRATA_STAGE_RING2_CHECK=1 (verification only): the copy is complete here; compare the first
                    // experts of the slot with their pinned sources byte for byte.  It serializes the copy, so it is
                    // never used for a timing reading.
                    if (ring_check && pin_layer_[(size_t) il]) {
                        ck(cudaStreamSynchronize(cs(up_)), "ring check");
                        std::vector<uint8_t> got(std::max(L.d_bytes, L.gu_bytes));
                        const size_t down0 = pad((size_t) n_ahead * gu2);
                        const int nv = std::min(n_ahead, 2);
                        for (int j = 0; j < nv; ++j) {
                            const size_t e = (size_t) L.ahead[(size_t) j];
                            ck(cudaMemcpy(got.data(), rbase + (size_t) j * gu2 + L.slot_up(), L.gu_bytes,
                                          cudaMemcpyDeviceToHost), "ring check");
                            if (std::memcmp(got.data(), L.exp_up + e * L.gu_stride, L.gu_bytes) != 0)
                                throw std::runtime_error("STRATA_STAGE_RING2_CHECK: layer " + std::to_string(il) +
                                                         " slot " + std::to_string(rs) + " expert " +
                                                         std::to_string(e) + " (up) differs from its source");
                            ck(cudaMemcpy(got.data(), rbase + down0 + (size_t) j * L.d_bytes, L.d_bytes,
                                          cudaMemcpyDeviceToHost), "ring check");
                            if (std::memcmp(got.data(), L.exp_down + e * L.d_bytes, L.d_bytes) != 0)
                                throw std::runtime_error("STRATA_STAGE_RING2_CHECK: layer " + std::to_string(il) +
                                                         " slot " + std::to_string(rs) + " expert " +
                                                         std::to_string(e) + " (down) differs from its source");
                        }
                        std::fprintf(stderr, "strata-poly: RING2_CHECK: layer %d slot %d: %d experts byte-equal to "
                                             "their sources\n", il, rs, nv);
                    }
                } else {
                    ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_ahead_, 0), "expert staging");
                }
            } else if (rows > 0) {
                if (cp_ && !ring) {
                    if (stage_dirty) ck(cudaStreamWaitEvent(cs(cp_), (cudaEvent_t) ev_krn_, 0), "expert staging");
                    stage(L, group_.data() + g0, m, cp_, n_cpu > 0);
                    ck(cudaEventRecord((cudaEvent_t) ev_copy_, cs(cp_)), "expert staging");
                    copied = true;
                    stage_dirty = false;
                } else {
                    stage(L, group_.data() + g0, m, st_, n_cpu > 0, rbase);
                }
                holds = r;
            }
            if (rows == 0) continue;   // nothing routed to the experts that go ahead
            if (copied) ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_copy_, 0), "expert staging");
            if (!tiled || fallback) {
                int most = 0;
                for (int g = g0; g < g0 + m; ++g) most = std::max(most, rows_of(g));
                mmq_->mul(sbuf, gt, ff2, in_w, gu2, m, mq_, d_bounds + g0, iota_, rg, most, gu_, ff2, st_);
                if (!L.gated) relu2(gu_ + (size_t) r0 * ff2, hx_ + (size_t) r0 * ff, (size_t) rows * n_ff_exp_, st_);
                else if (gelu_) geglu_rows(gu_ + (size_t) r0 * ff2, hx_ + (size_t) r0 * ff, n_ff_exp_, rows, st_);
                else swiglu_clamp_rows(gu_ + (size_t) r0 * ff2, hx_ + (size_t) r0 * ff, n_ff_exp_, rows, limit, st_);
                Mmq::quantize(hx_ + (size_t) r0 * ff, nullptr, mq2_, dt, ff, ff, rows, st_);
                mmq_->mul(sbuf + down_at, dt, out_w, ff, L.d_bytes, m, mq2_, d_rel + (g0 - n_cached + r),
                          d_place + r0, rows, most, dx_, out_w, st_);
            } else {
                // the round's sub-batches, cut before the launches (one interval each) and in the call's table
                for (const Sub& s : round_subs[(size_t) r]) {
                    const int msub = s.ge - s.gs, sub = bounds[s.ge] - bounds[s.gs];
                    int most = 0;
                    for (int g = s.gs; g < s.ge; ++g) most = std::max(most, bounds[g + 1] - bounds[g]);
                    const size_t at = (size_t) (s.gs - g0);   // the sub-batch's first expert in the staged round
                    Mmq::quantize(in, d_tok + bounds[s.gs], mq_tile_, gt, in_w, in_w, sub, st_);
                    mmq_->mul(sbuf + at * gu2, gt, ff2, in_w, gu2, msub, mq_tile_, d_rel_local_ + s.rel, iota_,
                              sub, most, gu_, ff2, st_);
                    if (!L.gated) relu2(gu_, hx_, (size_t) sub * n_ff_exp_, st_);
                    else if (gelu_) geglu_rows(gu_, hx_, n_ff_exp_, sub, st_);
                    else swiglu_clamp_rows(gu_, hx_, n_ff_exp_, sub, limit, st_);
                    Mmq::quantize(hx_, nullptr, mq2_, dt, ff, ff, sub, st_);
                    mmq_->mul(sbuf + down_at + at * L.d_bytes, dt, out_w, ff, L.d_bytes, msub, mq2_,
                              d_rel_local_ + s.rel, d_place + bounds[s.gs], sub, most, dx_, out_w, st_);
                }
            }
            if (cp_ && !ring) {
                ck(cudaEventRecord((cudaEvent_t) ev_krn_, cs(st_)), "expert staging");
                stage_dirty = true;
            }
        }
        ck(cudaEventRecord((cudaEvent_t) ev_free_, cs(st_)), "expert staging");
        if (ring) ck(cudaEventRecord((cudaEvent_t) ev_ring_free_[rs], cs(st_)), "expert staging");
        if (fallback) {
            std::swap(gu_, fb_gu_);
            std::swap(hx_, fb_hx_);
            std::swap(mq2_, fb_mq2_);
        }
    };
    if (n_cpu == 0) {
        card();
    } else {   // the card's part on a thread of its own, and the CPU's on the workers
        if (!stagers_) stagers_ = std::make_unique<Workers>(4);   // one thread copied slower than the bus
        Clock::time_point done;
        Behind behind;
        const Clock::time_point start = Clock::now();
        behind.go([&] {
            cudaSetDevice(device_);   // a thread starts on card 0
            card();
            ck(cudaStreamSynchronize(cs(st_)), "expert staging");
            done = Clock::now();
        });
        Relu2Active r2a;   // Stage D: CPU-share experts run the R57 T kernel on the pinned sidecar (no GGUF down)
        cpu_experts(*workers_, host_.data(), (int) host_.size(), n, gt, dt, in_w, n_ff_exp_, h_x_, limit, gelu_,
                    h_cpu_, 0, nullptr, nullptr, nullptr, nullptr, nullptr, r2d_base_ ? &r2a : nullptr);
        const double cpu = ms_since(start);
        behind.wait();
        const double gpu = std::chrono::duration<double, std::milli>(done - start).count();
        stats_.cpu_ms += cpu;
        stats_.cpu_experts += (int64_t) host_.size();
        // the share that would have had both done at once, if each one's time grows with its share
        if (opt_.cpu_share < 0 && n_cpu < n_spare) {
            const double s = (double) n_cpu / n_spare, c = cpu / s, g = gpu / (1 - s);
            share_[size] = std::clamp(0.5 * share_[size] + 0.5 * g / (c + g), 0.05, 0.95);
        }
    }
    moe_sum(dx_, d_w, (latent_rows && L.latent_up.p) ? moe_out_ : fo_, out_w, n_used_, n, st_);
    if (n_cpu > 0) moe_add(L, n);
    // STRATA_STAGE_RING2 (the design's 鈶?: the next layer's ahead list goes into the other buffer now, so its copy
    // runs on up_ under this layer's kernels; the next call waits on that slot's event instead of copying it again.
    if (ring && ring_hold_il_ == il) {   // this layer's own list is in place; issue its successor's copy once
        int nxt = il + 1;
        while (nxt < hi_ && !layers_[(size_t) nxt].moe) ++nxt;
        if (nxt < hi_ && prefetched_il_ != nxt && batched(layers_[(size_t) nxt], n)) {
            Layer& N = layers_[(size_t) nxt];
            plan_ahead(N, nxt);
            const int na = (int) N.ahead.size();
            if (na > 0) {
                const int s2 = ring_slot_;
                if (s2 == rs)   // the whole design rests on this: the prefetch never writes the slot being read
                    throw std::runtime_error("STRATA_STAGE_RING2: a prefetch would overwrite the slot it is for");
                ck(cudaStreamWaitEvent(cs(up_), (cudaEvent_t) ev_ring_free_[s2], 0), "expert staging");
                stage(N, N.ahead.data(), na, up_, false, slot_[s2]);
                ck(cudaEventRecord((cudaEvent_t) ev_ring_ahead_[s2], cs(up_)), "expert staging");
                prefetched_il_ = nxt;
                prefetched_slot_ = s2;
                prefetched_n_ = na;
                ++stats_.ring_prefetch;
            }
        }
    }
    if (ahead && holds >= 0) {
        staged_.assign(group_.begin() + round_[(size_t) holds], group_.begin() + round_[(size_t) holds + 1]);
        staged_il_ = il;
    }
}

// The rows a recorded attention launch is sized for, for n tokens from pos(): the power of two from 1024 that
// holds their rows and the ones before them, or the context if that is less.  room_key() numbers the sizes.
int Engine::room(int n) const { return (int) std::min<int64_t>(int64_t(1024) << room_key(n), opt_.max_context); }

int Engine::room_key(int n) const {
    int key = 0;
    for (int64_t r = 1024; r < pos_ + n; r *= 2) ++key;
    return key;
}

// Records what `work` hands to the stream, as a graph that is launched with one call from then on.
template <class F>
void* Engine::record(F&& work) {
    cudaGraph_t g = nullptr;
    ck(cudaStreamBeginCapture(cs(st_), cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    recording_ = recorded_ = true;
    try {
        work();
    } catch (...) {
        recording_ = false;
        if (cudaStreamEndCapture(cs(st_), &g) == cudaSuccess && g) cudaGraphDestroy(g);
        cudaGetLastError();   // a capture the failure broke does not end cleanly: the work's exception says why
        throw;
    }
    recording_ = false;
    ck(cudaStreamEndCapture(cs(st_), &g), "cudaStreamEndCapture");
    cudaGraphExec_t exec = nullptr;
    const cudaError_t made = cudaGraphInstantiateWithFlags(&exec, g, 0);
    cudaGraphDestroy(g);
    ck(made, "cudaGraphInstantiate");
    graphs_.push_back(exec);
    return exec;
}

// n tokens from pos() through every layer.  The sequence is not longer for it: the caller says by how much.
void Engine::run(const int32_t* tokens, int n) {
    if (n < 1 || n > rows_) throw std::logic_error("eval takes 1 to max_rows() tokens");
    if (pos_ + n > opt_.max_context)
        throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");
    poll_async_refill();   // STRATA_BORROW_ASYNC_REFILL: the slots whose copies have landed become resident
    cover(pos_ + n);

    // h_r_ and h_cpu_ may still be the source of the previous call's copies
    ck(cudaStreamSynchronize(cs(st_)), "eval");
    if (n_swap_ > 0) follow(n < kStageRows ? n : 0);
    tok_ = tokens;
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row;
    const float scale = embed_scale();
    if (pass_) {   // a later card: the card before it left the rows
        std::memcpy(h_r_, pass_ + (size_t) (pos_ - pass_at_) * r_row, (size_t) n * r_row * sizeof(float));
    } else {
        for (int t = 0; t < n; ++t) {
            float* r = h_r_ + (size_t) t * r_row;
            if (!input(tokens[t], pos_ + t, r) && scale != 1.f)
                for (size_t i = 0; i < e_row; ++i) r[i] *= scale;
            for (int h = 1; h < streams(); ++h) std::memcpy(r + (size_t) h * e_row, r, e_row * sizeof(float));
        }
    }
    if (n < kStageRows) copy2(h_r_, r_, n * (int) r_row, nullptr, nullptr, 0, st_);
    else ck(cudaMemcpyAsync(r_, h_r_, (size_t) n * r_row * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");

    // A token alone, or a window of guessed ones, hands the card the same launches every time, some thirty a
    // layer up to the router, and they cost this thread more than the card.  They are recorded once, a graph a
    // layer and a number of tokens, and replayed with one call.  Only for the layers that can be (Layer::fixed).
    // What a launch needs of the position it reads at at_; an attention over the rows so far is recorded again
    // when the rows outgrow what it was sized for (head_key()).  A prompt's few tokens are not a window: they
    // leave no snapshots of the recurrent state, so their launches are others.
    bool replay = opt_.graphs && n <= opt_.window && (windowed_ || n == 1) && !opt_.dump;
    for (int t = 0; t < n && replay; ++t) replay = !image_at(pos_ + t);
    const auto replayed = [&](std::vector<void*>& made, size_t key, auto&& work) {
        if (made.size() <= key) made.resize(key + 1, nullptr);
        if (!made[key]) made[key] = record(work);
        ck(cudaGraphLaunch((cudaGraphExec_t) made[key], cs(st_)), "layer graph");
    };
    if (replay) {   // a pair a token: a window's tokens may go one by one, each from its own
        for (int t = 0; t < n; ++t) {
            h_at_[2 * t] = pos_ + t;
            h_at_[2 * t + 1] = turn(pos_ + t);
        }
        copy2(h_at_, at_, n * 2 * (int) (sizeof(long long) / 4), nullptr, nullptr, 0, st_);
    }
    // A layer's launches around its routed experts.  up(): to its router's copy, with the shared expert behind it.
    // tail(): what the layer does with the experts' sum.  `rec` numbers the recording of m tokens' launches: a
    // call's tokens together are [0, window) by their number, and a half of a window comes after those, by the
    // window's tokens and which half it is.  Which tokens leave their state is the window's, so its halves' too.
    const size_t wn = (size_t) opt_.window, recs = 3 * wn;
    const auto up = [&](int il, int m, size_t rec) {
        Layer& L = layers_[(size_t) il];
        const size_t head = (size_t) head_key(il, m) * recs + rec;
        if (replay && L.fixed) replayed(L.heads, head, [&] { layer_head(L, il, m); });
        else layer_head(L, il, m);
        if (!L.moe || batched(L, m)) return;
        ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
        // STRATA_SHARED_OVERLAP: its recorded graph would run the shared expert on st_ (beside nothing), so a call
        // that overlaps takes the direct call - the shared expert on sh_ - instead of the old recording.
        if (shared_overlap_ && m <= 8 && !all_cached()) moe_shared_timed(L, il, m);
        else if (replay) replayed(L.shared, rec, [&] { moe_shared_timed(L, il, m); });
        else moe_shared_timed(L, il, m);
    };
    const auto tail = [&](int il, int m, size_t rec) {
        Layer& L = layers_[(size_t) il];
        dbg("ffn", il, fo_, n_embd_);
        if (replay && L.fixed) replayed(L.tail, rec, [&] { layer_tail(L, il, m); });
        else layer_tail(L, il, m);
    };
    // STRATA_CUT_HANDOFF: a layer's routed-experts sum and its latent projection are the same launches for a token
    // at a time, so the pair is recorded once as a graph a kind - with the CPU experts' sum and without it - and
    // replayed with one call.  Only a token at a time is recorded: a window moves the buffers (shift_rows()), so its
    // launches stay the ones they were.  The cache lives here, keyed by this engine and the layer: a graph recorded
    // for one engine's buffers is never replayed by another's.
    static const bool cut_handoff = [] {
        const char* e = std::getenv("STRATA_CUT_HANDOFF");
        return e && e[0] == '1' && e[1] == '\0';   // off by default: the delivery path is the old launches
    }();
    static std::map<std::pair<const void*, std::pair<int, int>>, std::vector<void*>> after_graphs;
    const auto replay_after = [&](int il, bool added, const std::function<void()>& work) {
        std::vector<void*>& made =
            after_graphs[std::make_pair((const void*) this, std::make_pair(il, added ? 1 : 0))];
        if (made.empty()) made.resize(1, nullptr);
        if (!made[0]) made[0] = record(work);
        ck(cudaGraphLaunch((cudaGraphExec_t) made[0], cs(st_)), "layer graph");
    };
    // A window's tokens, with experts on the CPU, go through the layers together or in two halves, one behind
    // the other (halves_, which window() sets).  Together, the card's part of a layer and the CPU's take turns:
    // the card waits out the CPU's experts and the CPU the card's layer.  In halves each has work while the
    // other is busy: this thread does nothing but route a half and compute its experts, half after half, and
    // while it does, one of the experts' threads hands the card the other half: its experts' sum, the rest of
    // its layer and the next layer up to the router.  That half's scores are there when its turn comes again.
    //
    // The card's part of a layer takes about as long for two tokens as for one: it reads the layer's dense
    // weights once for all of them, and its launches are the same.  So a window is two halves and not a part a
    // token, which had the card read every layer once for each token.  Halves have the card read it twice where
    // the tokens together have it read once, and that is what the choice between the two turns on.
    //
    // A half's launches are the ones its tokens have in a call of their own, at their rows of the buffers
    // (shift_rows()), so each token comes out as it would alone.
    if (n > 1 && windowed_ && !all_cached() && halves_) {
        const int first[3] = {0, (n + 1) / 2, n};   // half k is the tokens [first[k], first[k + 1])
        const unsigned bits = snap_bits_;           // the window's tokens that leave the recurrent state
        int at = 0;
        // The buffers at half k's rows, and its share of the window's states.  Returns the half's tokens.
        const auto half = [&](int k) {
            shift_rows(first[k] - at);
            at = first[k];
            const int m = first[k + 1] - at;
            snap_bits_ = bits >> at & ((1u << m) - 1u);
            snap_at_ = snap_ + size_t(std::popcount(bits & ((1u << at) - 1u))) * rec_n_;
            ev_ = ev_row_[k];
            return m;
        };
        // Half k on from layer il, whose experts it has (-1: from its start), to the next router's copy.
        const auto go_on = [&](int k, int il, bool add) {
            const int m = half(k);
            const size_t rec = wn + 2 * (size_t) (n - 1) + (size_t) k;
            if (il >= 0) {
                if (add) {
                    Layer& ell = layers_[(size_t) il];
                    moe_add(ell, m);
                    moe_latent(ell, m);
                }
                tail(il, m, rec);
            }
            for (int j = il < 0 ? lo_ : il + 1; j < hi_; ++j) {
                up(j, m, rec);
                if (layers_[(size_t) j].moe) break;
            }
        };
        const auto back = [&] {
            shift_rows(-at);
            at = 0;
            ev_ = ev_row_[0];
            snap_bits_ = bits;
            snap_at_ = snap_;
        };
        struct {
            int k = -1, il = -1;
            bool add = false;
        } late;   // the half whose experts are computed and that has not gone on
        std::exception_ptr failed;
        try {
            for (int k = 0; k < 2; ++k) go_on(k, -1, false);
            for (int il = lo_; il < hi_; ++il) {
                Layer& L = layers_[(size_t) il];
                if (!L.moe) continue;
                for (int k = 0; k < 2; ++k) {
                    // by another thread while this one computes: it must leave the rows as it found them
                    const std::function<void()> other = [&] {
                        try {
                            if (split_) cudaSetDevice(device_);
                            go_on(late.k, late.il, late.add);
                            half(k);
                        } catch (...) {
                            failed = std::current_exception();
                        }
                    };
                    const int m = half(k);
                    const bool add = moe(L, il, m, late.k >= 0 ? &other : nullptr);
                    if (failed) std::rethrow_exception(failed);
                    late = {k, il, add};
                }
            }
            if (late.k >= 0) go_on(late.k, late.il, late.add);
        } catch (...) {
            back();
            throw;
        }
        back();
        return;
    }
    // The routed experts on the card alone, when every one of them is in VRAM: recorded too, once a layer has
    // shown it routes there (Layer::card).
    const bool card = n < kStageRows && card_moe(n);
    const auto on_card = [&](int il, int m, size_t rec) {
        Layer& L = layers_[(size_t) il];
        if (!card) return false;
        if (replay && L.card == 1) replayed(L.card_moe, rec, [&] { moe_card(L, il, m); });
        else if (!moe_card(L, il, m)) return false;
        stats_.gpu_experts += m * n_used_;
        return true;
    };
    // Once every layer has shown that, a call's layers are one recording: on an RTX 5090 the card waited 2 to 4
    // microseconds from one layer's recording to the next, four of them a layer.
    bool whole = card && replay;
    for (int il = lo_; il < hi_ && whole; ++il) {
        const Layer& L = layers_[(size_t) il];
        whole = L.fixed && (!L.moe || L.card == 1);
    }
    if (whole) {
        std::vector<int> key{n};
        for (int il = lo_; il < hi_; ++il) key.push_back(head_key(il, n));
        void*& pass = passes_[key];
        if (!pass)
            pass = record([&] {
                for (int il = lo_; il < hi_; ++il) {
                    Layer& L = layers_[(size_t) il];
                    layer_head(L, il, n);
                    if (!L.moe) continue;
                    moe_shared_timed(L, il, n);
                    moe_card(L, il, n);
                    layer_tail(L, il, n);
                }
            });
        ck(cudaGraphLaunch((cudaGraphExec_t) pass, cs(st_)), "layer graph");
        for (int il = lo_; il < hi_; ++il) stats_.gpu_experts += layers_[(size_t) il].moe ? n * n_used_ : 0;
        return;
    }
    for (int il = lo_; il < hi_; ++il) {
        Layer& L = layers_[(size_t) il];
        up(il, n, (size_t) n - 1);
        if (!L.moe) continue;
        bool added = false;
        if (batched(L, n)) {
            moe_rows(L, il, n);
        } else {
            added = !on_card(il, n, (size_t) n - 1) && moe(L, il, n);
        }
        if (cut_handoff && replay && n == 1 && L.fixed) {
            replay_after(il, added, [&] {
                if (added) moe_add(L, n);
                moe_latent(L, n);
            });
        } else {
            if (added) moe_add(L, n);
            moe_latent(L, n);
        }
        tail(il, n, (size_t) n - 1);
    }
}

void Engine::shift_rows(ptrdiff_t by) {
    const ptrdiff_t e = n_embd_;
    r_ += by * streams() * e;
    xn_ += by * e;
    fo_ += by * e;
    rl_ += by * n_expert_;
    xq_ = (uint8_t*) xq_ + by * (ptrdiff_t) kernels::native_q8_1_bytes(n_embd_);
    h_rl_ += by * n_expert_;
    h_x_ += by * e;
    h_cpu_ += by * e;
    at_ += by * 2;
    pos_ += by;
    if (tok_) tok_ += by;
}

// Pinned memory for n rows of the residual on their way to the next card.
float* Engine::pass_rows(int n) {
    if (h_pass_rows_ < n) {
        if (h_pass_) {
            host_allocs_.erase(std::find(host_allocs_.begin(), host_allocs_.end(), (void*) h_pass_));
            ck(cudaFreeHost(h_pass_), "freeing pinned memory");
            h_pass_ = nullptr;
        }
        h_pass_rows_ = std::max(n, rows_);
        h_pass_ = (float*) hmalloc((size_t) h_pass_rows_ * streams() * n_embd_ * sizeof(float), "pinned memory");
    }
    return h_pass_;
}

// The n rows of the residual a call from `pos` leaves in r_, to the next card, which reads them for its layers.
void Engine::hand_on(int64_t pos, int n, float* h) {
    ck(cudaMemcpyAsync(h, r_, (size_t) n * streams() * n_embd_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "the next card");
    ck(cudaStreamSynchronize(cs(st_)), "the next card");
    next_->pass_ = h;
    next_->pass_at_ = pos;
}

void Engine::eval(const int32_t* tokens, int n, bool want_logits) {
    OnCard here(device_, split_);
    run(tokens, n);
    if (next_) {
        hand_on(pos_, n, pass_rows(n));
        next_->eval(tokens, n, want_logits);
        pos_ += n;
        stats_.tokens += n;
        return;
    }
    // The model's output: the last token's for the logits, and every token's for the draft block.
    if (want_logits || drafting()) {
        const bool replay = opt_.graphs && n == 1 && want_logits && !opt_.dump;
        const int m = drafting() ? n : 1;
        float* h = drafting() ? hn_ + n_embd_ : xn_;
        const auto tail = [&] {
            output_rows(n - m, m, h);
            dbg("out_norm", n_layer_, h, n_embd_);
            if (!want_logits) return;
            lin(output_, h + (size_t) (m - 1) * n_embd_, logits_, 1);
            finish_logits(logits_, 1);
            ck(cudaMemcpyAsync(h_logits_, logits_, (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost,
                               cs(st_)),
               "logits");
        };
        if (replay) {
            if (!tail_) tail_ = record(tail);
            ck(cudaGraphLaunch((cudaGraphExec_t) tail_, cs(st_)), "logits graph");
        } else {
            tail();
        }
        if (want_logits) {
            const Clock::time_point c0 = Clock::now();
            ck(cudaStreamSynchronize(cs(st_)), "eval");
            stats_.commit_ms += ms_since(c0);   // R51: the wait for the card, measured where it happens
        }
    }
    // behind the logits: the caller has them while the card does this
    if (drafting()) draft_rows(tokens, ahead_pos_ == pos_ && tokens[0] == ahead_tok_ ? 1 : 0, n);
    pos_ += n;
    stats_.tokens += n;
}

// A layer for n tokens, as run() hands it to the card when nothing is replayed or printed.
void Engine::layer(int il, int n) {
    Layer& L = layers_[(size_t) il];
    layer_head(L, il, n);
    if (!L.moe) return;
    if (batched(L, n)) {
        moe_rows(L, il, n);
    } else {
        ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
        moe_shared_timed(L, il, n);
        if (moe(L, il, n)) moe_add(L, n);
    }
    moe_latent(L, n);
    layer_tail(L, il, n);
}

// A prompt's tokens in the calls eval() would get, max_rows() at a time.  With experts outside the cache, each
// call copies every layer's through the staging buffer, which is most of what reading a prompt takes.  So the
// calls go through a layer one after the other before any goes on to the next, and the second one finds the
// layer's experts where the first one's copy left them (moe_rows()).  A call's tokens meet what they met in a
// call of their own, in the same order, so the result is eval()'s.  But a call under stage_ahead tokens takes
// those experts too, which on its own it would not, and its sums are split as a larger call's.  A read of fewer
// than stage_ahead tokens is the calls themselves: the copy of every layer's experts took a second, where its
// calls copy the experts they route to.
//
// All a call needs kept from one layer to the next is its rows of the residual.  r_ holds two calls' rows.  With
// more calls, a call's rows go to host memory when the card is done with them and come back before the call's
// turn at the next layer, in the half of r_ the call two before it has left: both copies are on a stream of
// their own, beside the card's work on the call between.
//
// A checkpoint inside the read (a mark) is taken a layer at a time: when a layer has run the tokens before it,
// the layer's state is what save_state() would find there, and it is copied before the next call changes it.
bool Engine::read(const int32_t* tokens, int n, const Mark* marks, int n_marks,
                  const std::function<bool(int, int)>* turn) {
    if (n < 1 || n > read_rows_) throw std::logic_error("read takes 1 to max_read() tokens");
    for (const Image& im : images_)
        if (im.at < pos_ + n && im.at + im.n > pos_)
            throw std::logic_error("read takes no rows of a picture: eval() does, a picture a call");
    for (int k = 0; k < n_marks; ++k)
        if (marks[k].at <= (k ? marks[k - 1].at : 0) || marks[k].at >= n)
            throw std::logic_error("read: the marks rise, above 0 and below the tokens read");

    // the calls: up to max_rows() tokens each, and none across a mark
    calls_.clear();
    for (int at = 0, k = 0; at < n;) {
        const int m = std::min(rows_, (k < n_marks ? marks[k].at : n) - at);
        const bool ends = k < n_marks && at + m == marks[k].at;
        calls_.push_back({at, m, ends ? k : -1});
        at += m;
        k += ends;
    }
    const int n_calls = (int) calls_.size();
    OnCard here(device_, split_);
    cover(pos_ + n);
    // On several cards, each reads through its own layers and hands the rows on, and `turn` hears of a card's
    // turns as its share of the model's layers: [lo_, to).
    const std::function<bool(int, int)>* asked = turn;
    std::function<bool(int, int)> share;
    const auto share_to = [&](int to) {
        if (!asked || (lo_ == 0 && to == n_layer_)) return asked;   // all of them
        share = [&, to](int done, int of) { return (*asked)(lo_ * of + done * (to - lo_), n_layer_ * of); };
        return (const std::function<bool(int, int)>*) &share;
    };
    if (n_calls == 1 || !mmq_ || all_cached() || n < opt_.stage_ahead) {
        // T8/P17: this branch leaves through eval() and never reaches the clear-outs of the several-call branch
        // below.  A ring hold or a prefetch left by the request before would survive into the next read, and
        // resize_cache() (main.cpp:721-735) may have changed L.slot[] since: the slot would point at a list that
        // no longer exists.  Cleared on the way in, so a return or a throw out of eval() leaves nothing to inherit.
        if (ring2_) {
            shared_ahead_ = false;
            prefetched_il_ = -1;
            ring_hold_il_ = -1;
        }
        turn = share_to(n_layer_);   // eval() takes the calls through the cards after this one
        // On several cards the next one takes a call on a thread of its own, while this one runs the call after
        // it.  Each card runs the calls as eval() does, in the same order.
        Behind behind;
        for (int c = 0; c < n_calls; ++c) {
            const Call& call = calls_[(size_t) c];
            if (next_) {
                run(tokens + call.at, call.m);
                behind.wait();
                hand_on(pos_, call.m, pass_rows(call.m));
                behind.go([this, t = tokens + call.at, m = call.m] { next_->eval(t, m, false); });
                pos_ += call.m;
                stats_.tokens += call.m;
            } else {
                eval(tokens + call.at, call.m, false);
            }
            if (call.mark >= 0 && marks[call.mark].state) {
                behind.wait();
                save_state(marks[call.mark].state);
            }
            if (turn && c + 1 < n_calls && !(*turn)(c + 1, n_calls)) {
                behind.wait();
                return false;
            }
        }
        behind.wait();
        return true;
    }
    turn = share_to(hi_);
    if (pos_ + n > opt_.max_context)
        throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");

    ck(cudaStreamSynchronize(cs(st_)), "read");
    if (n_swap_ > 0) follow(0);
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row, r_bytes = r_row * sizeof(float);
    const size_t part =
        std::max<size_t>((size_t) streams(), (size_t) ((draft_ && !draft_chunked()) ? 3 : 1)) * rows_ * e_row;   // r_'s half
    const float scale = embed_scale();
    float* const r0 = r_;
    const int64_t pos0 = pos_;
    const bool host = n_calls > 2;
    if (host && h_res_rows_ < n) {   // for this read's tokens and some more, so that a longer one seldom asks again
        if (h_res_) {
            host_allocs_.erase(std::find(host_allocs_.begin(), host_allocs_.end(), (void*) h_res_));
            ck(cudaFreeHost(h_res_), "freeing pinned memory");
            h_res_ = nullptr;
        }
        h_res_rows_ = std::min(read_rows_, (n + 4095) / 4096 * 4096);
        h_res_ = (float*) hmalloc((size_t) h_res_rows_ * r_bytes, "pinned memory");
    }
    const auto home = [&](int c) { return h_res_ + (size_t) calls_[(size_t) c].at * r_row; };
    const auto half = [&](int c) { return r0 + (size_t) (host ? c % 2 : c) * part; };
    // call c's rows back into its half of r_, behind whatever the stream has there of the half's last call
    const auto fetch = [&](int c) {
        ck(cudaMemcpyAsync(half(c), home(c), (size_t) calls_[(size_t) c].m * r_bytes, cudaMemcpyHostToDevice,
                           cs(rs_)),
           "read");
        ck(cudaEventRecord((cudaEvent_t) ev_in_[c % 2], cs(rs_)), "read");
    };
    for (int c = 0; c < n_calls; ++c) {
        const Call& call = calls_[(size_t) c];
        float* rows = host ? home(c) : h_r_;
        if (pass_)
            std::memcpy(rows, pass_ + (size_t) (pos0 + call.at - pass_at_) * r_row, (size_t) call.m * r_bytes);
        for (int t = 0; t < call.m && !pass_; ++t) {
            float* r = rows + (size_t) t * r_row;
            if (!input(tokens[call.at + t], pos0 + call.at + t, r) && scale != 1.f)
                for (size_t i = 0; i < e_row; ++i) r[i] *= scale;
            for (int h = 1; h < streams(); ++h) std::memcpy(r + (size_t) h * e_row, r, e_row * sizeof(float));
        }
        if (host) continue;
        ck(cudaMemcpyAsync(half(c), h_r_, (size_t) call.m * r_bytes, cudaMemcpyHostToDevice, cs(st_)), "embed");
        ck(cudaStreamSynchronize(cs(st_)), "embed");   // h_r_ is filled again
    }
    if (host) {
        fetch(0);
        fetch(1);
    }
    // a turn for each layer, and one for the model's output, which the draft block keeps a row of
    const int turns = hi_ - lo_ + (drafting() ? 1 : 0);
    shared_ahead_ = true;
    // T8/P6: a prefetch left by the request before is not this read's - the cache may have been resized since
    // (main.cpp:721-735 -> resize_cache), so the slot would point at a list that no longer exists.
    prefetched_il_ = -1;
    ring_hold_il_ = -1;
    try {
        for (int k = 0; k < turns; ++k) {
            const int il = lo_ + k;
            // Between two turns the caller may give the read up.  The layers so far have the read's tokens in
            // their state and the others have not: the sequence cannot go on from here, nor from where it was.
            if (turn && k > 0 && !(*turn)(k, turns)) {
                if (host) ck(cudaStreamSynchronize(cs(rs_)), "read");
                ck(cudaStreamSynchronize(cs(st_)), "read");
                shared_ahead_ = false;
                prefetched_il_ = -1;
                ring_hold_il_ = -1;
                r_ = r0;
                pos_ = pos0;
                return false;
            }
            for (int c = 0; c < n_calls; ++c) {
                const Call& call = calls_[(size_t) c];
                r_ = half(c);
                pos_ = pos0 + call.at;
                tok_ = tokens + call.at;
                if (host) ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_in_[c % 2], 0), "read");
                if (il < n_layer_) {
                    layer(il, call.m);
                } else {
                    output_rows(0, call.m, hn_ + n_embd_);
                    draft_rows(tokens + call.at, ahead_pos_ == pos_ && tokens[call.at] == ahead_tok_ ? 1 : 0, call.m);
                    ++read_draft_calls_;
                }
                if (uint8_t* state = call.mark >= 0 ? marks[call.mark].state : nullptr) {
                    ck(cudaStreamSynchronize(cs(st_)), "read");
                    const int64_t pos = pos_ + call.m;
                    if (il < n_layer_) save_layer(il, pos, state);
                    if (k == turns - 1) {   // what no turn is for: a layer behind the model's, the last output
                        for (int j = n_layer_; j < (int) layers_.size(); ++j) save_layer(j, pos, state);
                        save_last(state);
                    }
                }
                // The card's work on the next call starts now.  This call's rows leave beside it, unless this was
                // the last turn, and the half is the next call's that uses it: the one two on, or the one with
                // this half at the next turn.  After the last turn the next card reads them.
                const bool last = k == turns - 1, out = last && next_;
                if (!host || (last && c + 2 >= n_calls && !out)) continue;
                ck(cudaEventRecord((cudaEvent_t) ev_out_[c % 2], cs(st_)), "read");
                ck(cudaStreamWaitEvent(cs(rs_), (cudaEvent_t) ev_out_[c % 2], 0), "read");
                if (!last || out)
                    ck(cudaMemcpyAsync(home(c), half(c), (size_t) call.m * r_bytes, cudaMemcpyDeviceToHost,
                                       cs(rs_)),
                       "read");
                if (!last || c + 2 < n_calls) fetch(c + 2 < n_calls ? c + 2 : c % 2);
            }
        }
        if (host) ck(cudaStreamSynchronize(cs(rs_)), "read");
        if (next_ && !host) {   // the rows are in r_, a call's in each half
            float* h = pass_rows(n);
            for (int c = 0; c < n_calls; ++c)
                ck(cudaMemcpyAsync(h + (size_t) calls_[(size_t) c].at * r_row, half(c), calls_[(size_t) c].m * r_bytes,
                                   cudaMemcpyDeviceToHost, cs(st_)),
                   "the next card");
            ck(cudaStreamSynchronize(cs(st_)), "the next card");
        }
    } catch (...) {
        if (host) cudaStreamSynchronize(cs(rs_));
        shared_ahead_ = false;
        prefetched_il_ = -1;
        ring_hold_il_ = -1;
        r_ = r0;
        pos_ = pos0;
        throw;
    }
    shared_ahead_ = false;
    // T8/P6: a prefetch left by the request before is not this read's - the cache may have been resized since
    // (main.cpp:721-735 -> resize_cache), so the slot would point at a list that no longer exists.
    prefetched_il_ = -1;
    ring_hold_il_ = -1;
    // T8/P7: printed only with the switch on, so that a run without it has the stderr it always had.
    if (ring2_)
        std::fprintf(stderr, "strata-poly: T8 ring: %lld prefetches, %lld hold reuses, staging %lld calls / %.2f GB\n",
                     stats_.ring_prefetch, stats_.ring_reuse, stats_.stage_calls,
                     (double) stats_.stage_bytes / (1024.0 * 1024.0 * 1024.0));
    r_ = r0;
    pos_ = pos0;
    if (next_) {
        next_->pass_ = host ? h_res_ : h_pass_;
        next_->pass_at_ = pos0;
        std::vector<Mark> on(marks, marks + n_marks);   // the next card's part of each state is after this one's
        for (Mark& m : on)
            if (m.state) m.state += card_state_bytes();
        if (!next_->read(tokens, n, on.data(), n_marks, asked)) return false;
    }
    pos_ = pos0 + n;
    stats_.tokens += n;
    return true;
}

// The tokens of a window of t that leave the recurrent state as it is after them: each but the last when there
// is a place for each, else as many as there are places, the first token among them and the others spread over
// the window.
unsigned Engine::snap_tokens(int t) const {
    const int most = t - 1;
    if (most <= snap_places_) return (1u << most) - 1u;
    unsigned bits = 0;
    for (int i = 0; i < snap_places_; ++i) bits |= 1u << (i * most / snap_places_);
    return bits;
}

// The recurrent layers are the only ones a wrong guess would spoil: every other layer's row of a position is
// written again when the position gets its real token.  So they leave the state as it was after tokens of the
// window (snap_, snap_tokens()), and keep() puts back the one after the last token that was right.
void Engine::window(const int32_t* tokens, int t) {
    if (t < 1 || t > opt_.window) throw std::logic_error("a window takes 1 to max_window() tokens");
    OnCard here(device_, split_);
    // the two ways are a choice only with experts on the CPU; a call that records launches says nothing of them
    const bool timed = t > 1 && !all_cached();
    const bool recorded = std::exchange(recorded_, false);
    if (timed) halves_ = pick_way(t);
    const Clock::time_point t0 = Clock::now();
    windowed_ = true;
    snap_bits_ = snap_tokens(t);
    snap_at_ = snap_;
    snap_ok_ = false;   // a recurrent layer sets it again while this window's tokens go through (keep() checks it)
    try {
        run(tokens, t);
    } catch (...) {
        windowed_ = false;
        snap_bits_ = 0;
        recorded_ = recorded_ || recorded;
        throw;
    }
    windowed_ = false;
    snap_bits_ = 0;
    std::copy(tokens, tokens + t, win_tok_);
    if (next_) {   // a card's way is timed by its own layers
        hand_on(pos_, t, pass_rows(t));
        if (timed && !recorded_) took_way(t, ms_since(t0));
        recorded_ = recorded_ || recorded;
        win_ = t;
        next_->window(tokens, t);
        return;
    }
    float* h = drafting() ? hn_ + n_embd_ : xn_;
    output_rows(0, t, h);
    lin(output_, h, wlogits_, t);
    finish_logits(wlogits_, t);
    ck(cudaMemcpyAsync(h_wlogits_, wlogits_, (size_t) t * n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "logits");
    {
        const Clock::time_point c0 = Clock::now();
        ck(cudaStreamSynchronize(cs(st_)), "window");
        stats_.commit_ms += ms_since(c0);   // R51: the answer's wait for the window's logits, measured
    }
    if (timed && !recorded_) took_way(t, ms_since(t0));
    recorded_ = recorded_ || recorded;
    win_ = t;
}

// How a window of t tokens goes through the layers: in halves, or its tokens together.  Each way is timed a few
// times, then the one that took less is taken.  The other still gets a window now and then: what each takes
// moves with the experts the cache holds and with what else the machine is doing.
bool Engine::pick_way(int t) {
    const Way& whole = way_[0][t];
    const Way& split = way_[1][t];
    if (split.n < kWayFirst || whole.n < kWayFirst) return split.n < whole.n;   // in turn, the whole way first
    const bool halves = split.ms < whole.ms;
    if (++way_calls_[t] < std::min(kWayAgain << way_backoff_[t], kWayAgainMost)) return halves;
    way_calls_[t] = 0;
    if ((kWayAgain << way_backoff_[t]) < kWayAgainMost) ++way_backoff_[t];
    return !halves;
}

void Engine::took_way(int t, double ms) {
    Way& w = way_[halves_ ? 1 : 0][t];
    const Way& other = way_[halves_ ? 0 : 1][t];
    const bool behind = other.n >= kWayFirst && w.ms > other.ms;
    w.ms = w.n < kWayFirst ? (w.ms * w.n + ms) / (w.n + 1) : w.ms + (behind ? kWayBehind : kWayWeight) * (ms - w.ms);
    ++w.n;
    if (behind && w.ms < other.ms) way_calls_[t] = way_backoff_[t] = 0;   // the other way is behind now
}

void Engine::keep(int n) {
    if (n < 1 || n > win_) throw std::logic_error("keep takes 1 to the window's tokens");
    OnCard here(device_, split_);
    // The next card reads the window's rows again for the tokens it puts through its layers again: a right token's
    // row is the same, whatever came after it.
    if (n < win_) {
        // The state after the last right token, when that token left it.  Else the one of the last token before
        // it that did, and the right tokens after that one go through the layers again: they come out as they
        // did, and the state with them.
        const unsigned left = snap_tokens(win_) & ((1u << n) - 1u);   // the right tokens that left their state
        const int from = (int) std::bit_width(left);                  // the tokens the state at hand is after
        // Hard-fail rather than copy: a snapshot that no recurrent layer wrote is not there to return to (an
        // unrwritten snap_ over 40 Mamba layers would corrupt the state silently).  See t5, 澶歵oken绔埌绔?20261016.md.
        // left == 0 would index snap_ with SIZE_MAX (popcount(0) - 1): a wild pointer, not a state to return to.
        // snap_ok_ is NOT checked here: on a recorded-graph replay the host mamba() does not run (engine.cpp:2486),
        // so it stays false even though the replayed kernels did write snap_.  See t5-snap淇-20261016.md.
        if (left == 0) throw std::logic_error("keep: no token of the window left its state to return to");
        ck(cudaMemcpyAsync(state_, snap_ + size_t(std::popcount(left) - 1) * rec_n_, rec_n_ * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(st_)),
           "keep");
        if (from < n) {
            pos_ += from;
            try {
                run(win_tok_ + from, n - from);
            } catch (...) {
                pos_ -= from;
                throw;
            }
            pos_ -= from;
        }
    }
    if (drafting()) draft_rows(win_tok_, ahead_pos_ == pos_ && win_tok_[0] == ahead_tok_ ? 1 : 0, n);
    if (next_) next_->keep(n);
    pos_ += n;
    stats_.tokens += n;
    win_ = 0;
}

// The draft block's input for n tokens: each token's embedding and the model's output for the token before it
// (h, a row a token), each normed, the two side by side through the block's projection.  Returns the block's
// residual, and leaves its normed attention input in xn_ and xq_.  r_ is free between the model's calls, and the
// block works in it: n rows of the two halves, then the residual.
//
// h_r_ is written once the card has read the last copy out of it (ev_hr_): read()'s draft turn calls this for one
// call after another without waiting for the card in between.
float* Engine::draft_input(const int32_t* tokens, int64_t row, const float* h, int n) {
    const size_t e = (size_t) n_embd_, rows = (size_t) n;
    // STRATA_DRAFT_CHUNKED: the three row sets live in draft_work_ (a chunk of tokens at most), not in r_ - r_ keeps
    // the size a run without the draft block has.  Otherwise they are r_'s first three row sets, as they always were.
    float* const work = (draft_ && draft_chunked()) ? draft_work_ : r_;
    // read()'s draft turn calls this for one call after another without waiting for the card: the copy of the call
    // before may still be reading h_r_, so the host waits for it before writing the next call's rows
    if (!ev_hr_) {
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        ev_hr_ = ev;
    } else {
        ck(cudaEventSynchronize((cudaEvent_t) ev_hr_), "embed");
    }
    for (int t = 0; t < n; ++t) input(tokens[t], row + t, h_r_ + (size_t) t * e);
    if (n < kStageRows) copy2(h_r_, cur_, n * n_embd_, nullptr, nullptr, 0, st_);
    else ck(cudaMemcpyAsync(cur_, h_r_, rows * e * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");
    ck(cudaEventRecord((cudaEvent_t) ev_hr_, cs(st_)), "embed");
    rms_norm(cur_, fp(d_enorm_), work, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    rms_norm(h, fp(d_hnorm_), work + e, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    float* x = work + 2 * rows * e;
    lin(d_proj_, work, x, n);
    rms_norm(x, fp(layers_[(size_t) n_layer_].attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    return x;
}

// The rows the draft block's attention keeps for tokens [first, n) of a call that has run: token i is at pos() + i,
// and hn_'s row i is the model's output for the token before it.  A row is all a later guess reads of a token, so
// the rest of the block is not computed for it.  Then the call's last output is the one the next token pairs with.
void Engine::draft_rows(const int32_t* tokens, int first, int n) {
    const size_t e = (size_t) n_embd_;
    if (block_layer_ && first < n) {
        if (draft_ && draft_chunked()) {
            // STRATA_DRAFT_CHUNKED: a chunk of tokens at a time, in draft_work_.  Every chunk but the last has
            // kWayDraftChunk tokens and the last takes the remainder; when the remainder is under kStageRows it
            // joins the chunk before, so every chunk takes the same batched path a whole call takes, and a token's
            // rows depend on that token alone (block_rows writes only that token's k/v row).  The priming is the
            // same rows a whole call primes, in the same order.
            for (int off = first; off < n;) {
                int m = std::min(kWayDraftChunk, n - off);
                if (n - off - m > 0 && n - off - m < kStageRows) m = n - off;
                draft_input(tokens + off, pos_ + off, hn_ + (size_t) off * e, m);
                block_rows(pos_ + off, m);
                off += m;
            }
        } else {
            draft_input(tokens + first, pos_ + first, hn_ + (size_t) first * e, n - first);
            block_rows(pos_ + first, n - first);
        }
    }
    copy2(hn_ + (size_t) n * e, hn_, n_embd_, nullptr, nullptr, 0, st_);
    ahead_pos_ = -1;
}

// One token through the draft block, at pos() + i.  The first token's row is then what draft_rows() would leave
// for it, and eval() and keep() skip it; the rows of the guesses after it are written again when the positions get
// their real tokens.
void Engine::draft_step(int32_t token, int i) {
    Layer& L = layers_[(size_t) n_layer_];
    float* own = hn_ + n_embd_;   // the block's output, which stands in for the model's from the second guess on
    float* x = draft_input(&token, pos_ + i, i == 0 ? hn_ : own, 1);
    block_attend(pos_ + i, 1);
    axpy(ao_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(L.ffn_norm), xn_, n_embd_, 1, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_, 1, st_);
    moe_router(L, 1);
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
    moe_shared(L, n_layer_, 1);
    tok_ = nullptr;   // a guess's row has no token of the sequence
    if (moe(L, n_layer_, 1)) moe_add(L, 1);
    moe_latent(L, 1);
    axpy(fo_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(d_head_), own, n_embd_, 1, rms_eps_, st_);
    Dev head = output_;   // the matrix's first rows are the common tokens'
    head.n_out = draft_rows();
    lin(head, own, logits_, 1);
    finish_logits(logits_, 1);
}

int32_t Engine::draft(int32_t token, int i, float* prob) {
    if (next_) return next_->draft(token, i, prob);   // the last card has the draft block
    OnCard here(device_, split_);
    if (!draft_) throw std::logic_error("draft: this model has no block that guesses ahead");
    if (draft_load_only_)   // STRATA_DRAFT_LOAD_ONLY=1: the block's VRAM is there, its guesses are not
        throw std::logic_error("draft: STRATA_DRAFT_LOAD_ONLY=1 loads the block but does not draft from it");
    if (token < 0 || token >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    if (i < 0 || pos_ + i >= opt_.max_context) throw std::logic_error("draft: the position is outside the context");
    cover(pos_ + i + 1);
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    draft_step(token, i);
    // The guess and its probability are found on the card.  On the host a vocabulary's softmax takes up to 2 ms
    // where the block is unsure, four times what the block takes.
    likeliest(logits_, draft_rows(), h_guess_, st_);
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    if (i == 0) {
        ahead_pos_ = pos_;
        ahead_tok_ = token;
    }
    *prob = h_guess_->prob;
    return h_guess_->token;
}

int most_likely(const float* logits, int n) {
    const float* at = std::find(logits, logits + n, largest(logits, n));
    return at == logits + n ? 0 : (int) (at - logits);
}

}  // namespace strata::poly
