// src/poly/engine.hpp - the engine that runs a model from its GGUF.
//
// The engine reads the GGUF in place, in one file or in the numbered files of a split one.  Everything but the
// routed experts goes to VRAM; as many routed experts as fit go there too, and the rest are computed on the CPU
// from the mapped file while the GPU does its share.
//
// Engine is what every architecture shares.  An architecture is a class derived from it (glm5next.hpp, qwen35.hpp,
// gemma4.hpp) that reads the model's shape, loads a layer's weights and composes a layer from the kernels;
// Engine::open() picks it by the file's general.architecture.
//
// On several cards (Options::layer_split) each card has an engine of its own for a run of the layers, with their
// weights, caches, state and expert cache, and the last one has the output and the draft block.  The first one is
// the one the caller holds: it runs its layers, hands the rows of the residual on to the next one through host
// memory, and the next one goes on from them.
#pragma once

#include "cpu_experts.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata {
struct TensorInfo;
struct MetaValue;
}

namespace strata::poly {

class Shards;
struct Likeliest;

// follow()'s three counts were constants in runtime.hpp; they are Options' defaults now, so that the adaptive
// tier can be A/B'd without a build.  These values are the ones the engine shipped with.
constexpr float kSwapMinDefault = 2.f;
constexpr int kSwapHalfDefault = 128;
constexpr float kSwapRatioDefault = 2.f;

struct Options {
    std::string model;
    int max_context = 8192;
    int threads = 0;              ///< 0: the CPUs this process may use
    int vram_reserve_mib = -1;    ///< left free after the expert cache is filled; below 0: by the card's size
    std::string profile;          ///< expert use counts; orders the expert cache, and save_profile() adds to it
    bool profile_write = false;   ///< with --expert-profile: write it back (-written only on request; default read-only)
    bool gpu_experts = true;
    bool dump = false;            ///< print a checksum of every stage, for comparing against llama.cpp
    // A prompt batch.  More tokens a call and a larger staging buffer read a prompt faster, and both take VRAM
    // that the expert cache, and so decoding, does without.
    int prompt_batch = 1024;      ///< prompt tokens one eval() takes; the scratch is about 1.4 MiB of VRAM a token
    int stage_mib = 2048;         ///< the buffer the experts that are not cached pass through: by default about
                                  ///< what a layer of Q2_K experts leaves out of the cache, and never more than
                                  ///< a layer's experts take
    int stage_ahead = 1024;       ///< from this many tokens, a call routes to most of a layer's experts, and the
                                  ///< ones that are not cached start their copy before the layer has routed; in a
                                  ///< read of several calls and this many tokens every call does; 0: never.  Under
                                  ///< it a call copies only the experts it routes to, which with 8 GB of VRAM was
                                  ///< the faster of the two up to about 1,000 tokens
    float cpu_share = -1;         ///< of the experts outside VRAM that such a call routes to, the share the CPU
                                  ///< computes while the others are copied: the ones the fewest tokens chose;
                                  ///< below 0: as the two took the time between them; 0: none
    int expert_swaps = 64;        ///< experts on their way into the cache at a time while an answer is written, each
                                  ///< to the slot of one the answer uses less; 0: the cache stays as it was loaded
    bool stay_near = true;        ///< with several NUMA nodes, keep the CPU threads on the one that holds the experts
    bool graphs = true;           ///< a token's fixed launches are recorded once, a CUDA graph a layer, and replayed
    bool kv_grow = true;          ///< the attention caches get VRAM as the sequence grows, where the card can map it
                                  ///< so; false: for the whole context at load, as the expert cache gets the rest
    int window = 0;               ///< the most tokens window() takes: the known one and guesses after it; 1: none;
                                  ///< 0: 8, or 4 where every guess keeps a copy of the recurrent state
    bool draft = true;            ///< run the block that guesses ahead, when there is one and window is above 1
    int draft_vocab = -1;         ///< the guesses come from the vocabulary's first this many tokens; 0: all of it;
                                  ///< -1: as many as the architecture's vocabulary has in order of use (guess_vocab())
    int draft_dump = 0;           ///< --draft-dump: print the first this many rounds' guesses against the tokens the
                                  ///< model picked (stderr, a line a round); only prints, nothing else changes
    std::string draft_model;      ///< a file of the same model to take that block from, when the model's has none;
                                  ///< for an architecture that guesses with a model of its own, that model's file
    int tail_skip = 0;            ///< a routed expert outside VRAM that every token of a call ranks this far down or
                                  ///< further (the router's first choice is 0) is left out (--tail-skip,
                                  ///< STRATA_ROUTE_TAIL_SKIP); 0: none is
    std::string layer_split;      ///< the layers across the CUDA devices this process sees, in their order: "auto"
                                  ///< (by each card's free VRAM) or the first layer of each later card ("20,30");
                                  ///< empty, or one device: the current card alone
    // Where this engine's threads run (threads.hpp).  off is the default: nothing is pinned, which is what the
    // engine did before these switches existed - measured on this PC's 16 physical / 24 logical CPU, pinning every
    // thread there costs decode about 11%, so the switches are opt-in.
    std::string host_core = "off";     ///< off|first|last|<n>: the host thread's physical core; off pins nothing
    std::string aux_cpus = "off";      ///< off|auto|<list>: where the threads that are not CPU expert workers run
    // The adaptive tier's counts (follow()).  The defaults are the values the engine shipped with.
    float adapt_min_gain = kSwapMinDefault;  ///< a swap-in's decayed count beats the one it replaces by this much
    int swap_half = kSwapHalfDefault;        ///< tokens after which an expert's recent use is halved
    float swap_ratio = kSwapRatioDefault;    ///< and a swap-in's count is this many times the one it replaces
    // --pcie-frac: the share of the misses a later build's GPU/CPU split sends over PCIe.  0 (the default) keeps
    // the run the engine gave.  Above 0 this build only makes the staging slots the split will fill (pcie_stage_),
    // out of the expert cache's arena, and reports what it would cost; no expert is routed differently.
    double pcie_frac = 0.0;
    int pcie_slots = 16;          ///< staging slots for that split; 0 refuses one
    bool pcie_auto = false;       ///< Compile #3's share-fitting; parsed here, nothing uses it yet
    /// --pcie-mode: how a split's experts reach the card.  "dma" (the default): copied into the staging slots, then
    /// the kernels read the slots.  "direct": the kernels read the pinned host mapping where it is - no copy, no
    /// staging slot, no event (Strata's zero-copy half of the split).  "auto": direct where the card maps host
    /// memory into its address space, dma otherwise.
    std::string pcie_mode = "dma";
};

/// A thing a model's file may call by other names than the engine does: converters differ.
struct Names {
    const char* name;
    const char* other[2];
};

/// A tensor in VRAM in its GGUF form.  n_in is the contiguous axis.
struct Dev {
    void* p = nullptr;
    uint32_t type = 0;
    int n_in = 0, n_out = 0;
};

/// What a layer has in every architecture.  Its attention's weights, and a dense FFN's, are the architecture's.
struct Layer {
    bool moe = false;
    bool mmq = false;     ///< the card has MMQ tiles for its routed experts' types (Engine::batched())
    bool fixed = false;   ///< one token's launches can be recorded and replayed for the next ones (eval()): they
                          ///< are the same for every token, or read the position on the card (Engine::at_)
    Dev attn_norm, ffn_norm;   ///< the norms ahead of the attention and of the FFN
    // MoE
    Dev gate_inp, sh_gate, sh_up, sh_down;
    bool gated = true;    ///< the experts have a gate beside up (llama.cpp's SwiGLU); false: only up, relu2 (Nemotron)
    bool relu2 = false;   ///< the experts' activation is relu2 (LLM_FFN_RELU_SQR), not silu(gate)*up
    int exp_in = 0;       ///< the experts' input width (0: n_embd_); a latent MoE's is moe_latent_size (Nemotron 1024)
    int exp_out = 0;      ///< the experts' output width (0: n_embd_)
    Dev latent_down, latent_up;   ///< a latent MoE's projection into and out of the experts' width, once a layer
    ///< where an expert's up (a gated layer's: its gate) and down sit in a slot: gate | up | down, or up | down
    size_t slot_up() const { return gated ? gu_bytes : 0; }
    size_t slot_down() const { return (gated ? 2 : 1) * gu_bytes; }
    const uint8_t* exp_gate = nullptr;  ///< n_expert experts back to back, in the mapped file
    const uint8_t* exp_up = nullptr;
    const uint8_t* exp_down = nullptr;
    uint32_t gu_type = 0, d_type = 0;
    size_t gu_bytes = 0, d_bytes = 0;  ///< one expert's gate (or up), and its down
    size_t gu_stride = 0;              ///< from an expert's gate to the next one's: twice gu_bytes in a file that
                                       ///< keeps an expert's gate and up in one tensor
    std::vector<uint8_t*> slot;        ///< per expert: gate | up | down in VRAM, or null
    size_t slot_bytes = 0;             ///< from a slot of the layer to the one behind it: whole blocks of both types
    std::vector<int32_t> ahead;        ///< experts a full prompt batch stages before it routes
    int trade = -1;                    ///< layers with the same number can take each other's slots: same size
    int8_t card = -1;                  ///< a few tokens' experts routed and found on the card (moe_card()): 1 yes,
                                       ///< 0 no (the architecture routes this layer on the host), -1 not tried
    // What a call of a few tokens hands the card, recorded (Engine::run()): by the call's tokens, and the head
    // also by Engine::head_key()
    std::vector<void*> heads;          ///< up to the router
    std::vector<void*> shared;         ///< the shared expert
    std::vector<void*> tail;           ///< after the experts
    std::vector<void*> card_moe;       ///< the routed experts, routed on the card (moe_card())
};

/// A routed expert is run from the VRAM cache or on the CPU when tokens come a few at a time, as an answer's do
/// (gpu_experts, cpu_experts), and in a prompt batch from the cache or the staging buffer (batch_*).
struct Stats {
    int64_t tokens = 0, gpu_experts = 0, cpu_experts = 0, batch_cached = 0, batch_staged = 0;
    int64_t swapped_experts = 0;   ///< experts that took a cached one's slot (follow())
    int64_t tail_experts = 0;      ///< a token's experts left out by Options::tail_skip
    // --pcie-frac: the miss uses that came over PCIe to the GPU instead of the CPU, and the copies that took them there
    int64_t pcie_experts = 0, pcie_dmas = 0;
    int64_t pcie_direct = 0;   ///< the split's uses read in place (--pcie-mode direct): no copy at all
    size_t pcie_dma_bytes = 0;
    // R49's decode ledger (printed only): the four clocks of an answer's tokens, and the CPU expert pool's phases -
    // the teacher's q38_perf fields, so our decode can be read segment by segment beside its own.
    double pcie_wait_ms = 0;                                          ///< the compute stream waiting for the copies
    double pool_gate_up_ms = 0, pool_quant_ms = 0, pool_down_ms = 0;   ///< the CPU experts' three phases (all calls)
    // R51: the two segments R49 derived are measured here directly, plus the ReLU^2 sparsity probe's counts.
    double launch_ms = 0;   ///< the thread that hands launches to the card: the true wall time of the launch code
    double commit_ms = 0;   ///< true wall time spent waiting for the card (an answer's eval/window synchronize)
    long long probe_values = 0, probe_zeros = 0, probe_blocks = 0, probe_blocks_zero = 0, probe_lines = 0,
              probe_lines_zero = 0;
    // R56 (Stage B, STRATA_RELU2_DOWN=1): the neuron-major down sidecar's read-only probe.  All zero when off.
    // uses: expert uses whose live-neuron index was built (the decode path only); active: sum of h != 0 neurons;
    // active_q8: sum of hq != 0 neurons (a bit-exact skip skips exactly these); groups: sum over uses of the
    // 32-neuron IQ4_NL blocks with any hq != 0 (their fp16 scales must be read); bytes: sum over uses of
    // active_q8*512 + groups*2048 = the sidecar bytes a Stage C kernel would read for them.
    long long r2d_uses = 0, r2d_active = 0, r2d_active_q8 = 0, r2d_groups = 0, r2d_bytes = 0;
    double cpu_ms = 0, route_wait_ms = 0;
    // T8/P7: the staging ledger, so "the copy happened once a layer, not three times" is a number and not a claim.
    long long stage_calls = 0, ring_prefetch = 0, ring_reuse = 0;
    size_t stage_bytes = 0;
    double stage_wait_ms = 0;   ///< P7b only: the card's own time at the round-0 copy wait
    // STRATA_SHARED_TIMING=1: wall time of the shared expert; shared_n1_ms/shared_n1 are the decode steps (n == 1),
    // which is what "ms/token" is read from.  Default 0: never filled, nothing printed.
    double shared_ms = 0, shared_n1_ms = 0;
    long long shared_n1 = 0;
    /// cpu_ms less what the thread that handed the experts to the card spent: the CPU's own share of the phase, so
    /// that a split's PCIe work does not read as CPU time (an upper bound: the two overlap).
    double cpu_busy_ms = 0;
};

/// The most likely token of n logits: the first of the largest.
int most_likely(const float* logits, int n);

class Mmq;

/// A picture in a sequence: n rows that stand where n tokens would, each the vision encoder's output for a patch,
/// as wide as a token's embedding.  The patches come line after line, nx to a line.
struct Image {
    int64_t at = 0;   ///< the sequence's row its first one takes
    int n = 0, nx = 0;
    const float* rows = nullptr;
};

class Engine {
public:
    Engine();
    virtual ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// The engine of the architecture the model file names, loaded.  Throws std::runtime_error, as do eval and
    /// the other calls that touch the device.
    static std::unique_ptr<Engine> open(const Options& o);
    void load(const Options& o);

    int n_vocab() const { return n_vocab_; }
    /// The rows of the output a guess is picked from.  A guess that is never right costs a window's place and no
    /// token, so the rare tokens may be left out, where the vocabulary has the common ones first.
    int draft_rows() const {
        const int v = opt_.draft_vocab >= 0 ? opt_.draft_vocab : guess_vocab();
        return v > 0 ? std::min(v, n_vocab_) : n_vocab_;
    }
    int n_embd() const { return n_embd_; }
    int max_context() const { return opt_.max_context; }
    int threads() const { return workers_ ? workers_->threads() : 0; }
    /// The NUMA node the CPU threads were kept on, and the share of the experts' pages it holds; -1: no node chosen.
    int numa_node() const { return node_; }
    double numa_share() const { return node_share_; }
    int64_t pos() const { return pos_; }
    int64_t meta_u(const char* key, int64_t fallback) const;

    /// Forget the sequence.
    void reset();
    /// The most tokens one eval() takes.
    int max_rows() const { return rows_; }
    /// Feed n tokens from position pos().  With want_logits, logits() then holds the scores of the token after the
    /// last one.
    void eval(const int32_t* tokens, int n, bool want_logits);
    const float* logits() const { return next_ ? next_->logits() : h_logits_; }
    /// Feed n tokens of a prompt, none of them a picture's row, as eval() without logits would in calls of
    /// max_rows(): the same tokens a call.  With experts outside the cache it is faster than those calls, the
    /// more the more of them n is; under stage_ahead tokens it makes those calls.  The result is the calls' own
    /// bit for bit, but for a call of fewer than stage_ahead tokens beside others: its experts' sums are split as a
    /// larger call's, not as on its own.
    ///
    /// A mark is a place inside the read where the caller would have stopped between two calls: `at` of the
    /// read's tokens come before it.  The calls end there as they would have, and `state`, when it is not null,
    /// gets what save_state() gives at that place.  The marks rise, each above 0 and below n.
    struct Mark {
        int at;
        uint8_t* state;
    };
    int max_read() const { return read_rows_; }
    /// `turn`, when given, is asked each time the read's tokens are through one more layer (or, where the read
    /// is its calls one by one, through one more call), with how many of how many that is.  False gives the read
    /// up: read() returns false, and what the engine holds from pos() on is spoiled, its recurrent state and a
    /// window layer's rows among it.  The caller goes back to a state saved before (load_state()) or resets.
    bool read(const int32_t* tokens, int n, const Mark* marks = nullptr, int n_marks = 0,
              const std::function<bool(int, int)>* turn = nullptr);

    /// Whether the architecture reads pictures.
    virtual bool sees() const { return false; }
    /// The pictures of the sequence, in its order, in place of the ones set before.  eval() reads a row that lies
    /// in one from the picture and not from its token, so their rows must stay where they are while tokens are
    /// read.  A sequence that is come back to must get the pictures it was read with.
    void images(const Image* im, int n);
    /// Of m rows from row `at`, how many the next eval() takes.  An architecture whose pictures' rows see each
    /// other takes a picture in a call of its own, which can be more than m.
    virtual int64_t whole(int64_t at, int64_t m) const { (void) at; return m; }

    /// Guessed tokens.  window() feeds t tokens from pos(): the first is the one that is known, each of the others a
    /// guess at what follows the one before it.  logits_of(i) then holds the scores of the token after the i-th, so
    /// the guesses can be checked, and keep(n) makes the first n tokens part of the sequence and forgets the rest.
    /// What a kept token computed is what eval() would have computed for it.
    int max_window() const { return opt_.window; }
    void window(const int32_t* tokens, int t);
    const float* logits_of(int i) const { return next_ ? next_->logits_of(i) : h_wlogits_ + (size_t) i * n_vocab_; }
    void keep(int n);

    /// Some files carry a block that guesses ahead: one layer that takes a token and the model's output for the
    /// token before it, and gives what the model would likely say after the token.  drafts(): it is there and runs.
    /// draft(token, 0, &p) takes the token at pos(), picked and not fed yet, and returns the guess at the one after
    /// it; draft(guess, i, &p) for i = 1, 2, ... then guesses on from the i-th guess, with the block's own output
    /// in the model's place.  p is the probability the block gives what it returns.  pos() + i must be inside the
    /// context.  logits() is left alone.  An architecture may guess with a small model of its own instead of a
    /// block (draft_step()); the calls are the same.
    bool drafts() const { return next_ ? next_->drafts() : drafting(); }
    /// Test only: the calls read()'s layer-major path gave the draft block's rows (draft_rows()) on the card that has
    /// the block, since load.  Two or more in one read are back-to-back refills of its pinned input rows.
    int64_t read_draft_calls_for_test() const { return next_ ? next_->read_draft_calls_for_test() : read_draft_calls_; }
    int32_t draft(int32_t token, int i, float* prob);
    /// Whether a call since the last time this was asked recorded launches for replaying, which takes far longer
    /// than the call does afterwards: its time says nothing of the calls to come.
    bool took_record() {
        const bool later = next_ && next_->took_record();
        return std::exchange(recorded_, false) || later;
    }

    // A sequence's state, to leave the sequence and come back to it.  The recurrent layers have a state that
    // every token changes.  Each token also leaves a row in the attention caches, which stays as it was written.
    // On several cards: the first card's part, then the next one's.
    size_t state_bytes() const { return card_state_bytes() + (next_ ? next_->state_bytes() : 0); }
    size_t row_bytes() const { return card_row_bytes() + (next_ ? next_->row_bytes() : 0); }
    /// The recurrent state at pos(), state_bytes() long.
    void save_state(uint8_t* dst);
    /// Back to position `pos` with the recurrent state it had there.  The rows before pos must be the sequence's.
    void load_state(const uint8_t* src, int64_t pos);
    /// The rows of positions [first, first + n), n * row_bytes() long: layer after layer, a layer's rows together.
    void save_rows(int64_t first, int64_t n, uint8_t* dst);
    /// Rows [first, first + n) out of what save_rows wrote for [from, from + count).
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n);

    const Stats& stats() const;
    /// R56 (Stage B): whether STRATA_RELU2_DOWN=1 mapped the neuron-major down sidecar (never a compute path).
    bool relu2_down_on() const { return r2d_base_ != nullptr; }
    bool ring2() const { return ring2_; }   ///< T8: STRATA_STAGE_RING2's state, for a caller that prints its ledger
    int expert_ff() const { return n_ff_exp_; }   ///< R56: a routed expert's hidden width (the sidecar's neurons)
    /// R57c (D0): the pin budget's handover from the GGUF down half to the sidecar, at the answer's decode boundary.
    bool relu2_pin_switch();
    uint64_t pin_chunks() const { return pin_chunks_; }
    uint64_t pin_fallback_chunks() const { return pin_fallback_chunks_; }
    uint64_t pin_dmas() const { return pin_dmas_; }
    size_t pin_dma_bytes() const { return pin_dma_bytes_; }
    /// Segmented copy ledgers (R13): the load-time cache fill, the prompt's staging (the same DMAs pin_dmas() counts)
    /// and the swaps, so a trace's totals can be read against the engine without re-deriving the flight plan.
    uint64_t fill_dmas() const { return fill_dmas_; }
    size_t fill_dma_bytes() const { return fill_dma_bytes_; }
    uint64_t swap_dmas() const { return swap_dmas_; }
    size_t swap_dma_bytes() const { return swap_dma_bytes_; }
    int cached_experts() const { return cached_ + (next_ ? next_->cached_experts() : 0); }
    int total_experts() const { return n_moe_ * n_expert_ + (next_ ? next_->total_experts() : 0); }
    double cached_gib() const { return double(exp_at_) / double(1ull << 30) + (next_ ? next_->cached_gib() : 0); }
    /// The cache as load() filled it, which is as large as resize_cache() makes it again.
    int full_experts() const { return full_cached_ + (next_ ? next_->full_experts() : 0); }
    double full_gib() const { return double(full_bytes_) / double(1ull << 30) + (next_ ? next_->full_gib() : 0); }
    /// The VRAM the cache was last sized to leave free, and what is free now, in MiB: on several cards, the
    /// first card's reserve and the least that one of them has free.
    int reserve_mib() const { return reserve_mib_; }
    int64_t vram_free_mib() const;
    /// The expert cache sized again, to leave `mib` of VRAM free (below 0: what load() left), and filled with the
    /// experts used most so far.  Without VMM it gets no larger than load() made it.  Between calls that feed
    /// tokens: the experts it gives up are computed on the CPU from then on, as any expert outside it.
    void resize_cache(int mib);
    /// Between requests: the cache gives up VRAM when little is free, and takes VRAM that came free (see the
    /// definition for how much stays free).  True when it changed.
    bool mind_vram();
    void save_profile() const;
    /// STRATA_PROMPT_BORROW: the prompt path's staging buffers are lent the expert cache's tail slots for the
    /// length of a prompt, and the experts they stand in for are refilled after it, before the answer's first
    /// token.  begin() is called when a prompt batch is read; end() when the prompt is done (its time is prompt
    /// time).  Both do nothing unless the switch is on and a region was lent.
    void prompt_borrow_begin();
    void prompt_borrow_end();

protected:
    // ---- what an architecture supplies ----
    /// The model's shape from its metadata: load_shape() for what every architecture has, then its own.
    virtual void load_geometry() = 0;
    /// A layer's weights, to VRAM (upload()) or as places in the mapped file (load_moe()).  load() runs the layers
    /// through twice: begin_pass() is called before each.
    virtual void load_layer(int il) = 0;
    virtual void begin_pass() {}
    /// Throws when the draft block is not one the architecture runs.
    virtual void check_draft_block() {}
    /// The weights of what guesses ahead, to VRAM.  The engine's own: the draft block's parts beside its layer.
    virtual void load_drafter();
    /// One guess (draft()): the token through what guesses ahead, its scores into logits_.  hn_'s first row is the
    /// model's output for the token before pos(); the second is this step's to leave and the next step's to read.
    /// The engine's own runs the draft block.
    virtual void draft_step(int32_t token, int i);
    /// The floats of every recurrent layer's state together.  They are state_'s first.
    virtual size_t recurrent_floats() const = 0;
    /// Whether the recurrent layers leave their state after any tokens of a call, and not only after its first
    /// ones.  A window then takes more tokens than it has places for their states (snap_tokens()).
    virtual bool snaps_spread() const { return false; }
    /// How many of the vocabulary's first tokens hold nearly every token an answer writes; 0: the order says
    /// nothing of that.
    virtual int guess_vocab() const { return 0; }
    /// The layers' places in state_, and the attention's caches: a row a position.  Only for the layers this
    /// engine runs (runs()).
    virtual void alloc_caches() = 0;
    /// What row_bytes(), save_rows() and load_rows() are for the layers this engine runs.
    virtual size_t card_row_bytes() const = 0;
    virtual void save_card_rows(int64_t first, int64_t n, uint8_t* dst) = 0;
    virtual void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) = 0;
    /// The scratch of the architecture's own kernels, a fbuf() each.  Called twice: the first time only counts.
    virtual void assign_scratch() = 0;
    /// The widest input lin() gets, beside the model's width and the FFNs'.
    virtual int widest() const = 0;
    /// The copies of a token's embedding a layer's input starts from.
    virtual int streams() const = 0;
    /// A layer from its input to its router's copy (a layer without routed experts: to its end).
    virtual void layer_head(Layer& L, int il, int n) = 0;
    /// The routed experts' sum (fo_) into the layer's output.
    virtual void layer_tail(Layer& L, int il, int n) = 0;
    /// The model's output for `rows` tokens from row `first` of the call, normed: what the output matrix takes.
    virtual void output_rows(int first, int rows, float* h) = 0;
    /// A token's experts and their weights from its scores at layer il's router.  `token` is the token the row is
    /// for, or -1 where it has none (a picture's row, the draft block).  tmp: 2 * n_expert_ floats of scratch.  A
    /// prompt batch's tokens are routed by several threads at once: it changes nothing of the engine's.
    virtual void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const = 0;
    /// A prompt batch's routing on the card: n tokens' experts and weights from their scores in rl_, n_used_ a token
    /// into ids and w (on the card), what route() gives them up to the last bits of a weight.  False: the host routes.
    virtual bool route_card(int il, int n, int32_t* ids, float* w) {
        (void) il, (void) n, (void) ids, (void) w;
        return false;
    }
    /// The shared expert, into fo_.  It runs on the GPU while the host routes.
    virtual void moe_shared(const Layer& L, int il, int n) = 0;
    /// STRATA_SHARED_TIMING=1 only: moe_shared() with its wall time added to Stats (decode steps counted apart).
    void moe_shared_timed(const Layer& L, int il, int n);
    /// The scores of a layer's router for n tokens, into rl_.  The engine's own: the router's matrix times the
    /// experts' input (xn_).
    virtual void router_scores(const Layer& L, int n);
    /// What is done to `rows` rows of logits on the card before they are copied out.
    virtual void finish_logits(float* logits, int rows) { (void) logits, (void) rows; }
    /// What a token's embedding is multiplied by on its way in.
    virtual float embed_scale() const { return 1.f; }
    /// What a sequence has beside its recurrent state and its rows, in bytes: it is saved and brought back with
    /// the state (save_state(), load_state()).  load_extra() gets the position the state is for.
    virtual size_t extra_bytes() const { return 0; }
    virtual void load_extra(const uint8_t* src, int64_t pos) { (void) src, (void) pos; }
    /// Layer il's part of what save_state() writes, to its place in dst: the layer's recurrent state, and what
    /// extra_bytes() counts for it, as they are once the layer has run every token before `pos`.  The card is
    /// done with the layer when this is called.  save_state() is this for every layer; read() calls it for a
    /// layer when the layer gets to a checkpoint, which is before the layers after it do.
    virtual void save_layer(int il, int64_t pos, uint8_t* dst) = 0;
    /// The draft block's attention for n tokens from position `pos`.  xn_ and xq_ hold its normed input; ao_ gets
    /// its output, and the tokens leave their rows in its caches.  block_rows() only leaves the rows.
    virtual void block_attend(int64_t pos, int n) { (void) pos, (void) n; }
    virtual void block_rows(int64_t pos, int n) { (void) pos, (void) n; }
    /// Which recording of a layer's head fits n tokens from pos().  Launches that are sized by the rows they read
    /// are recorded once for each size (room()), and the architecture numbers the sizes from 0.
    virtual int head_key(int il, int n) const { (void) il, (void) n; return 0; }
    /// The rotary position of a row of text: its place in the sequence, unless pictures before it took fewer
    /// positions than rows.
    virtual int64_t turn(int64_t row) const { return row; }
    /// The buffers that hold a row a token, and pos(), moved `by` rows on (back, when negative): a call for one
    /// token then works in the place of another row of a window (run()).  An architecture adds its own buffers
    /// that a layer's head leaves for the layer's tail.
    virtual void shift_rows(ptrdiff_t by);
    /// The tokens of a window of t that leave the recurrent state as it is after them, a bit a token.
    unsigned snap_tokens(int t) const;
    /// The other names of a tensor or a metadata key in the files of this architecture, or null.
    virtual const Names* tensor_alias(const char* name) const { (void) name; return nullptr; }
    virtual const Names* meta_alias(const char* key) const { (void) key; return nullptr; }

    // ---- what it builds on ----
    /// Whether layer il is this engine's: one of [lo_, hi_), or the draft block on the card that ends the model.
    bool runs(int il) const { return il >= lo_ && (il < hi_ || hi_ == n_layer_); }
    size_t card_state_bytes() const { return state_n_ * sizeof(float) + extra_bytes(); }
    /// Every expert of this engine's layers is in its cache.
    bool all_cached() const { return cached_ == n_moe_ * n_expert_; }
    /// The rows a call of n tokens leaves in r_, to h (pinned) for the next card's engine, which goes on from them.
    void hand_on(int64_t pos, int n, float* h);
    float* pass_rows(int n);
    /// kGuessMat: quantized again to Q4_0, for a matrix only guesses are made with: a guess is checked, so a less
    /// exact one costs a wrong guess now and then and no token.
    enum Want { kF32, kFloatMat, kAnyMat, kGuessMat };
    Dev upload(const TensorInfo* t, const std::string& name, int64_t n_in, int64_t n_out, Want want,
               float (*fix)(float) = nullptr);
    const TensorInfo* tensor(int il, const char* name) const;
    const Shards& home(const TensorInfo& t) const;
    const MetaValue* meta_in(const Shards& g, const char* key) const;
    /// A number of the model's shape, which its file must have.
    int shape_i(const char* key) const;
    float shape_f(const char* key) const;
    void load_shape();
    /// A layer's routed experts.  `shared`: the layer has a shared expert.  `gated`: its experts have a gate beside
    /// up (swiglu, llama.cpp's usual shape); false: only up, activated by relu2.  exp_in/exp_out: the experts' width
    /// (0: the model's), which a latent MoE (Nemotron) makes smaller than n_embd_.
    void load_moe(int il, bool shared = true, bool gated = true, int exp_in = 0, int exp_out = 0);
    void mxfp4_to_q5_0(int il);
    void alloc();
    void shared_scratch();
    float* draft_input(const int32_t* tokens, int64_t row, const float* h, int n);
    void draft_rows(const int32_t* tokens, int first, int n);
    const Image* image_at(int64_t row) const;
    bool input(int32_t token, int64_t row, float* dst) const;
    void lin(const Dev& w, const float* x, float* y, int n, const void* xq = nullptr, int rows_per_x = 0,
             void* s = nullptr);
    void fill_expert_cache();
    /// A layer's slot: a whole number of blocks of both of the experts' types, so the slots that lie one behind
    /// the other are the matrices of one launch.  fill_expert_cache() sizes the cache with it, and --pcie-frac
    /// sizes the staging slots a later build's GPU/CPU split will fill with it.
    size_t expert_slot_bytes(const Layer& L) const;
    /// STRATA_PIN_ARENA_GIB: cudaHostRegisterReadOnly the first N GiB of expert sources, so stage() DMAs straight from
    /// the mapping instead of gathering it on the host.  All zeros / empty when the switch is off (the default).
    void pin_expert_arena();
    void start_swaps();
    void plan_ahead(Layer& L, int il);
    void run(const int32_t* tokens, int n);
    bool pick_way(int t);
    void took_way(int t, double ms);
    void layer(int il, int n);
    void save_last(uint8_t* dst);
    void moe_router(const Layer& L, int n);
    bool moe(Layer& L, int il, int n, const std::function<void()>* first = nullptr);
    bool card_moe(int n);
    bool moe_card(Layer& L, int il, int n);
    /// Whether n tokens take layer L's routed experts as a prompt batch, through MMQ (moe_rows()), and not a few at
    /// a time (moe()).
    bool batched(const Layer& L, int n) const;
    void moe_add(Layer& L, int n);
    /// A latent MoE's routed sum is in moe_out_ (exp_out wide); latent_up maps it into fo_.  A no-op without one.
    void moe_latent(Layer& L, int n);
    template <class F>
    void* record(F&& work);
    void moe_rows(Layer& L, int il, int n);
    void stage(const Layer& L, const int32_t* experts, int m, void* stream, bool alone = false,
               uint8_t* base = nullptr);
    void mind_memory();
    void follow(int written);
    void follow_loop();
    void embed(int32_t token, float* dst) const;
    void dbg(const char* name, int il, const float* dev, int n);
    float* fbuf(size_t n);
    int room(int n) const;
    int room_key(int n) const;
    void* dmalloc(size_t bytes, const char* what);
    void* hmalloc(size_t bytes, const char* what);
    /// An attention cache for alloc_caches(): `bytes` for every `per` rows, with addresses for the whole context
    /// and VRAM for the rows the sequence has reached (cover()).
    void* kv_alloc(size_t bytes, int per);
    /// VRAM in every attention cache for the rows below `end`, before a call writes them.
    void cover(int64_t end);

    // VRAM mapped onto addresses reserved for the most it can be, a piece at a time (CUDA's virtual memory
    // management): it grows and shrinks at its end and keeps its address, which recorded launches hold.
    struct Mapped {
        unsigned long long base = 0;   ///< a CUdeviceptr
        size_t reserved = 0, mapped = 0;
        std::vector<std::pair<unsigned long long, size_t>> pieces;   ///< allocation handles and their bytes, in order
    };
    struct Kv : Mapped {
        size_t bytes = 0;   ///< for every `per` rows
        int per = 1;
    };
    void vmm_reserve(Mapped& m, size_t bytes);
    void vmm_map(Mapped& m, size_t bytes, size_t piece, const char* what);
    void vmm_unmap(Mapped& m, size_t bytes);
    void vmm_free(Mapped& m);
    void shrink_cache(size_t bytes);
    int grow_cache(size_t bytes);
    void resize_card_cache(int mib);   ///< resize_cache() and mind_vram() for this engine's card
    bool mind_card_vram();
    void held_is_budget();
    void settle_swaps();

    Options opt_;
    std::unique_ptr<Shards> gg_;
    std::unique_ptr<Shards> dg_;      ///< Options::draft_model, while the draft block is read from it
    const Shards* block_ = nullptr;   ///< the file the draft block is in: gg_ or dg_
    std::shared_ptr<Workers> workers_;   ///< on several cards, one for all of them: they take turns
    int node_ = -1;
    // A layer split (open()).  The layers of the model are [lo_, hi_) here, and the engine of the next card has
    // the ones after.  A later card's engine reads its first layer's input from pass_, which the one before wrote:
    // a row of the residual a position, from pass_at_.
    int lo_ = 0, hi_ = 0;
    int end_ = -1;                        ///< what open() asked for hi_; -1: the model's end
    bool split_ = false;                  ///< one of several engines: a call sets its card first
    std::unique_ptr<Engine> next_;
    int64_t read_draft_calls_ = 0;               ///< read_draft_calls_for_test()
    const float* pass_ = nullptr;
    int64_t pass_at_ = 0;
    float* h_pass_ = nullptr;             ///< pinned; h_pass_rows_ tokens' rows
    int h_pass_rows_ = 0;
    mutable Stats all_stats_;             ///< stats() of every card's engine together
    double node_share_ = 0;
    std::thread mapper_;                  ///< reads the experts' pages once after load, so they are mapped
    std::atomic<bool> map_quit_{false};
    void* st_ = nullptr;   ///< the stream everything but the staging ahead runs on
    // STRATA_SHARED_OVERLAP=1: decode's calls (n <= 8 tokens) run the shared expert on a stream of its own, beside
    // the routed experts' work, and moe_latent() waits for its event before the routed sum is added into fo_.
    bool shared_overlap_ = false;      ///< the switch took effect (a stream, an event and the block's own rows)
    void* sh_ = nullptr;               ///< the stream it runs on
    void* ev_sh_ = nullptr;            ///< marks the end of its two products
    void* ev_st_ = nullptr;         ///< marks st_ done with xn_ (the head/rms_norm) before sh_ reads it
    bool sh_pending_ = false;          ///< its event must be waited for before fo_ is added to
    float* sh_g_ = nullptr;            ///< its up's rows (8 tokens: decode's window top)
    float* sh_h_ = nullptr;            ///< its relu2's rows
    void* q8_sh_ = nullptr;            ///< its quantized inputs (lin's scratch, of its own)
    float* wf_sh_ = nullptr;           ///< its BF16 matrix as floats (lin's scratch, of its own)
    void* blas_ = nullptr; ///< cuBLAS on st_
    void* ev_ = nullptr;   ///< marks the router's copy to the host: ev_row_'s first, or a window's row's own
    void* ev_row_[8] = {}; ///< one for each token a window can have
    std::vector<void*> dev_allocs_, host_allocs_;
    std::vector<void*> graphs_;   ///< every recorded graph, to destroy
    std::map<std::vector<int>, void*> passes_;   ///< a call's layers recorded as one (run()): by its tokens and
                                                 ///< every layer's head_key()
    void* tail_ = nullptr;        ///< one token's launches from the last layer to the logits, recorded
    // A recorded launch cannot be given the position as an argument: it reads it here when it runs.  eval() writes
    // the token's place in the sequence and its rotary position before it replays the layers.
    long long* at_ = nullptr;     ///< on the card: {pos(), turn(pos())}, and the same for a window's later tokens
    long long* h_at_ = nullptr;   ///< pinned
    bool recording_ = false;      ///< a layer's launches are being recorded: positions are read from at_
    bool recorded_ = false;       ///< took_record()

    // geometry
    int n_embd_ = 0, n_layer_ = 0, n_vocab_ = 0;
    int n_ff_ = 0;            ///< a dense FFN's width, in the architectures that have one
    int n_expert_ = 0, n_used_ = 0, n_ff_exp_ = 0, n_ff_shexp_ = 0, n_moe_ = 0;
    int rows_ = 1;            ///< tokens the scratch is sized for
    int read_rows_ = 1;       ///< the most tokens read() takes
    // read()'s calls between two layers: r_ holds two calls' rows, and with more calls than that the others'
    // wait in host memory, copied out and in beside the card's work on the call between
    struct Call {
        int at, m, mark;      ///< its first token, how many, and the mark it ends at or -1
    };
    std::vector<Call> calls_;
    float* h_res_ = nullptr;  ///< h_res_rows_ tokens' rows, pinned: as many as the longest read so far that needed it
    int h_res_rows_ = 0;
    void* rs_ = nullptr;      ///< the stream of those copies
    void* ev_in_[2] = {};     ///< marks a call's rows in each half of r_
    void* ev_out_[2] = {};    ///< marks the card done with a call in each half
    float rms_eps_ = 0;
    bool gelu_ = false;       ///< the routed experts' activation: GELU of the gate, else SiLU within the layer's limits
    std::vector<float> clamp_exp_, clamp_shexp_;   ///< a layer's SwiGLU limits: the routed experts', the shared one's

    std::vector<Layer> layers_;   ///< the model's, and after them the draft block
    std::vector<std::unique_ptr<uint8_t[]>> converted_;   ///< experts converted at load (mxfp4_to_q5_0), in RAM
    Dev out_norm_, output_;
    bool draft_ = false;
    bool block_layer_ = false;    ///< what guesses ahead is a block after the model's layers (layers_' last)
    /// STRATA_DRAFT_LOAD_ONLY=1: the block is loaded and wired and its VRAM kept, but nothing is computed from it and
    /// no guess is proposed - so a run has the numbers of one without a draft block and the VRAM of one with it.  An
    /// experiment switch for telling a draft block's VRAM cost from its computation; off unless the variable says 1.
    bool draft_load_only_ = false;
    /// Whether the block that guesses ahead computes and proposes: draft_ and not held back by STRATA_DRAFT_LOAD_ONLY.
    bool drafting() const { return draft_ && !draft_load_only_; }
    Dev d_proj_, d_enorm_, d_hnorm_, d_head_;   ///< the draft block's input projection and norms, and its output norm
    const uint8_t* tok_embd_ = nullptr;   ///< the token table in the mapped file, tok_row_ bytes a token
    size_t tok_row_ = 0;
    uint32_t tok_type_ = 0;

    uint8_t* dense_ = nullptr;   ///< the dense weights' arena
    size_t dense_bytes_ = 0, dense_at_ = 0;
    uint8_t* exp_ = nullptr;     ///< the routed experts' arena
    size_t exp_at_ = 0;
    Mapped arena_;               ///< its VRAM, when the card maps VRAM a piece at a time
    bool vmm_ = false;           ///< it does; without that every cache is as large as it can get from the start
    int device_ = 0;
    size_t grain_ = 0;           ///< what the card maps VRAM in
    std::vector<Kv> kv_;         ///< the attention caches that grow with the sequence
    int64_t kv_rows_ = 0;        ///< the rows they have VRAM for
    size_t budget_ = 0;          ///< their VRAM and the expert cache's together, as the cache was filled
    int asked_reserve_mib_ = -1; ///< Options::vram_reserve_mib as load() got it
    int reserve_mib_ = 0;        ///< what the cache was sized to leave free
    bool ram_tight_ = false;     ///< the machine's RAM is within 4 GiB of the model's files (fill_expert_cache())
    int minds_ = 0;              ///< mind_vram()'s calls
    int full_cached_ = 0;        ///< the cache as load() filled it: its experts and their bytes
    size_t full_bytes_ = 0;
    float* scratch_ = nullptr;
    size_t scratch_at_ = 0;
    float* state_ = nullptr;     ///< every recurrent layer's state; with a draft block, hn_'s first row
    size_t state_n_ = 0, rec_n_ = 0;   ///< all of it, and the recurrent layers' part
    // The draft block reads the model's output for the token before each of its tokens.  Row 0 is the output of
    // the last token of the sequence, and the outputs of the tokens of the call under way follow it.
    float* hn_ = nullptr;
    Likeliest* h_guess_ = nullptr;   ///< a guess and its probability, pinned
    int32_t win_tok_[8] = {};           ///< the window's tokens, for keep()
    int64_t ahead_pos_ = -1;            ///< draft() has left the block's row of this position as its token leaves it,
    int32_t ahead_tok_ = 0;             ///< which is this one
    float* snap_ = nullptr;      ///< window(): the recurrent state after tokens of it (snap_tokens()), rec_n_ apart
    int snap_places_ = 0;        ///< how many states it holds
    float* snap_at_ = nullptr;   ///< where the call under way leaves the first of its states
    unsigned snap_bits_ = 0;     ///< the tokens of the call under way that leave their state, a bit a token
    bool snap_ok_ = false;       ///< a recurrent layer wrote snap_ this window (keep() refuses to restore without it)
    bool windowed_ = false;      ///< the call under way is a window's
    int win_ = 0;                ///< the tokens of the window that keep() has not settled yet
    // A window with experts on the CPU goes through the layers whole, or in two halves one behind the other
    // (run()).  Which takes less time depends on the card, on the CPU and on how many of the experts the cache
    // has, so each way's time is kept for each window size (pick_way(), took_way()).
    struct Way {
        double ms = 0;           ///< what a window took this way, the later ones counting for more
        int n = 0;               ///< how many were timed
    };
    Way way_[2][9];              ///< [in halves][the window's tokens]
    int way_calls_[9] = {};      ///< windows of each size since the way behind had one
    int way_backoff_[9] = {};    ///< the way behind waits kWayAgain << this many windows of the size
    bool halves_ = false;        ///< the way of the window under way
    float *wlogits_ = nullptr, *h_wlogits_ = nullptr;   ///< a window's logits, token after token: VRAM and pinned
    int cached_ = 0;

    // device scratch: a layer's input streams, one stream, the normed input, the attention's and the FFN's output
    float *r_ = nullptr, *cur_ = nullptr, *xn_ = nullptr, *ao_ = nullptr, *fo_ = nullptr;
    // STRATA_DRAFT_CHUNKED=1: the draft block's priming rows (three row sets a chunk) in a scratch of their own, so
    // r_ keeps its non-draft size and the MTP's VRAM falls by the row sets it used to hold for the draft block.
    float* draft_work_ = nullptr;
    // A latent MoE (Nemotron): the experts read and write the layer's latent width, so their input is a projection
    // of the normed input and their sum is projected back before it is added to fo_.  moe_in_ is the experts' input
    // (xn_ itself when the layer has no latent_down); moe_out_ is their sum until latent_up maps it into fo_.
    float* moe_in_ = nullptr;
    float* moe_out_ = nullptr;
    float* moe_up_ = nullptr;   ///< a latent MoE's latent_up output, before it is added to fo_
    int moe_in_max_ = 0, moe_out_max_ = 0;   ///< the widest expert input/output any layer has (latent or n_embd_)
    float* wf_ = nullptr;      ///< the largest BF16 matrix as floats: lin() multiplies a prompt batch by cuBLAS
    size_t bf16_max_ = 0;      ///< that matrix's values
    float *g_ = nullptr, *u_ = nullptr, *h_ = nullptr, *d_ = nullptr, *rl_ = nullptr,
          *logits_ = nullptr;
    void *xq_ = nullptr, *q8_ = nullptr;

    // pinned host buffers
    float *h_r_ = nullptr, *h_rl_ = nullptr, *h_x_ = nullptr, *h_cpu_ = nullptr, *h_logits_ = nullptr;

    // A prompt batch's MoE (moe_rows).  A row is a token and one of the experts it routes to.
    std::unique_ptr<Mmq> mmq_;   ///< null when the card has no MMQ tile for the experts' types
    float *gu_ = nullptr, *hx_ = nullptr, *dx_ = nullptr;   ///< the rows' gate|up, hidden and down values
    void *mq_ = nullptr, *mq2_ = nullptr;                   ///< the rows quantized for the two products
    void* mqx_ = nullptr;                                   ///< lin()'s tokens quantized for a product
    int32_t *tab_ = nullptr, *h_tab_ = nullptr;   ///< a layer's row tables, in VRAM and pinned
    int32_t* iota_ = nullptr;                     ///< 0, 1, 2, ...
    int32_t* d_ids_ = nullptr;                    ///< a batch's experts and weights when it routes on the card
    float* d_wts_ = nullptr;
    int32_t* h_ids_ = nullptr;                    ///< pinned: the experts, back from the card
    void** d_slots_ = nullptr;                    ///< every layer's L.slot on the card, n_expert_ a layer (moe_card())
    std::vector<void*> h_slots_;                  ///< what d_slots_ holds
    uint8_t* stage_ = nullptr;                    ///< experts that are not cached, on their way through VRAM
    uint8_t* h_stage_[2] = {};                    ///< pinned; filled in turn
    void* ev_stage_[2] = {};                      ///< marks the upload out of each
    void* up_ = nullptr;                          ///< the stream of the uploads that start before the routing
    void* ev_ahead_ = nullptr;                    ///< marks the last of them
    void* ev_free_ = nullptr;                     ///< marks a layer's last read of the staging buffer
    // STRATA_MOE_ROWS_CP_STREAM: a prompt batch's round copies run on a stream of their own (cp_), so that a
    // round's copy overlaps the kernels of the round before.  The staging buffer is one, so a copy still waits
    // for the kernels that read it (ev_krn_) and the round's kernels wait for its copy (ev_copy_).  Null when the
    // switch is off: moe_rows() then takes exactly the path it took before the switch existed.
    void* cp_ = nullptr;
    void* ev_krn_ = nullptr;
    void* ev_copy_ = nullptr;
    // STRATA_STAGE_RING2: two round buffers, so that the next layer's ahead copy (on up_) runs while this layer's
    // kernels read the other.  slot_[0] is stage_; slot_[1] (stage2_) is made only when the switch is on.  Each
    // slot has its own event pair: a copy waits ev_ring_free_[s] (that slot's last reader), the round waits
    // ev_ring_ahead_[s] (that slot's copy).  Off: all null and the one-buffer path runs exactly as before.
    uint8_t* stage2_ = nullptr;
    uint8_t* slot_[2] = {nullptr, nullptr};
    bool ring2_ = false;       ///< the switch said on: the buffer itself is made after fill_expert_cache()
    void* ev_ring_ahead_[2] = {};
    void* ev_ring_free_[2] = {};
    int ring_slot_ = 0;        ///< the slot a ring call's round 0 reads when nothing was prefetched for it
    int prefetched_il_ = -1;   ///< the layer whose ahead list sits in slot prefetched_slot_, or -1
    int prefetched_slot_ = 0;
    // T8/P1: the several-call read copies ahead once a layer and reads that one copy from every call of it.
    // `prefetched_n_` is how many experts the copy in prefetched_slot_ took (checked when it is consumed);
    // `ring_hold_*` say which slot holds THIS layer's list for its later calls (-1: none).
    int prefetched_n_ = 0;     ///< how many experts that prefetch copied
    int ring_hold_il_ = -1;    ///< the layer whose list slot ring_hold_slot_ holds for its later calls, or -1
    int ring_hold_slot_ = -1;
    void* ev_hr_ = nullptr;                       ///< marks the card's last read of h_r_ by draft_input()
    int staged_il_ = -1;                          ///< the layer whose experts the buffer holds for its next call, or -1
    std::vector<int32_t> staged_;                 ///< those experts, in the buffer's order: one round of the layer's
    double ahead_ms_ = 0, route_ms_ = 0;          ///< what that layer's copy ahead took, and the card to its routing
    bool shared_ahead_ = false;                   ///< read() runs several calls: each copies ahead, whatever its size
    size_t stage_bytes_ = 0, chunk_bytes_ = 0;
    // STRATA_PROMPT_BORROW: the prompt's staging buffers (stage_, and with STRATA_STAGE_RING2 stage2_) are lent
    // slots of the expert cache's tail for the length of a prompt; the experts those slots would hold are refilled
    // after the prompt.  borrow_ is a mapping of its own, so cover() and shrink_cache() - which only ever cut
    // arena_ - cannot take it; the experts it holds are slot pointers outside arena_, which shrink_cache() leaves
    // alone.  Off (no mapping, no lent experts) unless the switch says 1: then the buffers hold their own VRAM
    // exactly as they always did.
    struct Lent { int il, e; size_t bytes; };
    bool prompt_borrow_ = false;       ///< STRATA_PROMPT_BORROW=1
    bool borrow_lend_ = false;         ///< prompt_borrow_ with at least one slot lent: stage_ is the region's
    bool borrow_refill_ = true;        ///< STRATA_PROMPT_BORROW_REFILL=0: lend, but put nothing back (an A/B)
    int borrow_slots_req_ = -1;        ///< STRATA_PROMPT_BORROW_SLOTS: the slots to lend; 0 = none, -1 = auto
    int borrow_verify_ = 0;            ///< STRATA_PROMPT_BORROW_CHECK=n: check n refilled and n resident slots
    size_t stage_alloc_bytes_ = 0;     ///< stage_bytes_ + 256 + kTail: what one staging buffer takes
    size_t borrow_one_buffer_ = 0;     ///< the VRAM one staging buffer takes, measured by a probe of its size
    size_t borrow_bytes_ = 0;          ///< the region's bytes
    size_t borrow_off_[2] = {0, 0};    ///< the two staging buffers' places in the region
    int borrow_buffers_ = 0;           ///< how many staging buffers the region holds (1, or 2 with ring2)
    Mapped borrow_;                    ///< the region's VRAM: a mapping of its own, outside arena_
    std::vector<Lent> borrow_cands_;   ///< the experts the cache would hold next, lowest rank first: refilled
    int borrow_lent_ = 0;              ///< experts of borrow_cands_ resident in the region now
    bool prompt_lent_ = false;         ///< the prompt is running: the region holds the staging buffers
    double borrow_refill_ms_ = 0;      ///< the last refill's time
    size_t borrow_refill_bytes_ = 0;   ///< and its bytes
    void map_prompt_borrow();           ///< fill_expert_cache(): maps the region and checks it fits (loud)
    void refill_prompt_borrow();        ///< prompt_borrow_end(): the experts back into the region, verified
    void verify_prompt_borrow(const std::vector<Lent>& took);   ///< STRATA_PROMPT_BORROW_CHECK
    // STRATA_BORROW_ASYNC_REFILL: the refill runs on a stream of its own while the answer's first rounds run.  The
    // experts it is filling in are not resident until their copies land (their slot pointers and the cache's count
    // are set then), so a token that routes to one takes the CPU path - correct, just slower - and nothing ever
    // reads half an expert.  Off by default: the refill blocks the prompt's end as it always did.
    bool borrow_async_ = false;        ///< STRATA_BORROW_ASYNC_REFILL=1
    bool async_refill_ = false;        ///< a refill is in flight: async_took_ is not resident yet
    void* as_ = nullptr;               ///< its stream (made the first time one is needed)
    void* ev_refill_ = nullptr;        ///< marks its last copy
    double async_t0_s_ = 0.0;          ///< steady_clock seconds when it started, for the done line
    int64_t async_pos0_ = 0;           ///< the sequence position the answer had when it started
    int64_t async_miss0_ = 0;          ///< stats_.cpu_experts when it started (the pending window's misses)
    int64_t async_tok0_ = 0;           ///< stats_.tokens when it started (the pending window's tokens)
    std::vector<Lent> async_took_;     ///< the experts it is filling in, in the region's order
    void start_async_refill(std::vector<Lent> took);   ///< prompt_borrow_end(): the copies go on as_
    void install_async_refill();       ///< they have landed: the slots become resident (and counted)
    void poll_async_refill();          ///< run()'s start: installs them once the event says they are in
    void finish_async_refill();        ///< prompt_borrow_begin()/~Engine: wait for them, then install
    // --pcie-frac (Compile #1 skeleton, Compile #2 the split): the staging slots a share of an answer's misses are
    // copied into, and the stream and event that take them there.  Null and unallocated unless the switch is above
    // 0, so the default run's VRAM, expert cache and arithmetic are exactly as before.
    uint8_t* pcie_stage_ = nullptr;
    int pcie_slots_ = 0;
    size_t pcie_slot_bytes_ = 0;
    void* pcie_st_ = nullptr;   ///< the copies run on it, beside the CPU's work
    void* ev_pcie_ = nullptr;   ///< marks the last of them: the compute stream waits for it before the split's experts
    bool pcie_direct_ = false;  ///< --pcie-mode direct (or auto that found it works): the kernels read the host mapping
    int stage_turn_ = 0;
    std::vector<int32_t> ids_, count_, first_, group_, round_;
    std::vector<float> route_tmp_;   ///< route()'s scratch, a share for each host thread
    struct Run {
        int c, j, m, most;   ///< the rows' class, its first cached expert and how many, the most rows one has
    };
    std::vector<Run> runs_;        ///< a layer's launches over cached experts whose slots lie one behind the other
    std::vector<uint8_t> pin_layer_;   ///< per layer: 1 when its expert sources are pinned (STRATA_PIN_ARENA_GIB)
    /// --pcie-mode direct: the device aliases of those pinned regions (cudaHostGetDevicePointer), so the split's
    /// kernels can read the mapping in place.  Null on the layers that are not pinned or when direct is off.
    std::vector<const uint8_t*> exp_up_dev_, exp_down_dev_;
    size_t pin_bytes_ = 0;
    // R56 (Stage B): the neuron-major down sidecar (STRATA_RELU2_DOWN=1).  Mapped read-only, NOT pinned and never
    // read whole (it is a second 31.7 GB copy; pinning it would push this box past its RAM - R53 搂0.3).  Stage B
    // only maps it, checks its size against the geometry and four experts' blocks against the GGUF bytes, and lets
    // cpu_experts() build the live-neuron index from it.  No kernel, slot, staging or GGUF pointer reads it.
    const uint8_t* r2d_base_ = nullptr;   ///< the mapping, null unless on
    size_t r2d_bytes_ = 0;                ///< its size
    void* r2d_map_ = nullptr;             ///< the mapping object (closed in ~Engine)
    void* r2d_file_ = nullptr;            ///< the file handle (closed in ~Engine)
    size_t r2d_expert_bytes_ = 0;         ///< one expert's sidecar block = its down bytes in the GGUF
    bool r2d_pinned_ = false;             ///< R57 option B: the sidecar is cudaHostRegisterReadOnly-pinned
    bool r2d_slots_t_ = false;            ///< Stage D D3: the VRAM slots' down segments hold the T layout
    std::vector<size_t> r2d_layer_off_;   ///< per layer: its experts' first sidecar byte (SIZE_MAX when not main MoE)
    void relu2_down_open();               ///< load(): map, pin and fault in the sidecar (loud; off by default)
    /// Stage D D2a: one expert's down, ROW-MAJOR, written into `dst` (device: a slot's down segment, a staging
    /// down area, a swap/borrow region).  Source = the pinned sidecar's T block: one DMA to a one-expert device
    /// scratch, then the R58-verified T->row kernel.  Every GPU consumer keeps reading today's bytes unchanged.
    void materialize_down(const Layer& L, int il, size_t e, uint8_t* dst, void* stream);
    /// Stage D: one 1.48 MiB device scratch a stream (fill's null stream, st_, up_, cp_, swap_st_): the four paths
    /// can be in flight together (ring2 ahead on up_ beside this layer's stage), so one shared scratch would race.
    void* r2d_scratch_for(void* stream);
    std::vector<std::pair<void*, void*>> r2d_scratch_;   ///< (stream, buffer)
    std::mutex r2d_scratch_m_;
    uint64_t pin_chunks_ = 0, pin_fallback_chunks_ = 0;   ///< stage() chunks taking the direct DMA / the old gather
    uint64_t pin_dmas_ = 0; size_t pin_dma_bytes_ = 0;    ///< the direct path's DMA count and bytes (size check)
    uint64_t fill_dmas_ = 0, swap_dmas_ = 0;              ///< R13: the load fill's and the swaps' copies
    size_t fill_dma_bytes_ = 0, swap_dma_bytes_ = 0;
    // STRATA_SCRATCH_TILE: gu_/hx_/mq2_ hold scratch_rows_ rows instead of rr (a batch of experts, never split).
    // The down mul's per-batch bounds table is bounds[b0+i]-bounds[b0] (0-based, first entry 0); the up keeps iota_.
    size_t scratch_rows_ = 0;
    int32_t* d_rel_local_ = nullptr;
    int32_t* h_rel_local_ = nullptr;
    // A lone expert with more rows than the tile cannot be split: that call goes back to the full-rr buffers.
    // They are made the first time it happens (never silent: moe_rows() warns), and kept for the next.
    float *fb_gu_ = nullptr, *fb_hx_ = nullptr;
    void* fb_mq2_ = nullptr;
    // The tiled up's input: mmq's q8 layout is [K block][row], so a batch's rows cannot be sliced out of mq_:
    // quantize() writes just the batch's rows here instead (row i = the token d_tok[r0 + i]).
    void* mq_tile_ = nullptr;
    std::vector<int32_t> chan_;    ///< a layer's cached experts in the order of their slots
    std::vector<uint8_t> sent_;
    // A call that copies only the experts it routes to leaves some of those outside VRAM to the CPU (moe_rows()).
    std::vector<uint8_t> on_cpu_;  ///< per expert of the layer: the CPU's
    std::vector<int32_t> spare_;   ///< the layer's routed experts outside VRAM, the fewest tokens' first
    std::vector<CpuExpert> host_;  ///< their uses, by token
    double share_[5] = {0.25, 0.25, 0.25, 0.25, 0.25};   ///< the CPU's share of spare_, by the call's tokens:
                                                         ///< under 64, from 64, 128, 256 and 512
    std::unique_ptr<Workers> stagers_;   ///< fill the staging buffers meanwhile

    // The cache follows what is written (follow()).  It is filled for the average answer; while one is written,
    // the experts it keeps routing to and the cache lacks are copied over the cached ones it uses least, by a
    // thread and a stream of their own.
    struct Swap;
    struct Pick {
        int il = -1, e = 0;
        float recent = 0;
        int64_t prior = 0;
    };
    std::unique_ptr<Swap[]> swap_;   ///< the experts on their way, opt_.expert_swaps at most
    int n_swap_ = 0, n_trade_ = 0;
    size_t slot_max_ = 0;            ///< the largest expert of the cache
    void* swap_st_ = nullptr;        ///< the stream of their uploads
    std::thread follower_;
    std::mutex swap_m_;
    std::condition_variable swap_cv_;
    bool swap_quit_ = false;
    std::string swap_error_;         ///< why the thread stopped; follow() throws it
    std::vector<float> recent_;      ///< per (layer, expert): its uses by written tokens, halved every swap_half
    std::vector<uint8_t> coming_;    ///< per (layer, expert): on its way into the cache
    std::vector<Pick> in_, out_;     ///< per trade number: the experts to bring in, and the ones to give up
    std::vector<size_t> trade_at_;   ///< per trade number: how many of its pairs follow() has taken

    int64_t pos_ = 0;
    const int32_t* tok_ = nullptr;       ///< the tokens of the call under way, a row each as h_rl_ has them (route())
    std::vector<Image> images_;          ///< the sequence's pictures, by row
    uint64_t images_set_ = 0;            ///< counts images(): what was derived from an earlier list is stale
    std::vector<int64_t> use_, prior_;   ///< routed count per (layer, expert): this run, and the profile's
    // STRATA_PROFILE_DECODE_ONLY: the routed uses of the answer's tokens and of the prompt's, counted apart - the
    // profile can then be written from the decode routing alone (the prompt path was 98.26% of every count the
    // table was built from).  use_ still counts both, so everything that orders or moves the cache reads exactly
    // what it always did; only save_profile() picks a side.
    std::vector<int64_t> use_decode_, use_prefill_;
    // STRATA_PROFILE_AUTO_LEARN=1: a missing/empty --expert-profile is learnt from this run's answers (decode
    // counts) and written back; the next start loads the same path and is faster.  Default 0: unchanged.
    bool profile_auto_learn_ = false;
    bool profile_decode_only_ = false;
    // STRATA_DUMP_ROUTE=1: one line a routed (layer, token): "il token id0 .. id21", for the offline 1.6/1.7
    // accuracy gate.  Default 0: no file, no writes, the routed path byte for byte unchanged.
    FILE* route_dump_ = nullptr;
    void dump_route(const int* ids, int il, int32_t token);
    Stats stats_;
};

}  // namespace strata::poly
