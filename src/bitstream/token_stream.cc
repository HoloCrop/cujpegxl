// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "token_stream.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <thread>

#include "ans.h"

namespace cujpegxl::bitstream {
namespace {

constexpr std::size_t NO_STREAM{SIZE_MAX};

// Streams coded in lockstep by one caller: enough independent state chains to
// hide the latency of the dependent state update.
constexpr std::size_t LANES{3};

// 64-bit reciprocal precision of libjxl's PutSymbol (state bits + table bits).
constexpr std::uint32_t RECIPROCAL_PRECISION{32 + ANS_LOG_TAB_SIZE};

// Builds a stream from its end: prepend() puts bits before all bits written so
// far, so a backward pass over the tokens emits them in forward order. The
// earliest bits wait in `acc`, full words are stored descending from the slot
// end.
struct BackWriter {
    std::uint64_t* out{nullptr};
    std::uint64_t acc{0};
    std::uint32_t fill{0};

    void prepend(std::uint64_t bits, std::uint32_t n) {
        assert(n < 64 && (bits >> n) == 0);
        // The spare word in the capacity bound makes this store safe even
        // when it is speculative. Only advance the cursor on overflow.
        const std::uint32_t total{fill + n};
        const std::uint32_t excess{total & 63};  // overflow bits when total > 64
        out[-1] = (acc << ((64 - fill) & 63)) | (bits >> excess);
        const bool flush{total > 64};
        out -= flush;
        fill = total - 64 * flush;
        // Only the low fill bits are live. The next flush shifts out stale
        // high bits; store() masks them when publishing the final head.
        acc = (acc << n) | bits;
    }
};

std::uint32_t read_raw(const std::uint64_t* raw, std::uint64_t pos, std::uint32_t n) {
    // TokenStream guarantees a readable trailing word for this unaligned load.
    std::uint64_t window;
    std::memcpy(&window, reinterpret_cast<const std::uint8_t*>(raw) + (pos >> 3), sizeof(window));
    return static_cast<std::uint32_t>((window >> (pos & 7)) & ((std::uint64_t{1} << n) - 1));
}

struct Lane {
    const std::uint16_t* entry{nullptr};  // one past the next token to code
    std::size_t left{0};
    const std::uint64_t* raw{nullptr};
    std::uint64_t raw_pos{0};
    std::uint32_t state{0};
    BackWriter out{};
    std::uint64_t* slot_end{nullptr};
    std::size_t stream{NO_STREAM};
};

std::size_t claim(StreamQueue& queue) {
    const std::size_t k{queue.next.fetch_add(1, std::memory_order_relaxed)};
    if (k >= queue.num_streams) {
        return NO_STREAM;
    }
    while (queue.ready.load(std::memory_order_acquire) <= k) {
        if (queue.cancelled.load(std::memory_order_acquire)) {
            return NO_STREAM;
        }
        std::this_thread::yield();
    }
    return queue.order[k];
}

void start(Lane& lane, const StreamQueue& queue, std::size_t i) {
    const TokenStream& s{queue.streams[i]};
    lane.entry = s.entries + s.num_tokens;
    lane.left = s.num_tokens;
    lane.raw = s.raw;
    lane.raw_pos = s.raw_bits;
    lane.state = ANS_INITIAL_STATE;
    lane.slot_end = queue.slot_words + queue.slot_end[i];
    lane.out = BackWriter{lane.slot_end};
    lane.stream = i;
}

void store(const Lane& lane, StreamQueue& queue) {
    const auto head{lane.out.acc & (~std::uint64_t{0} >> ((64 - lane.out.fill) & 63))};
    queue.coded[lane.stream] =
        CodedStream{lane.out.out, static_cast<std::size_t>(lane.slot_end - lane.out.out),
                    head, lane.out.fill};
}

template <typename Step, typename Finish>
void run_lanes(StreamQueue& queue, Step step, Finish finish) {
    std::array<Lane, LANES> lanes{};
    const auto refill = [&](Lane& lane) {
        for (;;) {
            const std::size_t i{claim(queue)};
            if (i == NO_STREAM) {
                lane.stream = NO_STREAM;
                return;
            }
            start(lane, queue, i);
            if (lane.left != 0) {
                return;
            }
            finish(lane);
        }
    };
    for (Lane& lane : lanes) {
        refill(lane);
    }
    for (;;) {
        std::size_t steps{SIZE_MAX};
        std::size_t active{0};
        for (const Lane& lane : lanes) {
            if (lane.stream != NO_STREAM) {
                steps = std::min(steps, lane.left);
                ++active;
            }
        }
        if (active == 0) {
            return;
        }
        if (active == LANES) {
            for (std::size_t t{0}; t < steps; ++t) {
                for (Lane& lane : lanes) {
                    step(lane);
                }
            }
            for (Lane& lane : lanes) {
                lane.left -= steps;
            }
        } else {
            // The queue is exhausted: drain the remaining lanes.
            for (Lane& lane : lanes) {
                if (lane.stream != NO_STREAM) {
                    for (; lane.left > 0; --lane.left) {
                        step(lane);
                    }
                }
            }
        }
        for (Lane& lane : lanes) {
            if (lane.stream != NO_STREAM && lane.left == 0) {
                finish(lane);
                refill(lane);
            }
        }
    }
}

}  // namespace

std::size_t coded_stream_capacity_words(const TokenStream& stream) {
    // The final state, at most 16 bits of renormalization or prefix code per
    // token, and the raw bits.
    return (32 + 16 * stream.num_tokens + stream.raw_bits) / 64 + 1;
}

void code_token_streams(const TokenCode& code, StreamQueue& queue) {
    if (code.use_prefix_code) {
        const PrefixEntryCode* const prefix{code.prefix.data()};
        run_lanes(
            queue,
            [prefix](Lane& lane) {
                --lane.entry;
                const PrefixEntryCode c{prefix[*lane.entry]};
                if (c.raw_bits != 0) {
                    lane.raw_pos -= c.raw_bits;
                    lane.out.prepend(read_raw(lane.raw, lane.raw_pos, c.raw_bits), c.raw_bits);
                }
                lane.out.prepend(c.bits, c.depth);
            },
            [&queue](const Lane& lane) { store(lane, queue); });
        return;
    }
    const AnsEntryCode* const ans{code.ans.data()};
    const std::uint16_t* const reverse_map{code.reverse_map.data()};
    run_lanes(
        queue,
        [ans, reverse_map](Lane& lane) {
            --lane.entry;
            const AnsEntryCode c{ans[*lane.entry]};
            lane.raw_pos -= c.raw_bits;
            const auto raw{read_raw(lane.raw, lane.raw_pos, c.raw_bits)};
            std::uint32_t x{lane.state};
            const bool renormalize{(x >> (32 - ANS_LOG_TAB_SIZE)) >= c.freq};
            const unsigned shift{renormalize ? 16u : 0u};
            lane.out.prepend((static_cast<std::uint64_t>(raw) << shift) | (renormalize ? x & 0xffff : 0),
                             c.raw_bits + shift);
            x >>= shift;
            const auto v{static_cast<std::uint32_t>((x * c.reciprocal) >> RECIPROCAL_PRECISION)};
            lane.state = (v << ANS_LOG_TAB_SIZE) + reverse_map[c.slot + x - v * c.freq];
        },
        [&queue](Lane& lane) {
            lane.out.prepend(lane.state, 32);
            store(lane, queue);
        });
}

void place_coded_stream(const CodedStream& stream, std::uint8_t* dst) {
    // Little-endian hosts only: the words' bytes are stored as they lie in
    // memory, which is the LSB-first byte order of the stream.
    const std::size_t size{stream.size_bytes()};
    if (size == 0) {
        return;
    }
    const std::uint32_t head_bits{stream.head_bits};
    assert(head_bits > 0);
    if (head_bits == 64) {
        std::memcpy(dst, &stream.head, 8);
        std::memcpy(dst + 8, stream.words, 8 * stream.num_words);
        return;
    }
    std::uint64_t carry{stream.head};
    for (std::size_t j{0}; j < stream.num_words; ++j) {
        const std::uint64_t word{carry | (stream.words[j] << head_bits)};
        carry = stream.words[j] >> (64 - head_bits);
        std::memcpy(dst + 8 * j, &word, 8);
    }
    std::memcpy(dst + 8 * stream.num_words, &carry, size - 8 * stream.num_words);
}

}  // namespace cujpegxl::bitstream
