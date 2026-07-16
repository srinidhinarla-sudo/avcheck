#pragma once

#include <cstddef>
#include <cstdint>

namespace avcheck {

// A brightness/gamma tone-curve filter, mirroring the kind of small,
// performance-critical image-processing primitive an SDK exposes to a host
// language. The 256-entry lookup table is built once at construction and
// owned by this object for its lifetime (RAII): the destructor frees it
// unconditionally, and copying is disabled so there is never more than one
// owner of a given buffer. Moves transfer ownership explicitly.
class ToneCurveFilter {
public:
    // skip_clamp intentionally disables range-clamping in the lookup table
    // build step. It exists to demonstrate, on purpose, what an unclamped
    // tone curve looks like (values wrap around mod 256) so the defect is
    // visible to AVCheck's own detectors in the Phase 5 end-to-end validation.
    ToneCurveFilter(double brightness, double gamma, bool skip_clamp = false);
    ~ToneCurveFilter();

    ToneCurveFilter(const ToneCurveFilter&) = delete;
    ToneCurveFilter& operator=(const ToneCurveFilter&) = delete;

    ToneCurveFilter(ToneCurveFilter&& other) noexcept;
    ToneCurveFilter& operator=(ToneCurveFilter&& other) noexcept;

    // Applies the lookup table in place to `length` bytes starting at `data`.
    void apply(uint8_t* data, size_t length) const;

    double brightness() const { return brightness_; }
    double gamma() const { return gamma_; }
    bool skip_clamp() const { return skip_clamp_; }

private:
    void build_lookup_table();

    uint8_t* lut_;  // owned; always exactly 256 entries, or nullptr if moved-from
    double brightness_;
    double gamma_;
    bool skip_clamp_;
};

}  // namespace avcheck
