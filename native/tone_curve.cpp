#include "tone_curve.hpp"

#include <algorithm>
#include <cmath>

namespace avcheck {

ToneCurveFilter::ToneCurveFilter(double brightness, double gamma, bool skip_clamp)
    : lut_(new uint8_t[256]), brightness_(brightness), gamma_(gamma), skip_clamp_(skip_clamp) {
    build_lookup_table();
}

ToneCurveFilter::~ToneCurveFilter() {
    delete[] lut_;
}

ToneCurveFilter::ToneCurveFilter(ToneCurveFilter&& other) noexcept
    : lut_(other.lut_), brightness_(other.brightness_), gamma_(other.gamma_), skip_clamp_(other.skip_clamp_) {
    other.lut_ = nullptr;
}

ToneCurveFilter& ToneCurveFilter::operator=(ToneCurveFilter&& other) noexcept {
    if (this != &other) {
        delete[] lut_;
        lut_ = other.lut_;
        brightness_ = other.brightness_;
        gamma_ = other.gamma_;
        skip_clamp_ = other.skip_clamp_;
        other.lut_ = nullptr;
    }
    return *this;
}

void ToneCurveFilter::build_lookup_table() {
    for (int i = 0; i < 256; ++i) {
        double normalized = static_cast<double>(i) / 255.0;
        double gamma_corrected = std::pow(normalized, 1.0 / gamma_);
        double brightened = gamma_corrected * 255.0 + brightness_ * 255.0;

        // std::lround(double) -> long is well-defined for any finite value in
        // long's range; the long -> uint8_t narrowing below wraps modulo 256
        // (well-defined for unsigned targets), which is exactly the visible
        // "bug" this flag exists to demonstrate when clamping is skipped.
        double value = skip_clamp_ ? brightened : std::clamp(brightened, 0.0, 255.0);
        lut_[i] = static_cast<uint8_t>(std::lround(value));
    }
}

void ToneCurveFilter::apply(uint8_t* data, size_t length) const {
    for (size_t i = 0; i < length; ++i) {
        data[i] = lut_[data[i]];
    }
}

}  // namespace avcheck
