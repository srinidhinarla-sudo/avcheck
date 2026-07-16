#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>

#include "tone_curve.hpp"

namespace py = pybind11;

namespace {

py::array_t<uint8_t> apply_frame(const avcheck::ToneCurveFilter& filter,
                                  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> input) {
    py::buffer_info buf = input.request();
    py::array_t<uint8_t> output(buf.shape);
    std::memcpy(output.mutable_data(), buf.ptr, static_cast<size_t>(buf.size));
    filter.apply(static_cast<uint8_t*>(output.mutable_data()), static_cast<size_t>(buf.size));
    return output;
}

}  // namespace

PYBIND11_MODULE(avcheck_native, m) {
    m.doc() = "AVCheck native module: a C++ tone-curve/brightness frame filter (RAII, pybind11)";

    py::class_<avcheck::ToneCurveFilter>(m, "ToneCurveFilter")
        .def(py::init<double, double, bool>(), py::arg("brightness") = 0.0, py::arg("gamma") = 1.0,
             py::arg("skip_clamp") = false)
        .def_property_readonly("brightness", &avcheck::ToneCurveFilter::brightness)
        .def_property_readonly("gamma", &avcheck::ToneCurveFilter::gamma)
        .def_property_readonly("skip_clamp", &avcheck::ToneCurveFilter::skip_clamp)
        .def("apply_frame", &apply_frame, py::arg("frame"),
             "Apply the tone curve to a uint8 numpy frame of any shape, returning a new array.");
}
