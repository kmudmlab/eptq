#include <pybind11/pybind11.h>
#include "eptq_kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "EPTQ CUDA inference kernels";

    m.def("e8_forward", &e8_forward,
          "E8 lattice fused dequant + matvec + masked correction + fp16 out.\n"
          "Args: Q(int16), codebook(fp16), g, h, x, O_vals, O_col, O_row_ptr, bias\n"
          "Returns: (B, M*8) fp16");

    m.def("e8d4_forward", &e8d4_forward,
          "E8-D4 fused dequant + matvec + masked correction + fp16 out.\n"
          "Args: Q(int16), codes(fp16,512x4), g, h, x, O_vals, O_col, O_row_ptr, bias\n"
          "Returns: (B, M*8) fp16");
}
