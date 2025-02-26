# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import xp

if xp.__name__ == "numpy":
    # TODO: We need CPU-compatible implementation for DSBanded kernels
    dsbanded_kernels = None
    from qttools.kernels.numba import dsbcoo as dsbcoo_kernels
    from qttools.kernels.numba import dsbcsr as dsbcsr_kernels
    from qttools.kernels.numba import dsbsparse as dsbsparse_kernels

elif xp.__name__ == "cupy":
    from qttools.kernels.triton import dsbanded as dsbanded_kernels
    from qttools.kernels.cuda import dsbcoo as dsbcoo_kernels
    from qttools.kernels.cuda import dsbcsr as dsbcsr_kernels
    from qttools.kernels.cuda import dsbsparse as dsbsparse_kernels

else:
    raise ValueError(f"Unrecognized ARRAY_MODULE '{xp.__name__}'")


__all__ = ["dsbsparse_kernels", "dsbcoo_kernels", "dsbcsr_kernels", "dsbanded_kernels", "obc_kernels"]
