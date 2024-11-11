import warnings

from qttools import xp
from qttools.datastructures.dsbsparse import _block_view
from qttools.nevp import NEVP
from qttools.obc.obc import OBCSolver


class Spectral(OBCSolver):
    """Spectral open-boundary condition solver.

    This technique of obtaining the surface Green's function is based on
    the solution of a non-linear eigenvalue problem (NEVP), defined via
    the system-matrix blocks in the semi-infinite contacts.

    Those eigenvalues corresponding to reflected modes are filtered out,
    so that only the ones that correspond to modes that propagate into
    the leads or those that decay away from the system are retained.

    The surface Green's function is then calculated from these filtered
    eigenvalues and eigenvectors.

    Parameters
    ----------
    nevp : NEVP
        The non-linear eigenvalue problem solver to use.
    block_sections : int, optional
        The number of sections to split the periodic matrix layer into.
    min_decay : float, optional
        The decay threshold after which modes are considered to be
        evanescent.
    max_decay : float, optional
        The maximum decay to consider for evanescent modes. If not
        provided, the maximum decay is set to the logarithm of the outer
        radius of the contour annulus if applicable. Otherwise, it is
        set to log(10).
    num_ref_iterations : int, optional
        The number of refinement iterations to perform on the surface
        Green's function.
    x_ii_formula : str, optional
        The formula to use for the calculation of the surface Green's
        function. The default is via the boundary "self-energy". The
        other option is "direct". The "self-energy" formula
        corresponds to Equation (13.1) in the paper [1]_ and the "direct"
        formula corresponds to Equation (15).

    References
    ----------
    .. [1] S. Brück, et al., Efficient algorithms for large-scale
       quantum transport calculations, The Journal of Chemical Physics,
       2017.

    """

    def __init__(
        self,
        nevp: NEVP,
        block_sections: int = 1,
        min_decay: float = 1e-6,
        max_decay: float | None = None,
        num_ref_iterations: int = 2,
        x_ii_formula: str = "self-energy",
    ) -> None:
        """Initializes the spectral OBC solver."""
        self.nevp = nevp

        self.min_decay = min_decay
        if max_decay is None:
            max_decay = xp.log(getattr(nevp, "r_o", 10.0))
        self.max_decay = max_decay

        self.num_ref_iterations = num_ref_iterations
        self.block_sections = block_sections
        self.x_ii_formula = x_ii_formula

    def _extract_subblocks(
        self,
        a_ji: xp.ndarray,
        a_ii: xp.ndarray,
        a_ij: xp.ndarray,
    ) -> list[xp.ndarray]:
        """Extracts the coefficient blocks from the periodic matrix.

        Parameters
        ----------
        a_ji : xp.ndarray
            The subdiagonal block of the periodic matrix.
        a_ii : xp.ndarray
            The diagonal block of the periodic matrix.
        a_ij : xp.ndarray
            The superdiagonal block of the periodic matrix.

        Returns
        -------
        blocks : list[xp.ndarray]
            The non-zero blocks making up the matrix layer.

        """
        # Construct layer of periodic matrix in semi-infinite lead.
        layer = (a_ji, a_ii, a_ij)
        if self.block_sections == 1:
            return layer

        # Get a nested block view of the layer.
        view = _block_view(xp.concatenate(layer, axis=-1), -1, 3 * self.block_sections)
        view = _block_view(view, -2, self.block_sections)

        # Make sure that the reduction leads to periodic sublayers.
        # NOTE: I'm not 100% sure that this is really necessary.
        for i in range(self.block_sections):
            if not xp.allclose(view[0, :], xp.roll(view[i, :], -i, axis=0)):
                raise ValueError("Requested block sectioning is not periodic.")

        # Select relevant blocks and remove empty ones.
        blocks = view[0, : -self.block_sections + 1]
        blocks = [block for block in blocks if xp.any(block)]

        return blocks

    def _find_reflected_modes(
        self,
        ws: xp.ndarray,
        vs: xp.ndarray,
        a_xx: list[xp.ndarray],
    ) -> tuple[xp.ndarray, xp.ndarray]:
        """Determines which eigenvalues correspond to reflected modes.

        For the computation of the surface Green's function, only the
        eigenvalues corresponding to modes that propagate or decay into
        the leads are retained.

        Parameters
        ----------
        ws : xp.ndarray
            The eigenvalues of the NEVP.
        vs : xp.ndarray
            The eigenvectors of the NEVP.
        a_ij : xp.ndarray
            The superdiagonal contact block.
        a_ji : xp.ndarray
            The subdiagonal contact block.

        Returns
        -------
        mask : xp.ndarray
            A boolean mask indicating which eigenvalues correspond to
            reflected modes.

        """
        batchsize = a_xx[0].shape[0]
        b = len(a_xx) // 2

        # Calculate the group velocity to select propagation direction.
        # The formula can be derived by taking the derivative of the
        # polynomial eigenvalue equation with respect to k.
        # NOTE: This is actually only correct if we have no overlap.
        dEk_dk = xp.zeros_like(ws)
        with warnings.catch_warnings(
            action="ignore", category=RuntimeWarning
        ):  # Ignore division by zero.
            # TODO: Replace this for loop with a faster implementation.
            for i in range(batchsize):
                for j, w in enumerate(ws[i]):
                    a = -sum(
                        (1j * n) * w**n * a_xn[i]
                        for a_xn, n in zip(a_xx, range(-b, b + 1))
                    )
                    phi = vs[i, :, j]
                    dEk_dk[i, j] = (phi.conj().T @ a @ phi) / (phi.conj().T @ phi)

            ks = -1j * xp.log(ws)

        # Find eigenvalues that correspond to reflected modes. These are
        # modes that either propagate into the leads or decay away from
        # the system.
        return ((dEk_dk.real < 0) & (xp.abs(ks.imag) < self.min_decay)) | (
            (ks.imag < -self.min_decay) & (ks.imag > -self.max_decay)
        )

    def _upscale_eigenmodes(
        self,
        ws: xp.ndarray,
        vs: xp.ndarray,
    ) -> xp.ndarray:
        """Upscales the eigenvectors to the full periodic matrix layer.

        The extraction of subblocks and hence the solution of a higher-
        ordere, but smaller, NEVP leads to eigenvectors that are only
        defined on the reduced matrix layer. This function upscales the
        eigenvectors back to the full periodic matrix layer.

        Parameters
        ----------
        ws : xp.ndarray
            The eigenvalues of the NEVP.
        vs : xp.ndarray
            The eigenvectors of the (potentially) higher order NEVP.

        Returns
        -------
        vs : xp.ndarray
            The upscaled eigenvectors.

        """
        if self.block_sections == 1:
            return vs

        batchsize, subblock_size, num_modes = vs.shape
        block_size = subblock_size * self.block_sections

        vs_upscaled = xp.zeros((batchsize, block_size, num_modes), dtype=vs.dtype)
        for i in range(batchsize):
            for j, w in enumerate(ws[i]):
                vs_upscaled[i, :, j] = xp.kron(
                    xp.array([w**n for n in range(self.block_sections)]), vs[i, :, j]
                )
                vs_upscaled[i, :, j] /= xp.linalg.norm(vs_upscaled[i, :, j])

        return vs_upscaled

    def _compute_x_ii(
        self,
        a_ii: xp.ndarray,
        a_ij: xp.ndarray,
        a_ji: xp.ndarray,
        ws: xp.ndarray,
        vs: xp.ndarray,
        mask: xp.ndarray,
    ) -> xp.ndarray:
        """Computes the surface Green's function.

        Parameters
        ----------
        a_ii : xp.ndarray
            The diagonal block of the periodic matrix.
        a_ij : xp.ndarray
            The superdiagonal block of the periodic matrix.
        a_ji : xp.ndarray
            The subdiagonal block of the periodic matrix.
        ws : xp.ndarray
            The eigenvalues of the NEVP.
        vs : xp.ndarray
            The eigenvectors of the NEVP.
        mask : xp.ndarray
            A boolean mask indicating which eigenvalues correspond to
            reflected modes.

        Returns
        -------
        x_ii : xp.ndarray
            The surface Green's function.

        """
        if self.x_ii_formula == "self-energy":
            # Equation (13.1).
            x_ii_a_ij = xp.zeros((mask.shape[0], *a_ij.shape[-2:]), dtype=a_ij.dtype)
            for i, m in enumerate(mask):
                v = vs[i][:, m]
                w = ws[i, m]
                # Moore-Penrose pseudoinverse.
                v_inv = xp.linalg.inv(v.T @ v) @ v.T
                x_ii_a_ij[i] = v / w @ v_inv

            # Calculate the surface Green's function.
            return xp.linalg.inv(a_ii + a_ji @ x_ii_a_ij)

        if self.x_ii_formula == "direct":
            # Equation (15).
            x_ii = xp.zeros((mask.shape[0], *a_ij.shape[-2:]), dtype=a_ij.dtype)
            for i, m in enumerate(mask):
                v = vs[i][:, m]
                w = ws[i, m]
                # "More stable" computation of the surface Green's function.
                inverse = xp.linalg.inv(
                    v.conj().T @ a_ii[i] @ v + v.conj().T @ a_ji[i] @ v / w
                )
                x_ii[i] = v @ inverse @ v.conj().T

            return x_ii

        raise ValueError(
            f"Unknown formula: {self.x_ii_formula}" "Choose 'self-energy' or 'direct'."
        )

    def __call__(
        self,
        a_ii: xp.ndarray,
        a_ij: xp.ndarray,
        a_ji: xp.ndarray,
        contact: str,
        out: None | xp.ndarray = None,
    ) -> xp.ndarray | None:

        if a_ii.ndim == 2:
            a_ii = a_ii[xp.newaxis, :, :]
            a_ij = a_ij[xp.newaxis, :, :]
            a_ji = a_ji[xp.newaxis, :, :]

        blocks = self._extract_subblocks(a_ji, a_ii, a_ij)
        ws, vs = self.nevp(blocks)
        mask = self._find_reflected_modes(ws, vs, blocks)
        vs = self._upscale_eigenmodes(ws, vs)
        x_ii = self._compute_x_ii(a_ii, a_ij, a_ji, ws, vs, mask)

        # Perform a number of refinement iterations.
        for __ in range(self.num_ref_iterations):
            x_ii = xp.linalg.inv(a_ii - a_ji @ x_ii @ a_ij)

        # Return the surface Green's function.
        if out is not None:
            out[...] = x_ii
            return

        return x_ii