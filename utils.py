from types import SimpleNamespace
import numpy as np
import ipywidgets
import sys
import matplotlib.pyplot as plt
import itertools
import functools
import warnings
import kwant
from kwant import system, _common


__all__ = ["pauli", "interact", "make_slab", "plot_spectrum", "plot_wfs", "WSM_BC"]

pauli = SimpleNamespace(
    s0=np.array([[1.0, 0.0], [0.0, 1.0]]),
    sx=np.array([[0.0, 1.0], [1.0, 0.0]]),
    sy=np.array([[0.0, -1j], [1j, 0.0]]),
    sz=np.array([[1.0, 0.0], [0.0, -1.0]]),
)

pauli.s0s0 = np.kron(pauli.s0, pauli.s0)
pauli.s0sx = np.kron(pauli.s0, pauli.sx)
pauli.s0sy = np.kron(pauli.s0, pauli.sy)
pauli.s0sz = np.kron(pauli.s0, pauli.sz)
pauli.sxs0 = np.kron(pauli.sx, pauli.s0)
pauli.sxsx = np.kron(pauli.sx, pauli.sx)
pauli.sxsy = np.kron(pauli.sx, pauli.sy)
pauli.sxsz = np.kron(pauli.sx, pauli.sz)
pauli.sys0 = np.kron(pauli.sy, pauli.s0)
pauli.sysx = np.kron(pauli.sy, pauli.sx)
pauli.sysy = np.kron(pauli.sy, pauli.sy)
pauli.sysz = np.kron(pauli.sy, pauli.sz)
pauli.szs0 = np.kron(pauli.sz, pauli.s0)
pauli.szsx = np.kron(pauli.sz, pauli.sx)
pauli.szsy = np.kron(pauli.sz, pauli.sy)
pauli.szsz = np.kron(pauli.sz, pauli.sz)

#------------------------------------------------------------
### Modified Kwant's spectrum function
#------------------------------------------------------------

_p = _common.lazy_import("_plotter")

def spectrum(
    syst,
    x,
    y=None,
    params=None,
    mask=None,
    file=None,
    show=True,
    dpi=None,
    fig_size=None,
    ax=None,
    num_bands=None,
):
    """Plot the spectrum of a Hamiltonian as a function of 1 or 2 parameters

    Parameters
    ----------
    syst : `kwant.system.FiniteSystem` or callable
        If a function, then it must take named parameters and return the
        Hamiltonian as a dense matrix.
    x : pair ``(name, values)``
        Parameter to ``ham`` that will be varied. Consists of the
        parameter name, and a sequence of parameter values.
    y : pair ``(name, values)``, optional
        Used for 3D plots (same as ``x``). If provided, then the cartesian
        product of the ``x`` values and these values will be used as a grid
        over which to evaluate the spectrum.
    params : dict, optional
        The rest of the parameters to ``ham``, which will be kept constant.
    mask : callable, optional
        Takes the parameters specified by ``x`` and ``y`` and returns True
        if the spectrum should not be calculated for the given parameter
        values.
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether ``matplotlib.pyplot.show()`` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    dpi : float
        Number of pixels per inch.  If not set the ``matplotlib`` default is
        used.
    fig_size : tuple
        Figure size `(width, height)` in inches.  If not set, the default
        ``matplotlib`` value is used.
    ax : ``matplotlib.axes.Axes`` instance or `None`
        If `ax` is not `None`, no new figure is created, but the plot is done
        within the existing Axes `ax`. in this case, `file`, `show`, `dpi`
        and `fig_size` are ignored.
    num_bands : int
        Number of bands that should be plotted, only works for 2D plots. If
        None all bands are plotted.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if `ax` is not set, else None.
    """

    if not _p.mpl_available:
        raise RuntimeError(
            "matplotlib was not found, but is required " "for plot_spectrum()"
        )
    if y is not None and not _p.has3d:
        raise RuntimeError("Installed matplotlib does not support 3d plotting")

    if isinstance(syst, system.FiniteSystem):

        def ham(**kwargs):
            return syst.hamiltonian_submatrix(params=kwargs, sparse=False)

    elif callable(syst):
        ham = syst
    else:
        raise TypeError("Expected 'syst' to be a finite Kwant system " "or a function.")

    params = params or dict()
    keys = (x[0],) if y is None else (x[0], y[0])
    array_values = (x[1],) if y is None else (x[1], y[1])

    # calculate spectrum on the grid of points
    spectrum = []
    bound_ham = functools.partial(ham, **params)
    for point in itertools.product(*array_values):
        p = dict(zip(keys, point))
        if mask and mask(**p):
            spectrum.append(None)
        else:
            h_p = np.atleast_2d(bound_ham(**p))
            spectrum.append(np.linalg.eigvalsh(h_p))
    # massage masked grid points into a list of NaNs of the appropriate length
    n_eigvals = len(next(filter(lambda s: s is not None, spectrum)))
    nan_list = [np.nan] * n_eigvals
    spectrum = [nan_list if s is None else s for s in spectrum]
    # make into a numpy array and reshape
    new_shape = [len(v) for v in array_values] + [-1]
    spectrum = np.array(spectrum).reshape(new_shape)

    # set up axes
    if ax is None:
        fig = _make_figure(dpi, fig_size, use_pyplot=(file is None))
        if y is None:
            ax = fig.add_subplot(1, 1, 1)
        else:
            warnings.filterwarnings("ignore", message=r".*mouse rotation disabled.*")
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            warnings.resetwarnings()
        ax.set_xlabel(keys[0])
        if y is None:
            ax.set_ylabel("Energy")
        else:
            ax.set_ylabel(keys[1])
            ax.set_zlabel("Energy")
        ax.set_title(
            ", ".join(
                "{} = {}".format(key, value)
                for key, value in params.items()
                if not callable(value)
            )
        )
    else:
        fig = None

    # actually do the plot
    if y is None:
        ax.plot(array_values[0], spectrum)
    else:
        if not hasattr(ax, "plot_surface"):
            msg = (
                "When providing an axis for plotting over a 2D domain the "
                'axis should be created with \'projection="3d"'
            )
            raise TypeError(msg)
        # plot_surface cannot directly handle rank-3 values, so we
        # explicitly loop over the last axis
        grid = np.meshgrid(*array_values)

        # modified: added num_bands functionality
        if num_bands is None:
            for i in range(spectrum.shape[-1]):
                spec = spectrum[:, :, i].transpose()  # row-major to x-y ordering
                ax.plot_surface(*(grid + [spec]), cstride=1, rstride=1)
        else:
            mid = spectrum.shape[-1] // 2
            num_bands //= 2
            for i in range(mid - num_bands, mid + num_bands):
                spec = spectrum[:, :, i].transpose()  # row-major to x-y ordering
                ax.plot_surface(*(grid + [spec]), cstride=1, rstride=1)

    _maybe_output_fig(fig, file=file, show=show)

    return fig

def _make_figure(dpi, fig_size, use_pyplot=False):
    if "matplotlib.backends" not in sys.modules:
        warnings.warn(
            "Kwant's plotting functions have\nthe side effect of "
            "selecting the matplotlib backend. To avoid this "
            "warning,\nimport matplotlib.pyplot, "
            "matplotlib.backends or call matplotlib.use().",
            RuntimeWarning,
            stacklevel=3,
        )
    if use_pyplot:
        # We import backends and pyplot only at the last possible moment (=now)
        # because this has the side effect of selecting the matplotlib backend
        # for good.  Warn if backend has not been set yet.  This check is the
        # same as the one performed inside matplotlib.use.
        from matplotlib import pyplot

        fig = pyplot.figure()
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = _p.Figure()
        fig.canvas = FigureCanvasAgg(fig)
    if dpi is not None:
        fig.set_dpi(dpi)
    if fig_size is not None:
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])
    return fig

def _maybe_output_fig(fig, file=None, show=True):
    """Output a matplotlib figure using a given output mode.

    Parameters
    ----------
    fig : matplotlib.figure.Figure instance
        The figure to be output.
    file : string or a file object
        The name of the target file or the target file itself
        (opened for writing).
    show : bool
        Whether to call ``matplotlib.pyplot.show()``.  Only has an effect if
        not saving to a file.

    Notes
    -----
    The behavior of this function producing a file is different from that of
    matplotlib in that the `dpi` attribute of the figure is used by defaul
    instead of the matplotlib config setting.
    """
    if fig is None:
        return

    if file is not None:
        fig.canvas.print_figure(file, dpi=fig.dpi)
    elif show:
        # If there was no file provided, pyplot should already be available and
        # we can import it safely without additional warnings.
        from matplotlib import pyplot

        pyplot.show()

#------------------------------------------------------------
### Custom functions
#------------------------------------------------------------

def interact(func, params, step_size=0.05):
    """
    A convenience function for varying parameters.

    Args:
        - func: function
        - params: dict -- (Other) paramters that the function to be plotted depends on.
    """
    params_spec = {
        key: ipywidgets.FloatText(value=value, step=step_size)
        for key, value in params.items()
    }
    return ipywidgets.interactive(func, **params_spec)

def make_slab(syst, ts_dir, L):
    """
    Construct a finite system with translational symmetry in two directions.

    Args:
        - ts_dir: string -- Direction(s) in which the system possesses translational symmetry.
        - L: int -- Number of layers.

    Returns:
        - template: kwant Builder -- A finite lattice system.
    """
    if (ts_dir == "xy") or (ts_dir == "yx"):
        ts_vec = ([1, 0, 0], [0, 1, 0])
        nts_idx = 2
    elif (ts_dir == "xz") or (ts_dir == "zx"):
        ts_vec = ([1, 0, 0], [0, 0, 1])
        nts_idx = 1
    elif (ts_dir == "yz") or (ts_dir == "zy"):
        ts_vec = ([0, 1, 0], [0, 0, 1])
        nts_idx = 0
    else:
        raise ValueError("Directions must be x, y, or z.")
    template = kwant.Builder(kwant.TranslationalSymmetry(*ts_vec))
    template.fill(
        syst, shape=(lambda site: 0 <= site.pos[nts_idx] < L), start=(0, 0, 0)
    )
    return template

def plot_spectrum(
    syst, params, kx=None, ky=None, kz=None, ts_dir=None, L=None, num_bands=None, fig_size=(6,4)
):
    """
    Plot the energy spectrum of a 3D finite/infinite system (with translational symmetry in two directions).

    Args:
        - syst: kwant Builder -- An infinite lattice system.
        - params: dict -- Parameters of the system.
        - kx, ky, kz: tuple(name, values) -- The k-component(s) vs. which the spectrum is plotted.
        - ts_dir: string -- Direction(s) in which the system possesses translational symmetry.
        - L: int -- Number of layers.
        - num_bands: int --  Number of bands to be plotted (only works for 3D plots, i.e., spectrum as a function of two different k-components).

    Returns:
        - fig: matplotlib figure
    """

    if ts_dir is None and L is None:
        final_syst = kwant.wraparound.wraparound(syst).finalized()
        fixed_list = []
        var_list = []
        for item in [kx, ky, kz]:
            if isinstance(item[1], int) or isinstance(item[1], float):
                fixed_list.append(item)
            else:
                var_list.append(item)
        spectrum(
            final_syst,
            *var_list,
            params={**dict(fixed_list), **params},
            fig_size=fig_size
        )

    else:
        final_syst = kwant.wraparound.wraparound(
            make_slab(syst, ts_dir, L), coordinate_names=ts_dir
        ).finalized()
        # Drop None element
        k_list = [i for i in [kx, ky, kz] if i]
        fixed_list = []
        var_list = []
        for item in k_list:
            if isinstance(item[1], int) or isinstance(item[1], float):
                fixed_list.append(item)
            else:
                var_list.append(item)
        if len(var_list) == 1:
            spectrum(
                final_syst,
                *var_list,
                params={**dict(fixed_list), **params},
                fig_size=fig_size
            )
        elif len(var_list) == 2:
            spectrum(
                final_syst,
                *var_list,
                params={**dict(fixed_list), **params},
                fig_size=fig_size,
                num_bands=num_bands
            )

def plot_wfs(syst, params, L, kx=None, ky=None, kz=None, fig_size=(6,4)):
    """
    Plot wavefunction density (corresponding to the two lowest eigenstates) on each site of a finite slab.

    Args:
        - syst: kwant Builder -- An infinite lattice system.
        - params: dict -- Parameters of the system.
        - L: int -- Number of layers.
        - kx, ky, kz: tuple(name, values) -- The k-component(s) vs. which the spectrum is plotted.

    Returns:
        - fig: matplotlib figure -- Figure showing the two lowest eigenstates (ev0 and ev1).
    """
    if kx is None:
        ts_dir = "yz"
        fixed_list = [ky, kz]
    elif ky is None:
        ts_dir = "xz"
        fixed_list = [kx, kz]
    elif kz is None:
        ts_dir = "xy"
        fixed_list = [kx, ky]
    else:
        raise ValueError("Only finite systems are supported.")

    final_syst = kwant.wraparound.wraparound(
        make_slab(syst, ts_dir, L), coordinate_names=ts_dir
    ).finalized()
    density = kwant.operator.Density(final_syst)
    ham = final_syst.hamiltonian_submatrix(params={**dict(fixed_list), **params})
    evals, evecs = np.linalg.eigh(ham)
    # Sort according to the absolute values of energy
    evecs = evecs[:, np.argsort(np.abs(evals))]

    plt.figure(figsize=fig_size)
    # One band with positive energy and one band with negative energy due to particle-hole symmetry
    plt.plot(density(evecs[:, 0]) + density(evecs[:, 1]), label="ev0")
    plt.plot(density(evecs[:, 2]) + density(evecs[:, 3]), label="ev1")
    plt.xlabel("Layer index")
    plt.ylabel("Wavefunction density")
    plt.legend()

def WSM_BC(syst, params, ts_dir, L, band_indices, ks):
    """
    Compute the Berry curvature of a particular layer (in the y-direction) in a WSM slab using the Fukui-Hatsugai-Suzuki method.

    Args:
        - syst: kwant.Builder -- A 3D infinite lattice system.
        - params: dict -- The parameters expected by the system.
        - ts_dir: string -- Direction(s) in which the system possesses translational symmetry.
        - L: int -- Number of layers.
        - band_indices: list -- Indices of the layers on which the (total) Berry curvature is computed.
        - ks: 1D array -- Values of momentum grid to be used for Berry curvature calculation.

    Returns:
        - bc: 2D array -- Berry curvature on each square in a ks x ks grid.
    """

    syst_final = kwant.wraparound.wraparound(
        make_slab(syst, ts_dir, L), coordinate_names=ts_dir
    ).finalized()

    def target_band(ki, kj, ts_dir, band_idx):
        if ts_dir == "yz" or ts_dir == "zy":
            fixed_list = ["k_y", "k_z"]
        elif ts_dir == "xz" or ts_dir == "zx":
            fixed_list = ["k_x", "k_z"]
        elif ts_dir == "xy" or ts_dir == "yx":
            fixed_list = ["k_x", "k_y"]
        else:
            raise ValueError(
                "A two-dimensional translationally invariant system is required."
            )
        ham_mat = syst_final.hamiltonian_submatrix(
            params=dict(**{fixed_list[0]: ki, fixed_list[1]: kj}, **params),
            sparse=False,
        )
        evals, evecs = np.linalg.eigh(ham_mat)
        # sorted evecs from the lowest (negative) to highest (positive) energies
        evecs = evecs[:, np.argsort(evals)]
        return evecs[:, band_idx]

    bc = np.zeros((len(ks) - 1) * (len(ks) - 1))
    for band_idx in band_indices:

        wf_grid = np.array(
            [[target_band(ki, kj, ts_dir, band_idx) for ki in ks] for kj in ks]
        )

        F_grid = []
        for i in range(len(ks) - 1):
            for j in range(len(ks) - 1):
                S12 = np.linalg.det(
                    np.reshape(wf_grid[i, j].T.conj() @ wf_grid[i + 1, j], (1, 1))
                )
                S23 = np.linalg.det(
                    np.reshape(
                        wf_grid[i + 1, j].T.conj() @ wf_grid[i + 1, j + 1], (1, 1)
                    )
                )
                S34 = np.linalg.det(
                    np.reshape(
                        wf_grid[i + 1, j + 1].T.conj() @ wf_grid[i, j + 1], (1, 1)
                    )
                )
                S41 = np.linalg.det(
                    np.reshape(wf_grid[i, j + 1].T.conj() @ wf_grid[i, j], (1, 1))
                )
                F_grid.append(-np.imag(np.log(S12 * S23 * S34 * S41)))

        bc += np.array(F_grid)

    bc = bc.reshape(len(ks) - 1, -1)

    return bc