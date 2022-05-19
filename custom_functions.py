from types import SimpleNamespace
import numpy as np
import ipywidgets
import matplotlib.pyplot as plt
import kwant


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
    if (ts_dir == 'xy') or (ts_dir == 'yx'):
        ts_vec = ([1, 0, 0], [0, 1, 0])
        nts_idx = 2
    elif (ts_dir == 'xz') or (ts_dir == 'zx'):
        ts_vec = ([1, 0, 0], [0, 0, 1])
        nts_idx = 1
    elif (ts_dir == 'yz') or (ts_dir == 'zy'):
        ts_vec = ([0, 1, 0], [0, 0, 1])
        nts_idx = 0
    else:
        raise ValueError("Directions must be x, y, or z.")
    template = kwant.Builder(kwant.TranslationalSymmetry(*ts_vec))
    template.fill(syst, shape=(lambda site: 0 <= site.pos[nts_idx] < L), start=(0, 0, 0))
    return template



def plot_spectrum(syst, params, kx=None, ky=None, kz=None, ts_dir=None, L=None, num_bands=None):
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
        kwant.plotter.spectrum(final_syst, *var_list, params={**dict(fixed_list), **params}, fig_size=(8,6))

    else:
        final_syst = kwant.wraparound.wraparound(make_slab(syst, ts_dir, L), coordinate_names=ts_dir).finalized()
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
            kwant.plotter.spectrum(final_syst, *var_list, params={**dict(fixed_list), **params}, fig_size=(8,6))
        elif len(var_list) == 2:
            kwant.plotter.spectrum(final_syst, *var_list, params={**dict(fixed_list), **params}, fig_size=(8,6), num_bands=num_bands)



def plot_wfs(syst, params, kx=None, ky=None, kz=None, L=None):
    """
    Plot wavefunction density (corresponding to the two lowest eigenstates) on each site of a finite slab.

    Args: 
        - syst: kwant Builder -- An infinite lattice system.
        - params: dict -- Parameters of the system.
        - kx, ky, kz: tuple(name, values) -- The k-component(s) vs. which the spectrum is plotted.
        - L: int -- Number of layers.
    
    Returns:
        - fig: matplotlib figure -- Figure showing the two lowest eigenstates (ev0 and ev1).
    """
    if kx is None:
        ts_dir = 'yz'
        fixed_list = [ky, kz]
    elif ky is None:
        ts_dir = 'xz'
        fixed_list = [kx, kz]
    elif kz is None:
        ts_dir = 'xy'
        fixed_list = [kx, ky]
    else:
        raise ValueError("Only finite systems are supported.")
    
    final_syst = kwant.wraparound.wraparound(make_slab(syst, ts_dir, L), coordinate_names=ts_dir).finalized()
    density = kwant.operator.Density(final_syst)
    ham = final_syst.hamiltonian_submatrix(params={**dict(fixed_list), **params})
    evals, evecs = np.linalg.eigh(ham)
    # Sort according to the absolute values of energy
    evecs = evecs[:, np.argsort(np.abs(evals))]

    plt.figure(figsize=(8,6))
    plt.plot(density(evecs[:, 0]) + density(evecs[:, 1]), label='ev0')
    plt.plot(density(evecs[:, 2]) + density(evecs[:, 3]), label='ev1')
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

    syst_final = kwant.wraparound.wraparound(make_slab(syst, ts_dir, L), coordinate_names=ts_dir).finalized()

    def target_band(ki, kj, ts_dir, band_idx):
        if ts_dir == 'yz' or ts_dir == 'zy':
            fixed_list = ["k_y", "k_z"]
        elif ts_dir == 'xz' or ts_dir == 'zx':
            fixed_list = ["k_x", "k_z"]
        elif ts_dir == 'xy' or ts_dir == 'yx':
            fixed_list = ["k_x", "k_y"]
        else:
            raise ValueError("A two-dimensional translationally invariant system is required.")
        ham_mat = syst_final.hamiltonian_submatrix(params=dict(**{fixed_list[0]: ki, fixed_list[1]: kj}, **params), sparse=False)
        evals, evecs = np.linalg.eigh(ham_mat)
        # sorted evecs from the lowest (negative) to highest (positive) energies
        evecs = evecs[:, np.argsort(evals)]
        return evecs[:, band_idx]

    bc = np.zeros((len(ks) - 1) * (len(ks) - 1))
    for band_idx in band_indices:

        wf_grid = np.array([[target_band(ki, kj, ts_dir, band_idx) for ki in ks] for kj in ks])

        F_grid = []
        for i in range(len(ks) - 1): 
            for j in range(len(ks) - 1):
                S12 = np.linalg.det(np.reshape(wf_grid[i,j].T.conj() @ wf_grid[i+1,j], (1,1)))
                S23 = np.linalg.det(np.reshape(wf_grid[i+1,j].T.conj() @ wf_grid[i+1,j+1], (1,1)))
                S34 = np.linalg.det(np.reshape(wf_grid[i+1,j+1].T.conj() @ wf_grid[i,j+1], (1,1)))
                S41 = np.linalg.det(np.reshape(wf_grid[i,j+1].T.conj() @ wf_grid[i,j], (1,1)))
                F_grid.append(-np.imag(np.log(S12 * S23 * S34 * S41)))
        
        bc += np.array(F_grid)
        
    bc = bc.reshape(len(ks) - 1, -1)

    return bc