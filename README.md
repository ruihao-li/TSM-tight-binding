# TSM-tight-binding

Study of finite-size topolgical semimetals, including Weyl and Dirac semimetals, with the use of tight-binding models. All the Python dependencies used in this project are contained in `conda_env.yml`. Create the Conda environment with `conda env create -f conda_env.yml`.

---

## Additional notes:

- Project demo is in `TSM_TB_demo.ipynb`.
- Some necessary functions used in the demo are defined in `utils.py`.
- Note that in `utils.py` we also include a modified version of the function `kwant.plotter.spectrum()` in Kwant. The new function enables an additional option `num_bands` which specifies the number of bands to show in 3D plots.