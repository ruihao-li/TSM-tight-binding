# TSM-tight-binding

Study of various properties of topolgical semimetals, including Weyl and Dirac semimetals, with the use of tight-binding models. All the Python dependencies used in this project (so far) are contained in `requirements.txt`. 

---

### Additional notes:

- Project demo is in `TSM_TB_demo.ipynb`.
- Some necessary functions used in the demo are defined in `custom_functions.py`.
- The `plotter.py` module in Kwant is modified to enable an additional option `num_bands` (specify the number of bands to be plotted) in the `kwant.plotter.spectrum` function for 3D plots. To enable this functionality, simply replace the original file in the local directory with `plotter_mod.py` and rename it to `plotter.py`.