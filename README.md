Biologically-informed neural networks (BINNs) are an extension of physics-informed neural networks which were introduced to discover the underlying dynamics of biological systems from sparse experimental data. 
BINNs are trained in a supervised learning framework to approximate in vitro cell biology assay experiments while respecting a generalized form of governing equation. 
In this work, we approximate the solution to population growth experiments, simulated using synthetic sigmoidal growth dynamics. 
We extend the existing BINNs framework to account for the natural heteroscedastic noise found across biology.
As such, as well as learning the solution and the governing dynamics, we also learn the underlying noise structure. 

For more information, see the associated manuscript: "A likelihood-based framework for simultaneously learning both noise and growth dynamics using biologically-informed neural networks" by Rebecca M. Crossley and Ruth E. Baker.

All of the code is implemented using Python and the open source machine learning framework, PyTorch.

The Jupyter notebook "figure_creation_lessdata_nodgdu.ipynb" demonstrates the commands needed to reproduce the results based on the synthetic data in the manuscript.
The notebook ""cr-analysis.ipynb" contains the BINN implementation for the example data from Lady Musgrove Island, which is saved as "CoralReef-Data.xlsx".
Both Jupyter notebooks rely on the functions defined in "ODE_BINN_fcts_lessdata_nodgdu.py".
Synthetic data can be found in the folder entitled "synethtic_datasets_power", whilst the outputs from BINN training can be found in the other folders. 
These are included such that users can save time and upload the best trained BINN to interrogate results, rather than having to rerun the full training process every time.  
