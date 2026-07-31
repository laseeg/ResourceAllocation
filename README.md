# ResourceAllocation

This repository contains Jupyter notebooks related to the publication "Enzyme kinetics shapes the growth response of metabolic networks" by Leon Seeger, Fernanda Pinheiro and Michael Lässig. 

This work was supported by Deutsche Forschungsgemeinschaft Grant SFB 1310 (to ML). FP acknowledges funding by Human Technopole. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

The repository contains code files and intermediate results produced by some of the notebooks. Dependencies of each notebooks and the data it accesses are described in the notebook's header. The relevant code files are jupyter notebooks with a .ipynb file extension. They contain commented code for generating a physiologically parameterized ensemble of metabolic chains, solving for optimal balanced growth states and for all data figures presented in the manuscript. The notebooks and modules are implemented in Python 3. They may be executed using a local installation with the relevant Python packages. Required non-standard packages are

numpy        2.4.6
pandas       2.3.2
scipy        1.18.0
matplotlib   3.11.1
seaborn      0.13.2
requests     2.32.5

All figures apart from pathway illustrations and plots based on previously published data can be generated with the notebook "FigureCode_20260615.ipynb". To this end, please first download the relevant public datasets as cited in the manuscript and replace the file paths to match the downloaded files (BennettTableRaw.txt from Bennett et al. 2009 Supplementary tables 3 and 7, msb20209536-sup-0010-datasetev9.xlsx from Mori et al. 2021 and bi2002289_si_003.xls from Bar-Even et al. 2011). Then execute the notebooks "IterativeConstruction_SharedFunctions.ipynb" and "IterativeConstruction_NutrientRichEnsemble.ipynb", as they create simulated data analyzed by the figure code. Lastly execute the "FigureCode_20260615.ipynb". Note that figures are not created in the order they appear in the text for computational ease.

Note that the simulated data is generated from stochastic parameterizations, and among these, representative examples are randomly chosen (at times with suitable constraints to display the whole chain in the fixed field of view of individual panels). For this reason, representative illustrations will look different from the figures displayed in the manuscript, but should still exhibit the key features of the printed figures.
