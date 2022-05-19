# Cancer cell line proteomic map

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Description
--

![image-20220331110835124](./graphical_abstract/Graphical_abstract.pdf)

Pan-cancer proteomic map of 949 human cell lines.
Goncalves, Poulos & Cai, et al 2022

A collaboration beteen CMRI ProCan &amp; Sanger GDSC.


Raw data processing
--
Raw mass spectrometry (MS) data were processed with DIA-NN - see associated publication for details. Code for generating final protein data matrices from the DIA-NN output is found under the "Raw_data_processing" subfolder. 


Major analyses
--
Code relating to major analyses found in the associated publication are documented in labelled subfolders in this repository. For example, the `exploratory_analysis` subfolder contains basic statistical analysis, dimensionality reduction analysis and differential analysis of the proteomic dataset. The `machine_learning` subfolder contains the code for deep learning and benchmarking related analysis.

The README file in each subfolder contains a list that summarises the figures associated with the code in that subfolder for easy navigation. 


Contact
--
For more information, please contact the study authors. Contact details are available in the associated publication. 