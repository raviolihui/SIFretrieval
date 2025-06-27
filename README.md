### SIFTER algorithm sensitivity analysis and adaptation for TROPOMI sensor

## Overview

Performing a sensitivity analysis on the SIFTER's algorithm performance on retrieving SIF from TROPOMI sensor. We will implement the SIFTER algorithm using various methods and compare them. 

## Features

- **Modelling optical thicness and obtaining reflectance** Obtaining optical thickness from reference scenes and reflectance from Amazonia of the valid pixels for the SIF retrieval: See file "tau_values.py"
- **SIF retrieval with different methods** SIF retrieval with various methods. Namely Mean (see file "mean.py"), PCA (see file "PCA.py"), KPCA (see file "KPCA.py"). Each file contains different ways of modeling surface albedo such as taking the surface albedo computed directly from the Amazonia reflectance and using different polynomial albedo orders
- **Comparison** A comparison analysis of all the retrieval methods can be found in file "comparison_analysis.py"
- **Computing validation SIF value** In order to validate our retrievals we compare to the TROPOSIF dataset of already retrieved SIF. Therefore in file "TROPOSIF_mean.py" you can compute the retrieved SIF signal with the TROPOSIF algorithm on an approximation of the valid pixels. 

## Background

- This file can be potentially be addapted to retrieve SIF over other relevant areas from the TROPOMI sensor

## Data

- **Input:**
  Satellite SIF dataset from TROPOMI sensor. To obtain these files contact the owner of this repository. 
  
- **Output:**  
  Anomaly maps highlighting regions and periods of significant SIF deviations.


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/raviolihui/SIFretrieval.git
   cd SIFretrieval
   ```

2. **Install dependencies:**  
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
## Usage 

- Make sure to have relevant data in your computer and modify the "paths" when necessary
- Run the files mentioned in the features one by one to obtain results in the master thesis

#### Acknowledgements

This repository was developed as part of a research project on remote sensing-based vegetation monitoring in the Amazon. For questions, contact the repository maintainer via GitHub Issues.

  
