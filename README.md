# SPEER
SPEER (SPecific tissuE variant Effect predictoR) predicts tissue-specific regulatory effects of rare genetic variants using a hierarchial Bayesian model in a transfer learning framework. SPEER's advantages include:
* integration of functional genomic annotations (from DNA sequence alone) with tissue-specific gene expression
* separate predictions in each tissue while flexibly sharing information across tissues 
* computationally efficient algorithm that scales well to a large number of variants.

## Installation
To download the code:
```
git clone https://github.com/farhand7/SPEER
```
SPEER is written in Python and requires the following packages: `pandas, sklearn, numpy`.

## Usage

For a complete example of the SPEER pipeline using simulated data, see the ipython notebook
```
src/example.ipynb
```
For details on the SPEER algorithm, see
```
src/SPEER.py
```
