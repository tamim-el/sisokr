# Sketch In Sketch Out: Accelerating both Learning and Inference for Structured Prediction with Kernels

This Python package contains code to use sketching on the input and output kernels for Structured Prediction.

## Installation

Necessary Python packages:
- numpy
- scikit-learn
- scipy
- liac-arff
- scikit-multilearn

To install them, run the following command:
pip install -r requirements.txt

## Details

Data: contains
- The data set Bibtex (Katakis et al., 2008)
- The data set Bookmarks (Katakis et al., 2008) (please unzip bookmarks.zip file before running run_bookmarks.py)
- The data set Mediamill (Snoek et al., 2008) (please unzip mediamill-train.zip file before running run_mediamill.py)

Methods: contains 2 files:
- Sketch.py contains Python classes to implement sub-sampling (Rudi et al., 2015) and p-sparsified (El Ahmad et al., 2022) sketches
- SketchedIOKR.py contains Python classes to implement IOKR, SIOKR, ISOKR and SISOKR models

Utils: contains 2 python files:
- load_data.py where data loading functions are implemented

## Use

RUN python files:
- run_bibtex.py: reproduces all results reported in the paper on Bibtex dataset with:
  python run_bibtex.py
- run_bookmarks.py: reproduces all results reported in the paper on Bookmarks dataset with:
  python run_bookmarks.py
- run_mediamill.py: reproduces all results reported in the paper on Mediamill dataset with:
  python run_mediamill.py
