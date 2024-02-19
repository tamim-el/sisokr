# Sketch-In-Sketch-Out-Accelerating-both-Learning-and-Inference-for-Structured-Prediction-with-Kernels

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

Methods: contains 2 files:
- Sketch.py contains a python class to implement sub-sampling (Rudi et al., 2015) and p-sparsified (El Ahmad et al., 2022) sketches
- SketchedIOKR.py contains a python class to implement IOKR, SIOKR, ISOKR and SISOKR models

Utils: contains 2 python files:
- load_data.py where data loading function is implemented

## Use

RUN python files:
- run_bibtex.py: reproduces all results reported in the paper on Bibtex dataset with:
  python run_bibtex.py
