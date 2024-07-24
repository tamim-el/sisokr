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

## Acknowledgments

This work was supported by the Télécom Paris research chair on Data Science and Artificial Intelligence for Digitalized Industry and Services (DSAIDIS) and the French National Research Agency (ANR) through ANR-18-CE23-0014 APi (Apprivoiser la Pré-image). Funded by the European Union. Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or European Commission. Neither the European Union nor the granting authority can be held responsible for them. This work received funding from the European Union’s Horizon Europe research and innovation program under grant agreement 101120237 (ELIAS).

## Cite

If you use this code, please cite the corresponding work:

```bibtex
@InProceedings{elahmad2024sketch,
  title = 	 { Sketch In, Sketch Out: Accelerating both Learning and Inference for Structured Prediction with Kernels },
  author =       {El Ahmad, Tamim and Brogat-Motte, Luc and Laforgue, Pierre and d'Alch\'{e}-Buc, Florence},
  booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {109--117},
  year = 	 {2024},
  editor = 	 {Dasgupta, Sanjoy and Mandt, Stephan and Li, Yingzhen},
  volume = 	 {238},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--04 May},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v238/el-ahmad24a/el-ahmad24a.pdf},
  url = 	 {https://proceedings.mlr.press/v238/el-ahmad24a.html},
  abstract = 	 { Leveraging the kernel trick in both the input and output spaces, surrogate kernel methods are a flexible and theoretically grounded solution to structured output prediction. If they provide state-of-the-art performance on complex data sets of moderate size (e.g., in chemoinformatics), these approaches however fail to scale. We propose to equip surrogate kernel methods with sketching-based approximations, applied to both the input and output feature maps. We prove excess risk bounds on the original structured prediction problem, showing how to attain close-to-optimal rates with a reduced sketch size that depends on the eigendecay of the input/output covariance operators. From a computational perspective, we show that the two approximations have distinct but complementary impacts: sketching the input kernel mostly reduces training time, while sketching the output kernel decreases the inference time. Empirically, our approach is shown to scale, achieving state-of-the-art performance on benchmark data sets where non-sketched methods are intractable. }
}
```


![Poster](https://github.com/tamim-el/sisokr/blob/main/Misc/Poster.png?raw=true)
