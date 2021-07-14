# EdIE-viz: Information Extraction from Brain Radiology Reports

Herein lies the code for our [paper](https://www.aclweb.org/anthology/2020.louhi-1.4.pdf) we presented at [LOUHI 2020](https://louhi2020.fbk.eu/):

### [Not a cute stroke: Analysis of Rule- and Neural Network-Based Information Extraction Systems for Brain Radiology Reports](https://www.aclweb.org/anthology/2020.louhi-1.4)

### Demo: [EdIE-viz](http://jekyll.inf.ed.ac.uk/edieviz)
[Our group's](https://www.ltg.ed.ac.uk/) web visualisation, can be found online [here](http://jekyll.inf.ed.ac.uk/edieviz) and an overview of its functionality is available [here](http://jekyll.inf.ed.ac.uk/edieviz/about).

## Contents

This repository contains systems and tools for information extraction from brain radiology reports.

* **EdIE-R**: Our rule-based system
* **EdIE-BiLSTM**: Our neural network system with a character-aware BiLSTM sentence encoder
* **EdIE-BERT**: Our neural network system with a BERT encoder
* **EdIE-viz**: Contains code to run our web interface
* **paper**: Contains scripts for reproducing our results
  * [Reproduce Acute Ischemic Stroke dataset evaluation](paper/ais)
  * [K Nearest Neighbours for error analysis](paper/knn)

## Installation

```bash
git clone https://github.com/Edinburgh-LTG/edieviz
```

For instructions on how to install and run each system refer to the README files in their corresponding folders.
The instructions **assume that you are in the corresponding folder** (eg. [EdIE-N](EdIE-N/README.md) folder to install EdIE-N)

Tested on:

* Debian Linux (buster - kernel: 4.19)
* Python 3.7
* x86_64 processor (See EdIE-R/bin for other binaries)

OS dependencies:

* wget
* dos2unix


## Citation
```
    @inproceedings{grivas-etal-2020-cute,
        title = "Not a cute stroke: Analysis of Rule- and Neural Network-based Information Extraction Systems for Brain Radiology Reports",
        author = "Grivas, Andreas  and
          Alex, Beatrice  and
          Grover, Claire  and
          Tobin, Richard  and
          Whiteley, William",
        booktitle = "Proceedings of the 11th International Workshop on Health Text Mining and Information Analysis",
        month = nov,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2020.louhi-1.4",
        doi = "10.18653/v1/2020.louhi-1.4",
        pages = "24--37",
        abstract = "We present an in-depth comparison of three clinical information extraction (IE) systems designed to perform entity recognition and negation detection on brain imaging reports: EdIE-R, a bespoke rule-based system, and two neural network models, EdIE-BiLSTM and EdIE-BERT, both multi-task learning models with a BiLSTM and BERT encoder respectively. We compare our models both on an in-sample and an out-of-sample dataset containing mentions of stroke findings and draw on our error analysis to suggest improvements for effective annotation when building clinical NLP models for a new domain. Our analysis finds that our rule-based system outperforms the neural models on both datasets and seems to generalise to the out-of-sample dataset. On the other hand, the neural models do not generalise negation to the out-of-sample dataset, despite metrics on the in-sample dataset suggesting otherwise.",
    }
```

## Contact
If you have any questions or feedback we would love to hear from you, please get in touch with [Andreas Grivas](https://grv.overfit.xyz/).
