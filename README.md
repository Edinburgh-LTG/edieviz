## EdIE-viz

Herein lies the code for our paper due to appear at EMNLP LOUHI:
>
	Andreas Grivas, Beatrice Alex, Claire Grover, Richard Tobin, William Whiteley

    Not a cute stroke: Analysis of Rule- and Neural Network-Based Information Extraction Systems for Brain Radiology Reports

The code is currently being updated to reflect recent changes to the paper.

Our webtool can be accessed online [here](http://jekyll.inf.ed.ac.uk/edieviz) and an overview of its functionality
is available [here](http://jekyll.inf.ed.ac.uk/edieviz/about).

### Contents

This repository contains systems and tools for information extraction from brain radiology reports.

* **EdIE-R**: Our rule-based system
* **EdIE-BiLSTM**: Our neural network system with a character-aware BiLSTM sentence encoder
* **EdIE-BERT**: Our neural network system with a BERT encoder
* **EdIE-viz**: Contains code to run our web interface
* **paper**: Contains scripts related to extracting results and plots

### Installation

```bash
git clone https://github.com/Edinburgh-LTG/edieviz
```

For instructions on how to install and run each system refer to the README files in their corresponding folders.
The instructions **assume that you are in the corresponding folder** (eg. EdIE-N folder to install EdIE-N)
