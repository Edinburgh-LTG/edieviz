# K Nearest Neighbour annotations for error analysis

This folder contains code to reproduce sections 6.1 and 6.2 from the paper.
However, since the ESS dataset isn't widely available, we use the [NCBI Disease](http://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)
as an example of how to apply this approach to any NER dataset in CoNLL format.
Note that the NCBI dataset isn't clinical text and the sentences are generally much longer than the ones in the ESS dataset.
The results here are for illustration purposes only (more appropriate matches between model and datasets exist).

The general idea is to use a pretrained model as a contextual similarity function between tokens.
This can then help judge whether similar tokens in the dataset have been annotated in different ways.

## Installation

* Download the NCBI Corpus by running [the get_dataset.sh script](../../datasets/ncbi-disease/get_dataset.sh).
* Make sure you have followed the installation steps (first block) in the [EdIE-N folder](../../EdIE-N/README.md).

# 1. Fit the index

From within the knn folder, run:

```bash
python fit_index.py --save_index bluebert-ncbi --load_blueprint blueprint.yaml
```

This script will create a `bluebert-ncbi` folder with the faiss index fit
on the [NCBI Disease](http://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) training corpus as encoded by [BlueBERT](https://github.com/ncbi-nlp/bluebert).  This index can then be loaded for the following two scripts.

### Details of what the script does
We use a pretrained BERT encoder (BlueBERT) to encode the sentences of a training corpus into contextual representations.
For each subword token, we will obtain a vector from the encoder.
Following the BERT paper (NER section), for words that are split into many subword tokens, we use the representation of the first subtoken as a representative.
We therefore have a vector per word, that we use as a key in a KNN search index (faiss).
For each key, we keep the token's annotated label as well as additional information relating to the sentence (e.g. sentence id). 

# 2. Evaluate the faiss index

We can use the faiss index with the KNN method to obtain interpolated probabilities over the labels.
The primary purpose of this script is to visualise the resulting probabilities.
However, we can also imagine creating a classifier that takes the argmax as a prediction - so we can check how well this model predicts the dev data. The accuracy of exact matches is reported at the end.

```bash
python predict_index.py --load_index bluebert-ncbi --load_blueprint blueprint.yaml
```

# 3. Visualise Entropy Plots

This script recreates the visualisation from Figure 5.
It is run on CoNLL style output file where the gold and predicted labels are specified.
See [../../datasets/ncbi-disease/example.tsv](../../datasets/ncbi-disease/example.tsv) for an example of the format.
For each token that has been mislabelled, the script will also print the top K most similar sentences along with their distance.

```bash
python print_error_breakdown.py  --load_index bluebert-ncbi --conll_file ../../datasets/ncbi-disease/example.tsv --load_blueprint blueprint.yaml
```

# Adapting to another setting
## Using a different dataset

You should be able to point the paths in the `blueprint.yaml` file to any CoNLL format NER dataset.
You can either use an absolute path, or a relative path if you place the files under the [datasets folder](../../datasets).
Lastly, you will likely also need to adapt the `size` variable of the `ner_tags` entry in `blueprint.yaml` to reflect the number of possible NER labels.

## Using a different encoder

If you would like to change what model is used to construct the representations, you can make changes to the `blueprint.yaml` file.
Specifically, the first line `pretrained_model` can be changed to any supported model from Huggingface.
However, note that the tokenisation has only been tested for BERT models, when using different models the `TransformerCoder` class in `edien/vocab.py` may need to be adapted.
