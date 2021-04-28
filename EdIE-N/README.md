# EdIE-N


## Install

```bash
export EDIEN_ROOT="$(pwd)"
cd EdIE-N
python3.7 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install -e .
# EDIEN_PATH is the absolute path to the EdIE-N folder
export EDIEN_PATH="$EDIEN_ROOT"
export EDIER_RUN="$EDIEN_ROOT/EdIE-R/scripts/run-edier"
```


## Download trained models

```bash
cd ../models
bash download_models.sh
cd EdIE-N
```


## Run released models on text from stdin

```bash
edien_eval_stdin --experiment ../models/LOUHI-EdIE-N-v1 <<< "No masses or extra-axial collections."
```

The above should print the following json containing both EdIE-N predictions and EdIE-R predictions:

```js
{
    "EdIE-N": [
        {
            "mod_tags": [ "O", "O", "O", "O", "O", "O" ],
            "negation": [ "pos", "neg", "pos", "neg", "neg", "pos" ],
            "ner_tags": [ "O", "B-tumour", "O", "B-subdural_haematoma", "I-subdural_haematoma", "O" ],
            "tokens": [ "No", "masses", "or", "extra-axial", "collections", "." ]
        }
    ],
    "EdIE-R": [
        {
            "mod_tags": [ "O", "O", "O", "O", "O", "O" ],
            "negation": [ "pos", "neg", "pos", "neg", "neg", "pos" ],
            "ner_tags": [ "O", "B-tumour", "O", "B-subdural_haematoma", "I-subdural_haematoma", "O" ],
            "tokens": [ "No", "masses", "or", "extra-axial", "collections", "." ]
        }
    ]
}
```

For the model that uses [NCBI-BERT](https://github.com/ncbi-nlp/bluebert):

```bash
edien_eval_stdin --experiment ../models/LOUHI-ncbi-bert/ <<< "No masses or extra-axial collections."
```

A better way to visualise model results, would be to run the model using the edieviz web app - see the README.md file in the EdIE-viz folder.


## Steps below require access to the Edinburgh Stroke Study dataset

Get in touch with @andreasgrv for more details agrivas at inf dot ed dot ac dot uk.


### Evaluating a model on dev and test

```bash
# Evaluate on dev set
edien_eval --experiment ../models/LOUHI-EdIE-N-v1 --dataset dev
# Evaluate on test set
edien_eval --experiment ../models/LOUHI-EdIE-N-v1 --dataset test
cd eval
bash eval_conll.sh ../../models/LOUHI-EdIE-N-v1/output/dev/
bash eval_conll.sh ../../models/LOUHI-EdIE-N-v1/output/test/
```


Scores that populate the EdIE-N section of Table 2 are in the produced files:
```bash
echo -e "Entity tags scores\n" &&
cat ../../models/LOUHI-EdIE-N-v1/output/test/ner_tags.conll.stats &&
echo -e "\nModifer tags scores\n" &&
cat ../../models/LOUHI-EdIE-N-v1/output/test/mod_tags.conll.stats &&
echo -e "\nNegation scores (including error propagation from modifier/entity tagging)\n" &&
cat ../../models/LOUHI-EdIE-N-v1/output/test/neg_results
```

Also run the same steps as above for `LOUHI-ncbi-bert`.

### Training a model

**Note: results will differ slightly if training is performed on CPU.**

In order to train a model on your own data you will need to define a data loader following the example in *edien/data/ess.py*.

The model choices specified in the paper are reflected in the blueprint **EdIE-N-v1.yaml**.

From the EdIE-N folder, run:

```bash
edien_train --load_blueprint path-to-blueprint
```


### Acknowledgements

We use Kemal Kurniawan's CRF implementation: [https://github.com/kmkurn/pytorch-crf](https://github.com/kmkurn/pytorch-crf)
