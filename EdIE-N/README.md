### Installation

```bash
cd EdIE-N
python3.7 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install -e .
# EDIEN_PATH is the absolute path to the EdIE-N folder
export EDIEN_PATH="$(pwd)"
```


### Training a model

The model choices specified in the paper are reflected in the blueprint **EdIE-N-v1.yaml**.

From the EdIE-N folder, run:

```bash
edien_train --load_blueprint blueprints/EdIE-N-v1.yaml
```


### Evaluating a model on dev and test
```bash
# Evaluate on dev set
edien_eval --experiment experiments/EdIE-N-v1 --dataset dev
# Evaluate on test set
edien_eval --experiment experiments/EdIE-N-v1 --dataset test
cd eval
bash eval_conll.sh ../experiments/EdIE-N-v1/output/dev/
bash eval_conll.sh ../experiments/EdIE-N-v1/output/test/
```

Scores that populate the EdIE-N section of Table 2 are in the produced files:
```bash
echo -e "Entity tags scores\n" &&
cat ../experiments/EdIE-N-v1/output/test/ner_tags.conll.stats &&
echo -e "\nModifer tags scores\n" &&
cat ../experiments/EdIE-N-v1/output/test/mod_tags.conll.stats &&
echo -e "\nNegation scores (including error propagation from modifier/entity tagging)\n" &&
cat ../experiments/EdIE-N-v1/output/test/neg_results
```

Note: results will differ slightly if training is performed on CPU.


### Acknowledgements

We use Kemal Kurniawan's CRF implementation: [https://github.com/kmkurn/pytorch-crf](https://github.com/kmkurn/pytorch-crf)
