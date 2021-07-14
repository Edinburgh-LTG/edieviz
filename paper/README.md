# Reproducing results in the paper

## How to obtain results on the AIS Dataset (Section 5)

#### 1. Add EdIE-R binaries to your `PATH`.
**You will need to choose the binaries that match your architecture**.

Assuming you are in this directory (paper), you can run:
```bash
export PATH=$PATH:$(pwd)/../EdIE-R/bin/sys-x86-64-el7
```

#### 2. Download [AIS dataset](https://doi.org/10.1371/journal.pone.0212778.s002) from [Kim et al 2019](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0212778)

```bash
cd ../datasets/kim-2019
bash get_dataset.sh
cd ../../paper
```

#### 3. Download trained models

*If you haven't already*

```bash
cd ../models
bash download_models.sh
cd ../paper
```

#### 4. Predict labels for AIS Dataset

```bash
# Run EdIE-R
cat ../datasets/kim-2019/documents.xml | ../EdIE-R/scripts/run-edier -t xml > EdIE-R-documents.xml
# Run Neural network models
cd ../EdIE-N
# This will take some time (run models on 3024 reports)
bin/edien_test_ais --experiment ../models/LOUHI-EdIE-N-v1 --in_folder ../datasets/kim-2019/documents
bin/edien_test_ais --experiment ../models/LOUHI-ncbi-bert --in_folder ../datasets/kim-2019/documents
# Aggregate the resulting documents into a single xml file
../EdIE-R/scripts/aggregate_xml ../models/LOUHI-EdIE-N-v1/output/documents  > ../models/LOUHI-EdIE-N-v1/output/documents.xml
../EdIE-R/scripts/aggregate_xml ../models/LOUHI-ncbi-bert/output/documents  > ../models/LOUHI-ncbi-bert/output/documents.xml
cd ../paper
```

#### 5. Score the predictions

Results should be as seen in Table 4 of the paper.

```bash
cd ais
# NOTE: the first column in the paper has a typo (96.64 should be 96.57)
bash print_ais_results.sh ../EdIE-R-documents.xml
bash print_ais_results.sh ../../models/LOUHI-EdIE-N-v1/output/documents.xml
bash print_ais_results.sh ../../models/LOUHI-ncbi-bert/output/documents.xml
```

## How to obtain results on the ESS Dataset (Section 4)

#### 1. Follow instructions in EdIE-N README file to obtain conll output files
#### 2. Execute the following to obtain scores on test set

	```bash
	cd paper
	export EDIER_RUN="../EdIE-R/scripts/run-edier"
	mkdir -p results/EdIE-R/test
	python run_edier.py --data test --outfolder results/EdIE-R/test
	bash ../EdIE-N/eval/eval_conll.sh results/EdIE-R/test

	# Assuming you already did step 1. following EdIE-N README
	mkdir -p results/EdIE-N/test
	cp ../models/LOUHI-EdIE-N-v1/output/test/* results/EdIE-N/test

	mkdir -p results/EdIE-N-BERT/test
	cp ../models/LOUHI-ncbi-bert/output/test/* results/EdIE-N-BERT/test
	```

	Print EdIE-R scores

	```bash
	echo -e "Entity tags scores\n" &&
	cat results/EdIE-R/test/ner_tags.conll.stats &&
	echo -e "\nModifer tags scores\n" &&
	cat results/EdIE-R/test/mod_tags.conll.stats &&
	echo -e "\nNegation scores (including error propagation from modifier/entity tagging)\n" &&
	cat results/EdIE-R/test/neg_results
	```

	Print EdIE-N scores

	```bash
	echo -e "Entity tags scores\n" &&
	cat results/EdIE-N/test/ner_tags.conll.stats &&
	echo -e "\nModifer tags scores\n" &&
	cat results/EdIE-N/test/mod_tags.conll.stats &&
	echo -e "\nNegation scores (including error propagation from modifier/entity tagging)\n" &&
	cat results/EdIE-N/test/neg_results
	```

	Print EdIE-N-BERT scores

	```bash
	echo -e "Entity tags scores\n" &&
	cat results/EdIE-N-BERT/test/ner_tags.conll.stats &&
	echo -e "\nModifer tags scores\n" &&
	cat results/EdIE-N-BERT/test/mod_tags.conll.stats &&
	echo -e "\nNegation scores (including error propagation from modifier/entity tagging)\n" &&
	cat results/EdIE-N-BERT/test/neg_results
	```

#### 3. To generate Figure 5:

	```bash
	cd plots
	bash generate_error_results.sh
	python plot_error_breakdown.py --results all_results
	```

## Running KNN on a dataset

See [knn/README.md](knn/README.md)
