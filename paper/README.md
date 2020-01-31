### How to obtain results

1. Follow instructions in EdIE-N README file to obtain conll output files
2. Execute the following to obtain scores on test set

	```bash
	cd paper
	export EDIER_RUN="../EdIE-R/scripts/run-edier"
	mkdir -p results/EdIE-R/test
	python run_edier.py --data test --outfolder results/EdIE-R/test
	bash ../EdIE-N/eval/eval_conll.sh results/EdIE-R/test

	# Assuming you already did step 1. following EdIE-N README
	mkdir -p results/EdIE-N/test
	cp ../EdIE-N/experiments/EdIE-N-v1/output/test/* results/EdIE-N/test
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

3. To generate Figure 5:

	```bash
	cd plots
	bash generate_error_results.sh
	python plot_error_breakdown.py --results all_results
	```
