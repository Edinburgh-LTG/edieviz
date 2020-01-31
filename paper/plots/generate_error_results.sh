#!/bin/bash

python3 error_breakdown.py \
	--conll_file ../results/EdIE-N/test/ner_tags.conll \
	--name EdIE-N_ner | tail -1 > all_results
python3 error_breakdown.py \
	--conll_file ../results/EdIE-N/test/mod_tags.conll \
	--name EdIE-N_mod | tail -1 >> all_results
python3 error_breakdown.py \
	--conll_file ../results/EdIE-R/test/ner_tags.conll \
	--name EdIE-R_ner | tail -1 >> all_results
python3 error_breakdown.py \
	--conll_file ../results/EdIE-R/test/mod_tags.conll \
	--name EdIE-R_mod | tail -1 >> all_results
echo -e "Written output to file: all_results"
