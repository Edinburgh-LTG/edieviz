#!/bin/bash

python3 error_breakdown.py \
    --conll_file ../results/EdIE-N-BERT/test/mod_tags.conll \
    --name EdIE-BERT_mod | tail -1 > all_results
python3 error_breakdown.py \
    --conll_file ../results/EdIE-N-BERT/test/ner_tags.conll \
    --name EdIE-BERT_find | tail -1 >> all_results
python3 error_breakdown.py \
    --conll_file ../results/EdIE-N/test/mod_tags.conll \
    --name EdIE-BiLSTM_mod | tail -1 >> all_results
python3 error_breakdown.py \
    --conll_file ../results/EdIE-N/test/ner_tags.conll \
    --name EdIE-BiLSTM_find | tail -1 >> all_results
python3 error_breakdown.py \
    --conll_file ../results/EdIE-R/test/mod_tags.conll \
    --name EdIE-R_mod | tail -1 >> all_results
python3 error_breakdown.py \
    --conll_file ../results/EdIE-R/test/ner_tags.conll \
    --name EdIE-R_find | tail -1 >> all_results
echo -e "Written output to file: all_results"
