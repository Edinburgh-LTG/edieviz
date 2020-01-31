if [ $# != 1 ]
then
	echo "Please pass conll output folder as single argument"
	exit 1
fi

# Run perl eval on all conll outputs
# $1 has the folder containing conll files
for file in $(ls "$1"/?*.conll)
do
	perl $EDIEN_PATH/eval/conlleval.pl < $file > "$file.stats"
done

# Run stats.py to aggregate results
out_folder=$(dirname "$1/a")
cat $(ls "$1"/{ner_tags.conll.stats,mod_tags.conll.stats}) > "${out_folder}/total"
python3 $EDIEN_PATH/eval/stats.py "${out_folder}/total" > "${out_folder}/results"
cat "${out_folder}/results"
echo -e "Also wrote results to: ${out_folder}/results\n"

cat $(ls "$1"/{negation_ner_tags.conll.stats,negation_mod_tags.conll.stats}) > "${out_folder}/neg_total"
python3 $EDIEN_PATH/eval/stats.py "${out_folder}/neg_total" > "${out_folder}/neg_results"
cat "${out_folder}/neg_results"

echo "Also wrote results to: ${out_folder}/neg_results"
