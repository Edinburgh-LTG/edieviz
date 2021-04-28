#!/bin/bash
set -e
set -o pipefail

FNAME="kim-2019.csv"

wget -O $FNAME -q --show-progress https://doi.org/10.1371/journal.pone.0212778.s002

if [ $? -eq 0 ]; then
	echo '## Downloaded AIS dataset successfully...'
else
	echo 'Failed downloading AIS dataset, exiting...'
	exit 1
fi

# Simple preprocessing to get simple sentences, one per line
cat $FNAME |
   	dos2unix |
   	cut -f4 -d , |
   	tail -n +2 |
   	sed 's/"//g' |
   	sed '/^[[:space:]]*$/d' > sentences.txt

if [ $? -eq 0 ]; then
	echo '## Written AIS sentences to sentences.txt...'
else
	echo 'Error writing sentences, exiting...'
	exit 1
fi

cat $FNAME |
	dos2unix |
	sed -r 's/([0-9]+,[0-9]+,[0-9]+,)([^"].*)$/\1"\2"/' |
	sed -r 's/\&/\&amp;/g' |
	sed -r 's/</\&lt;/g' |
	sed -r 's/^([0-9]+),([0-9]+),([0-9]+),"/<record type="CT" order_item="CT Head" origid="\2" goldlabel="\3">\nClinical Details\n\nReport\n\n/' |
	sed -r 's/\"$/\n<\/record>\n/' |
	sed -r 's/,ID,Label,Text/<records>/' |
	(cat - ; echo '</records>' ) > documents.xml

if [ $? -eq 0 ]; then
	echo '## Written AIS documents in xml format to documents.xml...'
else
	echo 'Error writing documents, exiting...'
	exit 1
fi

echo '## Processing documents with EdIE-R...'
# Run EdIE-R on the documents
cat documents.xml | ../../EdIE-R/scripts/run-edier -t xml > edier-documents.xml

if [ $? -eq 0 ]; then
	echo '## Written AIS documents with EdIE-R processing in xml format to edier-documents.xml...'
else
	echo 'Error processing documents with EdIE-R, exiting...'
	exit 1
fi

cat edier-documents.xml |
	../../EdIE-R/bin/sys-x86-64-el7/lxreplace -q "relations|ents|@choice" -t "" |
	../../EdIE-R/bin/sys-x86-64-el7/lxreplace -q ent -t "&children;" |
	../../EdIE-R/scripts/makeannotationdata-isch -d documents > /dev/null 2>&1

rm -f documents/*[0-9].xml

if [ $? -eq 0 ]; then
	echo '## Split xml file to ann-xml documents in the documents folder...'
else
	echo 'Error splitting documents with EdIE-R, exiting...'
	exit 1
fi
