#!/bin/bash
set -e
set -o pipefail

# wget -q --show-progress http://jekyll.inf.ed.ac.uk/edieviz/static/models/EdIE-LOUHI.zip
curl http://jekyll.inf.ed.ac.uk/edieviz/static/models/EdIE-LOUHI.zip -o EdIE-LOUHI.zip -#
if [ $? -eq 0 ]; then
	echo '## Downloaded models successfully...'
	echo '## Unzipping...'
	unzip EdIE-LOUHI.zip
	if [ $? -eq 0 ]; then
		echo '## Tidying up directory...'
		rm EdIE-LOUHI.zip
		mv EdIE-LOUHI/LOUHI-EdIE-N-v1 .
		mv EdIE-LOUHI/LOUHI-ncbi-bert .
		rmdir EdIE-LOUHI
	fi
else
	echo 'Failed downloading models, exiting...'
	exit 1
fi
