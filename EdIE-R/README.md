### How to run EdIE-R

Some synthetic radiology reports are provided in example/synth-in, as well as the expected output for these files in example/synth-out.

The main command is scripts/run-edier, which takes input from STDIN and outputs to STDOUT. The run-edier command processes either files in the same XML format as those in example/synth-in or plain text. In both cases the output is in an XML format which shows the results of all levels of processing, with document labels in the <labels> element at the top and entities, relations and negation in the <standoff> element at the bottom.

The -t flag is used to indicate whether the input is xml or plain, for example:

```bash
cd EdIE-R
cat example/synth-in/record1.xml | ./scripts/run-edier -t xml 
echo 'Moderate small vessel change.' | ./scripts/run-edier -t plain
```

Multiple XML records can be concatenated and grouped inside a top-level <records> element, as in example/synth-in/allrecords.xml. The run-edier command will process these in the same way as single records:

```bash
cat example/synth-in/allrecords.xml | ./scripts/run-edier -t xml
```
