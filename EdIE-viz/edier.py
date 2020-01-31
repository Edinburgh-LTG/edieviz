import os
import subprocess
from xml.etree import ElementTree as ET
from edien.data.ess import EdIEDoc


def prepare_input(text):
    xml = """<records><record>%s</record></records>""" % text
    return xml


def get_edier_output(text):
    EDIER_VAR = 'EDIER_RUN'
    edier = os.environ[EDIER_VAR]
    inp_str = prepare_input(text)

    p = subprocess.Popen([edier, '-t', 'xml'],
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)

    (output, _) = p.communicate(input=inp_str.encode('utf-8'))
    root = ET.fromstring(output)
    # print(root)

    documents = []
    for doc_tag in root.iter('document'):
        parsed_doc = EdIEDoc.from_xml(doc_tag, proc_all=True)
        documents.append(parsed_doc)

    return documents
