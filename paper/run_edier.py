import os
import subprocess
from argparse import ArgumentParser
from xml.etree import ElementTree as ET
from xml.dom.minidom import Text, Element
from edien import EdIENPath
from edien.data.ess import EdIEDoc, EdIEDataset


def make_neg_predict_consistent(negs, ents):
    """Propagate decision of B- token to following I- tokens"""
    assert len(negs) == len(ents)
    neg_decision = None
    for i in range(len(negs)):
        assert negs[i] is not None
        if ents[i][:2] == 'B-':
            neg_decision = negs[i]
        elif ents[i][:2] == 'I-':
            if neg_decision is not None:
                negs[i] = neg_decision
        else:
            neg_decision = None
    return negs, ents


def prepare_input(texts):

    records = Element('records')

    for text in texts:
        t = Text()
        t.data = text
        record = Element('record')
        record.appendChild(t)
        records.appendChild(record)

    return records.toxml()


def get_edier_output(texts):
    EDIER_VAR = 'EDIER_RUN'
    edier = os.environ[EDIER_VAR]
    inp_str = prepare_input(texts)

    p = subprocess.Popen([edier, '-t', 'xml'],
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)

    (output, _) = p.communicate(input=inp_str.encode('utf-8'))
    root = ET.fromstring(output)

    documents = []
    for doc_tag in root.iter('document'):
        parsed_doc = EdIEDoc.from_xml(doc_tag, proc_all=True)
        documents.append(parsed_doc)

    return documents


# NOTE: this is just a HACK to get around detokenization / tokenization
def prepare_text(text):
    text = text.replace(" 's", "'s")
    text = text.replace(" `s", "`s")
    text = text.replace("P.M ", "P.M")
    return text


if __name__ == "__main__":
    parser = ArgumentParser(description='Calculate EdIE-R predictions.')

    parser.add_argument('--outfolder',
                        type=str,
                        required=True,
                        help='Path to results files')
    parser.add_argument('--data',
                        type=str,
                        required=True,
                        choices=('dev', 'test'),
                        help='Which data to apply to')
    args = parser.parse_args()

    path = EdIENPath()
    DATA_FOLDER = path.datasets_folder
    ESS_FOLDER = os.path.join(DATA_FOLDER, 'radiology', 'ess', 'train')
    ess = EdIEDataset([ESS_FOLDER],
                      os.path.join(DATA_FOLDER, 'radiology', 'ess', 'dev'),
                      os.path.join(DATA_FOLDER, 'radiology', 'ess', 'test'))
    if args.data == 'dev':
        eval_sents = [s for s in ess.dev_sents]
    elif args.data == 'test':
        eval_sents = [s for s in ess.test_sents]
    else:
        raise ValueError('This should never happen.!')
    texts = [prepare_text(s.text) for s in eval_sents]
    docs = get_edier_output(texts)

    assert len(docs) == len(eval_sents)

    for field in ['ner_tags', 'mod_tags']:
        lines = []
        for i, (gold_sent, pred_doc) in enumerate(zip(eval_sents, docs)):
            pred_sent = pred_doc.sentences[0]
            if len(pred_sent) != len(gold_sent):
                # print(i, pred_sent.ner_tags, gold_sent.ner_tags)
                continue
            lines.append(gold_sent.to_conll(getattr(pred_sent, field), field))
        filename = '%s.conll' % field
        with open(os.path.join(args.outfolder, filename), 'w') as f:
            f.write('\n'.join(lines))

        neg_lines = []
        for i, (gold_sent, pred_doc) in enumerate(zip(eval_sents, docs)):
            pred_sent = pred_doc.sentences[0]
            if len(pred_sent) != len(gold_sent):
                # print(pred_sent.tokens)
                # print(gold_sent.tokens)
                # print(i)
                continue
            sent_ner_gold = getattr(gold_sent, field)
            sent_neg_gold = gold_sent.negation

            sent_ner_preds = list(getattr(pred_sent, field))
            sent_neg_preds = list(pred_sent.negation)

            sent_neg_preds, sent_ner_preds = make_neg_predict_consistent(sent_neg_preds, sent_ner_preds)

            comb_preds = [('%sneg_%s' % (ner[:2], ner[2:])) if (neg == 'neg' and ner != 'O') else ner
                          for neg, ner in zip(sent_neg_preds, sent_ner_preds)]
            comb_gold = [('%sneg_%s' % (ner[:2], ner[2:])) if (neg == 'neg' and ner != 'O') else ner
                         for neg, ner in zip(sent_neg_gold, sent_ner_gold)]
            inlines = []
            for parts in zip(gold_sent.tokens, gold_sent.pos_tags, comb_gold, comb_preds):
                inlines.append('%s\n' % ' '.join(parts))
            chunk = ''.join(inlines)
            chunk = '%s\n' % chunk
            neg_lines.append(chunk)

        filename = 'negation_%s.conll' % field
        with open(os.path.join(args.outfolder, filename), 'w') as f:
            f.write('\n'.join(neg_lines))
