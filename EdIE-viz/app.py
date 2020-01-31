import os
import subprocess
import argparse
import torch
from itertools import chain
from xml.etree import ElementTree as ET
from flask import Flask, render_template, request
from mlconf import Blueprint
from edien import EdIENPath
from edien.train_utils import only_pseudorandomness_please, eval_loop
from edien.data.ess import EdIEDoc, EdIELoader
from edien.data.base import Sentence
from edier import prepare_input, get_edier_output
from reverse_proxy import ReverseProxied


class InputSentence(Sentence):

    @property
    def mod_tags(self):
        return tuple('O' for t in self.tokens)

    @property
    def ner_tags(self):
        return tuple('O' for t in self.tokens)

    @property
    def dataset(self):
        return 'edien.data.ess.EdIEDataset'


app = Flask(__name__)
app.wsgi_app = ReverseProxied(app.wsgi_app, script_name='/edieviz')
app.jinja_env.filters['zip'] = zip
bp = None
model = None


EXAMPLE = \
    "There is loss of the neuronal tissue in the left inferior frontal and " \
    "superior temporal lobes, consistent with a prior infarct. There is generalised " \
    "cerebral volume loss which appears within normal limits for the patient’s age, " \
    "with no focal element to the generalised atrophy. Major intracranial vessels " \
    "appear patent. White matter of the brain appears largely normal, with no " \
    "evidence of significant small vessel disease. No mass lesion, hydrocephalus " \
    "or extra axial collection "


@app.route('/')
def homepage():
    return render_template('home.html', example=EXAMPLE, results=dict())


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    results = dict()

    if request.method == 'POST':
        text = request.form['submit-text']

        if len(text.strip()) > 1:

            docs = get_edier_output(text)

            original_sents = tuple(chain(*(doc.sentences for doc in docs)))

            results['EdIE-R'] = [dict(tokens=sent.tokens,
                                      negation=sent.negation,
                                      ner_tags=sent.ner_tags,
                                      mod_tags=sent.mod_tags)
                                 for sent
                                 in original_sents]

            # Compute EdIE-N predictions
            sents = bp.data.encode(original_sents, bp.paths)
            preds = model.predict(sents)
            preds = bp.data.decode(preds, axis=0)

            results['EdIE-N'] = [dict(tokens=sent.tokens,
                                      ner_tags=ner_tags,
                                      mod_tags=mod_tags,
                                      negation=negation)
                                 for sent, ner_tags, mod_tags, negation
                                 in zip(original_sents,
                                        preds.ner_tags,
                                        preds.mod_tags,
                                        preds.negation)]

    return render_template('home.html', example=EXAMPLE, results=results, zip=zip)


def setup_model(experiment):
    global bp, model
    blueprint_file = os.path.join(experiment, EdIENPath.BP_FILE)
    if not os.path.isfile(blueprint_file):
        raise ValueError("Couldn't find blueprint file %s" % blueprint_file)
    conf = Blueprint.from_file(blueprint_file)
    print('Successfully loaded blueprint from %s' % blueprint_file)

    conf.paths.experiment_name = conf.name
    # Make sure we aren't trying to create vocab on dev
    conf.data.vocab_encoder.load_if_exists = True
    bp = conf.build()

    model_path = bp.paths.model_path
    print('Loading model from %s' % model_path)
    model = bp.model.load(model_path)

    if conf.device.startswith('cuda'):
        # Set default gpu
        torch.cuda.set_device(conf.device)
        only_pseudorandomness_please(conf.seed)
        model.cuda()
    else:
        only_pseudorandomness_please(conf.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dependency parser explorer')

    parser.add_argument('--experiment', type=str, required=True,
                        help='Path to folder of experiment')

    args = parser.parse_args()

    setup_model(args.experiment)
    app.run(host='0.0.0.0', port=3001, debug=True)
