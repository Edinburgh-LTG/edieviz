# [EdIE-viz](http://jekyll.inf.ed.ac.uk/edieviz/)

### Installation

From the EdIE-viz folder run:
```bash
cd EdIE-viz
python3.7 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
# Install the EdIE-N package
pip install -r ../EdIE-N/requirements.txt
pip install -e ../EdIE-N
```

The webapp also relies on some environment variables that specify where to find specific resources.
In the environment variables below you need to specify the **absolute** paths to:

* the EdIE-R run-edier script (eg. ../EdIE-R/scripts/run-edier)
* the specific experiment folder (eg. ../models/LOUHI-EdIE-N-v1)
* the absolute path to the EdIE-N folder

```bash
export EDIER_RUN=..
export EDIEN_PATH=..
```

*Note*: to avoid caveats it is better to use **absolute** paths for env variables.

### Running the webapp

Assuming you followed the EdIE-N README.md instructions and downloaded the models, you can then run:

```bash
python app.py --experiment ../models/LOUHI-EdIE-N-v1/
```

Then navigate to http://localhost:3000 in your browser.
