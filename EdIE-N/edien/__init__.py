import os


class EdIENPath(object):

    BP_FILE = 'blueprint.yaml'
    MODEL_FILE = 'model.bin'
    OUT_FOLDER = 'output'

    # We set some basic paths in the following environment variables
    EDIEN_PATH = 'EDIEN_PATH'
    EXP_PATH = 'models'
    PREPROCESS_PATH = 'preprocess'
    DATA_PATH = 'datasets'
    EXT_DATA_PATH = 'data'

    """Utility class that provides easy access to common paths."""
    def __init__(self, experiment_name='default', root_folder=None):
        super(EdIENPath, self).__init__()
        self.experiment_name = experiment_name
        self.root_folder = root_folder or os.environ[self.EDIEN_PATH]

        if not os.path.isdir(self.experiment_folder):
            print('Creating directory %s' % self.experiment_folder)
            os.makedirs(self.experiment_folder)

    @property
    def experiment_folder(self):
        return os.path.join(self.experiments_folder, self.experiment_name)

    @property
    def experiments_folder(self):
        return os.path.join(self.root_folder, self.EXP_PATH)

    @property
    def datasets_folder(self):
        return os.path.join(self.root_folder, self.DATA_PATH)

    @property
    def ext_data_folder(self):
        return os.path.join(self.root_folder, self.EXT_DATA_PATH)

    @property
    def taxonomy_path(self):
        return os.path.join(self.root_folder,
                            self.PREPROCESS_PATH,
                            'data', 'taxonomy.pickle')

    @property
    def vocab_folder(self):
        return self.experiment_folder

    @property
    def output_folder(self):
        return os.path.join(self.experiment_folder, EdIENPath.OUT_FOLDER)

    @property
    def model_path(self):
        return os.path.join(self.experiment_folder, EdIENPath.MODEL_FILE)

    @property
    def blueprint_path(self):
        return os.path.join(self.experiment_folder, EdIENPath.BP_FILE)

    def for_output(self, filename):
        file_path = os.path.join(self.output_folder, filename)
        return file_path
