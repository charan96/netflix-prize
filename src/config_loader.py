import json


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as fh:
            return json.load(fh)

    def get_datafiles(self):
        return self.config['data']

    def get_statefiles_dir(self):
        return self.config['statefiles']['dir']

    def get_data_dict_loc(self):
        return self.config['statefiles']['data_dict']

    def get_matrix_loc(self):
        return self.config['statefiles']['matrix']

    def get_train_data_loc(self):
        return self.config['statefiles']['train']

    def get_test_data_loc(self):
        return self.config['statefiles']['test']

    def get_tfr_data_dir_loc(self):
        return self.config['tfr_data_dir']

    def get_tfr_train_data_dir_loc(self):
        return self.config['tfr_train_dir']
	
    def get_tfr_test_data_dir_loc(self):
        return self.config['tfr_test_dir']
