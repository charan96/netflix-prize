import json
import pickle


class Util:
    def __init__(self):
        pass

    @staticmethod
    def pickle_object(filename, obj):
        with open(filename, 'wb') as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle_object(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def json_dump_object(filename, obj):
        with open(filename, 'w') as fh:
            json.dump(fp=fh, obj=obj, indent=4)

    @staticmethod
    def json_load_object(filename):
        with open(filename, 'r') as fh:
            return json.load(fh)

    @staticmethod
    def write_data_to_scratch_file(obj_dict):
        with open('scratch-file.out', 'w') as fh:
            for key in obj_dict:
                fh.write(key + ": " + str(obj_dict[key]))
                fh.write("\n")
