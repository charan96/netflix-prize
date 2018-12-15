import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from random import shuffle
from collections import Counter

from src.util import Util


class DataMatrix:
    def __init__(self, config):
        self.config = config
        self.raw_data = None
        self.matrix = None

        self.train = None
        self.test = None

    def save_data_dict(self, data_dict):
        Util.pickle_object(self.config.get_data_dict_loc(), data_dict)

    def load_data_dict(self):
        self.raw_data = Util.load_pickle_object(self.config.get_data_dict_loc())
        return self.raw_data

    def save_matrix(self):
        Util.pickle_object(self.config.get_matrix_loc(), self.matrix)

    def load_matrix(self):
        self.matrix = Util.load_pickle_object(self.config.get_matrix_loc())
        return self.matrix

    def load_train_matrix(self):
        self.train = Util.load_pickle_object(self.config.get_train_data_loc())
        return self.train

    def load_test_matrix(self):
        self.test = Util.load_pickle_object(self.config.get_test_data_loc())
        return self.test

    def combine_datafiles(self):
        """ for each data file, read all the lines and put it into a list """
        full_data = []

        for dfile in self.config.get_datafiles():
            with open(dfile, 'r') as fh:
                for line in fh:
                    full_data.append(line)
            break

        return full_data

    def parse_raw_data(self):
        """
        each element in self.raw_data contains either a movie_id (9211:) or the (user_id, rating, date) triplet
        So, we create a dict keyed by movie_id and valued by a list of data triplets
        """
        data_dict = {}
        parent = None

        for item in self.raw_data:
            item = item.rstrip("\n")

            if item.endswith(':'):
                item = item[:-1]  # removing the colon and newline at the end
                parent = item

                data_dict[item] = []
            else:
                data_triplet = item.split(',')  # splitting item into (user_id, rating, date)
                data_dict[parent].append(data_triplet)

        self.save_data_dict(data_dict)

        return data_dict

    def build_matrix(self, save=True):
        self.load_data_dict()

        movies = self.raw_data.keys()
        users = set([user_id for data_list in self.raw_data.values() for (user_id, rating, date) in data_list])

        data_dict_list = []

        for movie in movies:
            data_dict = {user_id: int(rating) for (user_id, rating, date) in self.raw_data[movie]}

            rated_users = set(data_dict.keys())
            unrated_users = users - rated_users

            for unrated_user in unrated_users:
                data_dict[unrated_user] = 0

            data_dict_list.append(data_dict)

        self.matrix = pd.DataFrame.from_dict(data_dict_list)
        self.preprocess_matrix()

        if save:
            self.save_matrix()

        return self.matrix

    def print_matrix(self):
        print(self.matrix)

    def build_matrix_from_scratch(self):
        self.raw_data = self.combine_datafiles()
        self.parse_raw_data()
        matrix = self.build_matrix()

        return matrix

    @staticmethod
    def num_of_movies_watched(count_dict):
        count_dict_without_zero = deepcopy(count_dict)
        del count_dict_without_zero[0]

        return sum(count_dict_without_zero.values())

    def old_preprocess_matrix(self, min_num_movies_watched):
        """ Here we drop all the users who watched less than N movies """
        users_to_drop = set()

        for idx, user_row in self.matrix.iterrows():
            if DataMatrix.num_of_movies_watched(Counter(user_row)) < min_num_movies_watched:
                users_to_drop.add(idx)

        self.matrix.drop(list(users_to_drop), axis=0, inplace=True)

    def preprocess_matrix(self):
        self.matrix = self.matrix.loc[:, self.matrix.any()]     # dropping columns with only 0s
        self.matrix = self.matrix[(self.matrix.T != 0).any()]   # dropping rows with only 0s

    def train_test_split_matrix(self, split=0.2, use_saved=True):
        """
        if use_saved is true, the pickle files of train and test are loaded. If not, we split the data into train and
        test datasets.

        :param split: the test proportion of the data; has to be a float between 0 and 1
        :param use_saved: if true, we use the local pickled copies of the data
        """
        if use_saved:
            self.train = Util.load_pickle_object(self.config.get_train_data_loc())
            self.test = Util.load_pickle_object(self.config.get_test_data_loc())
        else:
            matrix_copy = self.matrix.copy(deep=True)
            split_point = int(matrix_copy.shape[1] * split)

            # here we shuffle the columns to prevent overfitting
            shuffled_columns = matrix_copy.columns.tolist();
            shuffle(shuffled_columns)
            matrix_copy = matrix_copy[shuffled_columns]

            train_matrix = matrix_copy.copy(deep=True)
            train_matrix.iloc[:, split_point:] = 0

            test_matrix = matrix_copy.copy(deep=True)
            test_matrix.iloc[:, :split_point] = 0

            Util.pickle_object(self.config.get_train_data_loc(), train_matrix)
            Util.pickle_object(self.config.get_test_data_loc(), test_matrix)

            self.train = train_matrix
            self.test = test_matrix
