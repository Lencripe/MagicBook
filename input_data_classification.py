# Author : Dilpreet Singh <dilpreet.singh@monash.edu>

import json
import os
import numpy as np
from random import shuffle
from scipy import misc

class InputData:
    """ Wrapper to handle retrieving/batching of (train + test) input.
        **Note**: This class should be placed in the same directory as 'resized_128' folder (but not inside it!)

        Params
        ------
        one_hot : bool (default = True)
            Specifies if labels should be one-hot encoded
    """

    # Paths for the input images
    DATA_DIRECTORY = os.path.dirname(__file__) + "/resized_128/"
    TRAIN_DATA_FILE_NAME = "class_labels_train.json"
    TEST_DATA_FILE_NAME = "class_labels_test.json"

    # One hot vector mapping for image classes:
    # 0 = [1, 0]
    # 1 = [0, 1]
    ONE_HOT_MAP = {0 : np.array([1,0], dtype=np.float32), 1 : np.array([0,1], dtype=np.float32)}

    def __init__(self, one_hot=True):
        self.one_hot = one_hot

        # Load the data
        self.train_data = self._load_json_file(self.DATA_DIRECTORY + self.TRAIN_DATA_FILE_NAME, list_of_tuples=True)
        self.test_data = self._load_json_file(self.DATA_DIRECTORY + self.TEST_DATA_FILE_NAME, list_of_tuples=True)

        # Create shuffled copies - useful for retrieving random batches
        self.train_shuffled_data = self._shuffle_list(self.train_data)
        self.test_shuffled_data = self._shuffle_list(self.test_data)

        # Lists to hold loaded data for test set
        # This loading of data isn't performed here because if computation is run on the GPU we'll run out of memory
        # So data is loaded (`test_get_all_examples`) only when requested
        self.test_image_data = None
        self.test_label_data = None

        # Tracking batches
        self.current_index_train = 0
        self._train_data_length = len(self.train_data)
        self._test_data_length = len(self.test_data)

# ----- Train Data Methods -----

    @property
    def train_data_size(self):
        """Size of training data"""
        return self._train_data_length

    def train_data_next_batch(self, batch_size):
        """ Returns next batch from training set.
            **Note**: Sequential calls to this method return subsequent batches, i.e. first time it's called (assume batch_size=50) the method returns the first 50 images + labels. The next call to function (again batch_size=50), will return 50 images + labels *but* they'll be from index 50 to 100. The function will automatically reset the index once it has reached the end of the training set (see property `train_data_size`)
        """
        images, labels = self._get_batch(self.train_data, batch_size, self.current_index_train)

        self.current_index_train = self.current_index_train + batch_size

        # Reset index if we have exhausted all the images
        if(self.current_index_train >= self.train_data_size):
            self.current_index_train = 0

        return (images, labels)

    def train_data_random_batch(self, batch_size):
        """ Returns a random batch (of size 'batch_size') of images from training data.
            **Note**: This method doesn't impact the sequential batches returned by `train_data_next_batch`
        """
        rand_from_index = np.random.randint(0, self.train_data_size - batch_size)
        return self._get_batch(self.train_shuffled_data, batch_size, rand_from_index)

# ----- Test Data Methods-----

    @property
    def test_data_size(self):
        """Size of test data"""
        return self._test_data_length

    def test_data_all(self):
        """ Returns all data in the test set.
            **Note**: Loading all data requires a significant amount of memory. If you are running this on a GPU it's likely that you'll run out! Use `test_data_ramdom_batch` to retrieve smaller test sets
        """
        if self.test_image_data is None:
            self.test_image_data, self.test_label_data = self._get_batch(self.test_data, self.test_data_size, 0)
        
        return (self.test_image_data, self.test_label_data)

    def test_data_random_batch(self, batch_size):
        """Returns a random batch (of size 'batch_size') of images from test data."""
        rand_from_index = np.random.randint(0, self.test_data_size - batch_size)
        return self._get_batch(self.test_shuffled_data, batch_size, rand_from_index)

# ----- Utility Methods -----

    def _get_batch(self, data, batch_size, from_index):
        to_index = from_index + batch_size

        batch = data[from_index : to_index]
        image_data = []
        label_data = []

        for (image_path, image_class) in batch:
            path = self.DATA_DIRECTORY + image_path
            image_data.append(self._flattened_image(path))

            if self.one_hot:
                label_data.append(self.ONE_HOT_MAP[image_class])
            else:
                label_data.append(image_class)

        return (image_data, label_data)

    def _flattened_image(self, path):
        image = misc.imread(path)
        image = np.array(image, dtype=np.float32)
        return image.flatten()

    def _shuffle_list(self, vals):
        """Returns a shuffled copy of list (without altering the original)"""
        # Make a copy of the list
        shuffled_vals = list(vals)
        shuffle(shuffled_vals)
        return shuffled_vals

    def _load_json_file(self, file_path, list_of_tuples=False):
        """Returns a dictionary (or a list of tuples) representation of the json file"""
        try:
            json_dict = json.load(open(file_path))
            if list_of_tuples:
                return list(json_dict.items())
            else:
                return json_dict

        except FileNotFoundError:
            raise Exception("Could not find file at path: {}".format(file_path))