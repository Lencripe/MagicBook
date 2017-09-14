# Author : Dilpreet Singh <dilpreet.singh@monash.edu>

from input_data_classification import InputData

input_data = InputData(one_hot=True)

# Get a batch of test data
# 'images' is a 2d array - each index is a flattened image (if you don't know what a 'flattened' image is then look up before continuing)
# 'labels' is a 2d array - each index is a one-hot encoding of the image class (look up one-hot encoding)
images, labels = input_data.train_data_next_batch(batch_size=50)

# This will now return images+labels from 50 to 100
images, labels = input_data.train_data_next_batch(batch_size=50)

# 100 to 150, you get the idea
images, labels = input_data.train_data_next_batch(batch_size=50)

# ----- Retrieve random batches -----

# Random batch of images for testing on training data
images, labels = input_data.train_data_random_batch(batch_size=200)

# Random batch of images for testing on test data
images, labels = input_data.test_data_random_batch(batch_size=500)