import glob
import random
import pickle


def load_from_dir(path, filter_fn=None, shuffle=True, num_data=None):
    """ Helper function to load data from dataset directory
    Args:
        path: path to dataset directory
        filter_fn: function to filter by filenames
        shuffle: whether to shuffle the data
        num_data: number of data to load
    
    Returns:
        inputs: list of tuples of corresponding data
    """
    filenames = glob.glob(path + '/*.pkl')
    if filter_fn:
        filenames = list(filter(filter_fn, filenames))
        
    inputs = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            _inputs = pickle.load(f)
        inputs.extend(_inputs)

    if shuffle:
        random.shuffle(inputs)
    if num_data:
        inputs = inputs[:num_data]

    return inputs