import glob
import random
import pickle

from tqdm import tqdm


def load_from_dir(
    path, 
    num_data=None, 
    shuffle=True, 
    filename_load_fn=None, 
    filename_filter_fn=None, 
    verbose=True
):
    """ Helper function to load data from dataset directory
    
    Args:
        path: path to dataset directory
        num_data: number of data to load (default: None -> load all data)
        shuffle: whether to shuffle the data
        filename_load_fn: function to load data from file (default: None -> load with pickle)
        filename_filter_fn: function to filter by filenames (default: None -> load all files)
        verbose: whether to print progress bar and other status indicators
        
    Returns:
        inputs: list of tuples of corresponding data
    """
    filenames = glob.glob(path + '/*.pkl')
    if filename_filter_fn:
        filenames = list(filter(filename_filter_fn, filenames))
    if verbose:
        filenames = tqdm(filenames)
        
    inputs = []
    for filename in filenames:
        if filename_load_fn:
            _inputs = filename_load_fn(filename)
        else:
            with open(filename, 'rb') as f:
                _inputs = pickle.load(f)
        inputs.extend(_inputs)

    if shuffle:
        random.shuffle(inputs)
    if num_data:
        inputs = inputs[:num_data]
    return inputs


def compute_train_eval_split(num_data, train_ratio=0.85):
    """ Helper function to compute number of data in train and eval datasets """
    num_train = int(num_data * train_ratio)
    num_eval = num_data - num_train
    return num_train, num_eval
