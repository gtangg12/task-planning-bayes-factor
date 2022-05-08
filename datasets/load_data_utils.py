import glob
import random
import pickle


def load_from_dir(path, filter_fn=None, shuffle=True, num_data=None):
    """ Load BabyAI data from path, filtering filename by filter_fn 
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