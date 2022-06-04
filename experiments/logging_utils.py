import os
import time


def current_time_str() -> str:
    ''' Return current time as string '''
    return time.strftime("%Y%m%d-%H%M%S")
    

def default_log_filename(logging_dir: str, make_dir=True) -> str:
    ''' Return path of log with name current time located at logging_dir, creating
        logging_dir if necessary. 
    '''
    if make_dir:
        os.makedirs(logging_dir, exist_ok=True)
    logging_filename = current_time_str()
    logging_filename = f'{logging_dir}/{logging_filename}.log'
    return logging_filename