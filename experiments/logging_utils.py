import os
import time


def get_current_time_str() -> str:
    ''' Return current time as string '''
    return time.strftime("%Y%m%d-%H%M%S")
    

def default_make_log_filename(logging_dir: str) -> str:
    ''' Return path of log with name current time located at logging_dir, creating
        logging_dir if necessary. 
    '''
    os.makedirs(logging_dir, exist_ok=True)
    current_time_str = get_current_time_str()
    logging_filename = f'{logging_dir}/{current_time_str}.log'
    return logging_filename


