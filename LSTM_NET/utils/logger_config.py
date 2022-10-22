import logging
import os
import sys

def config_logger(name, log_file, file_level, console_level):
    """Configure the logger that should be used by all modules in this
    package.
    This method sets up a logger, such that all messages are written to console
    and to an extra logging file. Both outputs will be the same, except that
    a message logged to file contains the module name, where the message comes
    from.

    The implementation is based on an earlier implementation of a function I
    used in another project:

        https://git.io/fNDZJ

    Args:
        name: The name of the created logger.
        log_file: Path of the log file. If None, no logfile will be generated.
            If the logfile already exists, it will be overwritten.
        file_level: Log level for logging to log file.
        console_level: Log level for logging to console.

    Returns:
        The configured logger.
    """
    file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s' \
                                       + ' - %(module)s - %(message)s', \
                                       datefmt='%m/%d/%Y %I:%M:%S %p')
    stream_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s' \
                                         + ' - %(message)s', \
                                         datefmt='%m/%d/%Y %I:%M:%S %p')

    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir != '' and not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        if os.path.exists(log_file):
            os.remove(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(file_level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_level)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if log_file is not None:
        logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

if __name__ == '__main__':
    pass
