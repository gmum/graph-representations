import os
import sys
import time
import json
import shutil
import logging


def save_history(train_loss, valid_loss, fold_subdirectory):
    """saves model history in fold_subdirectory"""
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    # casting to float because float32 is not JSON-serializable
    hist = {'train': [float(item) for item in train_loss], 'valid': [float(item) for item in valid_loss]}
    with open(os.path.join(fold_subdirectory, f"{timestamp}-history.json"), 'w') as f:
        json.dump(hist, f)


def save_configs(model_config, data_confg, representation_config, directory):
    """stores a copy of config files in the experiment dir"""
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    for config_file in [model_config, data_confg, representation_config]:
        filename = f"{timestamp}-{os.path.basename(config_file)}"
        shutil.copyfile(config_file, os.path.join(directory, filename))


class LoggerWrapper:
    def __init__(self, path='.'):
        """
        Wrapper for logging.
        Allows to replace sys.stderr.write so that error massages are redirected
        to sys.stdout and also saved in a file.
        use: logger = LoggerWrapper(); sys.stderr.write = logger.log_errors
        :param: path: a directory to create the log file
        """
        # count spaces so that the output is nicely indented
        self.trailing_spaces = 0

        # create the log file
        timestamp = time.strftime('%Y-%m-%d-%H-%M')
        self.filename = os.path.join(path, f'{timestamp}.log')
        try:
            os.mknod(self.filename)
        except FileExistsError:
            pass

        # configure logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-6s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=self.filename,
                            filemode='w')
        formatter = logging.Formatter('%(name)-6s: %(levelname)-8s %(message)s')

        # make a handler to redirect stuff to std.out
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)  # bacause matplotlib throws lots of debug messages
        self.console = logging.StreamHandler(sys.stdout)
        self.console.setLevel(logging.INFO)
        self.console.setFormatter(formatter)
        self.logger.addHandler(self.console)

    def log_errors(self, msg):
        """Bind this function to sys.stderr.write if you want the error massages
        to be redirected to sys.stdout and also saved in a file"""
        msg = msg.strip('\n')  # don't add extra newlines

        if msg == ' ' * len(msg):  # if you only get spaces: don't print them, but do remember
            self.trailing_spaces += len(msg)
        elif len(msg) > 1:
            self.logger.error(' ' * self.trailing_spaces + msg)
            self.trailing_spaces = 0
