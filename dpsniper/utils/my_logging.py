import logging.config
import logging
import os
import sys
import json
import contextlib
import time

# LOG LEVELS
# existing:
# CRITICAL = 50
# ERROR = 40
# WARNING = 30
# INFO = 20
# DEBUG = 10
DATA = 5
logging.addLevelName(DATA, "DATA")


class OnlyDataFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == DATA


class MyLoggingConfig:
    def __init__(self):
        self.stdout_level = None
        self.file_level = None
        self.log_file = None
        self.data_file = None
        self.context = []

    def clone(self):
        config = MyLoggingConfig()
        config.stdout_level = self.stdout_level
        config.file_level = self.file_level
        config.log_file = self.log_file
        config.data_file = self.data_file
        config.context = self.context
        return config


class MyLogging:
    def __init__(self):
        self.the_log = logging.getLogger("")    # returns the root logger
        self.the_log.setLevel(DATA)  # logger should collect everything (handlers will perform filtering later)

        self.full_formatter = logging.Formatter('%(asctime)s [%(levelname)5s]: %(message)s',
                                                datefmt="%Y-%m-%d_%H-%M-%S")
        self.standard_formatter = logging.Formatter('[%(levelname)5s] %(message)s')
        self.minimal_formatter = logging.Formatter('%(message)s')

        self._config = MyLoggingConfig()

        # configure basic error logging over stdout
        self.configure("ERROR")

    def _flush_and_remove_all_handlers(self):
        to_remove = []
        for handler in self.the_log.handlers:
            handler.flush()
            handler.close()
            to_remove.append(handler)
        for handler in to_remove:
            self.the_log.removeHandler(handler)

    def _add_stdout_handler(self, level):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.setFormatter(self.standard_formatter)
        self.the_log.addHandler(stdout_handler)

    def _add_file_handler(self, level, log_file):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(self.full_formatter)
        self.the_log.addHandler(file_handler)

    def _add_data_handler(self, data_file):
        data_handler = logging.FileHandler(data_file)
        data_handler.setLevel(DATA)
        data_handler.setFormatter(self.minimal_formatter)
        data_handler.addFilter(OnlyDataFilter())
        self.the_log.addHandler(data_handler)

    def configure(self, stdout_level: str, log_file=None, file_level='INFO', data_file=None):
        """
        Reconfigure the logger, can be called multiple times at any point in the program
        """
        self._flush_and_remove_all_handlers()
        self._add_stdout_handler(stdout_level)

        self._config.stdout_level = stdout_level
        self._config.file_level = file_level
        self._config.log_file = log_file
        self._config.data_file = data_file

        if log_file is not None:
            log_dir = os.path.dirname(log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self._add_file_handler(file_level, log_file)

        if data_file is not None:
            log_dir = os.path.dirname(data_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self._add_data_handler(data_file)

    def configure_for_subprocess(self, parent_logging_config: MyLoggingConfig, process_id: int):
        """
        Configures the logger for a subprocess. The subprocess will use the same logging config as the
        parent_logging_config, but uses separate log files postfixed by "-proc{process_id}".

        Call this method in the child process right after it has been spawned.

        Args:
            parent_logging_config: The logging config of the parent process.
            process_id: An id of the spawned process.
        """
        new_log_file = None
        if parent_logging_config.log_file is not None:
            splitext = os.path.splitext(parent_logging_config.log_file)
            new_log_file = "{}-proc{}{}".format(splitext[0], process_id, splitext[1])

        new_data_file = None
        if parent_logging_config.data_file is not None:
            splitext = os.path.splitext(parent_logging_config.data_file)
            new_data_file = "{}-proc{}{}".format(splitext[0], process_id, splitext[1])

        self.configure(parent_logging_config.stdout_level, new_log_file, parent_logging_config.file_level,
                       new_data_file)
        self._config.context = parent_logging_config.context

    def get_config(self):
        return self._config

    def critical(self, msg, *args, **kwargs):
        self.the_log.critical(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.the_log.error(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.the_log.warning(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.the_log.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.the_log.debug(msg, *args, **kwargs)

    def data(self, key, value):
        d = {"ctx": self._config.context, key: value}
        self.the_log.log(DATA, json.dumps(d))

    @contextlib.contextmanager
    def temporarily_disable(self):
        self.the_log.disabled = True
        yield
        self.the_log.disabled = False


# On Windows, this line will be called in every new launched process (as processes are spawned, not forked),
# so the log is re-initialized.
# On Linux, this line will only be called once, in the root process (processes are forked)
log = MyLogging()


@contextlib.contextmanager
def log_context(key):
    log._config.context.append(key)
    yield
    log._config.context.pop()


@contextlib.contextmanager
def time_measure(key):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed = end - start
    log.data(key, elapsed)
