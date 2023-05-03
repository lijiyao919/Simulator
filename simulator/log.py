from simulator.config import *
from simulator.timer import Timer

class EpisodeFilter(logging.Filter):
    def __init__(self, ep_thred):
        self._ep_thred = ep_thred

    def filter(self, record):
        msg = record.getMessage()
        ep = int(msg.split('-')[0])
        if ep >= self._ep_thred:
            return True
        else:
            return False

class MyLogger:
    def __init__(self, logger_name):
        #create a logger
        self._logger = logging.getLogger(logger_name)

        #create a handler
        formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
        file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)

        #create a filter
        ep_f = EpisodeFilter(LOG_EP_LEVEL)

        # set up logger
        self._logger.setLevel(LOG_LEVEL)
        self._logger.addHandler(file_handler)
        self._logger.addFilter(ep_f)
        self.propagate = False

    def debug(self, message):
        self._logger.debug("%s-%s: %s", Timer.get_episode(), Timer.get_time_step(), message)

    def info(self, message):
        self._logger.info("%s-%s: %s", Timer.get_episode(), Timer.get_time_step(), message)

