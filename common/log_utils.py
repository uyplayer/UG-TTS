# -*- coding: utf-8 -*-


import os
import logging
import sys

from concurrent_log import ConcurrentTimedRotatingFileHandler
from common.path_config import log_dir

_log_level = "DEBUG"
# _log_level = "INFO"


class Logger:
    def __init__(self, logger_name='framework', path=log_dir):
        self._logger = logging.getLogger(logger_name)
        self._logger.propagate = False
        logging.root.setLevel(logging.NOTSET)

        self.log_path = path
        self.log_file_name = 'UG-TTS.log'  # 日志文件
        self.backup_count = 1  # 保留的日志数量
        # 日志输出级别
        self.console_output_level = _log_level
        self.file_output_level = _log_level
        # 日志输出格式
        pattern = '%(asctime)s - %(filename)s [Line:%(lineno)d] - %(levelname)s - %(message)s'
        self.formatter = logging.Formatter(pattern)

    def get_logger(self):
        """在logger中添加日志句柄并返回，如果logger已有句柄，则直接返回
        我们这里添加两个句柄，一个输出日志到控制台，另一个输出到日志文件。
        两个句柄的日志级别不同，在配置文件中可设置。
        """
        if not self._logger.handlers:  # 避免重复日志
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            console_handler.setLevel(self.console_output_level)
            self._logger.addHandler(console_handler)

            # 每天重新创建一个日志文件，最多保留backup_count份
            file_handler = ConcurrentTimedRotatingFileHandler(filename=os.path.join(self.log_path, self.log_file_name),
                                                              when='MIDNIGHT',
                                                              interval=1,
                                                              backupCount=self.backup_count,
                                                              delay=True,
                                                              encoding='utf-8'
                                                              )
            file_handler.setFormatter(self.formatter)
            file_handler.setLevel(self.file_output_level)
            self._logger.addHandler(file_handler)
        return self._logger


def get_logger(name):
    return Logger(logger_name=name).get_logger()
