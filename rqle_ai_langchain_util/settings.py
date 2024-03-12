import os

# define location of the prompt config data
PROMPT_CONFIG_FOLDER = os.getenv('PROMPT_CONFIG_FOLDER', f'{os.getcwd()}/prompt_configs')

# logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_DIR = os.getenv('LOG_DIR', f'{os.getcwd()}/logs')
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default_formatter': {
            '()': 'logging.Formatter',
            'fmt': '%(levelname)s [%(asctime)s] - %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            'datefmt': LOG_DATE_FORMAT
        },
        'file_log_formatter': {
            '()': 'logging.Formatter',
            'fmt': '%(levelname)s [%(asctime)s] - %(name)s:%(lineno)d - %(message)s',
            'datefmt': LOG_DATE_FORMAT
        }
    },
    'handlers': {
        'console': {
            'formatter': 'default_formatter',
            'class': 'logging.StreamHandler',
            'level': LOG_LEVEL,
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'formatter': 'file_log_formatter',
            'class': 'logging.handlers.RotatingFileHandler',
            'level': LOG_LEVEL,
            'filename': f'{LOG_DIR}/{LOG_LEVEL.lower()}.log',
            'mode': 'a',
            'encoding': 'utf-8',
            'maxBytes': 5000000,
            'backupCount': 10,
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': LOG_LEVEL
        }
    }
}