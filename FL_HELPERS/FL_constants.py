import os
import time
import h5py
import copy
import socket
import datetime
import numpy as np
from io import BytesIO
import concurrent.futures



import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger('TensorFlow').setLevel(logging.ERROR)


LOCAL_IP = "127.0.0.1"
PORT = 11111
BUFFER_SIZE = 10240
FORMAT = "utf-8"
LOG_PATH = "LOGS/FIT"

GENERAL_INFO = '⚽ [GENERAL INFO]'
SERVER_INFO = '🌎 [Server info]'
SERVER_INFO_TESTING = '🌎 [TESTING INFO]'
SERVER_INFO_TRAINING = '🌎 [TRAINING INFO]'
SERVER_INFO_CONNECTION = '🌎 [CONNECTION INFO]'


CLIENT_INFO_MODEL = '🌤️  [Model info]'
CLIENT_INFO_ERROR = '🌤️  [Error info]'
CLIENT_INFO_TESTING = '🌤️  [Testing info]'
CLIENT_INFO_VALIDATE = '🌤️  [Validation info]'
CLIENT_INFO_TRAINING = '🌤️  [Training info]'
CLIENT_INFO_CONNECTION = '🌤️  [Connection info]'


DEBUG_VERBOSE = False
DATA_INFO_VERBOSE = False
