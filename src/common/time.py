import datetime
import time


class Time:
    timestamp = time.time()
    string_timestamp = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d-%H%M%S")
