import os
from datetime import datetime

start_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
print(start_time)

start_time = datetime.today().strftime('%Y-%m-%d-%H%M')
print(start_time)