from tqdm import tqdm
import time

for i in tqdm(range(100000), total=100000, mininterval=10):
    time.sleep(0.0005)