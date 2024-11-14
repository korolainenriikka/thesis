# count samples in data dir

import os
import glob

versions = [4,5]
versions.extend(list(range(8,22)))

file_count = 0
for v in versions:
    path = f'/home/riikoro/fossil_data/tooth_samples/v{v}'
    counter = len(glob.glob1(path,"*.png"))
    print(counter)
    file_count += counter

print(f'Total: {file_count}')
# 1105