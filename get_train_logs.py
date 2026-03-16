import re
with open('sst/train.py', 'r') as f:
    for i, line in enumerate(f.readlines()):
        if 'print' in line and 'loss:' in line:
            print(f"{i+1}: {line.strip()}")
