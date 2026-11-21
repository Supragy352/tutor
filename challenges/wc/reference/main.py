import sys

data = sys.stdin.buffer.read()
flags = sys.argv[1:]
if "-c" in flags:
    print(len(data))
elif "-w" in flags:
    print(len(data.split()))
else:
    print(data.count(b"\n"))
