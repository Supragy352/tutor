import sys

data = sys.stdin.buffer.read()
if "-c" in sys.argv:
    print(len(data))
elif "-w" in sys.argv:
    print(len(data.split()))
else:
    print(data.count(b"\n"))
