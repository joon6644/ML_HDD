import sys
input = sys.stdin.readline

result = []

while True:
    line = input().strip()

    if not line: break

    result.append("\n" + line[1:-2])

for i in result:
    print(i.strip())