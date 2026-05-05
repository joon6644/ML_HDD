import sys
input = sys.stdin.readline

result = []

while True:
    line = input().rstrip()
    if not line: break
    
    result.append("\n\"" + line + "\",")

print(*result)
