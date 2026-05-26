import json

# Read the file
with open('modify_09d.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the incorrect split
content = content.replace("split('\\\\n')", "split('\\n')")

with open('modify_09d.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed modify_09d.py")
