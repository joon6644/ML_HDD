import sys
input = sys.stdin.readline

def dict_to_excel():
    result = []

    while True:
        line = input().rstrip()
        if not line: break

        result.append("\n" + line[1:-2])

    for i in result:
        print(i.strip())

def excel_to_dict():
    result = []

    while True:
        line = input().rstrip()
        if not line: break
    
        result.append("\n\"" + line + "\",")

    print(*result)

if __name__ == "__main__":
    print("e: Excel to Dict")
    print("d: Dict to Excel")
    print("입력 (e/d): ", end="")
    
    mode = input().rstrip()

    if mode == 'd':
        dict_to_excel()
    elif mode == 'e':
        excel_to_dict()
    else:
        print("잘못된 입력입니다.")