import os
import csv
import glob


def process_csv_file(csv_path):
    # 创建一个25x25的二维数组，初始化为0
    matrix = [[0 for _ in range(25)] for _ in range(25)]

    # 设置对角线元素为1
    for i in range(25):
        matrix[i][i] = 1

    # 读取CSV文件
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) == 3:  # 确保每行有三个值
                try:
                    a, b, c = int(row[0]), int(row[1]), int(row[2])
                    # 设置[a,b]位置的值为c
                    if 0 <= a < 25 and 0 <= b < 25:
                        matrix[a][b] = c
                except ValueError:
                    continue  # 跳过无法转换为整数的值

    # 创建对应的TXT文件
    txt_path = os.path.splitext(csv_path)[0] + '.txt'
    with open(txt_path, 'w') as txt_file:
        for row in matrix:
            txt_file.write(' '.join(map(str, row)) + '\n')

    print(f"已处理: {csv_path} -> {txt_path}")


def main():
    # 获取当前目录下的gt文件夹中所有的CSV文件
    csv_files = glob.glob("./gt/*.csv")

    if not csv_files:
        print("在./gt/目录下没有找到CSV文件")
        return

    # 处理每个CSV文件
    for csv_path in csv_files:
        process_csv_file(csv_path)

    print(f"总共处理了 {len(csv_files)} 个文件")


if __name__ == "__main__":
    main()