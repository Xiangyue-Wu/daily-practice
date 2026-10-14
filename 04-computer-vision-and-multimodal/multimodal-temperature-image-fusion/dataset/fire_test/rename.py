import os
import argparse


def batch_rename_advanced(folder_path, prefix="file", start_number=1, digit_count=4,
                          keep_extension=True, dry_run=False):
    """
    高级批量重命名函数

    参数:
    folder_path: 文件夹路径
    prefix: 文件名前缀
    start_number: 起始序号
    digit_count: 序号位数
    keep_extension: 是否保持原扩展名
    dry_run: 试运行模式（不实际重命名）
    """
    try:
        if not os.path.exists(folder_path):
            print(f"错误：文件夹 '{folder_path}' 不存在")
            return False

        # 获取所有文件
        files = [f for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f))]

        if not files:
            print("文件夹中没有文件")
            return True

        files.sort()

        print(f"找到 {len(files)} 个文件")

        if dry_run:
            print("试运行模式（不实际重命名文件）:")

        # 重命名操作
        for i, filename in enumerate(files, start_number):
            if keep_extension:
                file_extension = os.path.splitext(filename)[1]
            else:
                file_extension = ""

            # 格式化序号（如：0001, 0002等）
            number_format = f"{i:0{digit_count}d}"
            new_filename = f"{prefix}_{number_format}{file_extension}"

            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)

            if dry_run:
                print(f"[试运行] {filename} -> {new_filename}")
            else:
                # 检查新文件名是否已存在
                if os.path.exists(new_file_path):
                    print(f"警告：文件 {new_filename} 已存在，跳过重命名 {filename}")
                    continue

                os.rename(old_file_path, new_file_path)
                print(f"重命名: {filename} -> {new_filename}")

        if not dry_run:
            print("批量重命名完成！")

        return True

    except Exception as e:
        print(f"发生错误: {e}")
        return False


def main_advanced():
    parser = argparse.ArgumentParser(description='文件批量重命名工具')
    parser.add_argument('--folder', default="./labels", help='文件夹路径')
    parser.add_argument('-p', '--prefix', default='val_img', help='文件名前缀（默认: file）')
    parser.add_argument('-s', '--start', type=int, default=1, help='起始序号（默认: 1）')
    parser.add_argument('-d', '--digits', type=int, default=4, help='序号位数（默认: 4）')
    parser.add_argument('--dry-run', action='store_true', help='试运行模式')

    args = parser.parse_args()

    batch_rename_advanced(
        folder_path=args.folder,
        prefix=args.prefix,
        start_number=args.start,
        digit_count=args.digits,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    # 使用命令行参数
    main_advanced()

    # # 或者使用交互式输入
    # folder_path = input("请输入文件夹路径: ").strip()
    # prefix = input("请输入文件名前缀(默认为file): ").strip() or "file"
    # start_num = input("请输入起始序号(默认为1): ").strip()
    # start_num = int(start_num) if start_num.isdigit() else 1
    # digits = input("请输入序号位数(默认为4): ").strip()
    # digits = int(digits) if digits.isdigit() else 4

    # batch_rename_advanced(folder_path, prefix, start_num, digits)