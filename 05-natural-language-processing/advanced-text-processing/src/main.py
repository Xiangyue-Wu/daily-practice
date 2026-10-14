"""自然语言处理进阶实验：统一运行入口。"""

import argparse
import importlib
from pathlib import Path


# 项目根目录（src 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# 实验模块配置
EXPERIMENTS = {
    "similarity": ("text_similarity", "文本相似度"),
    "keywords": ("keyword_extraction", "关键词提取"),
    "edit_distance": ("edit_distance", "编辑距离与拼写纠错"),
}


def run_experiment(task):
    """运行指定实验。"""
    module_name, description = EXPERIMENTS[task]
    module_path = SRC_DIR / f"{module_name}.py"

    if not module_path.is_file():
        print(f"[待补全] {description}：缺少 {module_name}.py")
        return False

    print(f"\n{'=' * 50}")
    print(f"开始实验：{description}")
    print(f"{'=' * 50}")

    # 为每个实验单独创建结果目录
    output_dir = RESULTS_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        module = importlib.import_module(module_name)
        module.run_experiment(
            data_dir=DATA_DIR,
            output_dir=output_dir,
        )
    except Exception as error:
        print(f"[失败] {description}：{error}")
        return False

    print(f"[完成] {description}")
    print(f"结果目录：{output_dir}")
    return True


def main():
    """解析命令行参数并执行实验。"""
    parser = argparse.ArgumentParser(
        description="自然语言处理进阶实验"
    )

    parser.add_argument(
        "--task",
        choices=["all", *EXPERIMENTS.keys()],
        default="all",
        help="选择实验，默认运行全部实验",
    )

    args = parser.parse_args()

    # 默认依次执行三个实验
    tasks = (
        list(EXPERIMENTS.keys())
        if args.task == "all"
        else [args.task]
    )

    success_count = 0

    for task in tasks:
        if run_experiment(task):
            success_count += 1

    print(f"\n执行结束：成功 {success_count}/{len(tasks)} 项实验")

    if success_count != len(tasks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
