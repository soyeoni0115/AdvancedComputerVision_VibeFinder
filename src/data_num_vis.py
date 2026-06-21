from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

import matplotlib as mpl
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SPLITS = ["train", "valid_seen", "valid_unseen"]
COLORS = ["#378ADD", "#1D9E75", "#D85A30"]


def count_split(split: str):
    split_dir = IMAGE_DIR / split
    if not split_dir.exists():
        print(f"[경고] 디렉토리 없음: {split_dir}")
        return 0, 0, []

    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    total_images = 0
    per_class_counts = []

    for class_dir in sorted(class_dirs, key=lambda d: int(d.name)):
        count = sum(
            1 for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        per_class_counts.append(count)
        total_images += count

    return total_images, len(class_dirs), per_class_counts


def main():
    stats = {}
    for split in SPLITS:
        total, n_classes, per_class = count_split(split)
        avg = total / n_classes if n_classes > 0 else 0
        stats[split] = {
            "total": total,
            "classes": n_classes,
            "avg_per_class": avg,
            "per_class": per_class,
        }
        print(f"{split:15s} | 이미지: {total:4d}장 | 클래스: {n_classes:3d}개 | 클래스당 평균: {avg:.1f}장")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Dataset Distribution", fontsize=15, fontweight="bold", y=1.02)

    x = np.arange(len(SPLITS))
    width = 0.5
    split_labels = SPLITS

    # 1. 이미지 수
    totals = [stats[s]["total"] for s in SPLITS]
    bars = axes[0].bar(x, totals, width=width, color=COLORS, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Split별 이미지 수", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(split_labels, fontsize=10)
    axes[0].set_ylabel("이미지 수 (장)")
    axes[0].set_ylim(0, max(totals) * 1.2)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0].spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, totals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                     f"{val}장", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. 클래스 수
    classes = [stats[s]["classes"] for s in SPLITS]
    bars2 = axes[1].bar(x, classes, width=width, color=COLORS, edgecolor="white", linewidth=0.8)
    axes[1].set_title("Split별 클래스 수", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(split_labels, fontsize=10)
    axes[1].set_ylabel("클래스 수 (개)")
    axes[1].set_ylim(0, max(classes) * 1.25)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars2, classes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val}개", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. 클래스당 평균 이미지 수
    avgs = [stats[s]["avg_per_class"] for s in SPLITS]
    bars3 = axes[2].bar(x, avgs, width=width, color=COLORS, edgecolor="white", linewidth=0.8)
    axes[2].set_title("클래스당 평균 이미지 수", fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(split_labels, fontsize=10)
    axes[2].set_ylabel("평균 이미지 수 (장)")
    axes[2].set_ylim(0, max(avgs) * 1.25)
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")
    axes[2].spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars3, avgs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}장", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = Path(__file__).parent / "dataset_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n저장 완료: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()