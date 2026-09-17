import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from config import cfg
from models.custom_resnet import model_factory
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.models import resnet18

MODEL_NAMES = ("custom_scratch", "feature_extractor", "fine_tune")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(logits, targets):
    """在 CPU 上汇总整个数据集；混淆矩阵行为真实类别、列为预测类别。"""
    count, classes = logits.shape
    if count == 0:
        raise ValueError("验证集为空。")
    predicted = logits.argmax(1)
    top5 = logits.topk(min(5, classes), dim=1).indices
    hits = (top5 == targets[:, None]).any(1)
    matrix = torch.bincount(
        targets * classes + predicted, minlength=classes**2
    ).reshape(classes, classes)
    tp = matrix.diag().double()
    support = matrix.sum(1)
    predicted_count = matrix.sum(0)
    precision = tp / predicted_count.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + predicted_count).clamp_min(1)
    metrics = {
        "samples": count,
        "errors": int((predicted != targets).sum()),
        "loss": float(torch.nn.functional.cross_entropy(logits, targets)),
        "top1_acc": float((predicted == targets).double().mean()),
        "top5_acc": float(hits.double().mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
    per_class = []
    for c in range(classes):
        per_class.append({
            "class_id": c,
            "oxford_class_id": c + 1,
            "support": int(support[c]),
            "predicted_count": int(predicted_count[c]),
            "tp": int(tp[c]),
            "fp": int(predicted_count[c] - tp[c]),
            "fn": int(support[c] - tp[c]),
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "top5_recall": float(hits[targets == c].double().mean())
            if support[c]
            else 0.0,
        })
    pairs = [
        {
            "true_class": a,
            "predicted_class": b,
            "count": int(matrix[a, b]),
            "true_support": int(support[a]),
            "error_rate_within_true_class": int(matrix[a, b]) / int(support[a]),
        }
        for a in range(classes)
        for b in range(classes)
        if a != b and matrix[a, b] > 0
    ]
    pairs.sort(
        key=lambda row: (-row["count"], row["true_class"], row["predicted_class"])
    )
    assert int(matrix.sum()) == count
    assert sum(row["count"] for row in pairs) == metrics["errors"]
    return metrics, per_class, pairs, matrix


def gallery(records, destination, title):
    """展示模型实际看到的中心裁剪区域，而非未经预处理的完整原图。"""
    if not records:
        return
    columns = 4
    rows = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 3.8), squeeze=False)
    crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        with Image.open(record["image_path"]) as image:
            ax.imshow(crop(image.convert("RGB")))
        ax.set_title(
            f"{record['image_name']}\n"
            f"True {record['true_class']} -> Pred {record['predicted_class']} "
            f"({record['confidence']:.1%})\n"
            f"Top5: {record['top5_classes']}\n"
            f"True rank: {record['true_rank']}",
            fontsize=9,
        )
    fig.suptitle(title + " | class IDs: 0-101", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def analyze(model, dataset, loader, folder, examples):
    model.eval()
    outputs, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            outputs.append(model(images.to(cfg.device)).cpu())
            labels.append(targets)
    logits, targets = torch.cat(outputs), torch.cat(labels)
    if not torch.isfinite(logits).all():
        raise ValueError("预测含 NaN 或 Inf。")
    metrics, per_class, pairs, matrix = summarize(logits, targets)
    probabilities = logits.softmax(1)
    predicted = logits.argmax(1)
    top5 = logits.topk(5, dim=1).indices
    true_scores = logits.gather(1, targets[:, None])
    # 并列分数使用同一排名：1 + 严格高于真实类别的类别数。
    ranks = (logits > true_scores).sum(1) + 1
    records = []
    for i, path in enumerate(dataset._image_files):
        top_ids = top5[i].tolist()
        records.append({
            "sample_index": i,
            "image_name": path.name,
            "image_path": str(path.resolve()),
            "true_class": int(targets[i]),
            "predicted_class": int(predicted[i]),
            "confidence": float(probabilities[i, predicted[i]]),
            "true_probability": float(probabilities[i, targets[i]]),
            "true_rank": int(ranks[i]),
            "top1_correct": int(predicted[i] == targets[i]),
            "top5_correct": int(int(targets[i]) in top_ids),
            "top5_classes": " ".join(map(str, top_ids)),
            "top5_probabilities": " ".join(
                f"{float(probabilities[i, c]):.6f}" for c in top_ids
            ),
        })
    folder.mkdir()
    write_csv(folder / "per_class_metrics.csv", per_class, list(per_class[0]))
    pair_fields = [
        "true_class",
        "predicted_class",
        "count",
        "true_support",
        "error_rate_within_true_class",
    ]
    write_csv(folder / "confused_pairs.csv", pairs, pair_fields)
    write_csv(folder / "predictions.csv", records, list(records[0]))
    with (folder / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_class / predicted_class", *range(cfg.NUM_CLASSES)])
        for c, row in enumerate(matrix.tolist()):
            writer.writerow([c, *row])
    errors = sorted(
        (r for r in records if not r["top1_correct"]),
        key=lambda r: (-r["confidence"], r["sample_index"]),
    )
    groups = {
        "confident_errors": errors[:examples],
        "top5_misses": [r for r in errors if not r["top5_correct"]][:examples],
        "frequent_confusions": [],
    }
    # 每个高频有向混淆类别对选一张，避免整个面板被单一类别占满。
    for pair in pairs[:examples]:
        groups["frequent_confusions"].append(
            next(
                r
                for r in errors
                if r["true_class"] == pair["true_class"]
                and r["predicted_class"] == pair["predicted_class"]
            )
        )
    selected = []
    for category, items in groups.items():
        gallery(items, folder / f"{category}.png", f"{folder.name}: {category}")
        selected.extend({"selection": category, **item} for item in items)
    write_csv(folder / "selected_errors.csv", selected, ["selection", *records[0]])
    return metrics, per_class, pairs, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=cfg.out_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples 必须为正整数")
    output = args.output_dir or args.checkpoint_dir / (
        "validation_analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    # 新目录，防止误覆盖历史分析。
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = Flowers102(cfg.data_dir, split="val", download=False, transform=transform)
    if len(dataset) == 0:
        raise ValueError("验证集为空")
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "split": "val",
        "samples": len(dataset),
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "class_ids": "0-based (Oxford IDs = class_id + 1); no name mapping assumed",
        "transform": str(transform),
        "torch_version": torch.__version__,
        "zero_division": 0,
        "macro_labels": list(range(cfg.NUM_CLASSES)),
        "checkpoints": {},
    }
    summary = []
    predictions = {}
    details = []
    for name in MODEL_NAMES:
        checkpoint = args.checkpoint_dir / f"{name}_best_model.pth"
        metadata["checkpoints"][name] = {
            "path": str(checkpoint.resolve()),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        model = (
            model_factory(cfg.NUM_CLASSES)
            if name == "custom_scratch"
            else resnet18(weights=None, num_classes=cfg.NUM_CLASSES)
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.to(cfg.device)
        metrics, classes, pairs, records = analyze(
            model, dataset, loader, output / name, args.examples
        )
        summary.append({"model": name, **metrics})
        predictions[name] = records
        print(
            f"{name}: val Top1={metrics['top1_acc']:.2%}, errors={metrics['errors']}",
            flush=True,
        )
        details.extend([
            f"## {name}",
            "",
            "### 最低 F1 类别",
            "",
            "| 类别 ID | 样本数 | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in sorted(classes, key=lambda r: (r["f1"], r["class_id"]))[:10]:
            details.append(
                f"| {row['class_id']} | {row['support']} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.2%} |"
            )
        details.extend([
            "",
            "### 高频有向混淆对",
            "",
            "| 真实 ID → 预测 ID | 数量 | 占真实类别比例 |",
            "|---|---:|---:|",
        ])
        for pair in pairs[:10]:
            details.append(
                f"| {pair['true_class']} → {pair['predicted_class']} | {pair['count']} | {pair['error_rate_within_true_class']:.0%} |"
            )
        for category, title in (
            ("confident_errors", "高置信度错误"),
            ("top5_misses", "Top-5 未命中"),
            ("frequent_confusions", "高频混淆对样例"),
        ):
            if (output / name / f"{category}.png").exists():
                details.extend([
                    "",
                    f"### {title}",
                    "",
                    f"![{title}](./{name}/{category}.png)",
                ])
        details.append("")
        del model
        if cfg.use_gpu:
            torch.cuda.empty_cache()
    write_csv(output / "summary.csv", summary, list(summary[0]))
    comparison = []
    for i in range(len(dataset)):
        base = predictions["feature_extractor"][i]
        fine = predictions["fine_tune"][i]
        comparison.append({
            "image_name": base["image_name"],
            "true_class": base["true_class"],
            **{
                f"{name}_pred": predictions[name][i]["predicted_class"]
                for name in MODEL_NAMES
            },
            **{
                f"{name}_correct": predictions[name][i]["top1_correct"]
                for name in MODEL_NAMES
            },
            "fine_fixed_feature_error": int(
                not base["top1_correct"] and fine["top1_correct"]
            ),
            "fine_introduced_error": int(
                base["top1_correct"] and not fine["top1_correct"]
            ),
        })
    write_csv(output / "model_comparison.csv", comparison, list(comparison[0]))
    header = [
        "# Flowers102 验证集误分类分析",
        "",
        "仅分析官方验证集；不使用测试集、不训练、不下载权重。类别 ID 为 0～101，Oxford 原始 ID 为 ID+1，未假定花名映射。",
        "",
        "## 指标汇总",
        "",
        "| 模型 | 样本 | 错误数 | Top-1 | Top-5 | Macro Precision | Macro Recall | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        header.append(
            f"| {row['model']} | {row['samples']} | {row['errors']} | {row['top1_acc']:.2%} | {row['top5_acc']:.2%} | {row['macro_precision']:.2%} | {row['macro_recall']:.2%} | {row['macro_f1']:.2%} |"
        )
    fixed = sum(r["fine_fixed_feature_error"] for r in comparison)
    introduced = sum(r["fine_introduced_error"] for r in comparison)
    header.extend([
        "",
        f"微调相对冻结特征：修正 {fixed} 张错误，新增 {introduced} 张错误，净增加 {fixed - introduced} 张正确预测。",
        "",
        "## 阅读说明",
        "",
        "- 所有 CSV 指标保存为 0～1；每类指标在全验证集混淆矩阵上计算，分母为零时记为 0。",
        "- 官方验证集每类仅 10 张，单张图片使该类 Recall 变化 10 个百分点；低分类别排名不宜过度解读。",
        "- 混淆对为有向统计，A→B 与 B→A 分开；矩阵行为真实类别、列为预测类别。",
        "- 图片是模型实际输入的中心裁剪视图；选择规则为高置信度错误、Top-5 未命中错误、高频混淆对各一例，各组可能重叠，不代表随机样本。",
        "- 图中置信度为 softmax 分数，不代表已校准的正确概率。完整原图路径、Top-5 分数见 predictions.csv。",
        "- summary.csv 为总体指标；每个模型目录包含 per_class_metrics.csv、confused_pairs.csv、confusion_matrix.csv、predictions.csv、selected_errors.csv；metadata.json 记录配置和权重 SHA256。",
        "",
    ])
    (output / "report.md").write_text("\n".join(header + details), encoding="utf-8")
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Output: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
