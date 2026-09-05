import torch
from config import cfg
from dataset import build_dataloaders, build_datasets
from model import build_feature_extractor_model, build_scratch_model, unfreeze_layer4
from train import train_model
from utils import plot_history, save_to_csv


def main():
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.SEED)
    if cfg.use_gpu:
        torch.cuda.manual_seed_all(cfg.SEED)

    train_set, val_set, test_set = build_datasets()

    all_results = []

    # 实验一：从头训练 ResNet18
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set
    )
    scratch_model = build_scratch_model().to(cfg.device)
    scratch_optimizer = torch.optim.SGD(
        scratch_model.parameters(),
        lr=cfg.scratch_learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )

    scratch_model, scratch_history, scratch_result = train_model(
        name="scratch",
        model=scratch_model,
        train_scope="all",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=scratch_optimizer,
        num_epochs=cfg.SCRATCH_EPOCHS,
    )

    plot_history.plot_history(scratch_history, "scratch")
    all_results.append(scratch_result)

    del scratch_model, scratch_optimizer
    if cfg.use_gpu:
        torch.cuda.empty_cache()

    # 实验二：使用预训练模型，冻结卷积层，只训练全连接层
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set
    )
    feature_extractor_model = build_feature_extractor_model().to(cfg.device)
    feature_extractor_optimizer = torch.optim.SGD(
        feature_extractor_model.fc.parameters(),
        lr=cfg.feature_learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
    )

    feature_extractor_model, feature_extractor_history, feature_extractor_result = (
        train_model(
            name="feature_extractor",
            model=feature_extractor_model,
            train_scope="fc",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=feature_extractor_optimizer,
            num_epochs=cfg.FEATURE_EPOCHS,
        )
    )

    plot_history.plot_history(feature_extractor_history, "feature_extractor")
    all_results.append(feature_extractor_result)

    # 实验三：解冻最后一层卷积层，训练全连接层和最后一层卷积层
    del feature_extractor_optimizer
    unfreeze_layer4(feature_extractor_model)
    fine_tune_optimizer = torch.optim.SGD(
        [
            {
                "params": feature_extractor_model.layer4.parameters(),
                "lr": cfg.finetune_layer4_learning_rate,
            },
            {
                "params": feature_extractor_model.fc.parameters(),
                "lr": cfg.finetune_fc_learning_rate,
            },
        ],
        momentum=0.9,
        weight_decay=1e-4,
    )

    _, transfer_history, transfer_result = train_model(
        name="fine_tune",
        model=feature_extractor_model,
        train_scope="layer4",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=fine_tune_optimizer,
        num_epochs=cfg.FINETUNE_EPOCHS,
    )

    plot_history.plot_history(transfer_history, "fine_tune")
    all_results.append(transfer_result)

    # 保存所有结果到 CSV
    save_to_csv.save_results_to_csv(all_results)


if __name__ == "__main__":
    main()
