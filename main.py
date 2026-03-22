from src.config import SEED, DEVICE, NUM_CLASSES
from src.utils import set_seed, ensure_dir
from src.datasets import create_dataloaders
from src.model import TrafficCNN
from src.train import (
    train_model,
    evaluate,
    run_pruning_experiments,
    save_pruning_plot,
    save_training_plot,
)

def main():
    ensure_dir("outputs")
    set_seed(SEED)

    full_train_dataset, train_loader, val_loader, test_loader = create_dataloaders()

    print("Train+Val total:", len(full_train_dataset))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    for i in range(5):
        _, y = full_train_dataset[i]
        print(f"Sample {i} label:", y)

    model = TrafficCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model, history = train_model(model, train_loader, val_loader)

    baseline_val_loss, baseline_val_acc = evaluate(model, val_loader)
    baseline_test_loss, baseline_test_acc = evaluate(model, test_loader)

    print(f"\nBaseline Val Accuracy:  {baseline_val_acc:.4f}")
    print(f"Baseline Test Accuracy: {baseline_test_acc:.4f}")

    results_df = run_pruning_experiments(model, train_loader, val_loader, test_loader)

    print("\nPruning Results:")
    print(results_df)

    save_pruning_plot(results_df)
    save_training_plot(history)

if __name__ == "__main__":
    main()
