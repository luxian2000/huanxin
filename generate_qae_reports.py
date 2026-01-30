import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HISTORY_PATH = "/Users/luxian/GitSpace/huanxin/QAE_08/qae_training_history.pt"
OUTPUT_DIR = "/Users/luxian/GitSpace/huanxin/QAE_08"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = torch.load(HISTORY_PATH, map_location="cpu", weights_only=False)

    epoch_losses = history.get("epoch_losses", [])
    val_fidelity = history.get("val_fidelity", [])
    batch_losses = history.get("batch_losses", [])

    # Build dataframe
    epochs = [e["epoch"] for e in epoch_losses]
    avg_losses = [e["avg_loss"] for e in epoch_losses]
    val_epochs = [e["epoch"] for e in val_fidelity]
    val_vals = [e["val_fidelity"] for e in val_fidelity]

    df_epoch = pd.DataFrame({
        "epoch": epochs,
        "avg_loss": avg_losses,
    })

    df_val = pd.DataFrame({
        "epoch": val_epochs,
        "val_fidelity": val_vals,
    })

    df_epoch = df_epoch.merge(df_val, on="epoch", how="outer").sort_values("epoch")

    # Batch loss dataframe
    df_batch = pd.DataFrame(batch_losses)

    # Summary stats
    summary = {
        "avg_loss_min": [df_epoch["avg_loss"].min()],
        "avg_loss_max": [df_epoch["avg_loss"].max()],
        "avg_loss_final": [df_epoch["avg_loss"].iloc[-1]],
        "val_fidelity_min": [df_epoch["val_fidelity"].min()],
        "val_fidelity_max": [df_epoch["val_fidelity"].max()],
        "val_fidelity_final": [df_epoch["val_fidelity"].iloc[-1]],
        "best_val_epoch": [int(df_epoch.loc[df_epoch["val_fidelity"].idxmax(), "epoch"])],
    }
    df_summary = pd.DataFrame(summary)

    # Export tables
    df_epoch.to_csv(os.path.join(OUTPUT_DIR, "qae_epoch_metrics.csv"), index=False)
    df_batch.to_csv(os.path.join(OUTPUT_DIR, "qae_batch_losses.csv"), index=False)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "qae_summary.csv"), index=False)

    # Plot epoch loss + val fidelity
    plt.figure(figsize=(10, 6))
    plt.plot(df_epoch["epoch"], df_epoch["avg_loss"], label="Avg Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("QAE Avg Train Loss per Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qae_avg_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_epoch["epoch"], df_epoch["val_fidelity"], label="Val Fidelity", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity")
    plt.title("QAE Validation Fidelity per Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qae_val_fidelity.png"), dpi=200)
    plt.close()

    # Combined plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_epoch["epoch"], df_epoch["avg_loss"], color="tab:blue", label="Avg Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df_epoch["epoch"], df_epoch["val_fidelity"], color="tab:orange", label="Val Fidelity")
    ax2.set_ylabel("Fidelity", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("QAE Training Loss and Validation Fidelity")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "qae_loss_fidelity.png"), dpi=200)
    plt.close(fig)

    # Batch loss plot (smoothed by epoch)
    if not df_batch.empty:
        df_batch = df_batch.sort_values(["epoch", "batch"]).reset_index(drop=True)
        plt.figure(figsize=(10, 6))
        plt.plot(df_batch.index, df_batch["loss"], label="Batch Loss", color="tab:green")
        plt.xlabel("Batch Index")
        plt.ylabel("Loss")
        plt.title("QAE Batch Loss (All Batches)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "qae_batch_loss.png"), dpi=200)
        plt.close()

    print("Reports generated in QAE_08 directory.")


if __name__ == "__main__":
    main()
