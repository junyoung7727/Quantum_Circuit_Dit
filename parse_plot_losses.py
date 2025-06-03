import re
import sys
import matplotlib.pyplot as plt

"""
Usage:
    python parse_plot_losses.py path/to/train.log

Parses train and validation losses from training log and plots them.
"""

def parse_log(log_path):
    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)(?:/\d+)?\s+-\s*loss:\s*([0-9\.]+),.*?val_mse:\s*\[([^\]]+)\]"
    )
    train_losses = {}
    val_losses = {}
    with open(log_path, 'r') as f:
        for line in f:
            m = epoch_pattern.search(line)
            if m:
                epoch = int(m.group(1))
                t_loss = float(m.group(2))
                mse_vals = list(map(float, m.group(3).split()))
                v_loss = sum(mse_vals) / len(mse_vals)
                train_losses[epoch] = t_loss
                val_losses[epoch] = v_loss
    # sort by epoch
    epochs = sorted(train_losses.keys())
    t_list = [train_losses[e] for e in epochs]
    v_list = [val_losses[e] for e in epochs]
    return epochs, t_list, v_list


def plot_losses(epochs, train_losses, val_losses, output=None):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output:
        plt.savefig(output)
        print(f"Saved plot to {output}")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    log_file = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else None
    epochs, t_losses, v_losses = parse_log(log_file)
    plot_losses(epochs, t_losses, v_losses, out_png)
