import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw_path

app = typer.Typer()

def warp_t_along_path(t, path):
    """Warp time series t (shape: [T, D]) using a DTW path."""
    return np.array([t[i] for i, _ in path])

@app.command()
def warp(
    csv_a: str = typer.Option(..., "csv_a", help="Path to first CSV (reference time series)"),
    col_a: str = typer.Option(..., "col_a", help="Column name in CSV A"),
    csv_b: str = typer.Option(..., "csv_b", help="Path to second CSV (time series to warp)"),
    col_b: str = typer.Option(..., "col_b", help="Column name in CSV B"),
    save_path: str = typer.Option(..., "save_path", help="Path to save warped CSV"),
    plot_path: str = typer.Option("warped_plot.png", "plot_path", help="Path to save plot")
):
    # Load CSVs
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    # Convert to 1-D numeric arrays
    A = pd.to_numeric(df_a[col_a], errors='coerce').to_numpy().reshape(-1, 1)
    B = pd.to_numeric(df_b[col_b], errors='coerce').to_numpy().reshape(-1, 1)

    # DTW
    path, _ = dtw_path(B, A)  # warp B onto A
    B_warped = warp_t_along_path(B, path)

    # Save warped CSV
    pd.DataFrame({col_b: B_warped}).to_csv(save_path, index=False)

    # Plot vertically
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(A, color='blue')
    axes[0].set_title(f"A ({col_a})")
    axes[1].plot(B, color='red')
    axes[1].set_title(f"B ({col_b}) original")
    axes[2].plot(B_warped, color='green')
    axes[2].set_title(f"B ({col_b}) warped onto A")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

    typer.echo(f"Warped series saved to {save_path}")
    typer.echo(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    app()
