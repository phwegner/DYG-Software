import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

app = typer.Typer()

@app.command()
def warp(
    csv_a: str = typer.Argument(..., help="CSV file containing time series A"),
    col_a: str = typer.Argument(..., help="Column name for time series A"),
    csv_b: str = typer.Argument(..., help="CSV file containing time series B"),
    col_b: str = typer.Argument(..., help="Column name for time series B"),
    save_path: str = typer.Argument(..., help="Path to save warped B as CSV"),
    plot_path: str = typer.Option(None, help="Optional path to save plot (PNG, PDF, etc.)"),
):
    """
    Warp time series B (from csv_b/col_b) onto time series A (from csv_a/col_a) using DTW.
    Saves warped B as CSV.
    Optionally saves a plot of A, B, and B_warped.
    """
    # Load both CSVs
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    if col_a not in df_a.columns:
        typer.echo(f"Error: {col_a} not found in {csv_a}")
        raise typer.Exit(code=1)
    if col_b not in df_b.columns:
        typer.echo(f"Error: {col_b} not found in {csv_b}")
        raise typer.Exit(code=1)

    A = df_a[col_a].to_numpy()
    B = df_b[col_b].to_numpy()

    # Run DTW
    distance, path = fastdtw(A, B, dist=euclidean)

    # Warp B onto A
    B_warped = np.array([B[j] for i, j in path])

    # Save warped series
    np.savetxt(save_path, B_warped, delimiter=",")
    typer.echo(f"Warped time series saved to {save_path}")

    # Save plot if requested
    if plot_path:
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(A, label=f"A ({col_a})", color="blue")
        axes[0].legend()
        axes[1].plot(B, label=f"B ({col_b}, original)", color="orange")
        axes[1].legend()
        axes[2].plot(B_warped, label=f"B_warped → A", color="green")
        axes[2].legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        typer.echo(f"Plot saved to {plot_path}")

    typer.echo(f"DTW distance: {distance}")

if __name__ == "__main__":
    app()
