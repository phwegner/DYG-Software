import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pathlib import Path

app = typer.Typer()

@app.command()
def warp(
    csv_a: Path = typer.Option(..., "--csv-a", help="CSV file containing time series A"),
    col_a: str = typer.Option(..., "--col-a", help="Column name for time series A"),
    csv_b: Path = typer.Option(..., "--csv-b", help="CSV file containing time series B"),
    col_b: str = typer.Option(..., "--col-b", help="Column name for time series B"),
    save_path: Path = typer.Option(None, "--save-path", help="Path to save warped B as CSV"),
    plot_path: Path = typer.Option(None, "--plot-path", help="Path to save plot (PNG, PDF, etc.)"),
):
    """
    Warp time series B onto A using DTW.
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

    A = pd.to_numeric(df_a[col_a], errors='coerce').to_numpy().reshape(-1, 1)
    B = pd.to_numeric(df_b[col_b], errors='coerce').to_numpy().reshape(-1, 1)

    # Run DTW
    distance, path = fastdtw(A, B, dist=euclidean)
    B_warped = np.array([B[j] for i, j in path])

    # Defaults for save paths
    if save_path is None:
        save_path = csv_b.parent / f"{col_b}_warped.csv"
    if plot_path is None:
        plot_path = csv_b.parent / f"{col_b}_warped.png"

    # Save warped series
    np.savetxt(save_path, B_warped, delimiter=",")
    typer.echo(f"Warped time series saved to {save_path}")

    # Save plot
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
