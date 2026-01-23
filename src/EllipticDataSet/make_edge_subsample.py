import argparse
import os
import numpy as np
import pandas as pd


def create_subsampled_edgelist(input_path, drop_percent):
    seed = 42

    # Dateiname & Ordner
    folder, fname = os.path.split(input_path)
    base, ext = os.path.splitext(fname)

    # Ziel-Dateiname nach deinem Schema
    out_name = f"{base}_{drop_percent}{ext}"
    out_path = os.path.join(folder, out_name)

    print(f"Loading edges from: {input_path}")
    df = pd.read_csv(input_path)

    n_edges = len(df)
    drop_ratio = drop_percent / 100.0
    keep_ratio = 1.0 - drop_ratio
    n_keep = int(round(keep_ratio * n_edges))

    print(f"Edges total: {n_edges}")
    print(f"Dropping {drop_percent}% -> keeping {n_keep} edges")

    rng = np.random.default_rng(seed)
    # Zufällige Indizes auswählen, die wir behalten
    keep_indices = rng.choice(n_edges, size=n_keep, replace=False)
    keep_indices.sort()

    df_sub = df.iloc[keep_indices].reset_index(drop=True)

    print(f"Saving subsampled edgelist to: {out_path}")
    df_sub.to_csv(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="elliptic_txs_edgelist.csv",
        help="Pfad zur Original-Edgelist-CSV",
    )
    parser.add_argument(
        "--drop_percent",
        type=int,
        required=True,
        help="Prozent der Kanten, die entfernt werden sollen (z.B. 25, 50)",
    )

    args = parser.parse_args()
    create_subsampled_edgelist(args.input, args.drop_percent)
