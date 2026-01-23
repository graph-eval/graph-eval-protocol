import os
import sys
import subprocess
import pandas as pd
import numpy as np
import glob

OUTPUT_SUFFIX = "_w111000_l2_a0_m_ud_nw10_wl20_cc1_cf0"


def get_paths(drop_rate=None, variant=None):
    """
    Ermittelt alle relevanten Pfade relativ zu diesem Skript.

    drop_rate: Prozent der gelöschten Kanten (z.B. 25, 50) oder None für vollen Graphen.
    seed:      Seed, der beim Erzeugen der gestrippten Edgelist verwendet wurde (nur für Namen).
    """
    here = os.path.dirname(os.path.abspath(__file__))

    # ffstruc2vec-Projektordner
    ff_dir = os.path.join(here, "ffstruc2vec")

    if variant is not None:
        variant_suffix = f"_{variant}"
        elliptic_csv = os.path.join(
            here,
            "..",
            "EllipticDataSet",
            f"elliptic_txs_edgelist{variant_suffix}.csv",
        )
    elif drop_rate is None:
        variant_suffix = ""
        elliptic_csv = os.path.join(
            here, "..", "EllipticDataSet", "elliptic_txs_edgelist.csv"
        )
    else:
        variant_suffix = f"_{drop_rate}"
        elliptic_csv = os.path.join(
            here,
            "..",
            "EllipticDataSet",
            f"elliptic_txs_edgelist{variant_suffix}.csv",
        )

    # Edgelist für ffstruc2vec (space-separiert, ohne Header)
    ff_graph_dir = os.path.join(ff_dir, "graph")
    os.makedirs(ff_graph_dir, exist_ok=True)
    ff_edgelist = os.path.join(
        ff_graph_dir, f"elliptic_txs_ff{variant_suffix}.edgelist"
    )

    # ffstruc2vec-Output (.emb)
    ff_emb_dir = os.path.join(ff_dir, "emb")
    os.makedirs(ff_emb_dir, exist_ok=True)
    ff_emb_txt = os.path.join(
        ff_emb_dir, f"elliptic_txs_ff{OUTPUT_SUFFIX}{variant_suffix}.emb"
    )

    # Ziel-Parquet für dein Projekt
    artifacts_dir = os.path.join(here, "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    output_parquet = os.path.join(
        artifacts_dir,
        f"ffstruc2vec_embeddings{OUTPUT_SUFFIX}{variant_suffix}.parquet",
    )

    # Mapping auch pro Variante (damit nichts überschrieben wird)
    mapping_parquet = os.path.join(
        artifacts_dir, f"ffstruc2vec_node_mapping{variant_suffix}.parquet"
    )

    return {
        "here": here,
        "ff_dir": ff_dir,
        "elliptic_csv": elliptic_csv,
        "ff_edgelist": ff_edgelist,
        "ff_emb_txt": ff_emb_txt,
        "output_parquet": output_parquet,
        "mapping_parquet": mapping_parquet,
        "variant_suffix": variant_suffix,
    }


def convert_csv_to_edgelist(
    csv_path: str, edgelist_path: str, mapping_path: str
):
    print(f"[FFSTRUC2VEC] Lade CSV-Edgelist: {csv_path}")
    df = pd.read_csv(csv_path)

    if not {"txId1", "txId2"}.issubset(df.columns):
        raise ValueError(
            f"Erwarte Spalten 'txId1' und 'txId2' in {csv_path}, "
            f"gefunden: {list(df.columns)}"
        )

    # Alle vorkommenden IDs sammeln
    all_ids = pd.unique(df[["txId1", "txId2"]].values.ravel())
    all_ids = np.asarray(all_ids, dtype=np.int64)

    # Sortieren (optional, aber nett)
    all_ids.sort()

    # Mapping: original ID -> 0..N-1
    id2idx = {orig_id: idx for idx, orig_id in enumerate(all_ids)}

    # Auf neue IDs mappen
    df2 = pd.DataFrame(
        {
            "src": df["txId1"].map(id2idx),
            "dst": df["txId2"].map(id2idx),
        }
    )

    print(f"[FFSTRUC2VEC] Schreibe remappte Edgelist nach: {edgelist_path}")
    df2.to_csv(edgelist_path, sep=" ", header=False, index=False)

    # Mapping speichern, um später wieder zurück auf txId zu kommen
    mapping_df = pd.DataFrame(
        {
            "node_idx": np.arange(len(all_ids), dtype=np.int64),
            "txId": all_ids.astype(str),
        }
    )
    mapping_df.to_parquet(mapping_path, index=False)
    print(
        f"[FFSTRUC2VEC] Mapping node_idx->txId gespeichert nach: {mapping_path}"
    )


def run_ffstruc2vec(
    ff_dir: str, edgelist_rel: str, emb_rel: str, dimensions: int = 64
):
    """
    Startet ffstruc2vec via subprocess.
    edgelist_rel und emb_rel sind Pfade RELATIV zum ff_dir (so wie im README).
    """
    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        "src/main.py",
        "--input",
        edgelist_rel,
        "--output",
        emb_rel,
        "--dimensions",
        str(dimensions),
        "--until_layer",
        "2",
        "--method",
        "3",
        "--OPT1",
        "0",
        "--OPT2",
        "1",
        "--OPT3",
        "1",
        "--num-walks",
        "10",
        "--walk-length",
        "20",
        "--workers",
        "1",
        "--random_seed",
        "42",
        "--active_feature",
        "0",
        "--cost_calc",
        "1",
        "--cost_function",
        "0",
        "--mean_factor",
        "1.0",
        "--weight_layer_0",
        "1.0",
        "--weight_layer_1",
        "1.0",
        "--weight_layer_2",
        "1.0",
        "--weight_layer_3",
        "0.0",
        "--weight_layer_4",
        "0.0",
        "--weight_layer_5",
        "0.0",
    ]

    print(f"[FFSTRUC2VEC] Starte ffstruc2vec im Ordner: {ff_dir}")
    print(f"[FFSTRUC2VEC] Command: {' '.join(cmd)}")

    # ffstruc2vec aus seinem Projektordner heraus starten
    result = subprocess.run(
        cmd,
        cwd=ff_dir,
    )

    print(f"[FFSTRUC2VEC] Returncode: {result.returncode}")

    if result.returncode != 0:
        print("[FFSTRUC2VEC] ERROR beim Ausführen von ffstruc2vec:")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError("ffstruc2vec execution failed")

    print("[FFSTRUC2VEC] ffstruc2vec erfolgreich ausgeführt.")
    if result.stdout:
        print("[FFSTRUC2VEC] STDOUT (gekürzt):")
        print("\n".join(result.stdout.splitlines()[:20]))
    if result.stderr:
        print("[FFSTRUC2VEC] STDERR (gekürzt):")
        print("\n".join(result.stderr.splitlines()[:20]))


def convert_emb_to_parquet(emb_path: str, output_path: str, mapping_path: str):
    """
    Lädt eine ffstruc2vec-Embedding-Datei (Word2Vec-Format) und konvertiert sie nach Parquet.
    emb_path ist der *Basisname*, z.B.
        /workspace/ffstruc2vec/emb/elliptic_txs_ff_w11111_l1_a0.emb
    Falls ffstruc2vec beim Speichern weitere Suffixe anhängt
        (z.B. ..._cost_1_costcalc_0_ps_1_... .emb),
    suchen wir mit einem Pattern nach passenden Dateien.
    """
    # 1) Exakten Namen probieren
    resolved_emb_path = emb_path
    if not os.path.exists(resolved_emb_path):
        # 2) Falls nicht vorhanden: mit Pattern suchen
        pattern = emb_path + "*"  # z.B. ..._w11111_l1_a0.emb*
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(
                f"Embedding-Datei nicht gefunden: {emb_path} "
                f"(auch kein Match für Pattern {pattern})"
            )
        if len(candidates) > 1:
            # Nimm die zuletzt geänderte Datei, falls mehrere existieren
            resolved_emb_path = max(candidates, key=os.path.getmtime)
            print(
                "[FFSTRUC2VEC] Mehrere Embedding-Dateien gefunden. "
                "Verwende die zuletzt geänderte:"
            )
            for c in candidates:
                print("  -", c)
        else:
            resolved_emb_path = candidates[0]

    print(f"[FFSTRUC2VEC] Lade Embedding-Datei: {resolved_emb_path}")

    with open(resolved_emb_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError(
                f"Unerwartetes Header-Format in {resolved_emb_path}: {header}"
            )
        n_nodes, dim = map(int, header)
        print(f"[FFSTRUC2VEC] Header: n_nodes={n_nodes}, dim={dim}")

        ids = []
        vecs = []
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            node_id = int(parts[0])  # das ist jetzt node_idx (0..N-1)
            emb = [float(x) for x in parts[1:]]
            if len(emb) != dim:
                raise ValueError(
                    f"Längenfehler bei Node {node_id}: "
                    f"erwarte {dim}, habe {len(emb)}"
                )
            ids.append(node_id)
            vecs.append(emb)

    X = np.asarray(vecs, dtype=np.float32)
    if X.shape[0] != len(ids):
        raise ValueError(
            "Anzahl Embeddings und Node-IDs passt nicht zusammen."
        )

    cols = [f"emb_ffstruc2vec_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["node_idx"] = ids

    # Mapping laden
    mapping_df = pd.read_parquet(mapping_path)
    mapping_df["node_idx"] = mapping_df["node_idx"].astype(int)
    mapping_df["txId"] = mapping_df["txId"].astype(str)

    df = df.merge(mapping_df, on="node_idx", how="left")

    df["txId"] = df["txId"].astype(str)
    df = df[["txId"] + cols]

    df.to_parquet(output_path, index=False)
    print(
        f"[FFSTRUC2VEC] Parquet gespeichert: {output_path} "
        f"(rows={len(df)}, dim={X.shape[1]})"
    )


def main(skip_train=False, drop_rate=None, seed=42, variant=None):
    paths = get_paths(drop_rate=drop_rate, variant=variant)

    print("=== ffstruc2vec Wrapper für Elliptic ===")
    print(f"Projektordner: {paths['here']}")
    print(f"ffstruc2vec-Ordner: {paths['ff_dir']}")
    if variant is not None:
        print(f"Modus: TARGETED DROP VARIANT (variant={variant})")
    elif drop_rate is None:
        print("Modus: FULL GRAPH (keine Kanten entfernt)")
    else:
        print(
            f"Modus: EDGE-DROP VARIANT (drop_rate={drop_rate}%, seed={seed})"
        )

    print(f"Verwendete Elliptic-Edgelist: {paths['elliptic_csv']}")
    print()

    if not skip_train:
        # 1) CSV -> space-separierte Edgelist + Mapping
        convert_csv_to_edgelist(
            paths["elliptic_csv"],
            paths["ff_edgelist"],
            paths["mapping_parquet"],
        )

        # 2) ffstruc2vec ausführen (Pfadangaben relativ zu ff_dir)
        run_ffstruc2vec(
            ff_dir=paths["ff_dir"],
            edgelist_rel=os.path.relpath(
                paths["ff_edgelist"], start=paths["ff_dir"]
            ),
            emb_rel=os.path.relpath(
                paths["ff_emb_txt"], start=paths["ff_dir"]
            ),
            dimensions=64,
        )
    else:
        print(
            "[FFSTRUC2VEC] --skip-train gesetzt: "
            "überspringe CSV->Edgelist und Training, "
            "verwende vorhandene Embedding- und Mapping-Dateien."
        )
        if not os.path.exists(paths["mapping_parquet"]):
            raise FileNotFoundError(
                f"Mapping-Datei nicht gefunden: {paths['mapping_parquet']} "
                f"(ohne Mapping kann nicht nach Parquet konvertiert werden)"
            )

    # 3) .emb -> Parquet (mit Rück-Mapping auf txId)
    convert_emb_to_parquet(
        paths["ff_emb_txt"],
        paths["output_parquet"],
        paths["mapping_parquet"],
    )

    print("=== ffstruc2vec Wrapper abgeschlossen ===")
    print(f"Output-Parquet: {paths['output_parquet']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wrapper zum Ausführen von ffstruc2vec auf dem Elliptic-Datensatz "
        "(voller Graph oder Edge-Drop-Varianten)."
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "Überspringt CSV->Edgelist und ffstruc2vec-Training und führt nur die "
            "Konvertierung der vorhandenen .emb-Datei nach Parquet aus."
        ),
    )
    parser.add_argument(
        "--drop_rate",
        type=int,
        default=None,
        help=(
            "Prozent der Kanten, die beim Erzeugen der Edgelist entfernt wurden "
            "(z.B. 25, 50). Wenn nicht gesetzt: voller Graph."
        ),
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Alternative zur drop_rate-Variante. "
            "Erwartet dann eine Datei "
            "elliptic_txs_edgelist_<variant>.csv im EllipticDataSet-Ordner."
        ),
    )

    args = parser.parse_args()

    main(
        skip_train=args.skip_train,
        drop_rate=args.drop_rate,
        seed=42,
        variant=args.variant,
    )
