# -*- coding: utf-8 -*-
"""
Utility functions for loading the Elliptic dataset and creating train/validation/test splits

- Expected raw data (standard Elliptic dataset):
    ../EllipticDataSet/elliptic_txs_features.csv
    ../EllipticDataSet/elliptic_txs_classes.csv
    ../EllipticDataSet/elliptic_txs_edgelist.csv
- Features: columns 3–95 (93 base features); preceding columns: txId, time_step, class
- Graph indicators: computed transductively on the full graph and merged as
  additional columns with the prefix gi_.

Important label definition (original Elliptic CSV):
    class == 1 → illicit (Fraud)
    class == 2 → licit (No-Fraud)
    class == "unknown" → unlabeled

Desired internal binary encoding:
    Fraud = 0
    No-Fraud = 1
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Tuple, List, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

WEEK_COL = "time_step"
CLASS_COL = "class"
TXID_COL = "txId"

DEBUG = True


def _read_csv_required(path: str, **read_kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    return pd.read_csv(path, **read_kwargs)


def _load_base_tables(
    base_dir: str = "..",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt Features- und Klassen-CSV (Standard Elliptic Layout)."""
    f_path = os.path.join(
        base_dir, "EllipticDataSet", "elliptic_txs_features.csv"
    )
    c_path = os.path.join(
        base_dir, "EllipticDataSet", "elliptic_txs_classes.csv"
    )

    df_feat = _read_csv_required(f_path, header=None)
    df_feat = df_feat.rename(columns={0: TXID_COL, 1: WEEK_COL, 2: CLASS_COL})

    df_cls = _read_csv_required(c_path)

    # Header-Absicherung
    if TXID_COL not in df_cls.columns or CLASS_COL not in df_cls.columns:
        df_cls.columns = [TXID_COL, CLASS_COL]

    # Einheitliche Typen für Join-Key
    df_feat[TXID_COL] = df_feat[TXID_COL].astype(str)
    df_cls[TXID_COL] = df_cls[TXID_COL].astype(str)

    # 'class' vereinheitlichen: "unknown" -> -1, dann numerisch
    df_cls[CLASS_COL] = df_cls[CLASS_COL].replace({"unknown": -1})
    df_cls[CLASS_COL] = pd.to_numeric(df_cls[CLASS_COL], errors="coerce")

    # Merge der class-Spalte (falls features.csv an Position 2 bereits class enthält,
    # überschreiben wir konsistent aus classes.csv):
    df = df_feat.drop(columns=[CLASS_COL], errors="ignore").merge(
        df_cls[[TXID_COL, CLASS_COL]], on=TXID_COL, how="left"
    )
    return df, df_feat


def _sanity_check_label_counts_raw(df_raw: pd.DataFrame) -> None:
    """Drucke Counts und warne, falls die bekannten Elliptic-Verhältnisse stark abweichen."""
    vc = df_raw[CLASS_COL].value_counts(dropna=False)
    n_unknown = int(vc.get(-1, 0))
    n_1 = int(vc.get(1, 0))  # illicit (Fraud)
    n_2 = int(vc.get(2, 0))  # licit (No-Fraud)

    if DEBUG:
        print("[CHK] label counts (raw 1=illicit, 2=licit):")
        print(vc)

    # Erwartung (grobe Plausibilitäten aus Originaldaten): 1 << 2
    if n_1 >= n_2:
        print(
            "[WARN] Unerwartetes Verhältnis in Rohlabels: class==1 (illicit) ist nicht kleiner als class==2 (licit).\n"
            "       Bitte prüfen: Wurde die Klassenkodierung irgendwo vertauscht?"
        )


def load_full_dataframe(
    base_dir: str = "..",
    week_max: Optional[int] = 42,
    labeled_only: bool = True,
    keep_columns: slice = slice(3, 96),  # 93 Basisfeatures
    struct_name_prefix: str = "emb_",
    edgelist_file: str = "EllipticDataSet/elliptic_txs_edgelist.csv",
    directed_graph: bool = True,
    gi_name_prefix: str = "gi_",
    gi_subset: Optional[List[str]] = None,
    emb_name_prefix: str = "emb_",
    embedding_dimensions: int = 64,
    # Graph Indicators
    add_degree_centrality: bool = False,
    add_in_out_degree: bool = False,
    add_pagerank: bool = False,
    add_betweenness_centrality: bool = False,
    add_eigenvector_centrality: bool = False,
    add_closeness_centrality: bool = False,
    add_clustering_coefficient: bool = False,
    add_square_clustering: bool = False,
    add_core_number: bool = False,
    add_triangles: bool = False,
    add_community_louvain: bool = False,
    add_community_leiden: bool = False,
    add_community_infomap: bool = False,
    # Proximity Embeddings
    add_node2vec_balanced: bool = False,
    add_node2vec_dfs: bool = False,
    add_node2vec_bfs: bool = False,
    add_deepwalk: bool = False,
    # Spectral Embeddings
    add_spectral_embeddings: bool = False,
    # Structural Embeddings
    add_ffstruc2vec_embeddings: bool = False,
    add_graphwave_embeddings: bool = False,
    add_role2vec_embeddings: bool = False,
    # GNNs
    add_gcn_embeddings: bool = False,
    add_gat_embeddings: bool = False,
    add_gcl_embeddings: bool = False,
    # Edge-Drop-Variante
    variant: Optional[str] = None,
) -> pd.DataFrame:
    """
    Lädt den vollen Feature-Frame (optional bis week_max) und hängt
    optional transduktive Graphindikatoren an (prefixed).
    """

    if variant == "" or variant == "0" or variant is None:
        suffix = ""
    else:
        # z.B. 0, 25, 50
        suffix = f"_{variant}"

    df, _raw_feat = _load_base_tables(base_dir)

    # Sanity-Check auf Rohlabels (1=illicit, 2=licit)
    _sanity_check_label_counts_raw(df)

    if week_max is not None:
        df = df[df[WEEK_COL] <= week_max].copy()

    if labeled_only:
        # -1 = unlabelled in Elliptic; wir behalten gelabelte (1 illicit / 2 licit).
        df = df[df[CLASS_COL].isin([1, 2])].copy()

    if DEBUG:
        print(f"[CHK] rows after filters: {len(df)}")
        print(
            f"[CHK] label counts (filtered 1 illicit / 2 licit):\n{df[CLASS_COL].value_counts(dropna=False)}"
        )
        print(
            f"[CHK] unique weeks (min..max): {df[WEEK_COL].min()}..{df[WEEK_COL].max()} (n={df[WEEK_COL].nunique()})"
        )
        print("[CHK] sample txId/week/class:")
        print(df[[TXID_COL, WEEK_COL, CLASS_COL]].head(5))

    # Featurescheibe extrahieren
    keep = df.columns[keep_columns]
    df_out = pd.concat(
        [
            df[[TXID_COL, CLASS_COL, WEEK_COL]].reset_index(drop=True),
            df[keep].reset_index(drop=True),
        ],
        axis=1,
    )

    if DEBUG:
        print(
            f"[CHK] df_out rows: {len(df_out)} | non-null {TXID_COL}: {df_out[TXID_COL].notna().sum()}"
        )

    # GRAPH INDICATORS (optional, unverändert)
    # Degree Centrality
    if add_degree_centrality:
        print("Loading degree centrality...")
        try:
            degree_path = os.path.join(
                base_dir, "artifacts", f"degree_centrality{suffix}.parquet"
            )
            df_degree = pd.read_parquet(degree_path)
            df_degree["txId"] = df_degree["txId"].astype(str)
            df_out = df_out.merge(df_degree, on="txId", how="left")
            df_out["gi_degree_centrality"] = (
                df_out["gi_degree_centrality"].fillna(0.0).astype(np.float32)
            )
            print("Loaded degree centrality")
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load degree centrality: {e}"
            )

    if add_in_out_degree:
        print("Loading in/out degree...")
        try:
            in_degree_path = os.path.join(
                base_dir, "artifacts", f"in_degree{suffix}.parquet"
            )
            df_in = pd.read_parquet(in_degree_path)
            df_in["txId"] = df_in["txId"].astype(str)
            df_out = df_out.merge(df_in, on="txId", how="left")
            df_out["gi_in_degree"] = (
                df_out["gi_in_degree"].fillna(0.0).astype(np.float32)
            )

            out_degree_path = os.path.join(
                base_dir, "artifacts", f"out_degree{suffix}.parquet"
            )
            df_out_deg = pd.read_parquet(out_degree_path)
            df_out_deg["txId"] = df_out_deg["txId"].astype(str)
            df_out = df_out.merge(df_out_deg, on="txId", how="left")
            df_out["gi_out_degree"] = (
                df_out["gi_out_degree"].fillna(0.0).astype(np.float32)
            )
            print("Loaded in/out degree")
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load in/out degree: {e}")

    if add_pagerank:
        print("Loading PageRank...")
        try:
            pagerank_path = os.path.join(
                base_dir, "artifacts", f"pagerank{suffix}.parquet"
            )
            df_pagerank = pd.read_parquet(pagerank_path)
            df_pagerank["txId"] = df_pagerank["txId"].astype(str)
            df_out = df_out.merge(df_pagerank, on="txId", how="left")
            df_out["gi_pagerank"] = (
                df_out["gi_pagerank"].fillna(0.0).astype(np.float32)
            )
            print("Loaded PageRank")
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load PageRank: {e}")

    if add_betweenness_centrality:
        print("Loading betweenness centrality...")
        try:
            betweenness_path = os.path.join(
                base_dir,
                "artifacts",
                f"betweenness_centrality{suffix}.parquet",
            )
            df_betweenness = pd.read_parquet(betweenness_path)
            df_betweenness["txId"] = df_betweenness["txId"].astype(str)
            df_out = df_out.merge(df_betweenness, on="txId", how="left")
            df_out["gi_betweenness_centrality"] = (
                df_out["gi_betweenness_centrality"]
                .fillna(0.0)
                .astype(np.float32)
            )
            print("Loaded betweenness centrality")
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load betweenness centrality: {e}"
            )

    if add_eigenvector_centrality:
        print("Loading eigenvector centrality...")
        try:
            eigenvector_path = os.path.join(
                base_dir,
                "artifacts",
                f"eigenvector_centrality{suffix}.parquet",
            )
            df_eigenvector = pd.read_parquet(eigenvector_path)
            df_eigenvector["txId"] = df_eigenvector["txId"].astype(str)
            df_out = df_out.merge(df_eigenvector, on="txId", how="left")
            df_out["gi_eigenvector_centrality"] = (
                df_out["gi_eigenvector_centrality"]
                .fillna(0.0)
                .astype(np.float32)
            )
            print("Loaded eigenvector centrality")
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load eigenvector centrality: {e}"
            )

    if add_closeness_centrality:
        print("Loading closeness centrality...")
        try:
            closeness_path = os.path.join(
                base_dir, "artifacts", f"closeness_centrality{suffix}.parquet"
            )
            df_closeness = pd.read_parquet(closeness_path)
            df_closeness["txId"] = df_closeness["txId"].astype(str)
            df_out = df_out.merge(df_closeness, on="txId", how="left")
            df_out["gi_closeness_centrality"] = (
                df_out["gi_closeness_centrality"]
                .fillna(0.0)
                .astype(np.float32)
            )
            print("Loaded closeness centrality")
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load closeness centrality: {e}"
            )

    if add_clustering_coefficient:
        print("Loading clustering coefficient...")
        try:
            clustering_path = os.path.join(
                base_dir,
                "artifacts",
                f"clustering_coefficient{suffix}.parquet",
            )
            df_clustering = pd.read_parquet(clustering_path)
            df_clustering["txId"] = df_clustering["txId"].astype(str)
            df_out = df_out.merge(df_clustering, on="txId", how="left")
            df_out["gi_clustering_coefficient"] = (
                df_out["gi_clustering_coefficient"]
                .fillna(0.0)
                .astype(np.float32)
            )
            print("Loaded clustering coefficient")
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load clustering coefficient: {e}"
            )

    if add_square_clustering:
        print("Loading square clustering...")
        try:
            square_path = os.path.join(
                base_dir, "artifacts", f"square_clustering{suffix}.parquet"
            )
            df_square = pd.read_parquet(square_path)
            df_square["txId"] = df_square["txId"].astype(str)
            df_out = df_out.merge(df_square, on="txId", how="left")
            df_out["gi_square_clustering"] = (
                df_out["gi_square_clustering"].fillna(0.0).astype(np.float32)
            )
            print("Loaded square clustering")
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load square clustering: {e}"
            )

    if add_core_number:
        print("Loading core number...")
        try:
            core_path = os.path.join(
                base_dir, "artifacts", f"core_number{suffix}.parquet"
            )
            df_core = pd.read_parquet(core_path)
            df_core["txId"] = df_core["txId"].astype(str)
            df_out = df_out.merge(df_core, on="txId", how="left")
            df_out["gi_core_number"] = (
                df_out["gi_core_number"].fillna(0.0).astype(np.float32)
            )
            print("Loaded core number")
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load core number: {e}")

    if add_triangles:
        print("Loading triangles...")
        try:
            triangles_path = os.path.join(
                base_dir, "artifacts", f"triangles{suffix}.parquet"
            )
            df_triangles = pd.read_parquet(triangles_path)
            df_triangles["txId"] = df_triangles["txId"].astype(str)
            df_out = df_out.merge(df_triangles, on="txId", how="left")
            df_out["gi_triangles"] = (
                df_out["gi_triangles"].fillna(0.0).astype(np.float32)
            )
            print("Loaded triangles")
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load triangles: {e}")

    if add_community_louvain:
        print("Loading Louvain communities...")
        try:
            extras_path = os.path.join(
                base_dir,
                "artifacts",
                f"community_louvain_extras{suffix}.parquet",
            )
            if os.path.exists(extras_path):
                df_ex = pd.read_parquet(extras_path)
                df_ex["txId"] = df_ex["txId"].astype(str)
                df_out = df_out.merge(df_ex, on="txId", how="left")

                # Fallbacks
                for c in ["gi_comm_size", "gi_comm_is_small"]:
                    if c in df_out.columns:
                        df_out[c] = (
                            df_out[c]
                            .fillna(0)
                            .astype(
                                "float32" if c == "gi_comm_size" else "Int8"
                            )
                        )
                if "gi_comm_density" in df_out.columns:
                    df_out["gi_comm_density"] = (
                        df_out["gi_comm_density"].fillna(0.0).astype("float32")
                    )
                for j in range(1, 6):
                    cc = f"gi_comm_code_{j}"
                    if cc in df_out.columns:
                        df_out[cc] = df_out[cc].fillna(0.0).astype("float32")

                print(
                    "Loaded Louvain extras (size, density, is_small, 5D code)"
                )
            else:
                raise FileNotFoundError(
                    f"[FATAL] Louvain extras not found: {extras_path}"
                )
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load Louvain extras: {e}")

    if add_community_leiden:
        print("Loading Leiden communities...")
        try:
            path = os.path.join(
                base_dir, "artifacts", f"leiden_communities{suffix}.parquet"
            )
            if os.path.exists(path):
                df_leiden = pd.read_parquet(path)
                # txId-Typ vereinheitlichen
                df_leiden["txId"] = df_leiden["txId"].astype(str)
                # Präfix für Graph-Indicator
                df_leiden.rename(
                    columns={"leiden_comm": f"{gi_name_prefix}leiden_comm"},
                    inplace=True,
                )
                df_out = df_out.merge(df_leiden, on="txId", how="left")

                # NaNs behandeln, sinnvoller Typ (kannst du nach Geschmack anpassen)
                col = f"{gi_name_prefix}leiden_comm"
                df_out[col] = df_out[col].fillna(-1).astype("Int32")

                print("Loaded Leiden communities")
            else:
                raise FileNotFoundError(
                    f"[FATAL] keine Leiden-Datei gefunden: {path}"
                )
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load Leiden communities: {e}"
            )

    if add_community_infomap:
        print("Loading Infomap communities.")
        try:
            path = os.path.join(
                base_dir, "artifacts", f"infomap_communities{suffix}.parquet"
            )
            if os.path.exists(path):
                df_info = pd.read_parquet(path)
                df_info["txId"] = df_info["txId"].astype(str)
                df_info.rename(
                    columns={"infomap_comm": f"{gi_name_prefix}infomap_comm"},
                    inplace=True,
                )
                df_out["txId"] = df_out["txId"].astype(str)
                df_out = df_out.merge(df_info, on="txId", how="left")
                col = f"{gi_name_prefix}infomap_comm"
                df_out[col] = df_out[col].fillna(-1).astype(int)
            else:
                raise FileNotFoundError(
                    f"[FATAL] keine Infomap-Datei gefunden: {path}"
                )
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Fehler beim Laden der Infomap-Communities: {e}"
            )

    if add_node2vec_balanced:
        print("Loading Node2Vec Balanced embeddings...")
        try:
            n2v_path = os.path.join(
                base_dir,
                "artifacts",
                f"node2vec_balanced_embeddings{suffix}.parquet",
            )
            df_n2v = pd.read_parquet(n2v_path)
            df_n2v["txId"] = df_n2v["txId"].astype(str)
            df_out = df_out.merge(df_n2v, on="txId", how="left")
            n2v_cols = [c for c in df_n2v.columns if "emb_n2v" in c]
            print(f"[NODE2VEC-BAL] loaded {len(n2v_cols)} features")
            df_out[n2v_cols] = df_out[n2v_cols].fillna(0.0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load Node2Vec Balanced: {e}"
            )

    if add_node2vec_dfs:
        print("Loading Node2Vec DFS embeddings...")
        try:
            n2v_path = os.path.join(
                base_dir,
                "artifacts",
                f"node2vec_dfs_embeddings{suffix}.parquet",
            )
            df_n2v = pd.read_parquet(n2v_path)
            df_n2v["txId"] = df_n2v["txId"].astype(str)
            rename_map = {
                col: col.replace("emb_node2vec", "emb_node2vec_dfs")
                for col in df_n2v.columns
                if "emb_n2v" in col
            }
            df_n2v = df_n2v.rename(columns=rename_map)
            df_out = df_out.merge(df_n2v, on="txId", how="left")
            n2v_cols = list(rename_map.values())
            print(f"[NODE2VEC-DFS] loaded {len(n2v_cols)} features")
            df_out[n2v_cols] = df_out[n2v_cols].fillna(0.0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load Node2Vec DFS: {e}")

    if add_node2vec_bfs:
        print("Loading Node2Vec BFS embeddings...")
        try:
            n2v_path = os.path.join(
                base_dir,
                "artifacts",
                f"node2vec_bfs_embeddings{suffix}.parquet",
            )
            df_n2v = pd.read_parquet(n2v_path)
            df_n2v["txId"] = df_n2v["txId"].astype(str)
            rename_map = {
                col: col.replace("emb_node2vec", "emb_node2vec_bfs")
                for col in df_n2v.columns
                if "emb_n2v" in col
            }
            df_n2v = df_n2v.rename(columns=rename_map)
            df_out = df_out.merge(df_n2v, on="txId", how="left")
            n2v_cols = list(rename_map.values())
            print(f"[NODE2VEC-BFS] loaded {len(n2v_cols)} features")
            df_out[n2v_cols] = df_out[n2v_cols].fillna(0.0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load Node2Vec BFS: {e}")

    if add_deepwalk:
        print("Loading DeepWalk embeddings...")
        try:
            dw_path = os.path.join(
                base_dir, "artifacts", f"deepwalk_embeddings{suffix}.parquet"
            )
            df_dw = pd.read_parquet(dw_path)
            df_dw["txId"] = df_dw["txId"].astype(str)
            df_out = df_out.merge(df_dw, on="txId", how="left")
            dw_cols = [c for c in df_dw.columns if "emb_dw" in c]
            print(f"[DEEPWALK] loaded {len(dw_cols)} features")
            df_out[dw_cols] = df_out[dw_cols].fillna(0.0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load DeepWalk: {e}")

    if add_spectral_embeddings:
        print("Loading spectral embeddings...")
        try:
            spectral_path = os.path.join(
                base_dir, "artifacts", f"spectral_embeddings{suffix}.parquet"
            )
            df_spectral = pd.read_parquet(spectral_path)
            df_spectral["txId"] = df_spectral["txId"].astype(str)
            df_out = df_out.merge(df_spectral, on="txId", how="left")
            spectral_cols = [
                c for c in df_spectral.columns if "emb_spectral" in c
            ]
            print(f"[SPECTRAL] loaded {len(spectral_cols)} features")
            df_out[spectral_cols] = (
                df_out[spectral_cols].fillna(0.0).astype(np.float32)
            )
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load spectral embeddings: {e}"
            )

    if add_ffstruc2vec_embeddings:
        print("Loading ffstruc2vec embeddings...")
        try:
            ff_paths = [
                os.path.join(
                    base_dir,
                    "artifacts",
                    f"ffstruc2vec_embeddings{suffix}.parquet",
                ),
            ]
            df_ff = None
            for ff_path in ff_paths:
                try:
                    if os.path.exists(ff_path):
                        df_ff = pd.read_parquet(ff_path)
                        print(f"Loaded ffstruc2vec embeddings from: {ff_path}")
                        break
                except Exception:
                    continue

            if df_ff is not None:
                df_ff["txId"] = df_ff["txId"].astype(str)
                df_out = df_out.merge(df_ff, on="txId", how="left")
                ff_cols = [c for c in df_ff.columns if c != "txId"]
                print(f"[FFSTRUC2VEC] loaded {len(ff_cols)} features")
                df_out[ff_cols] = (
                    df_out[ff_cols].fillna(0.0).astype(np.float32)
                )
            else:
                raise FileNotFoundError(
                    f"[FATAL] No ffstruc2vec embeddings file found in artifacts/"
                )
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load ffstruc2vec embeddings: {e}"
            )

    if add_graphwave_embeddings:
        print("Loading GraphWave embeddings...")
        try:
            graphwave_paths = [
                os.path.join(
                    base_dir,
                    "artifacts",
                    f"graphwave_embeddings_components{suffix}.parquet",
                ),
            ]
            df_graphwave = None
            for graphwave_path in graphwave_paths:
                try:
                    if os.path.exists(graphwave_path):
                        df_graphwave = pd.read_parquet(graphwave_path)
                        print(f"Loaded GraphWave from: {graphwave_path}")
                        break
                except Exception:
                    continue

            if df_graphwave is not None:
                df_graphwave["txId"] = df_graphwave["txId"].astype(str)
                df_out = df_out.merge(df_graphwave, on="txId", how="left")
                graphwave_cols = [
                    c for c in df_graphwave.columns if "emb_graphwave" in c
                ]
                print(
                    f"[GRAPHWAVE] loaded {len(graphwave_cols)} GraphWave features"
                )
                df_out[graphwave_cols] = (
                    df_out[graphwave_cols].fillna(0.0).astype(np.float32)
                )
            else:
                raise FileNotFoundError(
                    f"[FATAL] No GraphWave embeddings file found"
                )
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load GraphWave embeddings: {e}"
            )

    if add_role2vec_embeddings:
        print("Loading Role2Vec embeddings...")
        try:
            role2vec_path = os.path.join(
                base_dir, "artifacts", f"role2vec_embeddings{suffix}.parquet"
            )
            df_role2vec = pd.read_parquet(role2vec_path)
            df_role2vec["txId"] = df_role2vec["txId"].astype(str)
            df_out = df_out.merge(df_role2vec, on="txId", how="left")
            role2vec_cols = [
                c for c in df_role2vec.columns if "emb_role2vec" in c
            ]
            print(f"[ROLE2VEC] loaded {len(role2vec_cols)} features")
            df_out[role2vec_cols] = (
                df_out[role2vec_cols].fillna(0.0).astype(np.float32)
            )
        except Exception as e:
            raise RuntimeError(
                f"[FATAL] Could not load Role2Vec embeddings: {e}"
            )

    if add_gcn_embeddings:
        print("Loading GCN embeddings...")
        try:
            gcn_paths = [
                os.path.join(
                    base_dir, "artifacts", f"gcn_embeddings{suffix}.parquet"
                ),
            ]
            df_gcn = None
            for gcn_path in gcn_paths:
                try:
                    if os.path.exists(gcn_path):
                        df_gcn = pd.read_parquet(gcn_path)
                        print(f"Loaded GCN embeddings from: {gcn_path}")
                        break
                except Exception:
                    continue

            if df_gcn is not None:
                df_gcn["txId"] = df_gcn["txId"].astype(str)
                df_out = df_out.merge(df_gcn, on="txId", how="left")
                gcn_cols = [c for c in df_gcn.columns if c != "txId"]
                print(f"[GCN] loaded {len(gcn_cols)} GCN embedding features")
                df_out[gcn_cols] = (
                    df_out[gcn_cols].fillna(0.0).astype(np.float32)
                )
            else:
                raise FileNotFoundError(
                    f"[FATAL] No GCN embeddings file found in artifacts/"
                )
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load GCN embeddings: {e}")

    if add_gat_embeddings:
        print("Loading GAT embeddings...")
        try:
            gat_paths = [
                os.path.join(
                    base_dir, "artifacts", f"gat_embeddings{suffix}.parquet"
                ),
            ]
            df_gat = None
            for gat_path in gat_paths:
                try:
                    if os.path.exists(gat_path):
                        df_gat = pd.read_parquet(gat_path)
                        print(f"Loaded GAT embeddings from: {gat_path}")
                        break
                except Exception:
                    continue

            if df_gat is not None:
                df_gat["txId"] = df_gat["txId"].astype(str)
                df_out = df_out.merge(df_gat, on="txId", how="left")
                gat_cols = [c for c in df_gat.columns if c != "txId"]
                print(f"[GAT] loaded {len(gat_cols)} GAT embedding features")
                df_out[gat_cols] = (
                    df_out[gat_cols].fillna(0.0).astype(np.float32)
                )
            else:
                raise FileNotFoundError(
                    f"[FATAL] No GAT embeddings file found in artifacts/"
                )
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load GAT embeddings: {e}")

    if add_gcl_embeddings:
        print("Loading GCL embeddings.")
        try:
            gcl_path = os.path.join(
                base_dir, "artifacts", f"gcl_embeddings{suffix}.parquet"
            )

            if os.path.exists(gcl_path):
                df_gcl = pd.read_parquet(gcl_path)
                print(f"Loaded GCL embeddings from: {gcl_path}")

                # txId sauber als String
                df_gcl["txId"] = df_gcl["txId"].astype(str)
                df_out = df_out.merge(df_gcl, on="txId", how="left")

                # alle Embedding-Spalten (alles außer txId)
                gcl_cols = [c for c in df_gcl.columns if c != "txId"]
                print(f"[GCL] loaded {len(gcl_cols)} GCL embedding features")

                df_out[gcl_cols] = (
                    df_out[gcl_cols].fillna(0.0).astype(np.float32)
                )
            else:
                raise FileNotFoundError(
                    f"[FATAL] No GCL embeddings file found in artifacts/"
                )
        except Exception as e:
            raise RuntimeError(f"[FATAL] Could not load GCL embeddings: {e}")

    return df_out


def _encode_labels(
    series: pd.Series, mode: Literal["01", "raw"] = "01"
) -> pd.Series:
    if mode == "raw":
        return series
    # Elliptic-Original: 1 = illicit (Fraud), 2 = licit (No-Fraud), -1 = unknown
    # gewünschte Binärkodierung: Fraud=0, No-Fraud=1
    mapping = {1: 0, 2: 1}
    mapped = series.map(mapping)
    mapped = mapped.where(series.isin([1, 2]), -1).astype(int)
    if DEBUG:
        vc = mapped.value_counts(dropna=False)
        n_fraud = int(vc.get(0, 0))
        n_nofraud = int(vc.get(1, 0))
        if n_fraud >= n_nofraud:
            print(
                "[WARN] Unerwartetes Verhältnis nach Mapping (0=Fraud,1=No-Fraud): "
                f"Fraud={n_fraud} >= No-Fraud={n_nofraud}. "
                "Prüfe die Filter/Splits – in Elliptic ist Fraud deutlich seltener."
            )
    return mapped


def _temporal_masks(
    df: pd.DataFrame,
    t_train: Tuple[int, int],
    t_val: Tuple[int, int],
    t_test: Tuple[int, int],
):
    m_tr = (df[WEEK_COL] >= t_train[0]) & (df[WEEK_COL] <= t_train[1])
    m_va = (df[WEEK_COL] >= t_val[0]) & (df[WEEK_COL] <= t_val[1])
    m_te = (df[WEEK_COL] >= t_test[0]) & (df[WEEK_COL] <= t_test[1])
    return m_tr, m_va, m_te


def load_elliptic_splits(
    base_dir: str = "..",
    week_max: Optional[int] = 42,
    non_negative_shift: bool = True,  # historisches Flag – hier ohne Effekt
    encode_labels: Literal["01", "raw"] = "01",
    val_size: float = 0.33,
    test_size: float = 0.25,
    random_state: int = 42,
    split_mode: Literal["random", "temporal"] = "random",
    temporal_cfg: Optional[Dict[str, Tuple[int, int]]] = None,
    add_graph_indicators: bool = False,
    # Graph Indicators
    add_degree_centrality: bool = False,
    add_in_out_degree: bool = False,
    add_pagerank: bool = False,
    add_betweenness_centrality: bool = False,
    add_eigenvector_centrality: bool = False,
    add_closeness_centrality: bool = False,
    add_clustering_coefficient: bool = False,
    add_square_clustering: bool = False,
    add_core_number: bool = False,
    add_triangles: bool = False,
    add_community_louvain: bool = False,
    add_community_leiden: bool = False,
    add_community_infomap: bool = False,
    # Proximity Embeddings
    add_node2vec_balanced: bool = False,
    add_node2vec_dfs: bool = False,
    add_node2vec_bfs: bool = False,
    add_deepwalk: bool = False,
    # Spectral Embeddings
    add_spectral_embeddings: bool = False,
    # Structural Embeddings
    add_ffstruc2vec_embeddings: bool = False,
    add_graphwave_embeddings: bool = False,
    add_role2vec_embeddings: bool = False,
    # GNN
    add_gcn_embeddings: bool = False,
    add_gat_embeddings: bool = False,
    add_gcl_embeddings: bool = False,
    edgelist_file: str = "EllipticDataSet/elliptic_txs_edgelist.csv",
    directed_graph: bool = True,
    gi_name_prefix: str = "gi_",
    gi_subset: Optional[List[str]] = None,
    emb_name_prefix: str = "emb_",
    embedding_dimensions: int = 64,
    # Edge-Drop:
    variant: Optional[str] = None,
):
    """
    Gibt (X_train, X_val, X_test, y_train, y_val, y_test) zurück.
    """
    df = load_full_dataframe(
        base_dir=base_dir,
        week_max=week_max,
        labeled_only=True,
        keep_columns=slice(3, 96),
        # Graph Indicators
        add_degree_centrality=add_degree_centrality,
        add_in_out_degree=add_in_out_degree,
        add_pagerank=add_pagerank,
        add_betweenness_centrality=add_betweenness_centrality,
        add_eigenvector_centrality=add_eigenvector_centrality,
        add_closeness_centrality=add_closeness_centrality,
        add_clustering_coefficient=add_clustering_coefficient,
        add_square_clustering=add_square_clustering,
        add_core_number=add_core_number,
        add_triangles=add_triangles,
        add_community_louvain=add_community_louvain,
        add_community_leiden=add_community_leiden,
        add_community_infomap=add_community_infomap,
        # Proximity Embeddings
        add_node2vec_balanced=add_node2vec_balanced,
        add_node2vec_dfs=add_node2vec_dfs,
        add_node2vec_bfs=add_node2vec_bfs,
        add_deepwalk=add_deepwalk,
        # Spectral Embeddings
        add_spectral_embeddings=add_spectral_embeddings,
        # Structural Embeddings
        add_ffstruc2vec_embeddings=add_ffstruc2vec_embeddings,
        add_graphwave_embeddings=add_graphwave_embeddings,
        add_role2vec_embeddings=add_role2vec_embeddings,
        # GNNs
        add_gcn_embeddings=add_gcn_embeddings,
        add_gat_embeddings=add_gat_embeddings,
        add_gcl_embeddings=add_gcl_embeddings,
        edgelist_file=edgelist_file,
        directed_graph=directed_graph,
        gi_name_prefix=gi_name_prefix,
        gi_subset=gi_subset,
        emb_name_prefix=emb_name_prefix,
        embedding_dimensions=embedding_dimensions,
        # Edge-Drop:
        variant=variant,
    )

    # Label-Kodierung
    y = _encode_labels(df[CLASS_COL], encode_labels)
    X = df.drop(columns=[CLASS_COL])

    # Random vs. Temporal
    if split_mode == "random":

        assert (
            0 < test_size < 1 and 0 <= val_size < 1
        ), "sizes must be in (0,1)"
        assert test_size + val_size < 1, "test_size + val_size must be < 1"

        # Erst Train/Test, dann Train/Val (stratifiziert)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        val_relative = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_relative,
            random_state=random_state,
            stratify=y_train,
        )
    else:
        if not temporal_cfg:
            raise ValueError(
                "temporal_cfg muss gesetzt sein, z.B. "
                "{'train_weeks':(1,35), 'val_weeks':(36,42), 'test_weeks':(43,49)}"
            )
        m_tr, m_va, m_te = _temporal_masks(
            df,
            temporal_cfg["train_weeks"],
            temporal_cfg["val_weeks"],
            temporal_cfg["test_weeks"],
        )
        X_train, y_train = X[m_tr].copy(), y[m_tr].copy()
        X_val, y_val = X[m_va].copy(), y[m_va].copy()
        X_test, y_test = X[m_te].copy(), y[m_te].copy()

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )
