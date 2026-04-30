"""Evaluate FractalX on a reproducible MovieLens-1M split.

The harness intentionally stays small: NumPy for data handling and ALS,
scikit-learn for the MLP baseline, and matplotlib for the single headline plot.
"""

from __future__ import annotations

import argparse
import math
import shutil
import ssl
import urllib.request
import urllib.error
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prototype import FractalInterferenceScorer, cosine_mean_scores, l2_normalize_rows


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SEED = 20260430


@dataclass(frozen=True)
class RatingData:
    user_ids: np.ndarray
    item_ids: np.ndarray
    ratings: np.ndarray
    timestamps: np.ndarray
    n_users: int
    n_items: int


@dataclass(frozen=True)
class EvaluationUser:
    user_id: int
    train_positive_items: np.ndarray
    train_positive_weights: np.ndarray
    candidate_items: np.ndarray
    relevant: np.ndarray
    long_tail: np.ndarray


def ndcg_at_k(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Compute binary NDCG@k for one ranked candidate set."""

    if not np.any(relevant):
        return float("nan")
    order = np.argsort(-scores)[:k]
    gains = relevant[order].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_hits = min(int(np.sum(relevant)), k)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_hits + 2))
    ideal_dcg = float(np.sum(ideal_discounts))
    return dcg / ideal_dcg if ideal_dcg > 0.0 else float("nan")


def mean_reciprocal_rank(scores: np.ndarray, relevant: np.ndarray) -> float:
    """Compute reciprocal rank for one candidate set."""

    order = np.argsort(-scores)
    ranked_relevant = relevant[order]
    hit_positions = np.flatnonzero(ranked_relevant)
    if hit_positions.size == 0:
        return float("nan")
    return 1.0 / float(hit_positions[0] + 1)


def long_tail_recall_at_k(
    scores: np.ndarray,
    relevant: np.ndarray,
    long_tail: np.ndarray,
    k: int,
) -> float:
    """Recall@k restricted to relevant candidates in the long tail."""

    target = relevant & long_tail
    denominator = int(np.sum(target))
    if denominator == 0:
        return float("nan")
    order = np.argsort(-scores)[:k]
    return float(np.sum(target[order]) / denominator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Dataset cache directory")
    parser.add_argument("--plot-path", default="results.png", help="Output plot path")
    parser.add_argument("--max-users", type=int, default=350, help="Evaluation users")
    parser.add_argument("--candidate-negatives", type=int, default=250)
    parser.add_argument("--holdout-per-user", type=int, default=3)
    parser.add_argument("--min-positive-history", type=int, default=20)
    parser.add_argument("--max-history", type=int, default=100)
    parser.add_argument("--factors", type=int, default=24)
    parser.add_argument("--als-iterations", type=int, default=5)
    parser.add_argument("--als-regularization", type=float, default=0.12)
    parser.add_argument("--mlp-train-examples", type=int, default=80_000)
    parser.add_argument("--smax-values", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def ensure_movielens_1m(data_dir: Path) -> Path:
    """Download and extract MovieLens-1M if it is not already cached."""

    ratings_path = data_dir / "ml-1m" / "ratings.dat"
    if ratings_path.exists():
        return ratings_path

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-1m.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens-1M to {zip_path}...")
        download_file(MOVIELENS_1M_URL, zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)
    return ratings_path


def download_file(url: str, destination: Path) -> None:
    """Download a file, with an SSL fallback for local cert-chain breakage."""

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        return
    except urllib.error.URLError as error:
        if "CERTIFICATE_VERIFY_FAILED" not in str(error):
            raise

    print("TLS certificate verification failed locally; retrying download without verification.")
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=60, context=context) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def load_ratings(ratings_path: Path) -> RatingData:
    """Load MovieLens ratings and remap user/movie IDs to dense indices."""

    raw_users: list[int] = []
    raw_items: list[int] = []
    raw_ratings: list[float] = []
    raw_timestamps: list[int] = []

    with ratings_path.open("r", encoding="latin-1") as handle:
        for line in handle:
            user, item, rating, timestamp = line.strip().split("::")
            raw_users.append(int(user))
            raw_items.append(int(item))
            raw_ratings.append(float(rating))
            raw_timestamps.append(int(timestamp))

    user_map = {value: index for index, value in enumerate(sorted(set(raw_users)))}
    item_map = {value: index for index, value in enumerate(sorted(set(raw_items)))}

    user_ids = np.array([user_map[value] for value in raw_users], dtype=np.int32)
    item_ids = np.array([item_map[value] for value in raw_items], dtype=np.int32)
    ratings = np.array(raw_ratings, dtype=np.float32)
    timestamps = np.array(raw_timestamps, dtype=np.int64)

    return RatingData(
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
        timestamps=timestamps,
        n_users=len(user_map),
        n_items=len(item_map),
    )


def build_split(
    data: RatingData,
    *,
    max_users: int,
    holdout_per_user: int,
    min_positive_history: int,
    candidate_negatives: int,
    seed: int,
) -> tuple[list[tuple[int, int, float]], list[EvaluationUser], np.ndarray]:
    """Create a chronological positive holdout split for evaluation users."""

    rng = np.random.default_rng(seed)
    by_user: list[list[int]] = [[] for _ in range(data.n_users)]
    for row, user_id in enumerate(data.user_ids):
        by_user[int(user_id)].append(row)

    eligible_users: list[int] = []
    user_positive_rows: dict[int, list[int]] = {}
    for user_id, rows in enumerate(by_user):
        positives = [row for row in rows if data.ratings[row] >= 4.0]
        positives.sort(key=lambda row: int(data.timestamps[row]))
        if len(positives) >= min_positive_history + holdout_per_user:
            eligible_users.append(user_id)
            user_positive_rows[user_id] = positives

    rng.shuffle(eligible_users)
    selected_users = set(eligible_users[:max_users])
    holdout_rows: set[int] = set()
    for user_id in selected_users:
        holdout_rows.update(user_positive_rows[user_id][-holdout_per_user:])

    train_records: list[tuple[int, int, float]] = []
    train_seen: list[set[int]] = [set() for _ in range(data.n_users)]
    train_positive: dict[int, list[tuple[int, float, int]]] = {}
    train_positive_counts = np.zeros(data.n_items, dtype=np.int32)

    for row in range(data.ratings.size):
        user_id = int(data.user_ids[row])
        item_id = int(data.item_ids[row])
        rating = float(data.ratings[row])
        if row in holdout_rows:
            continue

        scaled_rating = (rating - 3.0) / 2.0
        train_records.append((user_id, item_id, scaled_rating))
        train_seen[user_id].add(item_id)
        if rating >= 4.0:
            train_positive.setdefault(user_id, []).append(
                (item_id, rating - 3.0, int(data.timestamps[row]))
            )
            train_positive_counts[item_id] += 1

    long_tail_items = bottom_popularity_items(train_positive_counts, fraction=0.80)
    all_items = np.arange(data.n_items, dtype=np.int32)
    eval_users: list[EvaluationUser] = []

    for user_id in sorted(selected_users):
        heldout = [
            int(data.item_ids[row])
            for row in sorted(holdout_rows)
            if int(data.user_ids[row]) == user_id
        ]
        if not heldout:
            continue

        excluded = set(train_seen[user_id])
        excluded.update(heldout)
        available = np.setdiff1d(all_items, np.fromiter(excluded, dtype=np.int32), assume_unique=False)
        if available.size == 0:
            continue
        negative_count = min(candidate_negatives, available.size)
        negatives = rng.choice(available, size=negative_count, replace=False)

        candidates = np.array(heldout + negatives.tolist(), dtype=np.int32)
        relevant = np.zeros(candidates.size, dtype=bool)
        relevant[: len(heldout)] = True
        order = rng.permutation(candidates.size)
        candidates = candidates[order]
        relevant = relevant[order]

        positives = sorted(train_positive.get(user_id, []), key=lambda row: row[2])
        positive_items = np.array([row[0] for row in positives], dtype=np.int32)
        positive_weights = np.array([row[1] for row in positives], dtype=float)
        eval_users.append(
            EvaluationUser(
                user_id=user_id,
                train_positive_items=positive_items,
                train_positive_weights=positive_weights,
                candidate_items=candidates,
                relevant=relevant,
                long_tail=long_tail_items[candidates],
            )
        )

    return train_records, eval_users, long_tail_items


def bottom_popularity_items(popularity: np.ndarray, fraction: float) -> np.ndarray:
    """Return a mask for items in the bottom popularity fraction."""

    order = np.argsort(popularity, kind="mergesort")
    cutoff = int(math.ceil(popularity.size * fraction))
    mask = np.zeros(popularity.size, dtype=bool)
    mask[order[:cutoff]] = True
    return mask


def train_explicit_als(
    train_records: list[tuple[int, int, float]],
    *,
    n_users: int,
    n_items: int,
    factors: int,
    iterations: int,
    regularization: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a small explicit-feedback ALS model."""

    rng = np.random.default_rng(seed)
    user_factors = 0.05 * rng.normal(size=(n_users, factors))
    item_factors = 0.05 * rng.normal(size=(n_items, factors))
    user_records: list[list[tuple[int, float]]] = [[] for _ in range(n_users)]
    item_records: list[list[tuple[int, float]]] = [[] for _ in range(n_items)]

    for user_id, item_id, rating in train_records:
        user_records[user_id].append((item_id, rating))
        item_records[item_id].append((user_id, rating))

    eye = np.eye(factors)
    for iteration in range(iterations):
        print(f"ALS iteration {iteration + 1}/{iterations}")
        for user_id, records in enumerate(user_records):
            if not records:
                continue
            items = np.array([item for item, _ in records], dtype=np.int32)
            values = np.array([value for _, value in records], dtype=float)
            factors_i = item_factors[items]
            lhs = factors_i.T @ factors_i + regularization * eye
            rhs = factors_i.T @ values
            user_factors[user_id] = np.linalg.solve(lhs, rhs)

        for item_id, records in enumerate(item_records):
            if not records:
                continue
            users = np.array([user for user, _ in records], dtype=np.int32)
            values = np.array([value for _, value in records], dtype=float)
            factors_u = user_factors[users]
            lhs = factors_u.T @ factors_u + regularization * eye
            rhs = factors_u.T @ values
            item_factors[item_id] = np.linalg.solve(lhs, rhs)

    return user_factors, item_factors


def build_user_profiles(eval_users: list[EvaluationUser], item_embeddings: np.ndarray) -> dict[int, np.ndarray]:
    """Build weighted mean positive-history profiles for evaluation users."""

    profiles: dict[int, np.ndarray] = {}
    for user in eval_users:
        if user.train_positive_items.size == 0:
            profiles[user.user_id] = np.zeros(item_embeddings.shape[1], dtype=float)
            continue
        history = item_embeddings[user.train_positive_items]
        weights = np.maximum(user.train_positive_weights, 1e-6)
        profiles[user.user_id] = np.average(history, axis=0, weights=weights)
    return profiles


def pair_features(user_profiles: np.ndarray, item_embeddings: np.ndarray) -> np.ndarray:
    """Feature map for the MLP baseline."""

    return np.hstack(
        [
            user_profiles,
            item_embeddings,
            user_profiles * item_embeddings,
            np.abs(user_profiles - item_embeddings),
        ]
    )


def train_mlp_baseline(
    eval_users: list[EvaluationUser],
    item_embeddings: np.ndarray,
    profiles: dict[int, np.ndarray],
    *,
    max_examples: int,
    seed: int,
) -> object:
    """Train one global MLP on positive histories and sampled unobserved items."""

    rng = np.random.default_rng(seed)
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    all_items = np.arange(item_embeddings.shape[0], dtype=np.int32)
    examples_per_user = max(2, max_examples // max(len(eval_users), 1))

    for user in eval_users:
        profile = profiles[user.user_id]
        positives = user.train_positive_items
        if positives.size == 0:
            continue
        positive_sample = rng.choice(
            positives,
            size=min(examples_per_user // 2, positives.size),
            replace=False,
        )
        seen = set(positives.tolist())
        seen.update(user.candidate_items[user.relevant].tolist())
        available = np.setdiff1d(
            all_items,
            np.fromiter(seen, dtype=np.int32),
            assume_unique=False,
        )
        negative_sample = rng.choice(
            available,
            size=min(positive_sample.size, available.size),
            replace=False,
        )
        items = np.concatenate([positive_sample, negative_sample])
        repeated_profile = np.repeat(profile[np.newaxis, :], items.size, axis=0)
        feature_rows.append(pair_features(repeated_profile, item_embeddings[items]))
        labels.extend([1] * positive_sample.size)
        labels.extend([0] * negative_sample.size)

    features = np.vstack(feature_rows)
    y = np.array(labels, dtype=np.int32)
    order = rng.permutation(y.size)
    features = features[order]
    y = y[order]

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=50,
            early_stopping=True,
            n_iter_no_change=6,
            random_state=seed,
        ),
    )
    model.fit(features, y)
    return model


def score_mlp(model: object, profile: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    """Score candidate embeddings with the global MLP baseline."""

    repeated_profile = np.repeat(profile[np.newaxis, :], candidate_embeddings.shape[0], axis=0)
    features = pair_features(repeated_profile, candidate_embeddings)
    return model.predict_proba(features)[:, 1]


def evaluate_scores(
    scored_users: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    k_values: tuple[int, int] = (10, 50),
) -> dict[str, float]:
    """Aggregate ranking metrics across users."""

    ndcg10: list[float] = []
    ndcg50: list[float] = []
    mrr: list[float] = []
    long_tail50: list[float] = []

    for scores, relevant, long_tail in scored_users:
        ndcg10.append(ndcg_at_k(scores, relevant, k_values[0]))
        ndcg50.append(ndcg_at_k(scores, relevant, k_values[1]))
        mrr.append(mean_reciprocal_rank(scores, relevant))
        long_tail50.append(long_tail_recall_at_k(scores, relevant, long_tail, k_values[1]))

    return {
        "NDCG@10": nanmean(ndcg10),
        "NDCG@50": nanmean(ndcg50),
        "MRR": nanmean(mrr),
        "LongTailRecall@50": nanmean(long_tail50),
    }


def nanmean(values: list[float]) -> float:
    array = np.array(values, dtype=float)
    if np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def evaluate_all(
    eval_users: list[EvaluationUser],
    item_embeddings: np.ndarray,
    profiles: dict[int, np.ndarray],
    mlp_model: object,
    *,
    smax_values: list[int],
    max_history: int,
    seed: int,
) -> dict[str, dict[int, dict[str, float]]]:
    """Evaluate cosine, MLP, and FractalX across Smax values."""

    cosine_scored: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    mlp_scored: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    fractal_by_smax: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        smax: [] for smax in smax_values
    }
    max_smax = max(smax_values)

    for index, user in enumerate(eval_users, start=1):
        if index % 50 == 0:
            print(f"Scoring user {index}/{len(eval_users)}")

        candidates = item_embeddings[user.candidate_items]
        profile = profiles[user.user_id]
        cosine_scored.append(
            (
                cosine_mean_scores(
                    item_embeddings[user.train_positive_items],
                    candidates,
                    user.train_positive_weights,
                ),
                user.relevant,
                user.long_tail,
            )
        )
        mlp_scored.append((score_mlp(mlp_model, profile, candidates), user.relevant, user.long_tail))

        history_items = user.train_positive_items[-max_history:]
        history_weights = user.train_positive_weights[-max_history:]
        scorer = FractalInterferenceScorer(
            smax=max_smax,
            lambda0=None,
            wave_mode="pca",
            n_wave_vectors=min(8, item_embeddings.shape[1]),
            random_state=seed + user.user_id,
        ).fit(item_embeddings[history_items], history_weights)
        by_scale = scorer.score_by_scale(candidates)

        for smax in smax_values:
            scale_weights = dimension_conditioned_scale_weights(
                smax=smax,
                dimension_ratio=scorer.dimension_ratio_,
                dimension_weight=scorer.dimension_weight,
            )
            scores = by_scale[:, :smax] @ scale_weights
            scores = scores.copy()
            scores[scores < 0.0] *= scorer.destructive_multiplier_
            fractal_by_smax[smax].append((scores, user.relevant, user.long_tail))

    cosine_metrics = evaluate_scores(cosine_scored)
    mlp_metrics = evaluate_scores(mlp_scored)
    results: dict[str, dict[int, dict[str, float]]] = {
        "CosineMean": {smax: cosine_metrics for smax in smax_values},
        "MLP": {smax: mlp_metrics for smax in smax_values},
        "FractalX": {
            smax: evaluate_scores(fractal_by_smax[smax]) for smax in smax_values
        },
    }
    return results


def dimension_conditioned_scale_weights(
    *,
    smax: int,
    dimension_ratio: float,
    dimension_weight: float,
) -> np.ndarray:
    scales = np.arange(1, smax + 1, dtype=float)
    base = 2.0 ** -(scales - 1.0)
    fine_position = (scales - 1.0) / max(smax - 1.0, 1.0)
    weights = base * (1.0 + dimension_weight * dimension_ratio * fine_position)
    return weights / np.sum(weights)


def print_results(results: dict[str, dict[int, dict[str, float]]], smax_values: list[int]) -> None:
    """Print a compact reproducible results table."""

    print("\nResults")
    print("Scorer,Smax,NDCG@10,NDCG@50,MRR,LongTailRecall@50")
    for scorer in ["CosineMean", "MLP", "FractalX"]:
        for smax in smax_values:
            metrics = results[scorer][smax]
            print(
                f"{scorer},{smax},"
                f"{metrics['NDCG@10']:.4f},"
                f"{metrics['NDCG@50']:.4f},"
                f"{metrics['MRR']:.4f},"
                f"{metrics['LongTailRecall@50']:.4f}"
            )


def save_plot(
    results: dict[str, dict[int, dict[str, float]]],
    smax_values: list[int],
    plot_path: Path,
) -> None:
    """Save long-tail recall vs. Smax for the three scorers."""

    plt.figure(figsize=(8, 5))
    for scorer, marker in [("CosineMean", "o"), ("MLP", "s"), ("FractalX", "^")]:
        values = [
            results[scorer][smax]["LongTailRecall@50"]
            for smax in smax_values
        ]
        plt.plot(smax_values, values, marker=marker, linewidth=2, label=scorer)

    plt.title("MovieLens-1M Long-Tail Recall@50 vs. Smax")
    plt.xlabel("Smax")
    plt.ylabel("Long-tail recall@50")
    plt.xticks(smax_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    print(f"Saved {plot_path}")


def main() -> None:
    args = parse_args()
    smax_values = [int(value) for value in args.smax_values.split(",")]
    rng_seed = int(args.seed)

    ratings_path = ensure_movielens_1m(Path(args.data_dir))
    data = load_ratings(ratings_path)
    print(
        f"Loaded MovieLens-1M: {data.ratings.size} ratings, "
        f"{data.n_users} users, {data.n_items} items"
    )

    train_records, eval_users, _ = build_split(
        data,
        max_users=args.max_users,
        holdout_per_user=args.holdout_per_user,
        min_positive_history=args.min_positive_history,
        candidate_negatives=args.candidate_negatives,
        seed=rng_seed,
    )
    print(f"Train ratings: {len(train_records)}")
    print(f"Evaluation users: {len(eval_users)}")

    _, item_factors = train_explicit_als(
        train_records,
        n_users=data.n_users,
        n_items=data.n_items,
        factors=args.factors,
        iterations=args.als_iterations,
        regularization=args.als_regularization,
        seed=rng_seed,
    )
    item_embeddings = l2_normalize_rows(item_factors)
    profiles = build_user_profiles(eval_users, item_embeddings)

    print("Training MLP baseline...")
    mlp_model = train_mlp_baseline(
        eval_users,
        item_embeddings,
        profiles,
        max_examples=args.mlp_train_examples,
        seed=rng_seed,
    )

    results = evaluate_all(
        eval_users,
        item_embeddings,
        profiles,
        mlp_model,
        smax_values=smax_values,
        max_history=args.max_history,
        seed=rng_seed,
    )
    print_results(results, smax_values)
    save_plot(results, smax_values, Path(args.plot_path))


if __name__ == "__main__":
    main()
