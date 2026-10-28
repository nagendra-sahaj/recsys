import numpy as np
import polars as pl

from recsys_large_scale.data.preprocess import compute_user_counts


def cold_user_mask(eval_users: np.ndarray, train: pl.DataFrame) -> np.ndarray:
    """True where a user in `eval_users` has zero rows in `train` -- a genuine
    cold-start case the model never saw anything about. Only meaningful against a split
    that can contain such cases (global_time_split's test set); leave_one_out_split's
    test set never does, by construction (every LOO test user needs >=1 train row just
    to be eligible for that split at all)."""
    train_users = set(train["user_idx"].unique().to_list())
    return np.array([u not in train_users for u in eval_users])


def bucket_by_activity(train: pl.DataFrame, n_users: int, labels: tuple[str, str, str] = ("low", "medium", "high")) -> np.ndarray:
    """Per-user_idx activity segment, split into terciles of each user's `train`
    interaction count -- adapts to the actual distribution instead of guessing fixed
    thresholds, since "a lot of interactions" means something very different on H&M's
    sparse purchase data than on MovieLens ratings. Users with zero train interactions
    (shouldn't occur for anyone actually present in an eval set built from the same
    processed data, but kept safe rather than assumed) get the lowest label."""
    counts = compute_user_counts(train, n_users)
    nonzero = counts[counts > 0]
    q1, q2 = np.quantile(nonzero, [1 / 3, 2 / 3])

    buckets = np.full(n_users, labels[0], dtype=object)
    buckets[counts > q1] = labels[1]
    buckets[counts > q2] = labels[2]
    return buckets


def map_customer_attribute(customers: pl.DataFrame, user_id_to_idx: dict, n_users: int, column: str) -> np.ndarray:
    """Per-user_idx array of a raw customers.csv column (e.g. "age",
    "club_member_status"), aligned to this project's user_idx indexing via
    `user_id_to_idx` (id_maps.json's mapping from raw customer_id to user_idx). Users
    with no matching customers.csv row, or a null value there, get None -- callers turn
    that into an explicit "unknown" segment rather than silently dropping those users,
    since H&M's age/membership columns do have real gaps worth reporting on."""
    out = np.full(n_users, None, dtype=object)
    for user_id, value in zip(customers["user_id"].to_list(), customers[column].to_list()):
        idx = user_id_to_idx.get(user_id)
        if idx is not None:
            out[idx] = value
    return out


def bucket_ages(ages: np.ndarray, edges: tuple[int, int, int, int] = (25, 35, 45, 55)) -> np.ndarray:
    """Buckets a per-user_idx age array (from map_customer_attribute(..., "age")) into
    labeled age groups; None ages become "unknown" rather than being dropped."""
    labels = []
    for age in ages:
        if age is None:
            labels.append("unknown")
        elif age < edges[0]:
            labels.append(f"<{edges[0]}")
        elif age < edges[1]:
            labels.append(f"{edges[0]}-{edges[1] - 1}")
        elif age < edges[2]:
            labels.append(f"{edges[1]}-{edges[2] - 1}")
        elif age < edges[3]:
            labels.append(f"{edges[2]}-{edges[3] - 1}")
        else:
            labels.append(f"{edges[3]}+")
    return np.array(labels, dtype=object)
