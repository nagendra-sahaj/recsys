import numpy as np
import polars as pl

GENRE_DELIMITER = "|"
NO_GENRES_LISTED = "(no genres listed)"
SECONDS_PER_DAY = 86_400
NEVER_PURCHASED_DAYS = 9999.0

FEATURE_NAMES = [
    "ann_score",
    "item_popularity_log",
    "user_activity_log",
    "itemknn_score",
    "genre_affinity",
    "is_injected",
]

# H&M's two-stage pipeline (scripts/run_two_stage_pipeline_hm.py, models/two_stage_hm.py)
# uses this feature set instead -- category_affinity stands in for genre_affinity
# (H&M articles have one category, not MovieLens's multi-genre lists), plus two
# recency features that only make sense on H&M's real daily timestamps. Kept
# separate from FEATURE_NAMES/build_candidate_features above so the existing,
# already-trained MovieLens two_stage.pkl is never put at risk by a shared-schema
# change -- see LEARNINGS.md for why.
FEATURE_NAMES_HM = [
    "ann_score",
    "item_popularity_log",
    "user_activity_log",
    "itemknn_score",
    "category_affinity",
    "is_injected",
    "days_since_category_purchase",
    "has_category_history",
]


def build_item_genre_matrix(
    movies: pl.DataFrame, item_id_to_idx: dict[int, int], n_items: int
) -> tuple[np.ndarray, list[str]]:
    """Binary (n_items, n_genres) matrix from MovieLens' pipe-delimited genre string
    (e.g. "Action|Adventure|Sci-Fi"). Movies with no id_maps entry (filtered out during
    Phase 0's min-interactions cleaning) are skipped entirely -- including from the
    genre vocabulary itself, so a genre that only ever appeared on a filtered-out movie
    doesn't leave a permanently-zero column behind."""
    mapped_rows = []
    for item_id, genres_str in zip(movies["item_id"].to_list(), movies["genres"].to_list()):
        item_idx = item_id_to_idx.get(item_id)
        if item_idx is None:
            continue
        genres = [g for g in genres_str.split(GENRE_DELIMITER) if g and g != NO_GENRES_LISTED]
        mapped_rows.append((item_idx, genres))

    all_genres = sorted({g for _, genres in mapped_rows for g in genres})
    genre_to_col = {g: i for i, g in enumerate(all_genres)}

    matrix = np.zeros((n_items, len(all_genres)), dtype=np.float32)
    for item_idx, genres in mapped_rows:
        for g in genres:
            matrix[item_idx, genre_to_col[g]] = 1.0
    return matrix, all_genres


def build_user_genre_affinity(train: pl.DataFrame, item_genre_matrix: np.ndarray, n_users: int) -> np.ndarray:
    """(n_users, n_genres): how many of this user's training interactions were with an
    item of each genre -- a simple content-based taste profile, used as a feature via
    dot product against a candidate item's own genre vector (see
    build_candidate_features)."""
    user_idx = train["user_idx"].to_numpy()
    item_idx = train["item_idx"].to_numpy()
    n_genres = item_genre_matrix.shape[1]
    affinity = np.zeros((n_users, n_genres), dtype=np.float32)
    np.add.at(affinity, user_idx, item_genre_matrix[item_idx])
    return affinity


def build_candidate_features(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    ann_score: np.ndarray,
    item_pop_counts: np.ndarray,
    user_activity_counts: np.ndarray,
    itemknn_score: np.ndarray,
    user_genre_affinity: np.ndarray,
    item_genre_matrix: np.ndarray,
    is_injected: np.ndarray,
) -> np.ndarray:
    """All arguments except the two count/affinity/matrix lookups are flat, parallel
    (n_pairs,) arrays -- one row per (user, candidate item) pair. Returns
    (n_pairs, len(FEATURE_NAMES)) float32, in FEATURE_NAMES order:
      - ann_score: the retrieval stage's own inner-product score for this pair
      - item_popularity_log: log1p(training interaction count) for the item
      - user_activity_log: log1p(training interaction count) for the user
      - itemknn_score: item-item CF's score for this pair (an independent signal)
      - genre_affinity: dot product of the user's genre taste profile and this item's
        genre vector -- how much has this user historically engaged with items of
        this item's genre(s)
      - is_injected: 1.0 if this (user, item) pair was force-inserted into the
        candidate set during *training* because ANN retrieval missed the true
        held-out item (see scripts/run_two_stage_pipeline.py) -- always 0.0 at
        inference time, since score_batch never injects anything, only ever scores
        candidates the retrieval stage genuinely found. Exists because an injected
        row's ann_score is *necessarily* low (if it scored well, ANN would have found
        it unaided) while its label is positive -- a majority-share contradiction of
        the "high ann_score -> relevant" pattern in the non-injected rows, found to
        badly corrupt the reranker's use of ann_score without this flag to let it
        separate the two cases (see LEARNINGS.md).
    """
    genre_affinity = (user_genre_affinity[user_idx] * item_genre_matrix[item_idx]).sum(axis=1)
    features = np.stack(
        [
            ann_score,
            np.log1p(item_pop_counts[item_idx]),
            np.log1p(user_activity_counts[user_idx]),
            itemknn_score,
            genre_affinity,
            is_injected,
        ],
        axis=1,
    )
    return features.astype(np.float32)


def build_item_category_matrix(
    articles: pl.DataFrame, item_id_to_idx: dict, n_items: int, category_column: str = "product_group_name"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """H&M's category matrix: unlike MovieLens's multi-genre movies, each article
    belongs to exactly *one* category. Defaults to `product_group_name` (H&M's
    broadest category level, ~19 groups like "Garment Upper body"/"Shoes"/
    "Accessories") rather than the far more granular `product_type_name` (~130
    types), so most users have *some* purchase history in a given candidate's
    category even with H&M's sparse per-user interaction counts.

    Returns:
      - a one-hot (n_items, n_categories) matrix, same shape/convention as
        build_item_genre_matrix, so build_user_genre_affinity works unchanged as a
        "category affinity" builder on either dataset's matrix
      - item_category_idx: (n_items,) int array, the column index of each item's
        category (-1 for an item with no id_maps entry or a null category) -- a
        direct lookup used by build_category_recency_features, cheaper than
        re-deriving "argmax of the one-hot row" on every call
      - the category name list, in column order
    """
    mapped_rows = []
    for item_id, category in zip(articles["item_id"].to_list(), articles[category_column].to_list()):
        item_idx = item_id_to_idx.get(item_id)
        if item_idx is None or category is None:
            continue
        mapped_rows.append((item_idx, category))

    categories = sorted({c for _, c in mapped_rows})
    category_to_col = {c: i for i, c in enumerate(categories)}

    matrix = np.zeros((n_items, len(categories)), dtype=np.float32)
    item_category_idx = np.full(n_items, -1, dtype=np.int64)
    for item_idx, category in mapped_rows:
        col = category_to_col[category]
        matrix[item_idx, col] = 1.0
        item_category_idx[item_idx] = col
    return matrix, item_category_idx, categories


def build_user_category_last_purchase(
    train: pl.DataFrame, item_category_idx: np.ndarray, n_users: int, n_categories: int
) -> np.ndarray:
    """(n_users, n_categories) int64 array: this user's most recent train-interaction
    timestamp with an item of that category, or -1 if they never bought one -- the
    lookup table behind the recency reranker feature (days since last purchase in
    this candidate's category). Items with no category (item_category_idx == -1,
    e.g. missing from articles.csv) are excluded from every user's history, not
    assigned their own "unknown category" column."""
    user_idx = train["user_idx"].to_numpy()
    item_idx = train["item_idx"].to_numpy()
    ts = train["timestamp"].to_numpy()
    category = item_category_idx[item_idx]

    has_category = category >= 0
    last_ts = np.full((n_users, n_categories), -1, dtype=np.int64)
    np.maximum.at(last_ts, (user_idx[has_category], category[has_category]), ts[has_category])
    return last_ts


def build_category_recency_features(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    item_category_idx: np.ndarray,
    user_category_last_ts: np.ndarray,
    reference_timestamp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each (user, candidate item) pair, how many days ago did this user last buy
    something in this item's category, as of `reference_timestamp` (the train split's
    own max timestamp -- a single fixed "now," not the target interaction's own
    timestamp, to avoid leaking exactly when the held-out purchase happens).

    Returns (days_since_category_purchase, has_category_history), both (n_pairs,)
    float32. A user with no prior purchase in that category (or an item with no
    category at all) gets the fixed NEVER_PURCHASED_DAYS sentinel with
    has_category_history=0.0 -- paired the same way is_injected pairs with ann_score,
    so the model can tell "a long time ago" apart from "never happened" instead of
    reading the sentinel as just an extreme case of the former."""
    category = item_category_idx[item_idx]
    valid = category >= 0
    last_ts = np.where(valid, user_category_last_ts[user_idx, np.clip(category, 0, None)], -1)
    has_history = ((last_ts >= 0) & valid).astype(np.float32)
    days_since = np.where(
        has_history > 0, (reference_timestamp - last_ts) / SECONDS_PER_DAY, NEVER_PURCHASED_DAYS
    ).astype(np.float32)
    return days_since, has_history


def build_candidate_features_hm(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    ann_score: np.ndarray,
    item_pop_counts: np.ndarray,
    user_activity_counts: np.ndarray,
    itemknn_score: np.ndarray,
    user_category_affinity: np.ndarray,
    item_category_matrix: np.ndarray,
    is_injected: np.ndarray,
    days_since_category_purchase: np.ndarray,
    has_category_history: np.ndarray,
) -> np.ndarray:
    """H&M's candidate feature matrix, in FEATURE_NAMES_HM order -- see that list's
    comment and build_candidate_features' docstring for the first 6 features (shared
    mechanism, category_affinity standing in for genre_affinity); the last two are
    documented in build_category_recency_features."""
    category_affinity = (user_category_affinity[user_idx] * item_category_matrix[item_idx]).sum(axis=1)
    features = np.stack(
        [
            ann_score,
            np.log1p(item_pop_counts[item_idx]),
            np.log1p(user_activity_counts[user_idx]),
            itemknn_score,
            category_affinity,
            is_injected,
            days_since_category_purchase,
            has_category_history,
        ],
        axis=1,
    )
    return features.astype(np.float32)
