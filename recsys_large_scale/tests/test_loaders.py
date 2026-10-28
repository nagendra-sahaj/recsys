import polars as pl

from recsys_large_scale.data.loaders import load_movielens1m_ratings


def test_load_movielens1m_ratings_parses_double_colon_format(tmp_path):
    """ML-1M's ratings.dat is "::"-separated with no header -- the format polars can't
    read directly, which is the whole reason this loader exists."""
    (tmp_path / "ratings.dat").write_text(
        "1::1193::5::978300760\n1::661::3::978302109\n2::1357::4::978298709\n"
    )

    df = load_movielens1m_ratings(tmp_path)

    assert df.columns == ["user_id", "item_id", "rating", "timestamp"]
    assert len(df) == 3
    assert df["user_id"].to_list() == [1, 1, 2]
    assert df["item_id"].to_list() == [1193, 661, 1357]
    assert df["rating"].to_list() == [5.0, 3.0, 4.0]
    assert df["timestamp"].to_list() == [978300760, 978302109, 978298709]


def test_load_movielens1m_ratings_schema_matches_the_25m_loader(tmp_path):
    """The two MovieLens loaders must agree on column names AND dtypes, because
    everything downstream (to_implicit_feedback, filter_min_interactions, the splits)
    is shared between the two datasets and reads these columns by name."""
    (tmp_path / "ratings.dat").write_text("1::1193::5::978300760\n")

    df = load_movielens1m_ratings(tmp_path)

    assert df.schema == pl.Schema(
        {
            "user_id": pl.Int64,
            "item_id": pl.Int64,
            "rating": pl.Float64,
            "timestamp": pl.Int64,
        }
    )
