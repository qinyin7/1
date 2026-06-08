import numpy as np
import pandas as pd
import torch

from src.candidate_features import FEATURE_COLUMNS
from src.deep_ranking_models import (
    DENSE_COLUMNS,
    CandidateFeatureEncoder,
    DIN,
    DeepFM,
    EnrichedDIN,
    MMoE,
    MultiBehaviorDIN,
    OOVAwareEnrichedDIN,
    build_history_indices,
    build_candidate_content_lookup,
    build_multibehavior_history_tables,
    encode_candidate_content,
    build_rolling_history_tables,
    build_pair_indices,
    build_listwise_groups,
    predict_torch_ranker,
    train_listwise_ranker,
    train_pairwise_ranker,
    train_torch_ranker,
)


def candidate_frame(rows: int = 16) -> pd.DataFrame:
    values = []
    for index in range(rows):
        row = {column: float(index % 4) for column in FEATURE_COLUMNS}
        row.update(
            user_id=index % 3,
            video_id=index,
            first_category=float(index % 2),
            author_id_hash=float(index % 5),
            itemcf_present=float(index % 2),
            content_present=1.0,
            tower_present=float(index % 3 == 0),
        )
        values.append(row)
    return pd.DataFrame(values)


def test_candidate_encoder_maps_unknown_values_to_zero():
    train = candidate_frame()
    encoder = CandidateFeatureEncoder.fit(train)
    evaluation = train.iloc[:1].copy()
    evaluation["video_id"] = 99999
    sparse, dense = encoder.encode(evaluation)

    assert sparse[0, 1] == 0
    assert dense.shape == (1, len(DENSE_COLUMNS))


def test_deep_rankers_have_expected_output_shapes():
    frame = candidate_frame()
    encoder = CandidateFeatureEncoder.fit(frame)
    sparse, dense = encoder.encode(frame)
    sparse_tensor = torch.from_numpy(sparse)
    dense_tensor = torch.from_numpy(dense)
    cardinalities = encoder.cardinalities

    deepfm = DeepFM(cardinalities, len(DENSE_COLUMNS), 8)
    din = DIN(cardinalities, len(DENSE_COLUMNS), 8)
    mmoe = MMoE(cardinalities, len(DENSE_COLUMNS), 8)
    multibehavior = MultiBehaviorDIN(cardinalities, len(DENSE_COLUMNS), 8)
    content_lookup = np.zeros((cardinalities[1], 4), dtype=np.float32)
    enriched = EnrichedDIN(cardinalities, len(DENSE_COLUMNS), 8, content_lookup)
    oov_enriched = OOVAwareEnrichedDIN(
        cardinalities, len(DENSE_COLUMNS) + 4, 8, content_dim=4, id_dropout=0.5
    )
    history = torch.zeros((len(frame), 5), dtype=torch.long)
    watch_history = torch.zeros((len(frame), 5), dtype=torch.float32)

    assert deepfm(sparse_tensor, dense_tensor).shape == (len(frame), 1)
    assert din(sparse_tensor, dense_tensor, history, history).shape == (len(frame), 1)
    assert mmoe(sparse_tensor, dense_tensor).shape == (len(frame), 3)
    assert multibehavior(
        sparse_tensor,
        dense_tensor,
        history,
        history,
        history,
        watch_history,
    ).shape == (len(frame), 1)
    content_history = torch.zeros((len(frame), 5, 4), dtype=torch.float32)
    assert enriched(
        sparse_tensor,
        dense_tensor,
        history,
        history,
        history,
        watch_history,
        history,
        content_history,
        watch_history,
    ).shape == (len(frame), 1)
    dense_with_content = torch.cat([dense_tensor, torch.zeros((len(frame), 4))], dim=1)
    assert oov_enriched(
        sparse_tensor,
        dense_with_content,
        history,
        history,
        history,
        watch_history,
        history,
        content_history,
        watch_history,
    ).shape == (len(frame), 1)


def test_deepfm_training_and_prediction_cpu_fast_check():
    frame = candidate_frame(32)
    encoder = CandidateFeatureEncoder.fit(frame)
    sparse, dense = encoder.encode(frame)
    labels = np.array([[index % 2] for index in range(len(frame))], dtype=np.float32)
    model = DeepFM(encoder.cardinalities, len(DENSE_COLUMNS), 8)

    history, _ = train_torch_ranker(
        model,
        sparse,
        dense,
        labels,
        epochs=1,
        batch_size=16,
        device=torch.device("cpu"),
    )
    predictions, _ = predict_torch_ranker(
        model,
        sparse,
        dense,
        batch_size=16,
        device=torch.device("cpu"),
    )

    assert len(history) == 1
    assert predictions.shape == (len(frame), 1)
    assert np.isfinite(predictions).all()


def test_pairwise_indices_stay_within_user_day_groups():
    frame = candidate_frame(24)
    frame["date"] = np.where(frame.index < 12, 20200827, 20200828)
    frame["label_complete"] = (frame.index % 5 == 0).astype("int8")
    positive, negative = build_pair_indices(frame, negatives_per_positive=3, seed=2026)

    assert len(positive) == len(negative)
    assert len(positive) > 0
    assert frame.loc[positive, "label_complete"].eq(1).all()
    assert frame.loc[negative, "label_complete"].eq(0).all()
    assert (
        frame.loc[positive, ["user_id", "date"]].reset_index(drop=True)
        == frame.loc[negative, ["user_id", "date"]].reset_index(drop=True)
    ).all().all()


def test_pairwise_training_cpu_fast_check():
    frame = candidate_frame(32)
    frame["date"] = 20200827
    frame["user_id"] = np.repeat(np.arange(4), 8)
    frame["label_complete"] = (frame.index % 8 == 0).astype("int8")
    encoder = CandidateFeatureEncoder.fit(frame)
    sparse, dense = encoder.encode(frame)
    positive, negative = build_pair_indices(frame, negatives_per_positive=2, seed=2026)
    model = DeepFM(encoder.cardinalities, len(DENSE_COLUMNS), 8)

    history, _ = train_pairwise_ranker(
        model,
        sparse,
        dense,
        positive,
        negative,
        epochs=1,
        batch_size=8,
        device=torch.device("cpu"),
    )

    assert len(history) == 1
    assert np.isfinite(history[0]["pairwise_loss"])


def test_listwise_groups_and_training_cpu_fast_check():
    frame = candidate_frame(32)
    frame["date"] = 20200827
    frame["user_id"] = np.repeat(np.arange(4), 8)
    frame["label_complete"] = (frame.index % 8 == 0).astype("int8")
    encoder = CandidateFeatureEncoder.fit(frame)
    sparse, dense = encoder.encode(frame)
    groups, labels = build_listwise_groups(frame)
    model = DeepFM(encoder.cardinalities, len(DENSE_COLUMNS), 8)

    history, _ = train_listwise_ranker(
        model,
        sparse,
        dense,
        groups,
        labels,
        epochs=1,
        group_batch_size=2,
        device=torch.device("cpu"),
    )

    assert groups.shape == labels.shape == (4, 8)
    assert len(history) == 1
    assert np.isfinite(history[0]["loss"])


def test_history_indices_combine_date_offsets_and_user_codes():
    frame = candidate_frame(4)
    frame["date"] = [20200827, 20200827, 20200828, 20200828]
    encoder = CandidateFeatureEncoder.fit(frame)
    offsets = {20200827: 0, 20200828: encoder.cardinalities[0]}
    indices = build_history_indices(frame, encoder, offsets)
    user_codes = encoder.encode_values("user_id", frame["user_id"].to_numpy())

    assert np.array_equal(indices[:2], user_codes[:2])
    assert np.array_equal(
        indices[2:], user_codes[2:] + encoder.cardinalities[0]
    )


def test_rolling_history_uses_only_strictly_earlier_dates():
    frame = candidate_frame(3)
    frame["user_id"] = 1
    frame["video_id"] = [10, 11, 12]
    encoder = CandidateFeatureEncoder.fit(frame)
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "video_id": [10, 11, 12],
            "date": [20200827, 20200828, 20200829],
            "timestamp": [1, 2, 3],
            "label_complete": [1, 1, 1],
        }
    )
    items = pd.DataFrame(
        {"video_id": [10, 11, 12], "first_category": [0, 1, 0]}
    )
    (history_items, _), offsets = build_rolling_history_tables(
        interactions,
        items,
        [20200828, 20200829],
        encoder,
        history_length=3,
    )
    user_code = encoder.encode_values("user_id", np.array([1]))[0]
    item_codes = encoder.encode_values("video_id", np.array([10, 11, 12]))

    assert history_items[offsets[20200828] + user_code].tolist() == [
        0,
        0,
        item_codes[0],
    ]
    assert history_items[offsets[20200829] + user_code].tolist() == [
        0,
        item_codes[0],
        item_codes[1],
    ]


def test_multibehavior_history_encodes_labels_and_watch_ratio():
    frame = candidate_frame(2)
    frame["user_id"] = 1
    frame["video_id"] = [10, 11]
    encoder = CandidateFeatureEncoder.fit(frame)
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1],
            "video_id": [10, 11],
            "date": [20200827, 20200828],
            "timestamp": [1, 2],
            "label_complete": [1, 0],
            "label_strong": [1, 0],
            "label_short": [0, 1],
            "watch_ratio": [2.5, 0.1],
        }
    )
    items = pd.DataFrame({"video_id": [10, 11], "first_category": [0, 1]})
    tables, offsets = build_multibehavior_history_tables(
        interactions, items, [20200829], encoder, history_length=2
    )
    _, _, behaviors, watch_ratios = tables
    user_code = encoder.encode_values("user_id", np.array([1]))[0]
    row = offsets[20200829] + user_code

    assert behaviors[row].tolist() == [4, 5]
    assert np.allclose(watch_ratios[row], [0.5, 0.02])


def test_candidate_content_lookup_keeps_oov_zero():
    frame = candidate_frame(3)
    encoder = CandidateFeatureEncoder.fit(frame)
    item_ids = np.array([frame.iloc[0]["video_id"], 99999], dtype=np.int64)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    lookup = build_candidate_content_lookup(encoder, item_ids, vectors)
    known_code = encoder.encode_values(
        "video_id", np.array([frame.iloc[0]["video_id"]])
    )[0]

    assert np.array_equal(lookup[0], np.zeros(2, dtype=np.float32))
    assert np.array_equal(lookup[known_code], vectors[0])


def test_candidate_content_encoding_preserves_oov_item_content():
    frame = candidate_frame(2)
    frame["video_id"] = [10, 99999]
    item_ids = np.array([10, 99999], dtype=np.int64)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    encoded = encode_candidate_content(frame, item_ids, vectors)

    assert np.array_equal(encoded, vectors)
