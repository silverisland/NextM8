"""Production station-name joins and scalar/first-element data capacities."""

import json
import numpy as np
import pandas as pd
import pytest
import torch

from forecast_backbones.m4_rum.evaluate import evaluate, load_member
from forecast_backbones.m4_rum import production
from forecast_backbones.m4_rum.infer import predict_run
from forecast_backbones.m4_rum.stations import first_capacity
from forecast_backbones.m4_rum.train import train_m4_rum
from forecast_backbones.m4_rum.transformer_models import TransformersAdaptedM4RumModel
from forecast_backbones.m4_rum.tests.conftest import make_frame
from pipelines.forecasting.data import dataset_with_station_id
from pipelines.forecasting.rows import row_stations
from pipelines.nanwang_metric import NanwangMetric
from pipelines.pipeline_wrapper_inference import unified_inference
from pipelines.transformer_dataset import UnifiedDataset


@pytest.mark.parametrize("as_list", [False, True])
def test_data_capacity_column_and_exact_first_value(cfg, batch, as_list):
    cfg["dataset"].update(
        station_meta=None, capacity_col="installed_mw", station_key="station"
    )
    model = TransformersAdaptedM4RumModel(cfg)
    inputs = dict(batch["batch_dict"], station=["A", "B"])
    inputs["installed_mw"] = (
        torch.tensor([[100.0] + [999.0] * 191, [200.0] + [1.0] * 191])
        if as_list
        else torch.tensor([100.0, 200.0])
    )
    f = model.fe.build(inputs, "cpu")
    torch.testing.assert_close(f.capacity[:, 0], torch.tensor([100.0, 200.0]))
    torch.testing.assert_close(f.history[:, :, 0], inputs["observe_power"] / f.capacity)
    # Incidental station_id values must not override the configured station key.
    other = dict(inputs, station_id=torch.tensor([999, 999]))
    np.testing.assert_array_equal(model.fe.build(other, "cpu").station_id, f.station_id)
    with pytest.raises(ValueError, match="capacity column"):
        model.fe.build({k: v for k, v in inputs.items() if k != "installed_mw"}, "cpu")


@pytest.mark.parametrize(
    "value", [0, -1, np.nan, [], [100, 200], [np.nan] + [100] * 191]
)
def test_bad_first_capacity_is_not_replaced_with_later_values(value):
    with pytest.raises(ValueError, match="Capacity"):
        first_capacity(value)


def test_no_csv_text_ids_keep_names_through_preparation(cfg, tmp_path):
    cfg["dataset"].update(station_meta=None)
    model = TransformersAdaptedM4RumModel(cfg)
    frame = make_frame(days=1).iloc[:2].assign(station_id=["A", "B"])
    source = tmp_path / "text_ids.parquet"
    frame.to_parquet(source, index=False)
    dataset = dataset_with_station_id(
        [str(source)], model.cfg, model.station_meta, tmp_path / "cache"
    )
    batch = UnifiedDataset.collate_fn(
        [dataset[0], dataset[1]], for_hf_trainer=True, drop_str_type=False
    )
    output = model(**batch)
    assert [model.station_meta.name(sid) for sid in output.aux["station_id"]] == [
        "A",
        "B",
    ]
    assert torch.isfinite(output.loss)


def test_station_only_csv_stable_ids_and_table_capacity_priority(cfg, batch, tmp_path):
    path = tmp_path / "station_only.csv"
    table = pd.DataFrame({"station": ["A", "B"], "capacity": [100.0, 200.0]})
    table.to_csv(path, index=False)
    cfg["dataset"].update(
        station_meta=str(path),
        station_key="station",
        station_meta_cols={"cap_mw": "capacity"},
    )
    first = TransformersAdaptedM4RumModel(cfg)
    inputs = dict(
        batch["batch_dict"], station=["B", "A"], cap_power_on=torch.tensor([1.0, 1.0])
    )
    f = first.fe.build(inputs, "cpu")
    torch.testing.assert_close(f.capacity[:, 0], torch.tensor([200.0, 100.0]))
    table.iloc[::-1].to_csv(path, index=False)
    second = TransformersAdaptedM4RumModel(cfg)
    np.testing.assert_array_equal(
        second.fe.build(inputs, "cpu").station_id, f.station_id
    )
    with pytest.raises(ValueError, match="Unknown/conflicting"):
        first.fe.build(dict(inputs, station=["unknown", "A"]), "cpu")
    frame = make_frame(days=1).iloc[:2].assign(station=["A", "B"])
    source = tmp_path / "names.parquet"
    frame.to_parquet(source, index=False)
    loaded = dataset_with_station_id(
        [str(source)], first.cfg, first.station_meta, tmp_path / "cache"
    )
    assert row_stations(loaded, "station_id").tolist() == [
        first.station_meta.id_of("A"),
        first.station_meta.id_of("B"),
    ]
    direct = unified_inference(
        first,
        frame,
        entity_cols=["timestamp_win", "station", "station_id"],
        error_policy="raise",
        progress=False,
        exclude_input_cols=["observe_power_future"],
    )
    assert len(direct) == 2


def test_train_save_reload_eval_and_unlabelled_infer_without_csv(cfg, tmp_path):
    cfg["dataset"].update(
        station_meta=None, station_key="station", capacity_col="installed"
    )
    cfg["model"].update(
        architecture="query_decoder", station_embedding=True, n_station_slots=16
    )
    cfg["train"].update(loss="pinball", slot_dropout=0.0)
    train = (
        make_frame(days=1, stations=(1,))
        .drop(columns="station_id")
        .assign(station="A", installed=100.0)
    )
    source = tmp_path / "train.parquet"
    train.to_parquet(source, index=False)
    valid = (
        make_frame(start="2025-06-20", days=5, stations=(1,), every=15)
        .drop(columns="station_id")
        .assign(station="A")
    )
    caps = np.where(np.arange(len(valid)) % 2, 200.0, 100.0)
    valid["installed"] = [[float(c)] + [999.0] * 191 for c in caps]
    validation = tmp_path / "valid.parquet"
    valid.to_parquet(validation, index=False)
    run = tmp_path / "run"
    train_m4_rum(
        cfg, train_parquet=[str(source)], valid_parquet=[str(validation)], out_dir=run
    )
    model = load_member(run)
    assert model.cfg["dataset"]["station_meta"] is None
    dataset = dataset_with_station_id(
        [str(source)], model.cfg, model.station_meta, tmp_path / "cache"
    )
    sid = int(row_stations(dataset, "station_id")[0])
    assert sid in model.cfg["dataset"]["known_stations"]
    assert sid == model.station_meta.identity({"station": ["A"]})[0]
    assert model.core.station_slots([sid], "cpu").item() != 0
    report = evaluate(
        [str(run)],
        [str(validation)],
        tmp_path / "evaluation",
        full=True,
        progress=False,
        device="cpu",
    )
    np.testing.assert_array_equal(report["result"].cap.to_numpy(), caps)
    assert np.isfinite(report["summary"]["station_mean_nanwang_short_D+1"])
    assert np.isfinite(report["summary"]["station_mean_nanwang_ultra_short"])
    station_report = pd.read_csv(tmp_path / "evaluation/reports/station_metrics.csv")
    assert "fixed station capacity" in station_report.ods_unavailable_reason.iloc[0]
    sample = valid.iloc[:2].drop(columns="observe_power_future")
    predicted = predict_run(run, sample, device="cpu", progress=False)
    np.testing.assert_allclose(
        np.stack(predicted.observe_power_predict),
        np.stack(report["result"].observe_power_predict.iloc[:2]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert predicted.station.tolist() == ["A", "A"]
    unseen = model.station_meta.identity(dict(station=["NEW"]))
    assert model.core.station_slots(unseen, "cpu").item() == 0
    # A scalar-capacity file and list-capacity file can share one loader.
    mixed = dataset_with_station_id(
        [str(source), str(validation)],
        model.cfg,
        model.station_meta,
        tmp_path / "mixed",
    )
    assert len(mixed) == len(train) + len(valid)
    mapping = tmp_path / "run_map.yaml"
    mapping.write_text(f'q6: "{run}"\n')
    production.main(
        [
            "--mode",
            "evaluate",
            "--experiments",
            "q6",
            "--run-map",
            str(mapping),
            "--test",
            str(validation),
            "--out",
            str(tmp_path / "comparison"),
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    manifest = json.loads(
        (tmp_path / "comparison/q6/seed_0/checkpoint_manifest.json").read_text()
    )
    assert manifest["station_meta"] is None
    assert manifest["capacity_source"] == "data_first_value"


def test_nanwang_row_capacity_denominators_and_fixed_equivalence():
    frame = pd.DataFrame(
        {
            "station": ["A", "A"],
            "timestamp_win": pd.to_datetime(["2025-06-01 10:00", "2025-06-02 10:00"]),
            "observe_power_future": [np.zeros(192), np.zeros(192)],
            "observe_power_predict": [np.ones(192), np.ones(192)],
            "cap": [100.0, 200.0],
        }
    )
    metric = NanwangMetric(capacity_col="cap")
    scores = metric.short_d1_daily(frame, gccap=None)
    np.testing.assert_allclose(scores.accuracy_pct, [95.0, 97.5])
    fixed = frame.assign(cap=100.0)
    pd.testing.assert_frame_equal(
        metric.short_d1_daily(fixed, None), NanwangMetric().short_d1_daily(fixed, 100.0)
    )
    frame = pd.concat([frame.iloc[:1]] * 18, ignore_index=True)
    frame["timestamp_win"] = pd.date_range(
        "2025-06-01 06:00", periods=len(frame), freq="15min"
    )
    frame["cap"] = np.where(np.arange(len(frame)) % 2, 200.0, 100.0)
    errors = metric._ultra_per_target(frame, 1.0)
    expected_caps = frame.set_index("timestamp_win").cap.reindex(
        errors.index - pd.Timedelta(minutes=15)
    )
    np.testing.assert_allclose(errors, 1 / (0.2 * expected_caps.to_numpy()))
    fixed = frame.assign(cap=100.0)
    pd.testing.assert_frame_equal(
        metric.ultra_short_daily(fixed, None),
        NanwangMetric().ultra_short_daily(fixed, 100.0),
    )
