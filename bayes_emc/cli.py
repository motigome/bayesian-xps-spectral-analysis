from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import os
import random
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence


class ConfigError(ValueError):
    """Raised when a user-facing configuration error is found."""


@dataclass(frozen=True)
class TuningParameterRef:
    model_name: str
    layer_name: str
    parameter_name: str
    parameter: dict[str, Any]


@dataclass(frozen=True)
class DiagnosticsSummary:
    parameter_rates: dict[tuple[str, str, str], list[tuple[float, bool, float]]]
    parameter_replica_rates: dict[tuple[str, str, str], list[tuple[int, float, bool, float]]]
    exchange_rates: list[tuple[int, int, float, float, float]]


@dataclass(frozen=True)
class NormalizedConfig:
    raw: dict[str, Any]
    config_path: Path
    project_dir: Path
    data_path: Path
    data_format: str
    data_header: bool
    input_columns: list[str]
    output_columns: list[str]
    result_dir: Path
    input_dim: int
    output_dim: int
    model_type_num: int
    base_nums: list[int]
    hierarchy_nums: list[int]
    parameter_nums: list[list[int]]
    replica_num: int
    gamma: float
    sample_num: int
    burnin_num: int
    parallel_workers: int
    likelihood_workers: int
    likelihood_parallel_min_rows: int
    progress: bool
    progress_interval_steps: int
    progress_bar_width: int
    tuning: list[list[float]]
    model_layout: list[dict[str, Any]]
    noise_type: str
    sigma2_min: float
    sigma2_candidate_max: float
    estimate_sigma2: bool
    build: dict[str, Any]

    @property
    def src_dir(self) -> Path:
        return self.project_dir / "src"

def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive integer.")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ConfigError(f"{name} must be a non-negative integer.")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive number.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ConfigError(f"{name} must be a positive finite number.")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a non-negative number.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ConfigError(f"{name} must be a non-negative finite number.")
    return parsed


def _bool_option(raw: dict[str, Any], keys: Iterable[str], fallback: bool, name: str) -> bool:
    value = fallback
    for key in keys:
        if key in raw:
            value = raw[key]
            break
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false.")
    return value


def _first_present(raw: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, Any]:
    for key in keys:
        if key in raw:
            return key, raw[key]
    return None, None


DEFAULT_PROPOSAL_DECAY = 0.5


def _normalize_model_structure(
    model: dict[str, Any],
    tuning_section: dict[str, Any],
) -> tuple[int, list[int], list[int], list[list[int]], list[list[float]], list[dict[str, Any]]]:
    if "models" in model:
        return _normalize_named_model_structure(model["models"], tuning_section)
    raise ConfigError("model.models is required. Use named models with basis_count and parameters.")


def _normalize_named_model_structure(
    models_value: Any,
    tuning_section: dict[str, Any],
) -> tuple[int, list[int], list[int], list[list[int]], list[list[float]], list[dict[str, Any]]]:
    if not isinstance(models_value, list) or not models_value:
        raise ConfigError("model.models must be a non-empty list.")

    base_nums: list[int] = []
    hierarchy_nums: list[int] = []
    parameter_nums: list[list[int]] = []
    inline_tuning: list[list[float] | None] = []
    layout: list[dict[str, Any]] = []

    for model_id, model_item in enumerate(models_value):
        if not isinstance(model_item, dict):
            raise ConfigError(f"model.models[{model_id}] must be an object.")
        model_name = str(model_item.get("name", f"model_{model_id}"))
        count_key, count_value = _first_present(
            model_item,
            ["basis_count", "component_count", "base_count", "components"],
        )
        if count_key is None:
            raise ConfigError(
                f"model.models[{model_id}] must define basis_count "
                "(alias: component_count/base_count/components)."
            )
        base_nums.append(_positive_int(count_value, f"model.models[{model_id}].{count_key}"))

        if "parameters" in model_item:
            layers_value: Any = [{"name": "default", "parameters": model_item["parameters"]}]
        else:
            layer_key, layers_value = _first_present(model_item, ["layers", "hierarchies"])
            if layer_key is None:
                raise ConfigError(f"model.models[{model_id}] must define parameters or layers.")

        if not isinstance(layers_value, list) or not layers_value:
            raise ConfigError(f"model.models[{model_id}] layers must be a non-empty list.")

        model_parameter_nums: list[int] = []
        layer_layout: list[dict[str, Any]] = []
        for layer_id, layer_item in enumerate(layers_value):
            if not isinstance(layer_item, dict):
                raise ConfigError(f"model.models[{model_id}].layers[{layer_id}] must be an object.")
            layer_name = str(layer_item.get("name", f"layer_{layer_id}"))
            parameters = layer_item.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise ConfigError(
                    f"model.models[{model_id}].layers[{layer_id}].parameters "
                    "must be a non-empty list."
                )
            model_parameter_nums.append(len(parameters))

            parameter_names: list[str] = []
            for parameter_id, parameter_item in enumerate(parameters):
                parameter_name, tuning = _parse_named_parameter(parameter_item, model_id, layer_id, parameter_id)
                parameter_names.append(parameter_name)
                inline_tuning.append(tuning)
            layer_layout.append({"name": layer_name, "parameters": parameter_names})

        hierarchy_nums.append(len(layers_value))
        parameter_nums.append(model_parameter_nums)
        layout.append({
            "name": model_name,
            "basis_count": base_nums[-1],
            "component_count": base_nums[-1],
            "layers": layer_layout,
        })

    if all(item is not None for item in inline_tuning):
        tuning = [item for item in inline_tuning if item is not None]
    elif "parameters" in tuning_section:
        tuning = _normalize_tuning(tuning_section, parameter_nums)
    else:
        raise ConfigError(
            "Each model.models parameter must include C, define a prior for auto C, "
            "or provide top-level tuning.parameters."
        )

    return len(models_value), base_nums, hierarchy_nums, parameter_nums, tuning, layout


def _parse_named_parameter(
    parameter_item: Any,
    model_id: int,
    layer_id: int,
    parameter_id: int,
) -> tuple[str, list[float] | None]:
    prefix = f"model.models[{model_id}].layers[{layer_id}].parameters[{parameter_id}]"
    if isinstance(parameter_item, str):
        return parameter_item, None
    if not isinstance(parameter_item, dict):
        raise ConfigError(f"{prefix} must be a string or an object.")
    name = str(parameter_item.get("name", f"p{parameter_id}"))
    tuning = parameter_item.get("tuning", parameter_item)
    if not isinstance(tuning, dict):
        raise ConfigError(f"{prefix}.tuning must be an object when provided.")
    has_c = "C" in tuning
    has_d = "d" in tuning
    if not has_c and "prior" in parameter_item:
        return name, [
            _proposal_c_from_prior(parameter_item["prior"], prefix),
            _nonnegative_float(tuning.get("d", DEFAULT_PROPOSAL_DECAY), f"{prefix}.d"),
        ]
    if has_c:
        return name, [
            _positive_float(tuning.get("C"), f"{prefix}.C"),
            _nonnegative_float(tuning.get("d", DEFAULT_PROPOSAL_DECAY), f"{prefix}.d"),
        ]
    if has_d:
        raise ConfigError(f"{prefix}.C can be omitted only when {prefix}.prior is available for auto tuning.")
    return name, None


def _parameter_replica_step_scales(parameter_item: Any, prefix: str) -> list[float]:
    if not isinstance(parameter_item, dict):
        return []
    tuning = parameter_item.get("tuning", parameter_item)
    if not isinstance(tuning, dict):
        raise ConfigError(f"{prefix}.tuning must be an object when provided.")
    value = tuning.get("replica_step_scales", tuning.get("step_size_scales"))
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{prefix}.replica_step_scales must be a list when provided.")
    return [
        _positive_float(item, f"{prefix}.replica_step_scales[{index}]")
        for index, item in enumerate(value)
    ]


def _validate_replica_step_scales_for_raw(raw: dict[str, Any], replica_num: int) -> None:
    model_section = raw.get("model")
    if not isinstance(model_section, dict):
        return
    models = model_section.get("models")
    if not isinstance(models, list):
        return
    for model_id, model_item in enumerate(models):
        if not isinstance(model_item, dict):
            continue
        if "parameters" in model_item:
            layer_items = [{"name": "default", "parameters": model_item["parameters"]}]
        else:
            layer_items = model_item.get("layers", model_item.get("hierarchies"))
        if not isinstance(layer_items, list):
            continue
        for layer_id, layer_item in enumerate(layer_items):
            if not isinstance(layer_item, dict):
                continue
            parameters = layer_item.get("parameters")
            if not isinstance(parameters, list):
                continue
            for parameter_id, parameter_item in enumerate(parameters):
                prefix = f"model.models[{model_id}].layers[{layer_id}].parameters[{parameter_id}]"
                scales = _parameter_replica_step_scales(parameter_item, prefix)
                if scales and len(scales) != replica_num:
                    raise ConfigError(
                        f"{prefix}.replica_step_scales length must match emc.replica_num "
                        f"({replica_num})."
                    )


def _proposal_c_from_prior(prior: Any, prefix: str) -> float:
    if not isinstance(prior, dict):
        raise ConfigError(f"{prefix}.prior must be an object when C is omitted.")
    kind = str(prior.get("type", prior.get("kind", ""))).lower()
    if kind == "uniform":
        lower = prior.get("lower", prior.get("min"))
        upper = prior.get("upper", prior.get("max"))
        if lower is None or upper is None:
            raise ConfigError(f"{prefix}.prior.lower and .upper are required for auto C with a uniform prior.")
        lower_value = float(lower)
        upper_value = float(upper)
        if not lower_value < upper_value:
            raise ConfigError(f"{prefix}.prior requires lower < upper.")
        return (upper_value - lower_value) / 3.0
    if kind in {"normal", "gaussian"}:
        sigma = prior.get("sigma", prior.get("std", prior.get("sd")))
        if sigma is None:
            raise ConfigError(f"{prefix}.prior.sigma is required for auto C with a normal prior.")
        return _positive_float(sigma, f"{prefix}.prior.sigma") / 3.0
    if kind == "gamma":
        shape = prior.get("shape", prior.get("alpha"))
        scale = prior.get("scale", prior.get("theta"))
        if shape is None or scale is None:
            raise ConfigError(f"{prefix}.prior.shape and .scale are required for auto C with a gamma prior.")
        shape_value = _positive_float(shape, f"{prefix}.prior.shape")
        scale_value = _positive_float(scale, f"{prefix}.prior.scale")
        return math.sqrt(shape_value) * scale_value / 3.0
    if kind == "beta":
        _alpha_value, _beta_value, lower_value, upper_value = _beta_prior_parameters(prior, prefix)
        return (upper_value - lower_value) / 3.0
    raise ConfigError(f"{prefix}.prior.type must be beta, normal, gamma, or uniform for auto C.")


def _beta_prior_parameters(prior: dict[str, Any], prefix: str) -> tuple[float, float, float, float]:
    alpha = prior.get("alpha", prior.get("shape1", prior.get("shape_alpha")))
    beta = prior.get("beta", prior.get("shape2", prior.get("shape_beta")))
    lower = prior.get("lower", prior.get("min"))
    upper = prior.get("upper", prior.get("max"))
    if alpha is None or beta is None:
        raise ConfigError(f"{prefix}.prior.alpha and .beta are required for a beta prior.")
    if lower is None or upper is None:
        raise ConfigError(f"{prefix}.prior.lower and .upper are required for a beta prior.")
    alpha_value = _positive_float(alpha, f"{prefix}.prior.alpha")
    beta_value = _positive_float(beta, f"{prefix}.prior.beta")
    lower_value = float(lower)
    upper_value = float(upper)
    if not lower_value < upper_value:
        raise ConfigError(f"{prefix}.prior requires lower < upper.")
    return alpha_value, beta_value, lower_value, upper_value


def _resolve_relative(path_value: str, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _normalize_data_format(value: Any, data_path: Path) -> str:
    parsed = str(value or "auto").lower()
    if parsed == "auto":
        suffix = data_path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix in {".tsv", ".tab"}:
            return "tsv"
        return "whitespace"
    aliases = {
        "space": "whitespace",
        "spaces": "whitespace",
        "txt": "whitespace",
        "text": "whitespace",
    }
    parsed = aliases.get(parsed, parsed)
    if parsed not in {"whitespace", "csv", "tsv"}:
        raise ConfigError("data.format must be auto, whitespace, csv, or tsv.")
    return parsed


def _normalize_column_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list when provided.")
    columns: list[str] = []
    for index, item in enumerate(value):
        column = str(item).strip()
        if not column:
            raise ConfigError(f"{name}[{index}] must not be empty.")
        columns.append(column)
    return columns


def load_config(config_path: Path) -> NormalizedConfig:
    config_path = config_path.expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc

    project_dir = config_path.parent
    return _normalize_config(raw, config_path, project_dir)


def _normalize_config(raw: dict[str, Any], config_path: Path, project_dir: Path) -> NormalizedConfig:
    if "emc" not in raw:
        raise ConfigError("config.json must use the V2 schema with emc, data, and model sections.")

    emc = _require_dict(raw, "emc")
    data = _require_dict(raw, "data")
    model = _require_dict(raw, "model")
    tuning_section = raw.get("tuning", {})
    if not isinstance(tuning_section, dict):
        raise ConfigError("tuning must be an object when provided.")
    project = raw.get("project", {})
    if not isinstance(project, dict):
        raise ConfigError("project must be an object when provided.")

    model_type_num, base_nums, hierarchy_nums, parameter_nums, tuning, model_layout = _normalize_model_structure(
        model,
        tuning_section,
    )
    data_path = _resolve_relative(str(data.get("path", "data/data.txt")), project_dir)
    data_format = _normalize_data_format(data.get("format", "auto"), data_path)
    data_header_raw = data.get("header", data.get("has_header", False))
    if not isinstance(data_header_raw, bool):
        raise ConfigError("data.header must be true or false when provided.")
    input_columns = _normalize_column_list(
        data.get("input_columns", data.get("x_columns")),
        "data.input_columns",
    )
    output_columns = _normalize_column_list(
        data.get("output_columns", data.get("y_columns")),
        "data.output_columns",
    )
    input_dim = _positive_int(data.get("input_dim"), "data.input_dim")
    output_dim = _positive_int(data.get("output_dim"), "data.output_dim")
    if input_columns and len(input_columns) != input_dim:
        raise ConfigError("data.input_columns length must match data.input_dim.")
    if output_columns and len(output_columns) != output_dim:
        raise ConfigError("data.output_columns length must match data.output_dim.")
    if (input_columns or output_columns) and not data_header_raw:
        raise ConfigError("data.input_columns/output_columns require data.header = true.")
    result_dir = _resolve_relative(str(project.get("result_dir", raw.get("result_dir", "result"))), project_dir)

    noise = model.get("noise", {"type": "gaussian", "sigma2_min": model.get("sigma2", 0.01)})
    if not isinstance(noise, dict):
        raise ConfigError("model.noise must be an object.")
    if "sigma2_max" in noise:
        raise ConfigError(
            "model.noise.sigma2_max is derived from the finite EMC temperature ladder and is not a config input. "
            "Set model.noise.sigma2_min as the beta=1 minimum noise variance instead."
        )
    noise_type = str(noise.get("type", "gaussian")).lower()
    if noise_type not in {"gaussian", "poisson"}:
        raise ConfigError("model.noise.type must be gaussian or poisson.")
    sigma2_min = _positive_float(
        noise.get("sigma2_min", noise.get("sigma2", 0.01)),
        "model.noise.sigma2_min",
    )
    estimate_sigma2 = _bool_option(
        noise,
        ["estimate_sigma2", "estimate"],
        True,
        "model.noise.estimate_sigma2",
    )

    gamma = _positive_float(emc.get("gamma"), "emc.gamma")
    if gamma < 1.0:
        raise ConfigError("emc.gamma must be greater than or equal to 1.0.")
    replica_num = _positive_int(emc.get("replica_num"), "emc.replica_num")
    if replica_num < 2:
        raise ConfigError("emc.replica_num must be at least 2 for Gaussian noise free-energy calculation.")
    _validate_replica_step_scales_for_raw(raw, replica_num)
    sigma2_candidate_max = _noise_sigma2_candidate_max(sigma2_min, gamma, replica_num)

    return NormalizedConfig(
        raw=raw,
        config_path=config_path,
        project_dir=project_dir,
        data_path=data_path,
        data_format=data_format,
        data_header=data_header_raw,
        input_columns=input_columns,
        output_columns=output_columns,
        result_dir=result_dir,
        input_dim=input_dim,
        output_dim=output_dim,
        model_type_num=model_type_num,
        base_nums=base_nums,
        hierarchy_nums=hierarchy_nums,
        parameter_nums=parameter_nums,
        replica_num=replica_num,
        gamma=gamma,
        sample_num=_positive_int(emc.get("sample_num"), "emc.sample_num"),
        burnin_num=_positive_int(emc.get("burnin_num"), "emc.burnin_num"),
        parallel_workers=_v2_nonnegative_optional_int(emc, "parallel_workers", 0),
        likelihood_workers=_v2_nonnegative_optional_int(emc, "likelihood_workers", 1),
        likelihood_parallel_min_rows=_v2_nonnegative_optional_int(emc, "likelihood_parallel_min_rows", 2048),
        progress=_bool_option(emc, ["progress", "show_progress"], True, "emc.progress"),
        progress_interval_steps=_v2_nonnegative_optional_int(emc, "progress_interval_steps", 0),
        progress_bar_width=_v2_positive_optional_int(emc, "progress_bar_width", 32),
        tuning=tuning,
        model_layout=model_layout,
        noise_type=noise_type,
        sigma2_min=sigma2_min,
        sigma2_candidate_max=sigma2_candidate_max,
        estimate_sigma2=estimate_sigma2,
        build=_normalize_build(raw.get("build", {})),
    )


def _require_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object.")
    return value


def _noise_sigma2_candidate_max(base_sigma2: float, gamma: float, replica_num: int) -> float:
    # Free-energy records are emitted at beta_right for each temperature interval.
    # Therefore the largest reported candidate is base_sigma2 / beta[1].
    exponent = max(0, replica_num - 2)
    return base_sigma2 * (gamma ** exponent)


def _normalize_tuning(tuning_section: dict[str, Any], parameter_nums: list[list[int]]) -> list[list[float]]:
    parameters = tuning_section.get("parameters")
    if not isinstance(parameters, list):
        raise ConfigError("tuning.parameters must be a list.")
    expected = sum(sum(model) for model in parameter_nums)
    if len(parameters) != expected:
        raise ConfigError(f"tuning.parameters must contain {expected} parameter setting(s).")
    tuning: list[list[float]] = []
    for idx, item in enumerate(parameters):
        if isinstance(item, dict):
            c_value = item.get("C")
            d_value = item.get("d")
        elif isinstance(item, list) and len(item) == 2:
            c_value, d_value = item
        else:
            raise ConfigError(f"tuning.parameters[{idx}] must be an object with C/d or a two-item list.")
        tuning.append([
            _positive_float(c_value, f"tuning.parameters[{idx}].C"),
            _nonnegative_float(d_value, f"tuning.parameters[{idx}].d"),
        ])
    return tuning


def _normalize_build(build: Any) -> dict[str, Any]:
    if build is None:
        build = {}
    if not isinstance(build, dict):
        raise ConfigError("build must be an object when provided.")
    return {
        "compiler": str(build.get("compiler", "c++")),
        "output": str(build.get("output", "main.out")),
        "include_dirs": list(build.get("include_dirs", [])),
        "library_dirs": list(build.get("library_dirs", [])),
        "flags": list(build.get("flags", ["-std=c++20", "-O3", "-ffast-math", "-funsafe-math-optimizations", "-pthread"])),
        "libs": list(build.get("libs", [])),
    }


def validate_config(config: NormalizedConfig, *, require_sources: bool = False) -> list[str]:
    warnings: list[str] = []

    if config.noise_type not in {"gaussian", "poisson"}:
        raise ConfigError("model.noise.type must be gaussian or poisson.")

    if not config.data_path.exists():
        raise ConfigError(f"Data file not found: {config.data_path}")

    checked_rows = _count_data_rows(config)

    if checked_rows == 0:
        raise ConfigError(f"Data file contains no rows: {config.data_path}")

    if require_sources:
        for relative in ["src/main.cpp", "src/target.hpp"]:
            source = config.project_dir / relative
            if not source.exists():
                raise ConfigError(f"Required C++ source file not found: {source}")

    return warnings


def _split_data_row(line: str, data_format: str) -> list[str]:
    if data_format == "whitespace":
        return line.split()
    delimiter = "," if data_format == "csv" else "\t"
    return [item.strip() for item in next(csv.reader([line], delimiter=delimiter))]


def _resolve_data_column_indices(
    *,
    config: NormalizedConfig,
    header: list[str] | None,
    selected: list[str],
    count: int,
    offset: int,
    label: str,
) -> list[int]:
    if not selected:
        return list(range(offset, offset + count))
    if header is None:
        raise ConfigError(f"data.{label}_columns requires data.header = true.")
    indices: list[int] = []
    for column in selected:
        try:
            indices.append(header.index(column))
        except ValueError as exc:
            raise ConfigError(f"{config.data_path}: data column not found: {column}") from exc
    return indices


def _count_data_rows(config: NormalizedConfig) -> int:
    expected_columns = config.input_dim + config.output_dim
    header: list[str] | None = None
    input_indices: list[int] | None = None
    output_indices: list[int] | None = None
    named_columns = bool(config.input_columns or config.output_columns)
    checked_rows = 0
    for line_number, line in enumerate(config.data_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = _split_data_row(stripped, config.data_format)
        if config.data_header and header is None:
            header = parts
            continue
        if input_indices is None or output_indices is None:
            input_indices = _resolve_data_column_indices(
                config=config,
                header=header,
                selected=config.input_columns,
                count=config.input_dim,
                offset=0,
                label="input",
            )
            output_indices = _resolve_data_column_indices(
                config=config,
                header=header,
                selected=config.output_columns,
                count=config.output_dim,
                offset=config.input_dim,
                label="output",
            )
        if not named_columns and len(parts) != expected_columns:
            raise ConfigError(
                f"{config.data_path}:{line_number} has {len(parts)} column(s); "
                f"expected {expected_columns}."
            )
        for column_index in [*input_indices, *output_indices]:
            if column_index >= len(parts):
                raise ConfigError(f"{config.data_path}:{line_number} is missing column {column_index}.")
            try:
                float(parts[column_index])
            except ValueError as exc:
                raise ConfigError(f"{config.data_path}:{line_number} contains a non-numeric value.") from exc
        checked_rows += 1
    return checked_rows


def _cpp_string_literal(value: str | Path) -> str:
    return json.dumps(str(value))


def _cpp_string_list(values: list[str]) -> str:
    return "{" + ", ".join(_cpp_string_literal(value) for value in values) + "}"


def _cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def _cpp_data_format(value: str) -> str:
    return {
        "whitespace": "Whitespace",
        "csv": "Csv",
        "tsv": "Tsv",
    }[value]


def _cpp_number(value: float | int) -> str:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ConfigError("Cannot write a non-finite number to generated C++.")
    return repr(parsed)


def _cpp_number_list(values: list[float]) -> str:
    return "{" + ", ".join(_cpp_number(value) for value in values) + "}"


def _v2_generated_header_path(config: NormalizedConfig) -> Path:
    return config.src_dir / "generated_v2_config.hpp"


def _v2_include_root() -> Path:
    source_checkout = Path(__file__).resolve().parents[1] / "cpp" / "include"
    installed_data = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "bayesian-xps-spectral-analysis"
        / "cpp"
        / "include"
    )
    for candidate in (source_checkout, installed_data):
        if (candidate / "bayes_emc" / "bayes_emc.hpp").is_file():
            return candidate
    raise ConfigError(
        "Bundled C++ headers were not found. Reinstall bayesian-xps-spectral-analysis "
        "or run from a complete source checkout."
    )


def _v2_positive_optional_int(raw: dict[str, Any], key: str, fallback: int) -> int:
    if key not in raw:
        return fallback
    return _positive_int(raw[key], f"emc.{key}")


def _v2_nonnegative_optional_int(raw: dict[str, Any], key: str, fallback: int) -> int:
    if key not in raw:
        return fallback
    return _nonnegative_int(raw[key], f"emc.{key}")


def _v2_prior_expression(parameter_item: Any, prefix: str) -> str:
    if not isinstance(parameter_item, dict):
        raise ConfigError(f"{prefix} must be an object with a prior for run.")
    prior = parameter_item.get("prior")
    if not isinstance(prior, dict):
        raise ConfigError(f"{prefix}.prior is required for run.")
    kind = str(prior.get("type", prior.get("kind", ""))).lower()
    if kind in {"normal", "gaussian"}:
        mean = prior.get("mean", prior.get("mu", 0.0))
        sigma = prior.get("sigma", prior.get("std", prior.get("sd")))
        if sigma is None:
            raise ConfigError(f"{prefix}.prior.sigma is required for a normal prior.")
        return (
            "PriorDistribution::Normal("
            f"{_cpp_number(float(mean))}, {_cpp_number(_positive_float(sigma, prefix + '.prior.sigma'))})"
        )
    if kind == "gamma":
        shape = prior.get("shape", prior.get("alpha"))
        scale = prior.get("scale", prior.get("theta"))
        if shape is None or scale is None:
            raise ConfigError(f"{prefix}.prior.shape and .scale are required for a gamma prior.")
        return (
            "PriorDistribution::Gamma("
            f"{_cpp_number(_positive_float(shape, prefix + '.prior.shape'))}, "
            f"{_cpp_number(_positive_float(scale, prefix + '.prior.scale'))})"
        )
    if kind == "uniform":
        lower = prior.get("lower", prior.get("min"))
        upper = prior.get("upper", prior.get("max"))
        if lower is None or upper is None:
            raise ConfigError(f"{prefix}.prior.lower and .upper are required for a uniform prior.")
        lower_value = float(lower)
        upper_value = float(upper)
        if not lower_value < upper_value:
            raise ConfigError(f"{prefix}.prior requires lower < upper.")
        return f"PriorDistribution::Uniform({_cpp_number(lower_value)}, {_cpp_number(upper_value)})"
    if kind == "beta":
        alpha_value, beta_value, lower_value, upper_value = _beta_prior_parameters(prior, prefix)
        return (
            "PriorDistribution::Beta("
            f"{_cpp_number(alpha_value)}, "
            f"{_cpp_number(beta_value)}, "
            f"{_cpp_number(lower_value)}, "
            f"{_cpp_number(upper_value)})"
        )
    raise ConfigError(f"{prefix}.prior.type must be beta, normal, gamma, or uniform.")


def _write_v2_generated_config(config: NormalizedConfig) -> Path:
    model = _require_dict(config.raw, "model")
    models_value = model.get("models")
    if not isinstance(models_value, list) or not models_value:
        raise ConfigError("run requires model.models with named parameters and priors.")
    if config.noise_type not in {"gaussian", "poisson"}:
        raise ConfigError("run currently supports model.noise.type = gaussian or poisson.")

    lines: list[str] = [
        "#pragma once",
        "",
        '#include "bayes_emc/bayes_emc.hpp"',
        "",
        "#include <string>",
        "#include <utility>",
        "#include <vector>",
        "",
        "namespace bayes_emc_generated {",
        "",
        "inline bayes_emc::AnalysisSpec MakeAnalysisSpec() {",
        "    using namespace bayes_emc;",
        "    AnalysisSpec spec;",
        f"    spec.input_dim = {config.input_dim};",
        f"    spec.output_dim = {config.output_dim};",
        "    spec.likelihood_type = "
        + (
            "LikelihoodType::Poisson;"
            if config.noise_type == "poisson"
            else "LikelihoodType::Gaussian;"
        ),
        f"    spec.gaussian_sigma2 = {_cpp_number(config.sigma2_min)};",
    ]

    for model_id, model_item in enumerate(models_value):
        if not isinstance(model_item, dict):
            raise ConfigError(f"model.models[{model_id}] must be an object.")
        model_name = str(model_item.get("name", f"model_{model_id}"))
        basis_count = config.base_nums[model_id]
        if "parameters" in model_item:
            layer_items: list[Any] = [{"name": "default", "parameters": model_item["parameters"]}]
            lines.append("    spec.models.push_back(ModelSpec::WithParameters(")
            lines.append(f"        {_cpp_string_literal(model_name)},")
            lines.append(f"        {basis_count},")
            lines.append("        {")
            parameters = layer_items[0]["parameters"]
            if not isinstance(parameters, list):
                raise ConfigError(f"model.models[{model_id}].parameters must be a list.")
            for parameter_id, parameter_item in enumerate(parameters):
                prefix = f"model.models[{model_id}].parameters[{parameter_id}]"
                parameter_name, tuning = _parse_named_parameter(parameter_item, model_id, 0, parameter_id)
                if tuning is None:
                    raise ConfigError(f"{prefix} must include C or define a prior for auto C.")
                prior_expression = _v2_prior_expression(parameter_item, prefix)
                replica_step_scales = _parameter_replica_step_scales(parameter_item, prefix)
                replica_scale_initializer = (
                    ""
                    if not replica_step_scales
                    else f", {_cpp_number_list(replica_step_scales)}"
                )
                lines.append(
                    "            ParameterSpec{"
                    f"{_cpp_string_literal(parameter_name)}, {prior_expression}, "
                    f"{_cpp_number(tuning[0])}, {_cpp_number(tuning[1])}"
                    f"{replica_scale_initializer}"
                    "},"
                )
            lines.append("        }")
            lines.append("    ));")
            continue

        layer_items = model_item.get("layers", model_item.get("hierarchies"))
        if not isinstance(layer_items, list) or not layer_items:
            raise ConfigError(f"model.models[{model_id}] must define parameters or layers for run.")
        lines.append("    {")
        lines.append("        ModelSpec model;")
        lines.append(f"        model.name = {_cpp_string_literal(model_name)};")
        lines.append(f"        model.basis_count = {basis_count};")
        for layer_id, layer_item in enumerate(layer_items):
            if not isinstance(layer_item, dict):
                raise ConfigError(f"model.models[{model_id}].layers[{layer_id}] must be an object.")
            layer_name = str(layer_item.get("name", f"layer_{layer_id}"))
            parameters = layer_item.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise ConfigError(f"model.models[{model_id}].layers[{layer_id}].parameters must be a non-empty list.")
            lines.append("        model.layers.push_back(LayerSpec{")
            lines.append(f"            {_cpp_string_literal(layer_name)},")
            lines.append("            {")
            for parameter_id, parameter_item in enumerate(parameters):
                prefix = f"model.models[{model_id}].layers[{layer_id}].parameters[{parameter_id}]"
                parameter_name, tuning = _parse_named_parameter(parameter_item, model_id, layer_id, parameter_id)
                if tuning is None:
                    raise ConfigError(f"{prefix} must include C or define a prior for auto C.")
                prior_expression = _v2_prior_expression(parameter_item, prefix)
                replica_step_scales = _parameter_replica_step_scales(parameter_item, prefix)
                replica_scale_initializer = (
                    ""
                    if not replica_step_scales
                    else f", {_cpp_number_list(replica_step_scales)}"
                )
                lines.append(
                    "                ParameterSpec{"
                    f"{_cpp_string_literal(parameter_name)}, {prior_expression}, "
                    f"{_cpp_number(tuning[0])}, {_cpp_number(tuning[1])}"
                    f"{replica_scale_initializer}"
                    "},"
                )
            lines.append("            }")
            lines.append("        });")
        lines.append("        spec.models.push_back(std::move(model));")
        lines.append("    }")

    emc = _require_dict(config.raw, "emc")
    sample_stride = _v2_positive_optional_int(emc, "sample_stride", 1)
    exchange_stride = _v2_positive_optional_int(emc, "exchange_stride", 1)
    lines.extend([
        "    return spec;",
        "}",
        "",
        "inline bayes_emc::EngineOptions MakeEngineOptions() {",
        "    bayes_emc::EngineOptions options;",
        f"    options.replica_count = {config.replica_num};",
        f"    options.gamma = {_cpp_number(config.gamma)};",
        f"    options.burnin_count = {config.burnin_num};",
        f"    options.sample_count = {config.sample_num};",
        f"    options.sample_stride = {sample_stride};",
        f"    options.exchange_stride = {exchange_stride};",
        f"    options.parallel_worker_count = {config.parallel_workers};",
        f"    options.likelihood_worker_count = {config.likelihood_workers};",
        f"    options.likelihood_parallel_min_rows = {config.likelihood_parallel_min_rows};",
        f"    options.progress_enabled = {_cpp_bool(config.progress)};",
        f"    options.progress_interval_steps = {config.progress_interval_steps};",
        f"    options.progress_bar_width = {config.progress_bar_width};",
        f"    options.seed = {_positive_int(emc.get('seed', 5489), 'emc.seed')};",
        "    return options;",
        "}",
        "",
        "inline const char * DefaultDataPath() {",
        f"    return {_cpp_string_literal(config.data_path)};",
        "}",
        "",
        "inline bayes_emc::DataTableOptions MakeDataOptions() {",
        "    bayes_emc::DataTableOptions options;",
        f"    options.format = bayes_emc::DataFormat::{_cpp_data_format(config.data_format)};",
        f"    options.header = {_cpp_bool(config.data_header)};",
        f"    options.input_columns = {_cpp_string_list(config.input_columns)};",
        f"    options.output_columns = {_cpp_string_list(config.output_columns)};",
        "    return options;",
        "}",
        "",
        "inline const char * DefaultResultDir() {",
        f"    return {_cpp_string_literal(config.result_dir)};",
        "}",
        "",
        "inline bool EstimateSigma2() {",
        f"    return {_cpp_bool(config.estimate_sigma2)};",
        "}",
        "",
        "inline const char * NoiseType() {",
        f"    return {_cpp_string_literal(config.noise_type)};",
        "}",
        "",
        "} // namespace bayes_emc_generated",
        "",
    ])

    config.src_dir.mkdir(parents=True, exist_ok=True)
    header_path = _v2_generated_header_path(config)
    header_path.write_text("\n".join(lines), encoding="utf-8")
    return header_path


def build_v2_command(config: NormalizedConfig) -> list[str]:
    build = config.build
    cmd = [build["compiler"], "main.cpp", "-o", build["output"]]
    include_dirs = [str(config.src_dir), str(_v2_include_root()), *build["include_dirs"]]
    for include_dir in include_dirs:
        if include_dir:
            cmd.append(f"-I{include_dir}")
    for library_dir in build["library_dirs"]:
        if library_dir:
            cmd.append(f"-L{library_dir}")
    cmd.extend(str(flag) for flag in build["flags"])
    cmd.extend(str(lib) for lib in build["libs"])
    return cmd


def validate_v2_sources(config: NormalizedConfig) -> None:
    for relative in ["src/main.cpp", "src/target.hpp"]:
        source = config.project_dir / relative
        if not source.exists():
            raise ConfigError(f"Required V2 C++ source file not found: {source}")


def run_engine_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    warnings = validate_config(config, require_sources=False)
    validate_v2_sources(config)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    header_path = _v2_generated_header_path(config)
    if args.dry_run:
        print("Config is valid for V2.")
        print("V2 generated header would be written to:", header_path)
        print("Build command:", " ".join(build_v2_command(config)))
        print("Run directory:", config.src_dir)
        return 0

    config.result_dir.mkdir(parents=True, exist_ok=True)
    (config.result_dir / "figures").mkdir(parents=True, exist_ok=True)
    header_path = _write_v2_generated_config(config)
    print(f"wrote {header_path}")

    if not args.skip_build:
        cmd = build_v2_command(config)
        print("building:", " ".join(cmd))
        subprocess.run(cmd, cwd=config.src_dir, check=True)

    if args.skip_exec:
        return 0

    executable = config.src_dir / config.build["output"]
    env = os.environ.copy()
    env["BAYES_EMC_DATA_PATH"] = str(config.data_path)
    env["BAYES_EMC_RESULT_DIR"] = str(config.result_dir)
    print("running:", executable)
    subprocess.run([str(executable)], cwd=config.src_dir, env=env, check=True)
    print("outputs:")
    for name in ["sample.json", "log.txt", "noise_estimation.txt", "diagnostics.tsv", "diagnostics_warnings.tsv"]:
        print(" ", config.result_dir / name)
    return 0


def run_command(args: argparse.Namespace) -> int:
    return run_engine_command(args)


def select_peaks_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    warnings = validate_config(config, require_sources=True)
    validate_v2_sources(config)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if config.noise_type != "gaussian":
        raise ConfigError("select-peaks currently supports model.noise.type = gaussian only.")
    if args.criterion == "estimated-noise" and not config.estimate_sigma2:
        raise ConfigError(
            "select-peaks --criterion estimated-noise requires model.noise.estimate_sigma2 = true. "
            "Use --criterion fixed-noise for fixed-sigma2 model selection."
        )

    min_peaks = _positive_int(args.min, "--min")
    max_peaks = _positive_int(args.max, "--max")
    if min_peaks > max_peaks:
        raise ConfigError("--min must be less than or equal to --max.")

    model_id = _resolve_peak_model_id(config, args.model)
    model_name = config.model_layout[model_id]["name"]
    selection_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config.result_dir / "model_selection" / "peak_count"
    )
    selection_root.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for peak_count in range(min_peaks, max_peaks + 1):
        run_dir = selection_root / f"peaks_{peak_count}"
        candidate_config = _config_with_peak_count(config, model_id, peak_count, run_dir)

        if args.dry_run:
            print(f"peak_count={peak_count}")
            print("  result_dir:", run_dir)
            print("  build:", " ".join(build_v2_command(candidate_config)))
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
        header_path = _write_v2_generated_config(candidate_config)
        print(f"wrote {header_path}")

        cmd = build_v2_command(candidate_config)
        print(f"building peaks={peak_count}:", " ".join(cmd))
        subprocess.run(cmd, cwd=candidate_config.src_dir, check=True)

        executable = candidate_config.src_dir / candidate_config.build["output"]
        env = os.environ.copy()
        env["BAYES_EMC_DATA_PATH"] = str(candidate_config.data_path)
        env["BAYES_EMC_RESULT_DIR"] = str(candidate_config.result_dir)
        print(f"running peaks={peak_count}:", executable)
        subprocess.run([str(executable)], cwd=candidate_config.src_dir, env=env, check=True)

        noise = _read_noise_estimation(run_dir / "noise_estimation.txt")
        beta_one = _noise_record_at_beta_one(noise["records"])
        score = (
            noise["min_free_energy"]
            if args.criterion == "estimated-noise"
            else beta_one["free_energy"]
        )
        candidates.append({
            "peak_count": peak_count,
            "model": model_name,
            "result_dir": str(run_dir),
            "score": score,
            "criterion_free_energy": score,
            "sigma2_mode": noise["sigma2_mode"],
            "estimated_sigma2": noise["estimated_sigma2"],
            "min_free_energy": noise["min_free_energy"],
            "beta_one_sigma2": beta_one["sigma2"],
            "beta_one_free_energy": beta_one["free_energy"],
        })

    if args.dry_run:
        return 0

    if not candidates:
        raise ConfigError("No peak-count candidates were evaluated.")
    selected = min(candidates, key=lambda item: item["score"])
    summary = {
        "schema_version": 1,
        "criterion": args.criterion,
        "model": model_name,
        "selected_peak_count": selected["peak_count"],
        "selected_free_energy": selected["score"],
        "candidates": candidates,
    }
    summary_json = selection_root / "peak_selection.json"
    summary_txt = selection_root / "peak_selection.txt"
    summary_svg = selection_root / "peak_selection.svg"
    summary_json.write_text(json.dumps(summary, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_peak_selection_table(summary_txt, summary)
    _write_peak_selection_svg(summary_svg, summary)
    restored_header = _write_v2_generated_config(config)
    print(f"selected_peak_count: {selected['peak_count']}")
    print(f"selected_free_energy: {selected['score']}")
    print(f"wrote {summary_json}")
    print(f"wrote {summary_txt}")
    print(f"wrote {summary_svg}")
    print(f"restored {restored_header}")
    return 0


def _resolve_peak_model_id(config: NormalizedConfig, model_name: str | None) -> int:
    if model_name:
        for model_id, model in enumerate(config.model_layout):
            if model["name"] == model_name:
                return model_id
        raise ConfigError(f"Model not found for peak selection: {model_name}")
    for model_id, model in enumerate(config.model_layout):
        if model["name"] == "spectral_peaks":
            return model_id
    if len(config.model_layout) == 1:
        return 0
    raise ConfigError("Specify --model for peak selection when config has multiple models.")


def _config_with_peak_count(
    config: NormalizedConfig,
    model_id: int,
    peak_count: int,
    result_dir: Path,
) -> NormalizedConfig:
    raw = copy.deepcopy(config.raw)
    project = raw.setdefault("project", {})
    if not isinstance(project, dict):
        raise ConfigError("project must be an object.")
    project["result_dir"] = str(result_dir)

    models = raw.get("model", {}).get("models")
    if not isinstance(models, list) or model_id >= len(models) or not isinstance(models[model_id], dict):
        raise ConfigError("model.models is invalid for peak selection.")
    model = models[model_id]
    for key in ["basis_count", "component_count", "base_count", "components"]:
        if key in model:
            model[key] = peak_count
    model["basis_count"] = peak_count
    return _normalize_config(raw, config.config_path, config.project_dir)


def _read_noise_estimation(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Noise estimation file not found: {path}")
    sigma2_mode = "estimated"
    estimated_sigma2: float | None = None
    replica_id: int | None = None
    min_free_energy: float | None = None
    records: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            parts = stripped[1:].strip().split()
            if len(parts) >= 2 and parts[0] == "sigma2_mode":
                sigma2_mode = parts[1]
            elif len(parts) >= 2 and parts[0] == "estimated_sigma2":
                estimated_sigma2 = float(parts[1])
            elif len(parts) >= 2 and parts[0] == "replica_id":
                replica_id = int(parts[1])
            elif len(parts) >= 2 and parts[0] == "min_free_energy":
                min_free_energy = float(parts[1])
            continue
        if stripped.startswith("sigma2"):
            continue
        parts = stripped.split()
        if len(parts) != 3:
            raise ConfigError(f"Invalid noise estimation row in {path}: {line}")
        records.append({
            "sigma2": float(parts[0]),
            "inverse_temperature": float(parts[1]),
            "free_energy": float(parts[2]),
        })
    if estimated_sigma2 is None or replica_id is None or min_free_energy is None or not records:
        raise ConfigError(f"Invalid noise estimation file: {path}")
    return {
        "sigma2_mode": sigma2_mode,
        "estimated_sigma2": estimated_sigma2,
        "replica_id": replica_id,
        "min_free_energy": min_free_energy,
        "records": records,
    }


def _noise_record_at_beta_one(records: list[dict[str, float]]) -> dict[str, float]:
    return min(records, key=lambda record: abs(record["inverse_temperature"] - 1.0))


def _write_peak_selection_table(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# criterion\t{summary['criterion']}",
        f"# selected_peak_count\t{summary['selected_peak_count']}",
        f"# selected_free_energy\t{summary['selected_free_energy']}",
        "peak_count\tscore\tsigma2_mode\testimated_sigma2\tmin_free_energy\tbeta_one_sigma2\tbeta_one_free_energy\tresult_dir",
    ]
    for item in summary["candidates"]:
        lines.append(
            f"{item['peak_count']}\t{item['score']}\t{item.get('sigma2_mode', 'estimated')}\t"
            f"{item['estimated_sigma2']}\t"
            f"{item['min_free_energy']}\t{item['beta_one_sigma2']}\t"
            f"{item['beta_one_free_energy']}\t{item['result_dir']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_peak_selection_svg(path: Path, summary: dict[str, Any]) -> None:
    candidates = summary["candidates"]
    if not candidates:
        return
    width = 720
    height = 360
    left = 72
    right = 32
    top = 36
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    peak_counts = [item["peak_count"] for item in candidates]
    scores = [item["score"] for item in candidates]
    min_peak, max_peak = min(peak_counts), max(peak_counts)
    min_score, max_score = min(scores), max(scores)
    if min_peak == max_peak:
        min_peak -= 1
        max_peak += 1
    if min_score == max_score:
        min_score -= 0.5
        max_score += 0.5

    def x_pos(peak_count: int) -> float:
        return left + (peak_count - min_peak) / (max_peak - min_peak) * plot_width

    def y_pos(score: float) -> float:
        return top + (max_score - score) / (max_score - min_score) * plot_height

    points = " ".join(f"{x_pos(item['peak_count']):.2f},{y_pos(item['score']):.2f}" for item in candidates)
    selected_peak = summary["selected_peak_count"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#222}.axis{stroke:#222;stroke-width:1}.line{fill:none;stroke:#2f6f8f;stroke-width:2}.point{fill:#2f6f8f}.selected{fill:#c33}</style>',
        f'<text x="{left}" y="22">Peak count selection by Bayesian free energy</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        f'<polyline class="line" points="{points}"/>',
    ]
    for item in candidates:
        cx = x_pos(item["peak_count"])
        cy = y_pos(item["score"])
        klass = "selected" if item["peak_count"] == selected_peak else "point"
        svg.append(f'<circle class="{klass}" cx="{cx:.2f}" cy="{cy:.2f}" r="4"/>')
        svg.append(f'<text x="{cx - 8:.2f}" y="{top + plot_height + 20}">{item["peak_count"]}</text>')
    svg.append(f'<text x="{left + plot_width / 2 - 48:.2f}" y="{height - 14}">peak count</text>')
    svg.append(f'<text x="8" y="{top + 12}">free energy</text>')
    svg.append(f'<text x="{left}" y="{top + plot_height + 40}">selected: {selected_peak}</text>')
    svg.append("</svg>\n")
    path.write_text("\n".join(svg), encoding="utf-8")


def benchmark_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    warnings = validate_config(config, require_sources=True)
    validate_v2_sources(config)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if config.noise_type != "gaussian":
        raise ConfigError("bayes-emc benchmark currently supports model.noise.type = gaussian only.")

    sample_num = _positive_int(args.sample_num, "--sample-num")
    burnin_num = _positive_int(args.burnin_num, "--burnin-num")
    repeat = _positive_int(args.repeat, "--repeat")
    parallel_worker_candidates = (
        _parse_nonnegative_int_list(args.parallel_workers_list, "--parallel-workers-list")
        if args.parallel_workers_list
        else [
            _nonnegative_int(args.parallel_workers, "--parallel-workers")
            if args.parallel_workers is not None
            else config.parallel_workers
        ]
    )
    likelihood_workers = (
        _nonnegative_int(args.likelihood_workers, "--likelihood-workers")
        if args.likelihood_workers is not None
        else config.likelihood_workers
    )
    likelihood_parallel_min_rows = (
        _nonnegative_int(args.likelihood_parallel_min_rows, "--likelihood-parallel-min-rows")
        if args.likelihood_parallel_min_rows is not None
        else config.likelihood_parallel_min_rows
    )
    benchmark_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config.project_dir / "result_benchmark"
    )
    benchmark_root.mkdir(parents=True, exist_ok=True)

    if args.skip_build and len(parallel_worker_candidates) > 1:
        raise ConfigError("--skip-build cannot be used with --parallel-workers-list.")

    summaries: list[dict[str, Any]] = []
    for parallel_workers in parallel_worker_candidates:
        variant_root = benchmark_root / f"parallel_{_worker_label(parallel_workers)}"
        summary = _run_benchmark_variant(
            config=config,
            benchmark_root=variant_root,
            sample_num=sample_num,
            burnin_num=burnin_num,
            repeat=repeat,
            parallel_workers=parallel_workers,
            likelihood_workers=likelihood_workers,
            likelihood_parallel_min_rows=likelihood_parallel_min_rows,
            skip_build=args.skip_build,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        selected = min(summaries, key=lambda item: item["mean_wall_time"])
        summary = {
            "schema_version": 1,
            "sample_num": sample_num,
            "burnin_num": burnin_num,
            "repeat": repeat,
            "likelihood_workers": likelihood_workers,
            "likelihood_parallel_min_rows": likelihood_parallel_min_rows,
            "recommended_parallel_workers": selected["parallel_workers"],
            "recommended_mean_wall_time": selected["mean_wall_time"],
            "candidates": summaries,
        }
        summary_json = benchmark_root / "benchmark_summary.json"
        summary_txt = benchmark_root / "benchmark_summary.txt"
        summary_json.write_text(json.dumps(summary, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_benchmark_summary_table(summary_txt, summary)
        print(f"recommended_parallel_workers: {selected['parallel_workers']}")
        print(f"recommended_mean_wall_time: {selected['mean_wall_time']:.3f}s")
        print(f"wrote {summary_json}")
        print(f"wrote {summary_txt}")

    restored_header = _write_v2_generated_config(config)
    print(f"restored {restored_header}")

    return 0


def _run_benchmark_variant(
    *,
    config: NormalizedConfig,
    benchmark_root: Path,
    sample_num: int,
    burnin_num: int,
    repeat: int,
    parallel_workers: int,
    likelihood_workers: int,
    likelihood_parallel_min_rows: int,
    skip_build: bool,
) -> dict[str, Any]:
    benchmark_root.mkdir(parents=True, exist_ok=True)
    benchmark_config = replace(
        config,
        sample_num=sample_num,
        burnin_num=burnin_num,
        parallel_workers=parallel_workers,
        likelihood_workers=likelihood_workers,
        likelihood_parallel_min_rows=likelihood_parallel_min_rows,
        progress=False,
        result_dir=benchmark_root,
    )
    header_path = _write_v2_generated_config(benchmark_config)
    print(f"wrote {header_path}")

    if not skip_build:
        cmd = build_v2_command(benchmark_config)
        print("building:", " ".join(cmd))
        subprocess.run(cmd, cwd=benchmark_config.src_dir, check=True)

    executable = benchmark_config.src_dir / benchmark_config.build["output"]
    wall_times: list[float] = []

    for run_id in range(1, repeat + 1):
        run_dir = benchmark_root / f"run_{run_id}"
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["BAYES_EMC_DATA_PATH"] = str(benchmark_config.data_path)
        env["BAYES_EMC_RESULT_DIR"] = str(run_dir)
        start = time.perf_counter()
        subprocess.run(
            [str(executable)],
            cwd=benchmark_config.src_dir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        wall_time = time.perf_counter() - start
        wall_times.append(wall_time)
        print(
            f"parallel_workers={parallel_workers} "
            f"likelihood_workers={likelihood_workers} "
            f"run {run_id}: wall={wall_time:.3f}s result_dir={run_dir}"
        )

    mean_wall = sum(wall_times) / len(wall_times)
    min_wall = min(wall_times)
    max_wall = max(wall_times)
    print(f"wall summary: mean={mean_wall:.3f}s min={min_wall:.3f}s max={max_wall:.3f}s")
    print(
        "benchmark config: "
        f"burnin={burnin_num} sample={sample_num} repeat={repeat} "
        f"parallel_workers={parallel_workers} "
        f"likelihood_workers={likelihood_workers} "
        f"likelihood_parallel_min_rows={likelihood_parallel_min_rows}"
    )
    return {
        "parallel_workers": parallel_workers,
        "likelihood_workers": likelihood_workers,
        "likelihood_parallel_min_rows": likelihood_parallel_min_rows,
        "mean_wall_time": mean_wall,
        "min_wall_time": min_wall,
        "max_wall_time": max_wall,
        "wall_times": wall_times,
        "result_dir": str(benchmark_root),
    }


def _parse_nonnegative_int_list(value: str, name: str) -> list[int]:
    parsed: list[int] = []
    for index, part in enumerate(value.split(",")):
        stripped = part.strip()
        if not stripped:
            raise ConfigError(f"{name} contains an empty item.")
        parsed.append(_nonnegative_int(stripped, f"{name}[{index}]"))
    if not parsed:
        raise ConfigError(f"{name} must contain at least one value.")
    return parsed


def _worker_label(worker_count: int) -> str:
    return "auto" if worker_count == 0 else str(worker_count)


def _write_benchmark_summary_table(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# recommended_parallel_workers\t{summary['recommended_parallel_workers']}",
        f"# recommended_mean_wall_time\t{summary['recommended_mean_wall_time']}",
        "parallel_workers\tmean_wall_time\tmin_wall_time\tmax_wall_time\tresult_dir",
    ]
    for item in summary["candidates"]:
        lines.append(
            f"{item['parallel_workers']}\t{item['mean_wall_time']}\t"
            f"{item['min_wall_time']}\t{item['max_wall_time']}\t{item['result_dir']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tune_command(args: argparse.Namespace) -> int:
    base_config = load_config(Path(args.config))
    warnings = validate_config(base_config, require_sources=True)
    validate_v2_sources(base_config)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    low_rate = _rate_option(args.low_rate, "--low-rate")
    high_rate = _rate_option(args.high_rate, "--high-rate")
    top_accept_rate = _rate_option(args.top_accept_rate, "--top-accept-rate")
    top_exchange_rate = _rate_option(args.top_exchange_rate, "--top-exchange-rate")
    d_target_rate = _rate_option(args.d_target_rate, "--d-target-rate")
    d_tolerance = _rate_option(args.d_tolerance, "--d-tolerance")
    if not low_rate < high_rate:
        raise ConfigError("--low-rate must be smaller than --high-rate.")
    if not (0.0 < d_target_rate - d_tolerance and d_target_rate + d_tolerance < 1.0):
        raise ConfigError("--d-target-rate +/- --d-tolerance must stay between 0 and 1.")

    sample_num = _positive_int(args.sample_num, "--sample-num")
    burnin_num = _positive_int(args.burnin_num, "--burnin-num")
    c_rounds = _nonnegative_int(args.c_rounds, "--c-rounds")
    d_rounds = _nonnegative_int(args.d_rounds, "--d-rounds")
    gamma_candidates = _positive_int(args.gamma_candidates, "--gamma-candidates")
    replica_step = _positive_int(args.replica_step, "--replica-step")
    initial_replica_num = _positive_int(args.initial_replica_num, "--initial-replica-num")
    max_replica_num = _positive_int(args.max_replica_num, "--max-replica-num") if args.max_replica_num else max(
        base_config.replica_num,
        initial_replica_num + 32,
    )
    initial_replica_num = min(initial_replica_num, max_replica_num)
    gamma_min = _positive_float(args.gamma_min, "--gamma-min")
    gamma_max = _positive_float(args.gamma_max, "--gamma-max")
    d_initial = _nonnegative_float(args.d_initial, "--d-initial")
    if gamma_min < 1.0:
        raise ConfigError("--gamma-min must be greater than or equal to 1.")
    if gamma_min > gamma_max:
        raise ConfigError("--gamma-min must be less than or equal to --gamma-max.")

    tuned_raw = copy.deepcopy(base_config.raw)
    parameter_refs = _tuning_parameter_refs(tuned_raw)
    if not args.keep_c:
        _initialize_c_from_priors(parameter_refs)
    _ensure_parameter_decay(parameter_refs)
    if not args.keep_replica_step_scales:
        _clear_replica_step_scales(parameter_refs)
    tuned_emc = _emc_dict(tuned_raw)
    tuned_emc["replica_num"] = initial_replica_num
    tuned_emc["gamma"] = min(max(base_config.gamma, gamma_min), gamma_max)

    tuning_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else base_config.result_dir / "tuning"
    )
    tuning_root.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, Any]] = []

    for round_id in range(c_rounds):
        trial = _run_tuning_trial(
            base_config,
            tuned_raw,
            tuning_root / f"c_{round_id + 1:02d}",
            sample_num,
            burnin_num,
            f"c_round_{round_id + 1}",
            quiet=args.quiet,
        )
        top_rates = _top_temperature_rates_for_refs(trial["summary"], parameter_refs)
        adjustments = _adjust_c_from_top_temperature(parameter_refs, trial["summary"], top_accept_rate, args.c_factor)
        min_top_rate, max_top_rate = _rate_range(top_rates)
        trial["action"] = (
            f"top_accept_min={_percent_text(min_top_rate)};"
            f"top_accept_max={_percent_text(max_top_rate)};"
            f"adjusted_c={adjustments}"
        )
        report_rows.append(_tuning_report_row("C", round_id + 1, trial, low_rate, high_rate))
        if adjustments == 0:
            break

    gamma_values = _gamma_candidate_values(gamma_min, gamma_max, gamma_candidates)
    current_replica_num = initial_replica_num
    best_gamma_trial: dict[str, Any] | None = None
    selected_gamma_trial: dict[str, Any] | None = None
    gamma_round = 0
    while current_replica_num <= max_replica_num:
        gamma_round += 1
        should_increase_replica = False
        for candidate_id, gamma in enumerate(gamma_values, start=1):
            candidate_raw = copy.deepcopy(tuned_raw)
            candidate_emc = _emc_dict(candidate_raw)
            candidate_emc["replica_num"] = current_replica_num
            candidate_emc["gamma"] = gamma
            trial = _run_tuning_trial(
                base_config,
                candidate_raw,
                tuning_root / f"gamma_r{gamma_round:02d}_{candidate_id:02d}",
                sample_num,
                burnin_num,
                f"gamma_round_{gamma_round}_{candidate_id}",
                quiet=args.quiet,
            )
            top_exchange = _top_exchange_rate(trial["summary"])
            non_top_low_count = _non_top_exchange_low_count(trial["summary"], low_rate)
            trial["action"] = (
                f"top_exchange={_percent_text(top_exchange)};"
                f"non_top_low={non_top_low_count}"
            )
            trial_score = _gamma_tuning_score(trial["summary"], low_rate, top_exchange_rate)
            best_score = (
                None
                if best_gamma_trial is None
                else _gamma_tuning_score(best_gamma_trial["summary"], low_rate, top_exchange_rate)
            )
            if best_score is None or trial_score < best_score:
                best_gamma_trial = trial
            if non_top_low_count > 0:
                should_increase_replica = True
                trial["action"] += ";increase_replica"
                report_rows.append(_tuning_report_row("gamma", gamma_round, trial, low_rate, high_rate))
                break
            if top_exchange is not None and top_exchange > top_exchange_rate:
                selected_gamma_trial = trial
                trial["action"] += ";selected"
                report_rows.append(_tuning_report_row("gamma", gamma_round, trial, low_rate, high_rate))
                break
            report_rows.append(_tuning_report_row("gamma", gamma_round, trial, low_rate, high_rate))
        if selected_gamma_trial is not None:
            break
        if not should_increase_replica and gamma_values:
            should_increase_replica = True
        if not should_increase_replica:
            break
        current_replica_num += replica_step

    gamma_trial = selected_gamma_trial or best_gamma_trial
    if gamma_trial is not None:
        tuned_emc = _emc_dict(tuned_raw)
        tuned_emc["replica_num"] = gamma_trial["replica_num"]
        tuned_emc["gamma"] = gamma_trial["gamma"]

    parameter_refs = _tuning_parameter_refs(tuned_raw)
    for round_id in range(c_rounds):
        trial = _run_tuning_trial(
            base_config,
            tuned_raw,
            tuning_root / f"c_high_{round_id + 1:02d}",
            sample_num,
            burnin_num,
            f"c_high_round_{round_id + 1}",
            quiet=args.quiet,
        )
        top_rates = _top_temperature_rates_for_refs(trial["summary"], parameter_refs)
        adjustments = _expand_c_from_top_temperature(parameter_refs, trial["summary"], high_rate, args.c_factor)
        min_top_rate, max_top_rate = _rate_range(top_rates)
        trial["action"] = (
            f"top_accept_min={_percent_text(min_top_rate)};"
            f"top_accept_max={_percent_text(max_top_rate)};"
            f"expanded_c={adjustments}"
        )
        report_rows.append(_tuning_report_row("C_high", round_id + 1, trial, low_rate, high_rate))
        if adjustments == 0:
            break

    _reset_parameter_decay(parameter_refs, d_initial, args.min_d, args.max_d)
    for round_id in range(d_rounds):
        trial = _run_tuning_trial(
            base_config,
            tuned_raw,
            tuning_root / f"d_{round_id + 1:02d}",
            sample_num,
            burnin_num,
            f"d_round_{round_id + 1}",
            quiet=args.quiet,
        )
        cold_rates = _cold_temperature_rates_for_refs(trial["summary"], parameter_refs)
        adjustments = _adjust_d_from_cold_temperature(
            parameter_refs,
            trial["summary"],
            d_target_rate,
            d_tolerance,
            args.d_step,
            args.min_d,
            args.max_d,
        )
        min_cold_rate, max_cold_rate = _rate_range(cold_rates)
        trial["action"] = (
            f"cold_accept_min={_percent_text(min_cold_rate)};"
            f"cold_accept_max={_percent_text(max_cold_rate)};"
            f"target={_percent_text(d_target_rate)};"
            f"tolerance={_percent_text(d_tolerance)};"
            f"adjusted_d={adjustments}"
        )
        report_rows.append(_tuning_report_row("d", round_id + 1, trial, low_rate, high_rate))
        if adjustments == 0:
            break

    local_step_rounds = _nonnegative_int(args.local_step_rounds, "--local-step-rounds")
    local_step_factor = _local_step_factor(args.local_step_factor)
    for round_id in range(local_step_rounds):
        trial = _run_tuning_trial(
            base_config,
            tuned_raw,
            tuning_root / f"local_step_{round_id + 1:02d}",
            sample_num,
            burnin_num,
            f"local_step_round_{round_id + 1}",
            quiet=args.quiet,
        )
        low_count = _low_acceptance_count_for_refs(trial["summary"], parameter_refs, low_rate)
        adjustments = _adjust_replica_step_scales_from_low_rates(
            parameter_refs,
            trial["summary"],
            _positive_int(_emc_dict(tuned_raw).get("replica_num"), "emc.replica_num"),
            low_rate,
            local_step_factor,
        )
        trial["action"] = (
            f"low_accept_below={low_count};"
            f"local_step_scaled={adjustments};"
            f"factor={local_step_factor}"
        )
        report_rows.append(_tuning_report_row("local_step", round_id + 1, trial, low_rate, high_rate))
        if adjustments == 0:
            break

    final_trial = _run_tuning_trial(
        base_config,
        tuned_raw,
        tuning_root / "final",
        sample_num,
        burnin_num,
        "final",
        quiet=args.quiet,
    )
    final_trial["action"] = "final"
    report_rows.append(_tuning_report_row("final", 0, final_trial, low_rate, high_rate))

    tuned_config_path = (
        Path(args.output_config).expanduser().resolve()
        if args.output_config
        else base_config.config_path.with_name("config.tuned.json")
    )
    tuned_config_path.write_text(json.dumps(tuned_raw, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "source_config": str(base_config.config_path),
        "tuned_config": str(tuned_config_path),
        "tuning_root": str(tuning_root),
        "low_rate": low_rate,
        "high_rate": high_rate,
        "sample_num": sample_num,
        "burnin_num": burnin_num,
        "final": _tuning_report_row("final", 0, final_trial, low_rate, high_rate),
        "trials": report_rows,
    }
    (tuning_root / "tune_report.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_tune_report_tsv(tuning_root / "tune_report.tsv", report_rows)
    _write_v2_generated_config(_normalize_config(tuned_raw, base_config.config_path, base_config.project_dir))

    print(f"wrote {tuned_config_path}")
    print(f"wrote {tuning_root / 'tune_report.tsv'}")
    print(f"final diagnostic warnings: {final_trial['warning_count']}")
    print(f"final replica_num: {final_trial['replica_num']}")
    print(f"final gamma: {final_trial['gamma']}")
    return 0


def _rate_option(value: Any, name: str) -> float:
    parsed = _positive_float(value, name)
    if parsed > 1.0:
        parsed /= 100.0
    if not 0.0 < parsed < 1.0:
        raise ConfigError(f"{name} must be a fraction between 0 and 1, or a percent between 0 and 100.")
    return parsed


def _emc_dict(raw: dict[str, Any]) -> dict[str, Any]:
    emc = raw.setdefault("emc", {})
    if not isinstance(emc, dict):
        raise ConfigError("emc must be an object.")
    return emc


def _tuning_parameter_refs(raw: dict[str, Any]) -> list[TuningParameterRef]:
    model_section = raw.get("model")
    if not isinstance(model_section, dict):
        raise ConfigError("model must be an object.")
    models = model_section.get("models")
    if not isinstance(models, list) or not models:
        raise ConfigError("model.models must be a non-empty list.")

    refs: list[TuningParameterRef] = []
    for model_id, model_item in enumerate(models):
        if not isinstance(model_item, dict):
            raise ConfigError(f"model.models[{model_id}] must be an object.")
        model_name = str(model_item.get("name", f"model_{model_id}"))
        if "parameters" in model_item:
            layer_items = [{"name": "default", "parameters": model_item["parameters"]}]
        else:
            layer_items = model_item.get("layers", model_item.get("hierarchies"))
        if not isinstance(layer_items, list) or not layer_items:
            raise ConfigError(f"model.models[{model_id}] must define parameters or layers.")
        for layer_id, layer_item in enumerate(layer_items):
            if not isinstance(layer_item, dict):
                raise ConfigError(f"model.models[{model_id}].layers[{layer_id}] must be an object.")
            layer_name = str(layer_item.get("name", f"layer_{layer_id}"))
            parameters = layer_item.get("parameters")
            if not isinstance(parameters, list) or not parameters:
                raise ConfigError(f"model.models[{model_id}].layers[{layer_id}].parameters must be a non-empty list.")
            for parameter_id, parameter_item in enumerate(parameters):
                if not isinstance(parameter_item, dict):
                    raise ConfigError(
                        "bayes-emc tune requires model.models parameters to be objects "
                        f"(found non-object at model {model_id}, layer {layer_id}, parameter {parameter_id})."
                    )
                parameter_name = str(parameter_item.get("name", f"p{parameter_id}"))
                refs.append(TuningParameterRef(model_name, layer_name, parameter_name, parameter_item))
    return refs


def _parameter_tuning_target(parameter: dict[str, Any]) -> dict[str, Any]:
    tuning = parameter.get("tuning")
    if tuning is None:
        return parameter
    if not isinstance(tuning, dict):
        raise ConfigError("parameter.tuning must be an object when provided.")
    return tuning


def _initialize_c_from_priors(parameter_refs: list[TuningParameterRef]) -> None:
    for index, ref in enumerate(parameter_refs):
        if "prior" not in ref.parameter:
            continue
        target = _parameter_tuning_target(ref.parameter)
        prefix = f"model parameter {ref.model_name}.{ref.layer_name}.{ref.parameter_name}"
        target["C"] = _proposal_c_from_prior(ref.parameter["prior"], prefix)
        target.setdefault("d", DEFAULT_PROPOSAL_DECAY)


def _ensure_parameter_decay(parameter_refs: list[TuningParameterRef]) -> None:
    for ref in parameter_refs:
        target = _parameter_tuning_target(ref.parameter)
        target.setdefault("d", DEFAULT_PROPOSAL_DECAY)


def _reset_parameter_decay(parameter_refs: list[TuningParameterRef], value: float, min_d: float, max_d: float) -> None:
    if not 0.0 <= min_d <= value <= max_d:
        raise ConfigError("--d-initial must be between --min-d and --max-d.")
    for ref in parameter_refs:
        target = _parameter_tuning_target(ref.parameter)
        target["d"] = value


def _clear_replica_step_scales(parameter_refs: list[TuningParameterRef]) -> None:
    for ref in parameter_refs:
        target = _parameter_tuning_target(ref.parameter)
        target.pop("replica_step_scales", None)
        target.pop("step_size_scales", None)


def _run_tuning_trial(
    base_config: NormalizedConfig,
    raw: dict[str, Any],
    result_dir: Path,
    sample_num: int,
    burnin_num: int,
    label: str,
    *,
    quiet: bool,
) -> dict[str, Any]:
    trial_raw = copy.deepcopy(raw)
    project = trial_raw.setdefault("project", {})
    if not isinstance(project, dict):
        raise ConfigError("project must be an object.")
    project["result_dir"] = str(result_dir)
    emc = _emc_dict(trial_raw)
    emc["sample_num"] = sample_num
    emc["burnin_num"] = burnin_num
    emc["progress"] = False

    trial_config = _normalize_config(trial_raw, base_config.config_path, base_config.project_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "figures").mkdir(parents=True, exist_ok=True)
    _write_v2_generated_config(trial_config)
    cmd = build_v2_command(trial_config)
    if not quiet:
        print(f"tuning {label}: building {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=trial_config.src_dir,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )

    executable = trial_config.src_dir / trial_config.build["output"]
    env = os.environ.copy()
    env["BAYES_EMC_DATA_PATH"] = str(trial_config.data_path)
    env["BAYES_EMC_RESULT_DIR"] = str(trial_config.result_dir)
    if not quiet:
        print(f"tuning {label}: running {executable}")
    subprocess.run(
        [str(executable)],
        cwd=trial_config.src_dir,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )

    diagnostics_path = result_dir / "diagnostics.tsv"
    warnings_path = result_dir / "diagnostics_warnings.tsv"
    summary = _read_diagnostics_summary(diagnostics_path)
    warning_count = max(0, len(warnings_path.read_text(encoding="utf-8").splitlines()) - 1) if warnings_path.exists() else 0
    return {
        "label": label,
        "result_dir": str(result_dir),
        "replica_num": trial_config.replica_num,
        "gamma": trial_config.gamma,
        "summary": summary,
        "warning_count": warning_count,
    }


def _read_diagnostics_summary(path: Path) -> DiagnosticsSummary:
    if not path.exists():
        raise ConfigError(f"Diagnostics file not found: {path}")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    if not rows:
        raise ConfigError(f"Diagnostics file is empty: {path}")
    header = rows[0]
    if len(header) < 4 or header[-2:] != ["Exchange %", "<Energy>"]:
        raise ConfigError(f"Unexpected diagnostics header in {path}")
    parameter_labels = header[1:-2]
    parameter_rates: dict[tuple[str, str, str], list[tuple[float, bool, float]]] = {}
    parameter_replica_rates: dict[tuple[str, str, str], list[tuple[int, float, bool, float]]] = {}
    exchange_rates: list[tuple[int, int, float, float, float]] = []
    for replica_id, row in enumerate(rows[1:]):
        if len(row) < len(header):
            continue
        raw_beta = row[0]
        unscaled = raw_beta.startswith("*")
        beta_text = raw_beta[1:] if unscaled else raw_beta
        beta = float(beta_text)
        for offset, label in enumerate(parameter_labels, start=1):
            key = _diagnostic_parameter_key(label)
            if key is None:
                continue
            rate = float(row[offset]) / 100.0
            parameter_rates.setdefault(key, []).append((beta, unscaled, rate))
            parameter_replica_rates.setdefault(key, []).append((replica_id, beta, unscaled, rate))
        exchange_text = row[1 + len(parameter_labels)]
        if exchange_text != "*****" and replica_id + 1 < len(rows) - 1:
            next_raw_beta = rows[replica_id + 2][0]
            next_beta = float(next_raw_beta[1:] if next_raw_beta.startswith("*") else next_raw_beta)
            exchange_rates.append((replica_id, replica_id + 1, beta, next_beta, float(exchange_text) / 100.0))
    return DiagnosticsSummary(
        parameter_rates=parameter_rates,
        parameter_replica_rates=parameter_replica_rates,
        exchange_rates=exchange_rates,
    )


def _diagnostic_parameter_key(label: str) -> tuple[str, str, str] | None:
    bracket = label.find("[")
    end_bracket = label.find("].", bracket)
    if bracket <= 0 or end_bracket < 0:
        return None
    model_name = label[:bracket]
    rest = label[end_bracket + 2:]
    parts = rest.split(".")
    if len(parts) < 2:
        return None
    layer_name = parts[0]
    parameter_name = parts[-1]
    return model_name, layer_name, parameter_name


def _rates_for_ref(
    summary: DiagnosticsSummary,
    ref: TuningParameterRef,
    *,
    unscaled: bool | None,
    exclude_beta_zero: bool = True,
) -> list[float]:
    key = (ref.model_name, ref.layer_name, ref.parameter_name)
    rates: list[float] = []
    for beta, is_unscaled, rate in summary.parameter_rates.get(key, []):
        if exclude_beta_zero and beta == 0.0:
            continue
        if unscaled is not None and is_unscaled != unscaled:
            continue
        rates.append(rate)
    return rates


def _top_temperature_rate_for_ref(summary: DiagnosticsSummary, ref: TuningParameterRef) -> float | None:
    key = (ref.model_name, ref.layer_name, ref.parameter_name)
    finite_rates = [
        (beta, rate)
        for beta, _, rate in summary.parameter_rates.get(key, [])
        if beta > 0.0
    ]
    if not finite_rates:
        return None
    _, rate = min(finite_rates, key=lambda item: item[0])
    return rate


def _top_temperature_rates_for_refs(summary: DiagnosticsSummary, refs: list[TuningParameterRef]) -> list[float]:
    return [
        rate
        for ref in refs
        if (rate := _top_temperature_rate_for_ref(summary, ref)) is not None
    ]


def _cold_temperature_rate_for_ref(summary: DiagnosticsSummary, ref: TuningParameterRef) -> float | None:
    key = (ref.model_name, ref.layer_name, ref.parameter_name)
    finite_rates = [
        (beta, rate)
        for beta, _, rate in summary.parameter_rates.get(key, [])
        if beta > 0.0
    ]
    if not finite_rates:
        return None
    _, rate = max(finite_rates, key=lambda item: item[0])
    return rate


def _cold_temperature_rates_for_refs(summary: DiagnosticsSummary, refs: list[TuningParameterRef]) -> list[float]:
    return [
        rate
        for ref in refs
        if (rate := _cold_temperature_rate_for_ref(summary, ref)) is not None
    ]


def _adjust_c_from_top_temperature(
    parameter_refs: list[TuningParameterRef],
    summary: DiagnosticsSummary,
    target_rate: float,
    factor: float,
) -> int:
    if not factor > 1.0:
        raise ConfigError("--c-factor must be greater than 1.")
    adjustment_count = 0
    for ref in parameter_refs:
        rate = _top_temperature_rate_for_ref(summary, ref)
        if rate is None:
            continue
        target = _parameter_tuning_target(ref.parameter)
        current_c = _positive_float(target.get("C"), f"{ref.parameter_name}.C")
        if rate <= target_rate:
            target["C"] = current_c / factor
            adjustment_count += 1
    return adjustment_count


def _expand_c_from_top_temperature(
    parameter_refs: list[TuningParameterRef],
    summary: DiagnosticsSummary,
    high_rate: float,
    factor: float,
) -> int:
    if not factor > 1.0:
        raise ConfigError("--c-factor must be greater than 1.")
    adjustment_count = 0
    for ref in parameter_refs:
        rate = _top_temperature_rate_for_ref(summary, ref)
        if rate is None:
            continue
        target = _parameter_tuning_target(ref.parameter)
        current_c = _positive_float(target.get("C"), f"{ref.parameter_name}.C")
        if rate >= high_rate:
            target["C"] = current_c * factor
            adjustment_count += 1
    return adjustment_count


def _adjust_d_from_cold_temperature(
    parameter_refs: list[TuningParameterRef],
    summary: DiagnosticsSummary,
    target_rate: float,
    tolerance: float,
    step: float,
    min_d: float,
    max_d: float,
) -> int:
    if not step > 0.0:
        raise ConfigError("--d-step must be positive.")
    if not 0.0 <= min_d <= max_d:
        raise ConfigError("--min-d must be non-negative and less than or equal to --max-d.")
    lower = target_rate - tolerance
    upper = target_rate + tolerance
    adjustment_count = 0
    for ref in parameter_refs:
        rate = _cold_temperature_rate_for_ref(summary, ref)
        if rate is None:
            continue
        target = _parameter_tuning_target(ref.parameter)
        current_d = _nonnegative_float(target.get("d", DEFAULT_PROPOSAL_DECAY), f"{ref.parameter_name}.d")
        if rate < lower:
            target["d"] = min(max_d, current_d + step)
            adjustment_count += int(target["d"] != current_d)
        elif rate > upper:
            target["d"] = max(min_d, current_d - step)
            adjustment_count += int(target["d"] != current_d)
    return adjustment_count


def _local_step_factor(value: Any) -> float:
    parsed = _positive_float(value, "--local-step-factor")
    if not parsed < 1.0:
        raise ConfigError("--local-step-factor must be greater than 0 and smaller than 1.")
    return parsed


def _low_acceptance_count_for_refs(
    summary: DiagnosticsSummary,
    parameter_refs: list[TuningParameterRef],
    low_rate: float,
) -> int:
    count = 0
    for ref in parameter_refs:
        key = (ref.model_name, ref.layer_name, ref.parameter_name)
        for _, beta, _, rate in summary.parameter_replica_rates.get(key, []):
            if beta == 0.0:
                continue
            if rate < low_rate:
                count += 1
    return count


def _replica_step_scales_for_ref(ref: TuningParameterRef, replica_num: int) -> list[float]:
    target = _parameter_tuning_target(ref.parameter)
    raw_scales = target.get("replica_step_scales", target.get("step_size_scales"))
    if raw_scales is None:
        scales = [1.0] * replica_num
    else:
        if not isinstance(raw_scales, list):
            raise ConfigError(f"{ref.parameter_name}.replica_step_scales must be a list.")
        if len(raw_scales) != replica_num:
            raise ConfigError(f"{ref.parameter_name}.replica_step_scales length must match emc.replica_num.")
        scales = [
            _positive_float(value, f"{ref.parameter_name}.replica_step_scales[{index}]")
            for index, value in enumerate(raw_scales)
        ]
    target["replica_step_scales"] = scales
    target.pop("step_size_scales", None)
    return scales


def _adjust_replica_step_scales_from_low_rates(
    parameter_refs: list[TuningParameterRef],
    summary: DiagnosticsSummary,
    replica_num: int,
    low_rate: float,
    factor: float,
) -> int:
    factor = _local_step_factor(factor)
    adjusted: set[tuple[str, str, str, int]] = set()
    adjustment_count = 0
    for ref in parameter_refs:
        key = (ref.model_name, ref.layer_name, ref.parameter_name)
        for replica_id, beta, _, rate in summary.parameter_replica_rates.get(key, []):
            if beta == 0.0 or rate >= low_rate:
                continue
            if not 0 <= replica_id < replica_num:
                continue
            adjustment_key = (*key, replica_id)
            if adjustment_key in adjusted:
                continue
            scales = _replica_step_scales_for_ref(ref, replica_num)
            scales[replica_id] *= factor
            adjusted.add(adjustment_key)
            adjustment_count += 1
    return adjustment_count


def _gamma_candidate_values(lower: float, upper: float, count: int) -> list[float]:
    lower = max(lower, 1.000001)
    if count == 1:
        candidates = [lower]
    else:
        candidates = [
            lower + (upper - lower) * index / (count - 1)
            for index in range(count)
        ]
    unique = sorted({round(value, 12) for value in candidates if lower <= value <= upper})
    return unique


def _exchange_warning_count(summary: DiagnosticsSummary, low_rate: float, high_rate: float) -> int:
    return sum(1 for *_, rate in summary.exchange_rates if rate < low_rate or rate > high_rate)


def _top_exchange_rate(summary: DiagnosticsSummary) -> float | None:
    for replica_id, _, _, _, rate in summary.exchange_rates:
        if replica_id == 0:
            return rate
    return None


def _non_top_exchange_rates(summary: DiagnosticsSummary) -> list[float]:
    return [rate for replica_id, *_rest, rate in summary.exchange_rates if replica_id != 0]


def _non_top_exchange_low_count(summary: DiagnosticsSummary, low_rate: float) -> int:
    return sum(1 for rate in _non_top_exchange_rates(summary) if rate < low_rate)


def _gamma_tuning_score(summary: DiagnosticsSummary, low_rate: float, top_exchange_target: float) -> tuple[float, float, float]:
    top_exchange = _top_exchange_rate(summary)
    non_top_rates = _non_top_exchange_rates(summary)
    low_count = float(sum(1 for rate in non_top_rates if rate < low_rate))
    low_severity = sum(max(0.0, low_rate - rate) / low_rate for rate in non_top_rates)
    top_deficit = 1.0 if top_exchange is None else max(0.0, top_exchange_target - top_exchange) / top_exchange_target
    return (low_count, top_deficit, low_severity)


def _percent_text(rate: float | None) -> str:
    if rate is None:
        return "NA"
    return f"{100.0 * rate:.2f}%"


def _rate_range(rates: list[float]) -> tuple[float | None, float | None]:
    if not rates:
        return None, None
    return min(rates), max(rates)


def _summary_rate_ranges(summary: DiagnosticsSummary) -> dict[str, float | None]:
    mh_rates = [rate for values in summary.parameter_rates.values() for beta, _, rate in values if beta != 0.0]
    exchange_rates = [rate for *_, rate in summary.exchange_rates]
    min_mh, max_mh = _rate_range(mh_rates)
    min_exchange, max_exchange = _rate_range(exchange_rates)
    return {
        "min_mh": min_mh,
        "max_mh": max_mh,
        "min_exchange": min_exchange,
        "max_exchange": max_exchange,
    }


def _tuning_report_row(
    phase: str,
    round_id: int,
    trial: dict[str, Any],
    low_rate: float,
    high_rate: float,
) -> dict[str, Any]:
    ranges = _summary_rate_ranges(trial["summary"])
    return {
        "phase": phase,
        "round": round_id,
        "label": trial["label"],
        "replica_num": trial["replica_num"],
        "gamma": trial["gamma"],
        "warning_count": trial["warning_count"],
        "exchange_warning_count": _exchange_warning_count(trial["summary"], low_rate, high_rate),
        "min_mh_percent": None if ranges["min_mh"] is None else 100.0 * ranges["min_mh"],
        "max_mh_percent": None if ranges["max_mh"] is None else 100.0 * ranges["max_mh"],
        "min_exchange_percent": None if ranges["min_exchange"] is None else 100.0 * ranges["min_exchange"],
        "max_exchange_percent": None if ranges["max_exchange"] is None else 100.0 * ranges["max_exchange"],
        "result_dir": trial["result_dir"],
        "action": trial.get("action", ""),
    }


def _write_tune_report_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "phase",
        "round",
        "label",
        "replica_num",
        "gamma",
        "warning_count",
        "exchange_warning_count",
        "min_mh_percent",
        "max_mh_percent",
        "min_exchange_percent",
        "max_exchange_percent",
        "result_dir",
        "action",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        values = []
        for key in header:
            value = row.get(key)
            values.append("" if value is None else str(value))
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_command(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    warnings = validate_config(config, require_sources=args.sources)
    print("Config is valid.")
    print(f"data rows: {_count_data_rows(config)}")
    print(f"data path: {config.data_path}")
    print(f"data format: {config.data_format}")
    print(f"data header: {config.data_header}")
    if config.input_columns or config.output_columns:
        print(f"input columns: {config.input_columns or 'positional'}")
        print(f"output columns: {config.output_columns or 'positional'}")
    print(f"result dir: {config.result_dir}")
    print(f"parallel workers: {config.parallel_workers} (0 means auto)")
    print(f"likelihood workers: {config.likelihood_workers} (0 means auto)")
    print(f"likelihood parallel min rows: {config.likelihood_parallel_min_rows}")
    print(f"progress: {config.progress}")
    print(f"progress interval steps: {config.progress_interval_steps} (0 means auto)")
    print(f"progress bar width: {config.progress_bar_width}")
    print(f"noise model: {config.noise_type}")
    print(f"noise estimate sigma2: {config.estimate_sigma2}")
    print(f"noise sigma2 min: {config.sigma2_min}")
    if config.estimate_sigma2:
        print(f"noise finite ladder max sigma2: {config.sigma2_candidate_max}")
        print(f"noise sigma2 candidate count: {config.replica_num - 1}")
    else:
        print(f"noise sigma2 fixed value: {config.sigma2_min}")
        print("noise sigma2 candidates: disabled for estimation")
    _print_model_summary(config)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _print_model_summary(config: NormalizedConfig) -> None:
    print("model layout:")
    for model in config.model_layout:
        print(f"  - {model['name']}: basis_count={model['basis_count']}")
        layers = model["layers"]
        for layer in layers:
            parameters = ", ".join(layer["parameters"])
            if len(layers) == 1 and layer["name"] == "default":
                print(f"    parameters: {parameters}")
            else:
                print(f"    {layer['name']}: {parameters}")
    print(
        "internal shape: "
        f"model_type_num={config.model_type_num}, "
        f"base_nums={config.base_nums}, "
        f"hierarchy_nums={config.hierarchy_nums}, "
        f"parameter_nums={config.parameter_nums}"
    )

def init_command(args: argparse.Namespace) -> int:
    template = args.template
    directory = args.directory
    if directory is None:
        if template == "linear":
            directory = "linear_project"
        elif template == "spectral":
            directory = "spectral_project"
        elif template == "background-spectral":
            directory = "background_spectral_project"
        else:
            raise ConfigError("Available templates: linear, spectral, background-spectral.")
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    (target / "src").mkdir(exist_ok=True)
    (target / "data").mkdir(exist_ok=True)
    (target / "result" / "figures").mkdir(parents=True, exist_ok=True)
    if template == "linear":
        config_template = LINEAR_CONFIG
        target_template = LINEAR_TARGET_HPP_TEMPLATE
        readme_template = LINEAR_README_TEMPLATE
        data_text = _linear_v2_demo_data()
        label = "linear"
    elif template == "spectral":
        config_template = SPECTRAL_CONFIG
        target_template = SPECTRAL_TARGET_HPP_TEMPLATE
        readme_template = SPECTRAL_README_TEMPLATE
        data_text = _spectral_demo_data()
        label = "spectral"
    elif template == "background-spectral":
        config_template = BACKGROUND_SPECTRAL_CONFIG
        target_template = BACKGROUND_SPECTRAL_TARGET_HPP_TEMPLATE
        readme_template = BACKGROUND_SPECTRAL_README_TEMPLATE
        data_text = _background_spectral_v2_demo_data()
        label = "background spectral"
    else:
        raise ConfigError("Available templates: linear, spectral, background-spectral.")

    files = {
        target / "config.json": json.dumps(config_template, indent=4, ensure_ascii=False) + "\n",
        target / "src" / "main.cpp": V2_MAIN_CPP_TEMPLATE,
        target / "src" / "target.hpp": target_template,
        target / "README.md": readme_template,
    }
    for path, content in files.items():
        _write_new_file(path, content, force=args.force)
    data_relative = Path(str(config_template.get("data", {}).get("path", "data/data.csv")))
    data_path = target / data_relative
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not data_path.exists():
        data_path.write_text(data_text, encoding="utf-8")
    print(f"initialized {label} project: {target}")
    return 0


def _write_new_file(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise ConfigError(f"Refusing to overwrite existing file: {path} (use --force)")
    path.write_text(content, encoding="utf-8")


def _spectral_demo_data() -> str:
    params = [
        (0.587, 1.210, 95.689),
        (1.522, 1.455, 146.837),
        (1.183, 1.703, 164.469),
    ]
    rows = ["x,y"]
    for i in range(120):
        x = 0.9 + i * (1.1 / 119.0)
        y = sum(a * math.exp(-0.5 * b * (x - mu) ** 2) for a, mu, b in params)
        rows.append(f"{x:.8f},{y:.8f}")
    return "\n".join(rows) + "\n"


def _linear_v2_demo_data() -> str:
    rows = ["x,y"]
    rng = random.Random(20260415)
    data_count = 100
    for i in range(data_count):
        x = -2.0 + 4.0 * i / (data_count - 1)
        noise = rng.gauss(0.0, 0.05)
        y = 1.25 - 0.80 * x + noise
        rows.append(f"{x:.10f},{y:.10f}")
    return "\n".join(rows) + "\n"


def _background_spectral_v2_demo_data() -> str:
    def peak(x: float, a: float, mu: float, b: float) -> float:
        return a * math.exp(-0.5 * b * (x - mu) ** 2)

    rows = ["x,y"]
    rng = random.Random(20260416)
    data_count = 100
    for i in range(data_count):
        x = -1.5 + 4.0 * i / (data_count - 1)
        noise = rng.gauss(0.0, 0.03)
        y = (
            0.25
            + 0.12 * x
            + peak(x, 0.75, -0.45, 18.0)
            + peak(x, 1.10, 1.15, 14.0)
            + noise
        )
        rows.append(f"{x:.10f},{y:.10f}")
    return "\n".join(rows) + "\n"


def plot_command(args: argparse.Namespace) -> int:
    sample_path = Path(args.sample_json).expanduser().resolve()
    with sample_path.open(encoding="utf-8") as file:
        sample_json = json.load(file)

    samples = _extract_samples(sample_json)
    if not samples:
        raise ConfigError(f"No parameter samples found in {sample_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else sample_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.labels and args.sort_peaks_by:
        raise ConfigError("--labels cannot be combined with --sort-peaks-by because sorted peak labels are generated from sample metadata.")
    labels = args.labels.split(",") if args.labels else _extract_parameter_labels(sample_json)
    if not labels:
        labels = [f"p{i}" for i in range(len(samples[0]))]
    if args.sort_peaks_by:
        samples, labels = _sort_repeated_basis_samples_by_parameter(
            sample_json,
            samples,
            labels,
            args.sort_peaks_by,
        )

    bin_count = _plot_bin_count(args.bins, len(samples))
    smooth = _nonnegative_float(args.smooth, "--smooth")
    density_power = _positive_float(args.density_power, "--density-power")
    dpi = _positive_int(args.dpi, "--dpi")
    plot_ranges = _parse_plot_ranges(args.ranges, len(samples[0])) if args.ranges else None
    posterior = _extract_posterior(sample_json)
    map_index: int | None = None
    map_values: list[float] | None = None
    if posterior and len(posterior) == len(samples):
        map_index = max(range(len(posterior)), key=posterior.__getitem__)
        map_values = list(samples[map_index])

    figure_path: Path
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        sample_array = np.array(samples, dtype=float)
        corner_path = output_dir / "corner.png"
        import corner

        figure = _write_corner_png(
            sample_array,
            labels,
            corner_path,
            corner_module=corner,
            bins=bin_count,
            smooth=smooth,
            density_power=density_power,
            dpi=dpi,
            plot_datapoints=args.plot_datapoints,
            plot_contours=args.plot_contours,
            map_values=map_values,
            ranges=plot_ranges,
        )
        plt.close(figure)
        figure_path = corner_path
    except ImportError:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            sample_array = np.array(samples, dtype=float)
            corner_path = output_dir / "corner.png"
            figure, axes = plt.subplots(sample_array.shape[1], 1, figsize=(6, 2.4 * sample_array.shape[1]))
            if sample_array.shape[1] == 1:
                axes = [axes]
            for idx, axis in enumerate(axes):
                hist_range = plot_ranges[idx] if plot_ranges is not None else None
                axis.hist(sample_array[:, idx], bins=bin_count, range=hist_range, color="#1f2937", alpha=0.75)
                if map_values is not None and idx < len(map_values):
                    axis.axvline(
                        map_values[idx],
                        color="#dc2626",
                        linewidth=1.4,
                        alpha=0.95,
                        zorder=8,
                    )
                axis.set_xlabel(labels[idx] if idx < len(labels) else f"p{idx}")
                axis.set_ylabel("count")
                if hist_range is not None:
                    axis.set_xlim(hist_range[0], hist_range[1])
            figure.tight_layout()
            figure.savefig(corner_path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            figure_path = corner_path
        except ImportError:
            svg_path = output_dir / "posterior.svg"
            _write_simple_svg_plot(samples, labels, svg_path, bin_count=bin_count, map_values=map_values, ranges=plot_ranges)
            figure_path = svg_path

    if posterior and map_index is not None:
        summary_path = output_dir / "posterior_max.json"
        summary_path.write_text(
            json.dumps(
                {
                    "sample_index": map_index,
                    "posterior": posterior[map_index],
                    "parameters": _sample_parameters_from_values(labels, samples[map_index]),
                },
                indent=4,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {summary_path}")

    print(f"wrote {figure_path}")
    return 0


def _plot_bin_count(value: Any, sample_count: int) -> int:
    if value is None or str(value).strip().lower() == "auto":
        sample_root = math.sqrt(max(1, sample_count))
        if sample_count < 50:
            return min(24, max(12, int(round(sample_root * 2.0))))
        return min(64, max(32, int(round(sample_root * 1.8))))
    parsed = _positive_int(value, "--bins")
    if parsed < 2:
        raise ConfigError("--bins must be at least 2.")
    return parsed


def _parse_plot_ranges(value: str, dimensions: int) -> list[tuple[float, float] | None]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != dimensions:
        raise ConfigError(f"--ranges must provide {dimensions} comma-separated ranges, got {len(parts)}")
    ranges: list[tuple[float, float] | None] = []
    for index, part in enumerate(parts):
        if part.lower() in {"auto", "*"}:
            ranges.append(None)
            continue
        bounds = part.split(":")
        if len(bounds) != 2:
            raise ConfigError(f"--ranges item {index + 1} must be 'min:max' or 'auto'")
        try:
            lower = float(bounds[0])
            upper = float(bounds[1])
        except ValueError as exc:
            raise ConfigError(f"--ranges item {index + 1} contains a non-numeric bound: {part}") from exc
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ConfigError(f"--ranges item {index + 1} must be finite: {part}")
        if lower >= upper:
            raise ConfigError(f"--ranges item {index + 1} must have min < max: {part}")
        ranges.append((lower, upper))
    return ranges


def _sort_repeated_basis_samples_by_parameter(
    sample_json: dict[str, Any],
    samples: list[list[float]],
    labels: list[str],
    sort_parameter: str,
) -> tuple[list[list[float]], list[str]]:
    parameter_name = sort_parameter.strip()
    if not parameter_name:
        raise ConfigError("--sort-peaks-by must name a parameter, e.g. mu.")
    if not samples:
        return samples, labels

    records = _plot_parameter_records(sample_json, labels)
    sample_width = len(samples[0])
    if len(records) != sample_width:
        raise ConfigError("--sort-peaks-by requires one sample.json parameters record per sample column.")
    if any(len(row) != sample_width for row in samples):
        raise ConfigError("sample.json contains rows with inconsistent parameter counts.")
    for record in records:
        if record["offset"] >= sample_width:
            raise ConfigError("sample.json parameter metadata does not match the sample column count.")

    groups = _repeated_basis_sort_groups(records, parameter_name)
    if not groups:
        raise ConfigError(
            f"--sort-peaks-by {parameter_name!r} requires a repeated-basis model "
            "where every basis has that parameter."
        )

    groups_by_model = {group["model_id"]: group for group in groups}
    multi_group = len(groups) > 1
    inserted_models: set[int] = set()
    output_specs: list[dict[str, Any]] = []
    output_labels: list[str] = []

    for record in sorted(records, key=lambda item: item["original_index"]):
        group = groups_by_model.get(record["model_id"])
        if group is None:
            output_specs.append({"kind": "fixed", "offset": record["offset"]})
            output_labels.append(record["label"])
            continue
        if record["model_id"] in inserted_models:
            continue
        inserted_models.add(record["model_id"])
        label_prefix = f"{group['model_label']}.peak_position" if multi_group else "peak_position"
        for position in range(len(group["basis_ids"])):
            for key in group["parameter_keys"]:
                suffix = group["suffix_by_key"][key]
                output_specs.append({
                    "kind": "sorted_basis",
                    "group": group,
                    "position": position,
                    "key": key,
                })
                output_labels.append(f"{label_prefix}[{position}].{suffix}")

    sorted_samples: list[list[float]] = []
    for row in samples:
        basis_order_by_model = {
            group["model_id"]: sorted(
                group["basis_ids"],
                key=lambda basis_id: row[group["sort_offset_by_basis"][basis_id]],
            )
            for group in groups
        }
        sorted_row: list[float] = []
        for spec in output_specs:
            if spec["kind"] == "fixed":
                sorted_row.append(row[spec["offset"]])
                continue
            group = spec["group"]
            basis_id = basis_order_by_model[group["model_id"]][spec["position"]]
            record = group["record_by_basis_key"][basis_id][spec["key"]]
            sorted_row.append(row[record["offset"]])
        sorted_samples.append(sorted_row)

    return sorted_samples, output_labels


def _plot_parameter_records(sample_json: dict[str, Any], labels: list[str]) -> list[dict[str, Any]]:
    parameters = sample_json.get("parameters")
    if not isinstance(parameters, list):
        raise ConfigError("--sort-peaks-by requires sample.json parameters metadata.")
    records: list[dict[str, Any]] = []
    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, dict):
            raise ConfigError("sample.json parameters metadata must contain objects.")
        fallback_label = labels[index] if index < len(labels) else f"p{index}"
        label = parameter.get("label", fallback_label)
        if not isinstance(label, str) or not label:
            label = fallback_label
        name = _plot_parameter_name(parameter, label)
        records.append({
            "original_index": index,
            "offset": _nonnegative_int(parameter.get("offset", index), f"parameters[{index}].offset"),
            "model_id": _nonnegative_int(parameter.get("model_id", 0), f"parameters[{index}].model_id"),
            "basis_id": _nonnegative_int(parameter.get("basis_id", 0), f"parameters[{index}].basis_id"),
            "layer_id": _nonnegative_int(parameter.get("layer_id", 0), f"parameters[{index}].layer_id"),
            "parameter_id": _nonnegative_int(parameter.get("parameter_id", index), f"parameters[{index}].parameter_id"),
            "label": label,
            "model_label": _plot_model_label(label),
            "name": name,
            "suffix": _plot_parameter_suffix(label, name),
        })
    return records


def _plot_parameter_name(parameter: dict[str, Any], label: str) -> str:
    name = parameter.get("name")
    if isinstance(name, str) and name:
        return name
    if label:
        return label.split(".")[-1]
    return "parameter"


def _plot_model_label(label: str) -> str:
    first = label.split(".", 1)[0]
    if "[" in first:
        return first.split("[", 1)[0]
    return first or "model"


def _plot_parameter_suffix(label: str, name: str) -> str:
    pieces = label.split(".")
    if len(pieces) >= 3 and pieces[-2] != "default":
        return f"{pieces[-2]}.{pieces[-1]}"
    return name


def _repeated_basis_sort_groups(records: list[dict[str, Any]], parameter_name: str) -> list[dict[str, Any]]:
    models: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        models.setdefault(record["model_id"], []).append(record)

    groups: list[dict[str, Any]] = []
    for model_id, model_records in models.items():
        basis_records: dict[int, list[dict[str, Any]]] = {}
        for record in model_records:
            basis_records.setdefault(record["basis_id"], []).append(record)
        if len(basis_records) < 2:
            continue

        sort_offset_by_basis: dict[int, int] = {}
        for basis_id, items in basis_records.items():
            matches = [item for item in items if item["name"] == parameter_name]
            if len(matches) != 1:
                break
            sort_offset_by_basis[basis_id] = matches[0]["offset"]
        else:
            reference_basis = min(basis_records)
            reference_records = sorted(basis_records[reference_basis], key=lambda item: item["original_index"])
            parameter_keys = [
                (item["layer_id"], item["parameter_id"], item["name"])
                for item in reference_records
            ]
            suffix_by_key = {
                (item["layer_id"], item["parameter_id"], item["name"]): item["suffix"]
                for item in reference_records
            }
            record_by_basis_key: dict[int, dict[tuple[int, int, str], dict[str, Any]]] = {}
            expected_keys = set(parameter_keys)
            for basis_id, items in basis_records.items():
                by_key = {
                    (item["layer_id"], item["parameter_id"], item["name"]): item
                    for item in items
                }
                if set(by_key) != expected_keys:
                    break
                record_by_basis_key[basis_id] = by_key
            else:
                groups.append({
                    "model_id": model_id,
                    "model_label": model_records[0]["model_label"],
                    "basis_ids": sorted(basis_records),
                    "sort_offset_by_basis": sort_offset_by_basis,
                    "parameter_keys": parameter_keys,
                    "suffix_by_key": suffix_by_key,
                    "record_by_basis_key": record_by_basis_key,
                })

    return groups


def _write_corner_png(
    sample_array: Any,
    labels: list[str],
    output_path: Path,
    *,
    corner_module: Any,
    bins: int,
    smooth: float,
    density_power: float,
    dpi: int,
    plot_datapoints: bool,
    plot_contours: bool,
    map_values: Sequence[float] | None = None,
    ranges: Sequence[tuple[float, float] | None] | None = None,
) -> Any:
    posterior_color = "#000000"
    posterior_edge_color = "#000000"
    corner_kwargs: dict[str, Any] = {
        "bins": bins,
        "labels": labels,
        "color": posterior_color,
        "plot_datapoints": plot_datapoints,
        "plot_density": True,
        "plot_contours": plot_contours,
        "fill_contours": False,
        "hist_kwargs": {
            "color": posterior_color,
            "alpha": 1.0,
            "edgecolor": posterior_edge_color,
            "linewidth": 1.15,
        },
        "pcolor_kwargs": {"alpha": 1.0},
    }
    if ranges is not None:
        corner_kwargs["range"] = ranges
    if plot_contours:
        corner_kwargs["contour_kwargs"] = {"colors": posterior_edge_color, "linewidths": 0.8, "alpha": 0.9}
    if smooth > 0.0:
        corner_kwargs["smooth"] = smooth
        corner_kwargs["smooth1d"] = smooth
    restore_density_norm = _install_independent_corner_density_norm(corner_module, density_power)
    try:
        try:
            figure = corner_module.corner(sample_array, **corner_kwargs)
        except ImportError:
            if smooth <= 0.0:
                raise
            corner_kwargs.pop("smooth", None)
            corner_kwargs.pop("smooth1d", None)
            figure = corner_module.corner(sample_array, **corner_kwargs)
    finally:
        if restore_density_norm is not None:
            restore_density_norm()
    _overlay_map_on_corner_figure(figure, map_values, color="#dc2626")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return figure


def _overlay_map_on_corner_figure(
    figure: Any,
    map_values: Sequence[float] | None,
    *,
    color: str,
) -> None:
    if map_values is None:
        return
    axes = list(getattr(figure, "axes", []) or [])
    dimensions = len(map_values)
    if dimensions <= 0 or len(axes) < dimensions * dimensions:
        return

    for row in range(dimensions):
        for column in range(row + 1):
            axis = axes[row * dimensions + column]
            x_value = float(map_values[column])
            if row == column:
                axis.axvline(
                    x_value,
                    color=color,
                    linewidth=1.4,
                    alpha=0.95,
                    zorder=8,
                )
            else:
                y_value = float(map_values[row])
                axis.axvline(
                    x_value,
                    color=color,
                    linewidth=1.4,
                    alpha=0.95,
                    zorder=8,
                )
                axis.axhline(
                    y_value,
                    color=color,
                    linewidth=1.4,
                    alpha=0.95,
                    zorder=8,
                )


def _install_independent_corner_density_norm(corner_module: Any, density_power: float) -> Any | None:
    norm_factory = _density_power_norm_factory(density_power)
    core_module = getattr(corner_module, "core", None)
    original_hist2d = getattr(core_module, "hist2d", None)
    if norm_factory is None or core_module is None or original_hist2d is None:
        return None

    def hist2d_with_fresh_norm(*args: Any, **kwargs: Any) -> Any:
        pcolor_kwargs = dict(kwargs.get("pcolor_kwargs") or {})
        pcolor_kwargs["norm"] = norm_factory()
        kwargs["pcolor_kwargs"] = pcolor_kwargs
        return original_hist2d(*args, **kwargs)

    core_module.hist2d = hist2d_with_fresh_norm

    def restore() -> None:
        core_module.hist2d = original_hist2d

    return restore


def _density_power_norm_factory(density_power: float) -> Any | None:
    if abs(density_power - 1.0) < 1e-12:
        return None
    try:
        from matplotlib.colors import PowerNorm
    except ImportError:
        return None
    return lambda: PowerNorm(gamma=density_power)


def _write_simple_svg_plot(
    samples: list[list[float]],
    labels: list[str],
    output_path: Path,
    *,
    bin_count: int | None = None,
    map_values: Sequence[float] | None = None,
    ranges: Sequence[tuple[float, float] | None] | None = None,
) -> None:
    dimensions = len(samples[0])
    panel_width = 640
    panel_height = 150
    left = 64
    top = 36
    gap = 42
    width = left + panel_width + 32
    height = top + dimensions * panel_height + max(0, dimensions - 1) * gap + 36
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#222}.axis{stroke:#222;stroke-width:1}.bar{fill:#1e3a8a;opacity:.92}</style>',
    ]

    for dimension in range(dimensions):
        values = [row[dimension] for row in samples]
        fixed_range = ranges[dimension] if ranges is not None else None
        if fixed_range is not None:
            lower, upper = fixed_range
        else:
            lower = min(values)
            upper = max(values)
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        panel_bin_count = bin_count if bin_count is not None else _plot_bin_count("auto", len(values))
        counts = [0 for _ in range(panel_bin_count)]
        for value in values:
            ratio = (value - lower) / (upper - lower)
            if ratio < 0.0 or ratio > 1.0:
                continue
            bin_id = min(panel_bin_count - 1, max(0, int(ratio * panel_bin_count)))
            counts[bin_id] += 1
        max_count = max(counts) or 1
        panel_top = top + dimension * (panel_height + gap)
        label = labels[dimension] if dimension < len(labels) else f"p{dimension}"
        svg.append(f'<text x="{left}" y="{panel_top - 12}">{html.escape(label)}</text>')
        svg.append(f'<line class="axis" x1="{left}" y1="{panel_top + panel_height}" x2="{left + panel_width}" y2="{panel_top + panel_height}"/>')
        svg.append(f'<line class="axis" x1="{left}" y1="{panel_top}" x2="{left}" y2="{panel_top + panel_height}"/>')
        bar_width = panel_width / panel_bin_count
        for bin_id, count in enumerate(counts):
            bar_height = panel_height * count / max_count
            x = left + bin_id * bar_width
            y = panel_top + panel_height - bar_height
            svg.append(
                f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" '
                f'width="{max(1.0, bar_width - 1):.2f}" height="{bar_height:.2f}"/>'
            )
        if map_values is not None and dimension < len(map_values):
            span = upper - lower
            if span <= 0.0:
                span = 1.0
            map_value = float(map_values[dimension])
            x_value = left + (map_value - lower) / span * panel_width
            svg.append(
                f'<line x1="{x_value:.2f}" y1="{panel_top:.2f}" '
                f'x2="{x_value:.2f}" y2="{panel_top + panel_height:.2f}" '
                'stroke="#dc2626" stroke-width="1.5" stroke-linecap="round"/>'
            )
        svg.append(f'<text x="{left}" y="{panel_top + panel_height + 18}">{lower:.6g}</text>')
        svg.append(f'<text x="{left + panel_width - 72}" y="{panel_top + panel_height + 18}">{upper:.6g}</text>')

    svg.append("</svg>\n")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def _extract_samples(sample_json: dict[str, Any]) -> list[list[float]]:
    if isinstance(sample_json.get("samples"), (list, dict)):
        return _extract_samples_v2(sample_json)
    raise ConfigError("sample.json must contain a V2 samples field.")


def _extract_samples_v2(sample_json: dict[str, Any]) -> list[list[float]]:
    raw_samples = sample_json.get("samples", [])
    if isinstance(raw_samples, dict):
        raw_values = raw_samples.get("values", [])
        if not isinstance(raw_values, list):
            return []
        samples: list[list[float]] = []
        for values in raw_values:
            if isinstance(values, list) and values and all(_is_number_like(item) for item in values):
                samples.append([float(item) for item in values])
        return samples

    samples: list[list[float]] = []
    for sample in raw_samples:
        values = sample.get("values")
        if isinstance(values, list) and values and all(_is_number_like(item) for item in values):
            samples.append([float(item) for item in values])
            continue

        groups = sample.get("parameter_groups", [])
        if not isinstance(groups, list):
            continue
        flattened: list[float] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            values = group.get("values")
            if isinstance(values, list) and values and all(_is_number_like(item) for item in values):
                flattened.extend(float(item) for item in values)
        if flattened:
            samples.append(flattened)
    return samples


def _is_number_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _extract_posterior(sample_json: dict[str, Any]) -> list[float]:
    if isinstance(sample_json.get("samples"), (list, dict)):
        return _extract_posterior_v2(sample_json)
    raise ConfigError("sample.json must contain a V2 samples field.")


def _extract_posterior_v2(sample_json: dict[str, Any]) -> list[float]:
    raw_samples = sample_json.get("samples", [])
    if isinstance(raw_samples, dict):
        scores = raw_samples.get("log_posterior", raw_samples.get("posterior", []))
        if not isinstance(scores, list):
            return []
        posterior: list[float] = []
        for score in scores:
            try:
                posterior.append(float(score))
            except (TypeError, ValueError):
                pass
        return posterior

    posterior: list[float] = []
    for sample in raw_samples:
        score = sample.get("posterior", sample.get("log_posterior"))
        if score is None:
            continue
        try:
            posterior.append(float(score))
        except (TypeError, ValueError):
            pass
    return posterior


def _extract_parameter_labels(sample_json: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    parameters = sample_json.get("parameters")
    if not isinstance(parameters, list):
        return labels
    for parameter in parameters:
        if not isinstance(parameter, dict):
            return []
        label = parameter.get("label")
        if not isinstance(label, str) or not label:
            return []
        labels.append(label)
    return labels


def _extract_sample_parameters(sample_json: dict[str, Any], sample_index: int) -> Any:
    raw_samples = sample_json.get("samples")
    if isinstance(raw_samples, dict):
        raw_values = raw_samples.get("values", [])
        if not isinstance(raw_values, list):
            return []
        values = raw_values[sample_index]
        if not isinstance(values, list):
            return []
        labels = _extract_parameter_labels(sample_json)
        if labels and len(labels) == len(values):
            return [
                {"label": labels[index], "value": values[index]}
                for index in range(len(values))
            ]
        return values
    if isinstance(raw_samples, list):
        sample = raw_samples[sample_index]
        return sample.get("parameter_groups", sample.get("values", []))
    raise ConfigError("sample.json must contain a V2 samples field.")


def _sample_parameters_from_values(labels: list[str], values: list[float]) -> Any:
    if labels and len(labels) == len(values):
        return [
            {"label": labels[index], "value": values[index]}
            for index in range(len(values))
        ]
    return values


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayes-emc",
        description="Configure, run, and plot the C++ EMC Bayesian inference engine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create a template analysis project")
    init_parser.add_argument(
        "template",
        choices=["linear", "spectral", "background-spectral"],
    )
    init_parser.add_argument("directory", nargs="?")
    init_parser.add_argument("--force", action="store_true", help="overwrite existing template files")
    init_parser.set_defaults(func=init_command)

    check_parser = subparsers.add_parser("check", help="validate config and data shape")
    check_parser.add_argument("config")
    check_parser.add_argument("--sources", action="store_true", help="also require C++ model source files")
    check_parser.set_defaults(func=check_command)

    run_parser = subparsers.add_parser("run", help="generate V2 config header, build C++, and execute EMC")
    run_parser.add_argument("config")
    run_parser.add_argument("--dry-run", action="store_true", help="validate and print commands without writing")
    run_parser.add_argument("--skip-build", action="store_true", help="reuse an existing compiled executable")
    run_parser.add_argument("--skip-exec", action="store_true", help="build only")
    run_parser.set_defaults(func=run_command)

    tune_parser = subparsers.add_parser("tune", help="tune C, gamma, and d using short EMC trial runs")
    tune_parser.add_argument("config")
    tune_parser.add_argument("--output-config", help="path for the tuned config; defaults to config.tuned.json")
    tune_parser.add_argument("--output-dir", help="directory for tuning trial outputs; defaults to result/tuning")
    tune_parser.add_argument("--sample-num", type=int, default=300, help="samples per tuning trial")
    tune_parser.add_argument("--burnin-num", type=int, default=300, help="burn-in steps per tuning trial")
    tune_parser.add_argument("--low-rate", type=float, default=0.10, help="low warning threshold as fraction or percent")
    tune_parser.add_argument("--high-rate", type=float, default=0.99, help="high warning threshold as fraction or percent")
    tune_parser.add_argument("--top-accept-rate", type=float, default=0.90, help="target acceptance for the first finite temperature layer")
    tune_parser.add_argument("--top-exchange-rate", type=float, default=0.90, help="target exchange rate between beta=0 and the first finite temperature")
    tune_parser.add_argument("--keep-c", action="store_true", help="keep existing C values instead of resetting from priors")
    tune_parser.add_argument("--keep-replica-step-scales", action="store_true", help="keep existing per-replica step-size scales before tuning")
    tune_parser.add_argument("--initial-replica-num", type=int, default=8, help="replica count to start tuning from")
    tune_parser.add_argument("--c-rounds", type=int, default=6, help="maximum high-temperature C tuning rounds")
    tune_parser.add_argument("--c-factor", type=float, default=1.5, help="multiplicative C adjustment factor")
    tune_parser.add_argument("--gamma-min", type=float, default=1.0)
    tune_parser.add_argument("--gamma-max", type=float, default=10.0)
    tune_parser.add_argument("--gamma-candidates", type=int, default=16, help="gamma candidates per replica count")
    tune_parser.add_argument("--max-replica-num", type=int, help="maximum replica count to try")
    tune_parser.add_argument("--replica-step", type=int, default=4, help="replica count increment when gamma cannot satisfy exchange rates")
    tune_parser.add_argument("--d-rounds", type=int, default=8, help="maximum coldest-temperature d tuning rounds")
    tune_parser.add_argument("--d-initial", type=float, default=0.5, help="initial d value before the d tuning phase")
    tune_parser.add_argument("--d-target-rate", type=float, default=0.30, help="target MH acceptance at the coldest temperature")
    tune_parser.add_argument("--d-tolerance", type=float, default=0.05, help="accepted deviation around --d-target-rate")
    tune_parser.add_argument("--d-step", type=float, default=0.25, help="additive d adjustment step")
    tune_parser.add_argument("--min-d", type=float, default=0.0)
    tune_parser.add_argument("--max-d", type=float, default=2.0, help="maximum d value during tuning; values above 1 are allowed")
    tune_parser.add_argument("--local-step-rounds", type=int, default=2, help="final rounds that shrink only low-acceptance replica/parameter step sizes")
    tune_parser.add_argument("--local-step-factor", type=float, default=0.5, help="multiplier for low-acceptance replica/parameter step-size scales")
    tune_parser.add_argument("--quiet", action="store_true", help="suppress per-trial build/run messages")
    tune_parser.set_defaults(func=tune_command)

    select_parser = subparsers.add_parser("select-peaks", help="select spectral peak count by free energy")
    select_parser.add_argument("config")
    select_parser.add_argument("--min", type=int, required=True, help="minimum peak count")
    select_parser.add_argument("--max", type=int, required=True, help="maximum peak count")
    select_parser.add_argument("--model", help="model name to vary; defaults to spectral_peaks")
    select_parser.add_argument(
        "--criterion",
        choices=["estimated-noise", "fixed-noise"],
        default="estimated-noise",
        help="compare each model at estimated noise or fixed sigma2_min",
    )
    select_parser.add_argument("--output-dir", help="directory for model selection outputs")
    select_parser.add_argument("--dry-run", action="store_true", help="validate and print candidate commands")
    select_parser.set_defaults(func=select_peaks_command)

    bench_parser = subparsers.add_parser("benchmark", help="run a short EMC benchmark without changing config.json")
    bench_parser.add_argument("config")
    bench_parser.add_argument("--sample-num", type=int, default=200)
    bench_parser.add_argument("--burnin-num", type=int, default=200)
    bench_parser.add_argument("--repeat", type=int, default=3)
    bench_parser.add_argument("--parallel-workers", type=int, help="override emc.parallel_workers; 0 means auto")
    bench_parser.add_argument("--parallel-workers-list", help="comma-separated worker counts to compare, e.g. 1,2,4,0")
    bench_parser.add_argument("--likelihood-workers", type=int, help="override emc.likelihood_workers; 0 means auto")
    bench_parser.add_argument("--likelihood-parallel-min-rows", type=int, help="minimum data rows before likelihood parallelism")
    bench_parser.add_argument("--skip-build", action="store_true", help="reuse an existing compiled executable")
    bench_parser.add_argument("--output-dir", help="benchmark result directory")
    bench_parser.set_defaults(func=benchmark_command)

    plot_parser = subparsers.add_parser("plot", help="plot posterior samples from sample.json")
    plot_parser.add_argument("sample_json")
    plot_parser.add_argument("--output-dir")
    plot_parser.add_argument("--labels", help="comma-separated parameter labels, e.g. a,mu,b")
    plot_parser.add_argument("--bins", default="auto", help="histogram bins for posterior plots; default chooses from sample count")
    plot_parser.add_argument("--smooth", type=float, default=0.0, help="Gaussian smoothing width for corner plots; 0 disables smoothing")
    plot_parser.add_argument("--density-power", type=float, default=1.6, help="density color normalization power; larger values make high-density regions darker")
    plot_parser.add_argument("--dpi", type=int, default=220, help="PNG resolution for matplotlib/corner output")
    plot_parser.add_argument("--plot-datapoints", action="store_true", help="show raw sample points in corner off-diagonal panels")
    plot_parser.add_argument("--plot-contours", action="store_true", help="show contour lines in corner off-diagonal panels")
    plot_parser.add_argument("--sort-peaks-by", help="sort repeated basis components in each sample by this parameter before plotting, e.g. mu")
    plot_parser.add_argument("--ranges", help="comma-separated fixed plot ranges, one per parameter, as min:max or auto")
    plot_parser.set_defaults(func=plot_command)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        return exc.returncode


PROPOSAL_TUNING_COMMENT = (
    "Proposal step width is C when data_size * beta < 1, otherwise C / (data_size * beta)^d. "
    "Omit C to infer a conservative initial value from the prior "
    "(uniform width/3, normal sigma/3, gamma standard deviation/3, beta width/3). "
    "Omit d to use 0.5; d may be 0 or greater than 1. "
    "bayes-emc tune first shrinks C until first finite-temperature acceptance exceeds 90%, "
    "then tunes gamma/replica_num from exchange rates, expands C again if that same acceptance is 99% or higher, "
    "resets d to 0.5 and adjusts it until coldest-temperature acceptance is about 30%, "
    "then optionally writes replica_step_scales to shrink only replica/parameter step sizes whose acceptance remains below 10%."
)


LINEAR_CONFIG: dict[str, Any] = {
    "project": {
        "name": "linear_1d",
        "model": "linear",
        "result_dir": "result",
    },
    "data": {
        "path": "data/data.csv",
        "format": "csv",
        "header": True,
        "input_columns": ["x"],
        "output_columns": ["y"],
        "input_dim": 1,
        "output_dim": 1,
    },
    "emc": {
        "_comment": "Exchange Monte Carlo settings. First tune sample_num and burnin_num; tune replica_num/gamma only when exploration or replica exchange is poor. The default ladder keeps the first finite beta very close to zero so the beta=0 prior replica remains exchangeable even with broad priors or small sigma2_min.",
        "_comment_replica_num": "Number of replicas in the inverse-temperature ladder. If estimate_sigma2 is true, sample.json uses the free-energy-selected temperature layer; otherwise it uses the beta=1 layer.",
        "replica_num": 36,
        "_comment_gamma": "Spacing ratio for inverse temperatures: beta_0 = 0, beta_i = gamma^(i + 1 - replica_num) for i >= 1, and the last beta is 1. gamma must be >= 1.0.",
        "gamma": 1.6,
        "_comment_sample_num": "Number of posterior samples saved after burn-in.",
        "sample_num": 800,
        "_comment_burnin_num": "Number of update steps discarded before saving samples.",
        "burnin_num": 2500,
        "_comment_sample_stride": "Number of update steps between saved samples. Default is 1. Increase only when you intentionally thin saved samples.",
        "sample_stride": 1,
        "_comment_exchange_stride": "How often adjacent replicas try to swap states. 1 means every update step.",
        "exchange_stride": 1,
        "_comment_parallel_workers": "Replica-level parallel workers. 0 means auto from CPU count, 1 means serial.",
        "parallel_workers": 0,
        "_comment_likelihood_workers": "Within-replica likelihood workers over data rows. 1 means serial, 0 means auto.",
        "likelihood_workers": 1,
        "_comment_likelihood_parallel_min_rows": "Likelihood parallelism is disabled below this number of data rows to avoid thread overhead.",
        "likelihood_parallel_min_rows": 2048,
        "_comment_progress": "Show a progress bar during long runs.",
        "progress": True,
        "_comment_progress_interval_steps": "Progress refresh interval in update steps. 0 means auto.",
        "progress_interval_steps": 0,
        "_comment_progress_bar_width": "Width of the text progress bar.",
        "progress_bar_width": 32,
        "_comment_seed": "Random seed. Fix it for reproducibility; change it to check robustness.",
        "seed": 12345,
    },
    "model": {
        "models": [
            {
                "name": "linear",
                "basis_count": 1,
                "parameters": [
                    {
                        "name": "intercept",
                        "prior": {"type": "normal", "mean": 0.0, "sigma": 5.0},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 0.1,
                        "d": 0.5,
                    },
                    {
                        "name": "slope",
                        "prior": {"type": "normal", "mean": 0.0, "sigma": 5.0},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 0.1,
                        "d": 0.5,
                    },
                ],
            }
        ],
        "noise": {
            "_comment": "Gaussian observation noise. sigma2_min is the beta=1 lower-bound variance. This tutorial sets it below the synthetic noise scale so free-energy estimation can move upward.",
            "_comment_estimate_sigma2": "true estimates sigma2 by free-energy comparison; false keeps sigma2_min fixed as a known value.",
            "type": "gaussian",
            "sigma2_min": 0.0001,
            "estimate_sigma2": True,
        },
    },
    "build": {
        "compiler": "c++",
        "include_dirs": [],
        "library_dirs": [],
        "flags": ["-std=c++20", "-O2", "-pthread"],
        "libs": [],
        "output": "v2_main.out",
    },
}


SPECTRAL_CONFIG: dict[str, Any] = {
    "project": {
        "name": "spectral_decomposition",
        "model": "spectral",
        "result_dir": "result",
    },
    "data": {
        "path": "data/data.csv",
        "format": "csv",
        "header": True,
        "input_columns": ["x"],
        "output_columns": ["y"],
        "input_dim": 1,
        "output_dim": 1,
    },
    "emc": {
        "_comment": "Exchange Monte Carlo settings. First tune sample_num and burnin_num; tune replica_num/gamma only when exploration or replica exchange is poor. The default ladder keeps the first finite beta very close to zero so the beta=0 prior replica remains exchangeable even with broad priors or small sigma2_min.",
        "_comment_replica_num": "Number of replicas in the inverse-temperature ladder. If estimate_sigma2 is true, sample.json uses the free-energy-selected temperature layer; otherwise it uses the beta=1 layer.",
        "replica_num": 36,
        "_comment_gamma": "Spacing ratio for inverse temperatures: beta_0 = 0, beta_i = gamma^(i + 1 - replica_num) for i >= 1, and the last beta is 1. gamma must be >= 1.0.",
        "gamma": 1.6,
        "_comment_sample_num": "Number of posterior samples saved after burn-in.",
        "sample_num": 240,
        "_comment_burnin_num": "Number of update steps discarded before saving samples.",
        "burnin_num": 600,
        "_comment_sample_stride": "Number of update steps between saved samples. Default is 1. Increase only when you intentionally thin saved samples.",
        "sample_stride": 1,
        "_comment_exchange_stride": "How often adjacent replicas try to swap states. 1 means every update step.",
        "exchange_stride": 1,
        "_comment_parallel_workers": "Replica-level parallel workers. 0 means auto from CPU count, 1 means serial.",
        "parallel_workers": 0,
        "_comment_likelihood_workers": "Within-replica likelihood workers over data rows. 1 means serial, 0 means auto.",
        "likelihood_workers": 1,
        "_comment_likelihood_parallel_min_rows": "Likelihood parallelism is disabled below this number of data rows to avoid thread overhead.",
        "likelihood_parallel_min_rows": 2048,
        "_comment_progress": "Show a progress bar during long runs.",
        "progress": True,
        "_comment_progress_interval_steps": "Progress refresh interval in update steps. 0 means auto.",
        "progress_interval_steps": 0,
        "_comment_progress_bar_width": "Width of the text progress bar.",
        "progress_bar_width": 32,
        "_comment_seed": "Random seed. Fix it for reproducibility; change it to check robustness.",
        "seed": 20260415,
    },
    "model": {
        "models": [
            {
                "name": "spectral_peaks",
                "basis_count": 3,
                "parameters": [
                    {
                        "name": "a",
                        "prior": {"type": "gamma", "shape": 2.0, "scale": 0.5},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 0.5,
                        "d": 0.5,
                    },
                    {
                        "name": "mu",
                        "prior": {"type": "normal", "mean": 1.5, "sigma": 0.5},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 0.5,
                        "d": 0.7,
                    },
                    {
                        "name": "b",
                        "prior": {"type": "gamma", "shape": 14.0, "scale": 10.0},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 40.0,
                        "d": 0.6,
                    },
                ],
            }
        ],
        "noise": {
            "_comment": "Gaussian observation noise. sigma2_min is the beta=1 minimum/base variance. If estimate_sigma2 is true, free-energy noise estimation compares sigma2_min / beta candidates.",
            "_comment_estimate_sigma2": "true estimates sigma2 by free-energy comparison; false keeps sigma2_min fixed as a known value.",
            "type": "gaussian",
            "sigma2_min": 0.01,
            "estimate_sigma2": True,
        },
    },
    "build": {
        "compiler": "c++",
        "include_dirs": [],
        "library_dirs": [],
        "flags": ["-std=c++20", "-O2", "-pthread"],
        "libs": [],
        "output": "v2_main.out",
    },
}


BACKGROUND_SPECTRAL_CONFIG: dict[str, Any] = {
    "project": {
        "name": "background_spectral",
        "model": "background-spectral",
        "result_dir": "result",
    },
    "data": {
        "path": "data/data.csv",
        "format": "csv",
        "header": True,
        "input_columns": ["x"],
        "output_columns": ["y"],
        "input_dim": 1,
        "output_dim": 1,
    },
    "emc": {
        "_comment": "Exchange Monte Carlo settings. First tune sample_num and burnin_num; tune replica_num/gamma only when exploration or replica exchange is poor. The default ladder keeps the first finite beta very close to zero so the beta=0 prior replica remains exchangeable even with broad priors or small sigma2_min.",
        "_comment_replica_num": "Number of replicas in the inverse-temperature ladder. If estimate_sigma2 is true, sample.json uses the free-energy-selected temperature layer; otherwise it uses the beta=1 layer.",
        "replica_num": 36,
        "_comment_gamma": "Spacing ratio for inverse temperatures: beta_0 = 0, beta_i = gamma^(i + 1 - replica_num) for i >= 1, and the last beta is 1. gamma must be >= 1.0.",
        "gamma": 1.6,
        "_comment_sample_num": "Number of posterior samples saved after burn-in.",
        "sample_num": 600,
        "_comment_burnin_num": "Number of update steps discarded before saving samples.",
        "burnin_num": 3000,
        "_comment_sample_stride": "Number of update steps between saved samples. Default is 1. Increase only when you intentionally thin saved samples.",
        "sample_stride": 1,
        "_comment_exchange_stride": "How often adjacent replicas try to swap states. 1 means every update step.",
        "exchange_stride": 1,
        "_comment_parallel_workers": "Replica-level parallel workers. 0 means auto from CPU count, 1 means serial.",
        "parallel_workers": 0,
        "_comment_likelihood_workers": "Within-replica likelihood workers over data rows. 1 means serial, 0 means auto.",
        "likelihood_workers": 1,
        "_comment_likelihood_parallel_min_rows": "Likelihood parallelism is disabled below this number of data rows to avoid thread overhead.",
        "likelihood_parallel_min_rows": 2048,
        "_comment_progress": "Show a progress bar during long runs.",
        "progress": True,
        "_comment_progress_interval_steps": "Progress refresh interval in update steps. 0 means auto.",
        "progress_interval_steps": 0,
        "_comment_progress_bar_width": "Width of the text progress bar.",
        "progress_bar_width": 32,
        "_comment_seed": "Random seed. Fix it for reproducibility; change it to check robustness.",
        "seed": 424242,
    },
    "model": {
        "models": [
            {
                "name": "linear_background",
                "basis_count": 1,
                "parameters": [
                    {
                        "name": "intercept",
                        "prior": {"type": "normal", "mean": 0.0, "sigma": 1.0},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 0.25,
                        "d": 0.5,
                    },
                    {
                        "name": "slope",
                        "prior": {"type": "normal", "mean": 0.0, "sigma": 0.5},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 0.15,
                        "d": 0.5,
                    },
                ],
            },
            {
                "name": "spectral_peaks",
                "basis_count": 2,
                "parameters": [
                    {
                        "name": "a",
                        "prior": {"type": "gamma", "shape": 3.0, "scale": 0.3},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 0.25,
                        "d": 0.5,
                    },
                    {
                        "name": "mu",
                        "prior": {"type": "normal", "mean": 0.4, "sigma": 1.0},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 0.25,
                        "d": 0.5,
                    },
                    {
                        "name": "b",
                        "prior": {"type": "gamma", "shape": 8.0, "scale": 2.0},
                        "_comment_C_d": PROPOSAL_TUNING_COMMENT,
                        "C": 2.0,
                        "d": 0.5,
                    },
                ],
            },
        ],
        "noise": {
            "_comment": "Gaussian observation noise. sigma2_min is the beta=1 minimum/base variance. If estimate_sigma2 is true, free-energy noise estimation compares sigma2_min / beta candidates.",
            "_comment_estimate_sigma2": "true estimates sigma2 by free-energy comparison; false keeps sigma2_min fixed as a known value.",
            "type": "gaussian",
            "sigma2_min": 0.0009,
            "estimate_sigma2": True,
        },
    },
    "build": {
        "compiler": "c++",
        "include_dirs": [],
        "library_dirs": [],
        "flags": ["-std=c++20", "-O2", "-pthread"],
        "libs": [],
        "output": "v2_main.out",
    },
}


V2_MAIN_CPP_TEMPLATE = """#include "bayes_emc/bayes_emc.hpp"
#include "generated_v2_config.hpp"
#include "target.hpp"

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::filesystem::path PathFromEnv(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\\0') return fallback;
    return value;
}

const bayes_emc::SampleRecord & MapSample(const std::vector<bayes_emc::SampleRecord> & samples) {
    if (samples.empty()) {
        throw std::runtime_error("No sample was generated.");
    }
    const bayes_emc::SampleRecord * best = &samples.front();
    for (const bayes_emc::SampleRecord & sample : samples) {
        if (sample.log_posterior > best->log_posterior) {
            best = &sample;
        }
    }
    return *best;
}

void WriteLog(
    const std::filesystem::path & path,
    const bayes_emc::EngineResult & result,
    const bayes_emc::SampleRecord & map,
    const std::size_t posterior_replica_id,
    const double posterior_sigma2,
    const char * noise_type,
    const std::size_t diagnostic_warning_count
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open V2 log output.");
    }
    out << "engine: bayes_emc_v2\\n";
    out << "parallel_worker_count: " << result.parallel_worker_count << "\\n";
    out << "likelihood_worker_count: " << result.likelihood_worker_count << "\\n";
    out << "sample_count: " << result.samples.size() << "\\n";
    out << "posterior_replica_id: " << posterior_replica_id << "\\n";
    out << "posterior_inverse_temperature: " << result.inverse_temperatures[posterior_replica_id] << "\\n";
    out << "noise_type: " << noise_type << "\\n";
    out << "posterior_sigma2: " << posterior_sigma2 << "\\n";
    out << "map_sample_id: " << map.sample_id << "\\n";
    out << "map_log_posterior: " << map.log_posterior << "\\n";
    out << "map_energy: " << map.energy << "\\n";
    out << "diagnostics_tsv: diagnostics.tsv\\n";
    out << "diagnostics_warnings_tsv: diagnostics_warnings.tsv\\n";
    out << "diagnostic_warning_count: " << diagnostic_warning_count << "\\n";
    out << "map_parameters:\\n";
    for (const bayes_emc::ParameterIndex & index : result.layout.Indices()) {
        const std::size_t offset = result.layout.Offset(index);
        out << "  " << result.layout.Label(index) << ": " << map.values[offset] << "\\n";
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    auto spec = bayes_emc_generated::MakeAnalysisSpec();
    auto options = bayes_emc_generated::MakeEngineOptions();
    const fs::path data_path = PathFromEnv("BAYES_EMC_DATA_PATH", bayes_emc_generated::DefaultDataPath());
    const fs::path result_dir = PathFromEnv("BAYES_EMC_RESULT_DIR", bayes_emc_generated::DefaultResultDir());

    fs::create_directories(result_dir / "figures");
    bayes_emc::DataSet data = bayes_emc::DataSet::LoadTable(
        data_path.string(),
        spec.input_dim,
        spec.output_dim,
        bayes_emc_generated::MakeDataOptions()
    );

    bayes_emc::ExchangeMonteCarlo engine(spec, data, bayes_emc_user::TargetFunction, options);
    bayes_emc::EngineResult result = engine.Run();
    std::size_t posterior_replica_id = result.inverse_temperatures.size() - 1;
    double posterior_sigma2 = spec.gaussian_sigma2;
    if (spec.likelihood_type == bayes_emc::LikelihoodType::Gaussian) {
        bayes_emc::NoiseEstimationResult noise = bayes_emc::EstimateGaussianNoiseByFreeEnergy(
            result,
            spec.gaussian_sigma2,
            data.Size(),
            spec.output_dim
        );
        if (!bayes_emc_generated::EstimateSigma2()) {
            noise = bayes_emc::SelectFixedGaussianNoise(noise, spec.gaussian_sigma2);
        }
        posterior_replica_id = noise.replica_id;
        posterior_sigma2 = noise.estimated_sigma2;
        bayes_emc::WriteNoiseEstimation(result_dir / "noise_estimation.txt", noise);
    } else {
        std::ofstream noise_out(result_dir / "noise_estimation.txt");
        if (!noise_out) {
            throw std::runtime_error("Could not open noise estimation output.");
        }
        noise_out << "# noise_type\\tpoisson\\n";
        noise_out << "# sigma2_mode\\tnot_applicable\\n";
        noise_out << "# posterior_replica_id\\t" << posterior_replica_id << "\\n";
    }
    bayes_emc::WriteSampleJson(result_dir / "sample.json", result, posterior_replica_id, posterior_sigma2);
    bayes_emc::WriteDiagnosticsTsv(result_dir / "diagnostics.tsv", result);
    const std::size_t diagnostic_warning_count =
        bayes_emc::WriteDiagnosticsWarningsTsv(result_dir / "diagnostics_warnings.tsv", result);
    const std::vector<bayes_emc::SampleRecord> & posterior_samples =
        bayes_emc::SamplesForReplica(result, posterior_replica_id);
    const bayes_emc::SampleRecord & map = MapSample(posterior_samples);
    WriteLog(
        result_dir / "log.txt",
        result,
        map,
        posterior_replica_id,
        posterior_sigma2,
        bayes_emc_generated::NoiseType(),
        diagnostic_warning_count
    );

    std::cout << "samples: " << result.samples.size() << "\\n";
    std::cout << "map_log_posterior: " << map.log_posterior << "\\n";
    std::cout << "noise_type: " << bayes_emc_generated::NoiseType() << "\\n";
    if (spec.likelihood_type == bayes_emc::LikelihoodType::Gaussian && bayes_emc_generated::EstimateSigma2()) {
        std::cout << "estimated_sigma2: " << posterior_sigma2 << "\\n";
    } else if (spec.likelihood_type == bayes_emc::LikelihoodType::Gaussian) {
        std::cout << "fixed_sigma2: " << posterior_sigma2 << "\\n";
    } else {
        std::cout << "poisson_noise: fixed by count likelihood\\n";
    }
    std::cout << "diagnostic_warnings: " << diagnostic_warning_count << "\\n";
    for (const bayes_emc::ParameterIndex & index : result.layout.Indices()) {
        const std::size_t offset = result.layout.Offset(index);
        std::cout << result.layout.Label(index) << ": " << map.values[offset] << "\\n";
    }
}
"""


LINEAR_TARGET_HPP_TEMPLATE = """#pragma once

#include "bayes_emc/bayes_emc.hpp"

#include <vector>

namespace bayes_emc_user {

inline void TargetFunction(
    const std::vector<double> & x,
    const bayes_emc::ParameterView & params,
    std::vector<double> & out
) {
    const double intercept = params.Value(0, 0, 0);
    const double slope = params.Value(0, 0, 1);
    out[0] = intercept + slope * x[0];
}

} // namespace bayes_emc_user
"""


SPECTRAL_TARGET_HPP_TEMPLATE = """#pragma once

#include "bayes_emc/bayes_emc.hpp"

namespace bayes_emc_user {

// Optimized standard target for a one-dimensional sum of Gaussian peaks.
// It still behaves like a normal TargetFunction, but V2 Core can also use its
// incremental residual cache to update the Gaussian likelihood by recomputing
// only the changed peak when one parameter is proposed.
inline const bayes_emc::GaussianPeakSumTarget TargetFunction =
    bayes_emc::GaussianPeakSumTarget::SpectralOnly(0);

} // namespace bayes_emc_user
"""


BACKGROUND_SPECTRAL_TARGET_HPP_TEMPLATE = """#pragma once

#include "bayes_emc/bayes_emc.hpp"

namespace bayes_emc_user {

// Optimized standard target for a linear background plus Gaussian peaks.
// Model 0 is the linear background and model 1 is the repeated spectral peak
// basis.  V2 Core uses the incremental residual cache when available.
inline const bayes_emc::GaussianPeakSumTarget TargetFunction =
    bayes_emc::GaussianPeakSumTarget::WithLinearBackground(0, 1);

} // namespace bayes_emc_user
"""


LINEAR_README_TEMPLATE = """# 1次元線形モデル サンプル

このディレクトリは `bayes-emc init linear` で生成される、V2 Core用の最小サンプルです。

```bash
bayes-emc check config.json
bayes-emc tune config.json
bayes-emc run config.tuned.json
bayes-emc plot result/sample.json
```

`config.json` はモデル構造、事前分布、EMC設定、データ形状を管理します。
データは `data/data.csv` のヘッダ付きCSVで、`input_columns` と `output_columns` で列名を指定します。
`src/target.hpp` だけがモデル固有の数式です。
`tune` は短いEMC試行で `C`, `gamma`, `d` を調整し、`config.tuned.json` を保存します。
本推論では `bayes-emc run config.tuned.json` を使えます。

人工データの真値は `intercept=1.25`, `slope=-0.80` です。
推論後は `result/sample.json`, `result/log.txt`, `result/noise_estimation.txt`,
`result/diagnostics.tsv`, `result/diagnostics_warnings.tsv` を確認してください。
`result/sample.json` には、推定または固定されたノイズ分散に対応する温度層のサンプルが保存されます。
`posterior_replica_id`, `posterior_inverse_temperature`, `posterior_sigma2` がその対応を示します。
`result/log.txt` はMAP解と実行要約です。各レプリカのMH採択率と隣接レプリカ間の交換率は
`result/diagnostics.tsv` に保存されます。採択率はパラメータごとの列です。
10%未満または99%超の採択率/交換率は `result/diagnostics_warnings.tsv` に警告として保存されます。
`bayes-emc tune` では、最初の有限温度層の採択率を `C` の90%下限と99%上限、
隣接交換率を `gamma` と `replica_num`、最低温度層の採択率30%を `d` の調整目安として使います。
`Inv Temp` の先頭に `*` が付く行は `data_size * beta < 1` の温度層です。
`beta=0` の最高温度レプリカは事前分布から直接サンプルされ、採択率は100%として出力されます。
`model.noise.estimate_sigma2` が `true` ならノイズ分散を自由エネルギーで推定し、
`false` なら `model.noise.sigma2_min` を既知値として固定します。
このサンプルでは推定の動きが見えやすいように、`sigma2_min` を人工データのノイズスケールより
小さめに置いています。
`emc.parallel_workers` は `0` でCPU数に合わせた自動並列、`1` で逐次実行です。
大きなデータでは `emc.likelihood_workers` で尤度計算もデータ点方向に並列化できます。
`emc.progress` が `true` の場合、実行中は標準エラーに進捗バーが表示されます。
"""


SPECTRAL_README_TEMPLATE = """# スペクトル分解サンプル

このディレクトリは `bayes-emc init spectral` で生成される、V2 Core用のスペクトル分解サンプルです。

```bash
bayes-emc check config.json
bayes-emc tune config.json
bayes-emc run config.tuned.json
bayes-emc plot result/sample.json
bayes-emc select-peaks config.json --min 1 --max 5
```

`config.json` はピーク数、事前分布、EMC設定、データ形状を管理します。
データは `data/data.csv` のヘッダ付きCSVで、`input_columns` と `output_columns` で列名を指定します。
`src/target.hpp` だけがスペクトルモデル固有の数式です。
`tune` は短いEMC試行で `C`, `gamma`, `d` を調整し、`config.tuned.json` を保存します。
本推論では `bayes-emc run config.tuned.json` を使えます。

V2では `src/prior.hpp` を書きません。
事前分布は `config.json` から `generated_v2_config.hpp` へ変換され、
サンプリングと確率密度計算の両方で共有されます。
`result/noise_estimation.txt` には、ベイズ自由エネルギーで評価したノイズ強度候補が保存されます。
`result/sample.json` には、推定または固定されたノイズ分散に対応する温度層のサンプルが保存されます。
`result/log.txt` はMAP解と実行要約です。各レプリカのMH採択率と隣接レプリカ間の交換率は
`result/diagnostics.tsv` に保存されます。採択率はパラメータごとの列です。
10%未満または99%超の採択率/交換率は `result/diagnostics_warnings.tsv` に警告として保存されます。
`bayes-emc tune` では、最初の有限温度層の採択率を `C` の90%下限と99%上限、
隣接交換率を `gamma` と `replica_num`、最低温度層の採択率30%を `d` の調整目安として使います。
`Inv Temp` の先頭に `*` が付く行は `data_size * beta < 1` の温度層です。
`beta=0` の最高温度レプリカは事前分布から直接サンプルされ、採択率は100%として出力されます。
`model.noise.estimate_sigma2` が `false` の場合は `model.noise.sigma2_min` を固定して評価します。
`emc.parallel_workers` は `0` でCPU数に合わせた自動並列、`1` で逐次実行です。
大きなデータでは `emc.likelihood_workers` で尤度計算もデータ点方向に並列化できます。
ピーク数を決めたい場合は `select-peaks` が候補ごとの自由エネルギーを
`result/model_selection/peak_count/` に保存します。
`emc.progress` が `true` の場合、実行中は標準エラーに進捗バーが表示されます。
"""


BACKGROUND_SPECTRAL_README_TEMPLATE = """# 線形背景 + スペクトルピークサンプル

このディレクトリは `bayes-emc init background-spectral` で生成される、V2 Core用の複数モデルサンプルです。

```bash
bayes-emc check config.json
bayes-emc tune config.json
bayes-emc run config.tuned.json
bayes-emc plot result/sample.json
```

`config.json` には `linear_background` と `spectral_peaks` の2モデルがあります。
データは `data/data.csv` のヘッダ付きCSVで、`input_columns` と `output_columns` で列名を指定します。
線形背景は1基底、ピークモデルは2基底です。
`src/target.hpp` は、線形背景とガウスピーク群の寄与を足し合わせます。
`tune` は短いEMC試行で `C`, `gamma`, `d` を調整し、`config.tuned.json` を保存します。
本推論では `bayes-emc run config.tuned.json` を使えます。

人工データの線形背景の真値は `intercept=0.25`, `slope=0.12` です。
ピークの順序はサンプリング中に入れ替わることがあります。
`result/noise_estimation.txt` には、ベイズ自由エネルギーで評価したノイズ強度候補が保存されます。
`result/sample.json` には、推定または固定されたノイズ分散に対応する温度層のサンプルが保存されます。
`result/log.txt` はMAP解と実行要約です。各レプリカのMH採択率と隣接レプリカ間の交換率は
`result/diagnostics.tsv` に保存されます。採択率はパラメータごとの列です。
10%未満または99%超の採択率/交換率は `result/diagnostics_warnings.tsv` に警告として保存されます。
`bayes-emc tune` では、最初の有限温度層の採択率を `C` の90%下限と99%上限、
隣接交換率を `gamma` と `replica_num`、最低温度層の採択率30%を `d` の調整目安として使います。
`Inv Temp` の先頭に `*` が付く行は `data_size * beta < 1` の温度層です。
`beta=0` の最高温度レプリカは事前分布から直接サンプルされ、採択率は100%として出力されます。
`model.noise.estimate_sigma2` が `false` の場合は `model.noise.sigma2_min` を固定して評価します。
`emc.parallel_workers` は `0` でCPU数に合わせた自動並列、`1` で逐次実行です。
大きなデータでは `emc.likelihood_workers` で尤度計算もデータ点方向に並列化できます。
`emc.progress` が `true` の場合、実行中は標準エラーに進捗バーが表示されます。
"""
