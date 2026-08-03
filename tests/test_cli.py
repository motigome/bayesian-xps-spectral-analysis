import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from bayes_emc import cli


def write_json(path, data):
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def minimal_config(data_path="data/data.txt"):
    return {
        "project": {
            "name": "test_project",
            "model": "spectral",
            "result_dir": "result",
        },
        "data": {
            "path": data_path,
            "input_dim": 1,
            "output_dim": 1,
        },
        "emc": {
            "replica_num": 4,
            "gamma": 1.2,
            "sample_num": 5,
            "burnin_num": 3,
        },
        "model": {
            "models": [
                {
                    "name": "test_basis_model",
                    "basis_count": 2,
                    "parameters": [
                        {"name": "a", "prior": {"type": "gamma", "shape": 2.0, "scale": 0.5}, "C": 0.5, "d": 0.5},
                        {"name": "mu", "prior": {"type": "normal", "mean": 1.5, "sigma": 0.5}, "C": 0.5, "d": 0.7},
                        {"name": "b", "prior": {"type": "gamma", "shape": 14.0, "scale": 10.0}, "C": 40.0, "d": 0.6},
                    ],
                },
            ],
            "noise": {
                "type": "gaussian",
                "sigma2_min": 0.01,
                "estimate_sigma2": True,
            },
        },
    }


class CliConfigTests(unittest.TestCase):
    def test_cpp_include_root_contains_bundled_core(self):
        self.assertTrue((cli._v2_include_root() / "bayes_emc" / "bayes_emc.hpp").is_file())

    def test_v2_generated_header_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_path = project / "config.json"
            write_json(config_path, minimal_config())

            config = cli.load_config(config_path)
            cli.validate_config(config)
            header_path = cli._write_v2_generated_config(config)
            header_text = header_path.read_text(encoding="utf-8")
            self.assertIn("MakeAnalysisSpec", header_text)
            self.assertIn("test_basis_model", header_text)
            self.assertIn("PriorDistribution::Gamma", header_text)
            self.assertIn("options.parallel_worker_count = 0;", header_text)
            self.assertIn("options.likelihood_worker_count = 1;", header_text)
            self.assertIn("options.progress_enabled = true;", header_text)
            self.assertIn("options.progress_interval_steps = 0;", header_text)
            self.assertIn("return true;", header_text)
            self.assertEqual(config.model_layout[0]["name"], "test_basis_model")
            self.assertEqual(config.model_layout[0]["basis_count"], 2)
            self.assertTrue(config.estimate_sigma2)
            self.assertAlmostEqual(config.sigma2_min, 0.01)
            self.assertAlmostEqual(config.sigma2_candidate_max, 0.01 * (1.2 ** 2))

    def test_legacy_sigma2_alias_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["noise"]["sigma2"] = config_data["model"]["noise"].pop("sigma2_min")
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            self.assertAlmostEqual(config.sigma2_min, 0.01)

    def test_poisson_noise_is_written_to_generated_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["noise"] = {
                "type": "poisson",
                "sigma2_min": 1.0,
                "estimate_sigma2": False,
            }
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            warnings = cli.validate_config(config)
            header_path = cli._write_v2_generated_config(config)
            header_text = header_path.read_text(encoding="utf-8")
            self.assertEqual(warnings, [])
            self.assertEqual(config.noise_type, "poisson")
            self.assertIn("spec.likelihood_type = LikelihoodType::Poisson;", header_text)
            self.assertIn('return "poisson";', header_text)

    def test_proposal_c_can_be_inferred_from_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["models"][0]["parameters"] = [
                {"name": "u", "prior": {"type": "uniform", "lower": -3.0, "upper": 6.0}},
                {"name": "n", "prior": {"type": "normal", "mean": 0.0, "sigma": 6.0}},
                {"name": "g", "prior": {"type": "gamma", "shape": 9.0, "scale": 2.0}, "d": 0.25},
            ]
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            self.assertEqual(config.tuning[0], [3.0, 0.5])
            self.assertEqual(config.tuning[1], [2.0, 0.5])
            self.assertEqual(config.tuning[2], [2.0, 0.25])

    def test_proposal_c_can_be_inferred_from_beta_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["models"][0]["parameters"] = [
                {
                    "name": "u",
                    "prior": {"type": "beta", "alpha": 2.0, "beta": 2.0, "lower": 0.0, "upper": 6.0},
                }
            ]
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            expected_c = 6.0 / 3.0
            self.assertAlmostEqual(config.tuning[0][0], expected_c)
            self.assertEqual(config.tuning[0][1], 0.5)

    def test_beta_prior_is_written_to_generated_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["models"][0]["parameters"][0]["prior"] = {
                "type": "beta",
                "alpha": 2.0,
                "beta": 2.0,
                "lower": 0.0,
                "upper": 0.6,
            }
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            header_path = cli._write_v2_generated_config(config)
            header_text = header_path.read_text(encoding="utf-8")
            self.assertIn("PriorDistribution::Beta(2.0, 2.0, 0.0, 0.6)", header_text)

    def test_proposal_decay_can_be_zero_or_greater_than_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["models"][0]["parameters"][0]["d"] = 0.0
            config_data["model"]["models"][0]["parameters"][1]["d"] = 1.5
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            self.assertEqual(config.tuning[0][1], 0.0)
            self.assertEqual(config.tuning[1][1], 1.5)

    def test_replica_step_scales_are_written_to_generated_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["models"][0]["parameters"][0]["replica_step_scales"] = [1.0, 0.5, 0.25, 1.0]
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            header_path = cli._write_v2_generated_config(config)
            header_text = header_path.read_text(encoding="utf-8")
            self.assertIn("{1.0, 0.5, 0.25, 1.0}", header_text)

    def test_replica_step_scales_must_match_replica_num(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["models"][0]["parameters"][0]["replica_step_scales"] = [1.0, 0.5]
            config_path = project / "config.json"
            write_json(config_path, config_data)

            with self.assertRaises(cli.ConfigError):
                cli.load_config(config_path)

    def test_local_step_tuning_scales_only_low_acceptance_replicas(self):
        parameter = {"name": "a", "C": 1.0, "d": 0.5}
        ref = cli.TuningParameterRef("test_basis_model", "default", "a", parameter)
        summary = cli.DiagnosticsSummary(
            parameter_rates={},
            parameter_replica_rates={
                ("test_basis_model", "default", "a"): [
                    (0, 0.0, True, 1.0),
                    (1, 0.05, True, 0.08),
                    (2, 0.5, False, 0.12),
                    (3, 1.0, False, 0.04),
                ]
            },
            exchange_rates=[],
        )

        adjustments = cli._adjust_replica_step_scales_from_low_rates([ref], summary, 4, 0.10, 0.5)

        self.assertEqual(adjustments, 2)
        self.assertEqual(parameter["replica_step_scales"], [1.0, 0.5, 1.0, 0.5])

    def test_plot_bin_count_uses_sample_count_for_auto(self):
        self.assertEqual(cli._plot_bin_count("auto", 4), 12)
        self.assertEqual(cli._plot_bin_count("auto", 625), 45)
        self.assertEqual(cli._plot_bin_count("auto", 10000), 64)
        self.assertEqual(cli._plot_bin_count("18", 100), 18)

    def test_plot_parser_disables_smoothing_by_default(self):
        parser = cli.make_parser()
        args = parser.parse_args(["plot", "result/sample.json"])

        self.assertEqual(args.smooth, 0.0)
        self.assertEqual(args.density_power, 1.6)
        self.assertIsNone(args.sort_peaks_by)

    def test_plot_can_sort_repeated_basis_samples_by_mu(self):
        sample_json = {
            "parameters": [
                {"offset": 0, "model_id": 0, "basis_id": 0, "layer_id": 0, "parameter_id": 0, "label": "spectral_peaks[0].default.a"},
                {"offset": 1, "model_id": 0, "basis_id": 0, "layer_id": 0, "parameter_id": 1, "label": "spectral_peaks[0].default.mu"},
                {"offset": 2, "model_id": 0, "basis_id": 0, "layer_id": 0, "parameter_id": 2, "label": "spectral_peaks[0].default.b"},
                {"offset": 3, "model_id": 0, "basis_id": 1, "layer_id": 0, "parameter_id": 0, "label": "spectral_peaks[1].default.a"},
                {"offset": 4, "model_id": 0, "basis_id": 1, "layer_id": 0, "parameter_id": 1, "label": "spectral_peaks[1].default.mu"},
                {"offset": 5, "model_id": 0, "basis_id": 1, "layer_id": 0, "parameter_id": 2, "label": "spectral_peaks[1].default.b"},
            ],
        }
        samples = [
            [10.0, 2.0, 100.0, 20.0, 1.0, 200.0],
            [11.0, 0.5, 101.0, 21.0, 1.5, 201.0],
        ]

        sorted_samples, sorted_labels = cli._sort_repeated_basis_samples_by_parameter(
            sample_json,
            samples,
            cli._extract_parameter_labels(sample_json),
            "mu",
        )

        self.assertEqual(
            sorted_labels,
            [
                "peak_position[0].a",
                "peak_position[0].mu",
                "peak_position[0].b",
                "peak_position[1].a",
                "peak_position[1].mu",
                "peak_position[1].b",
            ],
        )
        self.assertEqual(
            sorted_samples,
            [
                [20.0, 1.0, 200.0, 10.0, 2.0, 100.0],
                [11.0, 0.5, 101.0, 21.0, 1.5, 201.0],
            ],
        )

    def test_corner_plot_uses_smoothing_and_hides_datapoints_by_default(self):
        class Figure:
            def __init__(self):
                self.saved = False

            def savefig(self, *_args, **_kwargs):
                self.saved = True

        class CornerModule:
            def __init__(self):
                self.calls = []
                self.figure = Figure()

            def corner(self, _samples, **kwargs):
                self.calls.append(kwargs)
                return self.figure

        corner_module = CornerModule()
        figure = cli._write_corner_png(
            [[0.0, 1.0], [1.0, 2.0]],
            ["a", "b"],
            Path("corner.png"),
            corner_module=corner_module,
            bins=12,
            smooth=1.0,
            density_power=1.0,
            dpi=220,
            plot_datapoints=False,
            plot_contours=False,
        )

        self.assertIs(figure, corner_module.figure)
        self.assertTrue(figure.saved)
        self.assertEqual(corner_module.calls[0]["bins"], 12)
        self.assertEqual(corner_module.calls[0]["smooth"], 1.0)
        self.assertEqual(corner_module.calls[0]["smooth1d"], 1.0)
        self.assertFalse(corner_module.calls[0]["plot_datapoints"])
        self.assertFalse(corner_module.calls[0]["fill_contours"])
        self.assertTrue(corner_module.calls[0]["plot_density"])
        self.assertFalse(corner_module.calls[0]["plot_contours"])
        self.assertNotIn("histtype", corner_module.calls[0]["hist_kwargs"])
        self.assertNotIn("contour_kwargs", corner_module.calls[0])
        self.assertNotIn("cmap", corner_module.calls[0]["pcolor_kwargs"])
        self.assertEqual(corner_module.calls[0]["color"], "#000000")
        self.assertEqual(corner_module.calls[0]["pcolor_kwargs"]["alpha"], 1.0)
        self.assertEqual(corner_module.calls[0]["hist_kwargs"]["edgecolor"], "#000000")
        self.assertGreaterEqual(corner_module.calls[0]["hist_kwargs"]["alpha"], 0.95)

    def test_corner_plot_can_enable_contours_explicitly(self):
        class Figure:
            def savefig(self, *_args, **_kwargs):
                return None

        class CornerModule:
            def __init__(self):
                self.calls = []

            def corner(self, _samples, **kwargs):
                self.calls.append(kwargs)
                return Figure()

        corner_module = CornerModule()
        cli._write_corner_png(
            [[0.0, 1.0], [1.0, 2.0]],
            ["a", "b"],
            Path("corner.png"),
            corner_module=corner_module,
            bins=12,
            smooth=1.0,
            density_power=1.0,
            dpi=220,
            plot_datapoints=False,
            plot_contours=True,
        )

        self.assertTrue(corner_module.calls[0]["plot_contours"])
        self.assertIn("contour_kwargs", corner_module.calls[0])

    def test_corner_plot_overlays_map_cross_on_lower_triangle(self):
        class Axis:
            def __init__(self):
                self.vertical_lines = []
                self.horizontal_lines = []

            def axvline(self, x, **kwargs):
                self.vertical_lines.append((x, kwargs))

            def axhline(self, y, **kwargs):
                self.horizontal_lines.append((y, kwargs))

        class Figure:
            def __init__(self):
                self.axes = [
                    Axis(),
                    Axis(),
                    Axis(),
                    Axis(),
                ]

            def savefig(self, *_args, **_kwargs):
                return None

        class CornerModule:
            def __init__(self):
                self.calls = []
                self.figure = Figure()

            def corner(self, _samples, **kwargs):
                self.calls.append(kwargs)
                return self.figure

        corner_module = CornerModule()
        cli._write_corner_png(
            [[0.0, 1.0], [1.0, 2.0]],
            ["a", "b"],
            Path("corner.png"),
            corner_module=corner_module,
            bins=12,
            smooth=0.0,
            density_power=1.0,
            dpi=220,
            plot_datapoints=False,
            plot_contours=False,
            map_values=[0.25, 1.75],
        )

        self.assertEqual(len(corner_module.figure.axes[0].vertical_lines), 1)
        self.assertEqual(len(corner_module.figure.axes[0].horizontal_lines), 0)
        self.assertEqual(len(corner_module.figure.axes[1].vertical_lines), 0)
        self.assertEqual(len(corner_module.figure.axes[1].horizontal_lines), 0)
        self.assertEqual(len(corner_module.figure.axes[2].vertical_lines), 1)
        self.assertEqual(len(corner_module.figure.axes[2].horizontal_lines), 1)
        self.assertEqual(len(corner_module.figure.axes[3].vertical_lines), 1)
        self.assertEqual(len(corner_module.figure.axes[3].horizontal_lines), 0)
        diagonal_x, diagonal_kwargs = corner_module.figure.axes[0].vertical_lines[0]
        self.assertEqual(diagonal_x, 0.25)
        self.assertEqual(diagonal_kwargs["color"], "#dc2626")
        self.assertAlmostEqual(diagonal_kwargs["linewidth"], 1.4)
        offdiag_x, _offdiag_v_kwargs = corner_module.figure.axes[2].vertical_lines[0]
        offdiag_y, _offdiag_h_kwargs = corner_module.figure.axes[2].horizontal_lines[0]
        self.assertEqual(offdiag_x, 0.25)
        self.assertEqual(offdiag_y, 1.75)

    def test_corner_density_norm_is_independent_per_panel(self):
        class Figure:
            def savefig(self, *_args, **_kwargs):
                return None

        class CoreModule:
            def __init__(self):
                self.norms = []

            def hist2d(self, *_args, **kwargs):
                self.norms.append(kwargs["pcolor_kwargs"].get("norm"))

        class CornerModule:
            def __init__(self):
                self.core = CoreModule()
                self.calls = []

            def corner(self, _samples, **kwargs):
                self.calls.append(kwargs)
                self.core.hist2d(pcolor_kwargs=kwargs["pcolor_kwargs"])
                self.core.hist2d(pcolor_kwargs=kwargs["pcolor_kwargs"])
                return Figure()

        original_factory = cli._density_power_norm_factory
        try:
            cli._density_power_norm_factory = lambda _power: lambda: object()
            corner_module = CornerModule()
            cli._write_corner_png(
                [[0.0, 1.0], [1.0, 2.0]],
                ["a", "b"],
                Path("corner.png"),
                corner_module=corner_module,
                bins=12,
                smooth=0.0,
                density_power=1.6,
                dpi=220,
                plot_datapoints=False,
                plot_contours=False,
            )
        finally:
            cli._density_power_norm_factory = original_factory

        self.assertNotIn("norm", corner_module.calls[0]["pcolor_kwargs"])
        self.assertEqual(len(corner_module.core.norms), 2)
        self.assertIsNot(corner_module.core.norms[0], corner_module.core.norms[1])

    def test_corner_plot_retries_without_smoothing_when_scipy_is_unavailable(self):
        class Figure:
            def savefig(self, *_args, **_kwargs):
                return None

        class CornerModule:
            def __init__(self):
                self.calls = []

            def corner(self, _samples, **kwargs):
                self.calls.append(kwargs)
                if "smooth" in kwargs:
                    raise ImportError("no scipy")
                return Figure()

        corner_module = CornerModule()
        cli._write_corner_png(
            [[0.0], [1.0]],
            ["a"],
            Path("corner.png"),
            corner_module=corner_module,
            bins=12,
            smooth=1.0,
            density_power=1.0,
            dpi=220,
            plot_datapoints=False,
            plot_contours=False,
        )

        self.assertEqual(len(corner_module.calls), 2)
        self.assertIn("smooth", corner_module.calls[0])
        self.assertNotIn("smooth", corner_module.calls[1])

    def test_sigma2_candidate_max_is_derived_from_temperature_ladder(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            self.assertTrue(config.estimate_sigma2)
            self.assertAlmostEqual(config.sigma2_candidate_max, 0.01 * (1.2 ** 2))

    def test_sigma2_max_is_rejected_as_config_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["noise"]["sigma2_max"] = 999.0
            config_path = project / "config.json"
            write_json(config_path, config_data)

            with self.assertRaisesRegex(cli.ConfigError, "sigma2_max is derived"):
                cli.load_config(config_path)

    def test_sigma2_can_be_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["noise"]["estimate_sigma2"] = False
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            self.assertFalse(config.estimate_sigma2)
            header_path = cli._write_v2_generated_config(config)
            self.assertIn("return false;", header_path.read_text(encoding="utf-8"))

    def test_component_count_alias_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"]["models"][0]["component_count"] = config_data["model"]["models"][0].pop("basis_count")
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            self.assertEqual(config.base_nums, [2])
            self.assertEqual(config.model_layout[0]["basis_count"], 2)

    def test_csv_data_with_named_columns_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.csv").write_text("y,x\n1.0,0.0\n2.0,1.0\n", encoding="utf-8")
            config_data = minimal_config("data/data.csv")
            config_data["data"].update({
                "format": "csv",
                "header": True,
                "input_columns": ["x"],
                "output_columns": ["y"],
            })
            config_path = project / "config.json"
            write_json(config_path, config_data)

            config = cli.load_config(config_path)
            self.assertEqual(config.data_format, "csv")
            self.assertEqual(cli.validate_config(config), [])
            header_path = cli._write_v2_generated_config(config)
            header_text = header_path.read_text(encoding="utf-8")
            self.assertIn("DataFormat::Csv", header_text)
            self.assertIn('options.input_columns = {"x"};', header_text)
            self.assertIn('options.output_columns = {"y"};', header_text)

    def test_internal_mhbp_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
            config_data = minimal_config()
            config_data["model"] = {
                "model_type_num": 1,
                "base_num": 2,
                "hierarchy_num": 1,
                "parameter_num": 3,
                "noise": {
                    "type": "gaussian",
                    "sigma2_min": 0.01,
                    "estimate_sigma2": True,
                },
            }
            config_data["tuning"] = {
                "parameters": [
                    {"name": "a", "C": 0.5, "d": 0.5},
                    {"name": "mu", "C": 0.5, "d": 0.7},
                    {"name": "b", "C": 40.0, "d": 0.6},
                ],
            }
            config_path = project / "config.json"
            write_json(config_path, config_data)

            with self.assertRaises(cli.ConfigError):
                cli.load_config(config_path)

    def test_data_shape_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "data").mkdir()
            (project / "data" / "data.txt").write_text("0.0 1.0 2.0\n", encoding="utf-8")
            config_path = project / "config.json"
            write_json(config_path, minimal_config())

            config = cli.load_config(config_path)
            with self.assertRaises(cli.ConfigError):
                cli.validate_config(config)

    def test_sample_json_extractors(self):
        sample_json = {
            "schema_version": 2,
            "samples": [
                {
                    "sample_id": 0,
                    "posterior": "0.25",
                    "parameter_groups": [
                        {
                            "model_id": 0,
                            "base_id": 0,
                            "hierarchy_id": 0,
                            "values": [1.0, 2.0, 3.0],
                        }
                    ],
                }
            ],
        }

        self.assertEqual(cli._extract_samples(sample_json), [[1.0, 2.0, 3.0]])
        self.assertEqual(cli._extract_posterior(sample_json), [0.25])
        self.assertEqual(
            cli._extract_sample_parameters(sample_json, 0),
            [
                {
                    "model_id": 0,
                    "base_id": 0,
                    "hierarchy_id": 0,
                    "values": [1.0, 2.0, 3.0],
                }
            ],
        )

    def test_v2_sample_json_extractors(self):
        sample_json = {
            "schema_version": 4,
            "parameters": [
                {"offset": 0, "label": "linear[0].default.a"},
                {"offset": 1, "label": "linear[0].default.b"},
            ],
            "samples": {
                "format": "columnar_v1",
                "energy": [0.5],
                "log_posterior": [-1.25],
                "values": [[1.0, -0.5]],
            },
        }

        self.assertEqual(cli._extract_samples(sample_json), [[1.0, -0.5]])
        self.assertEqual(cli._extract_posterior(sample_json), [-1.25])
        self.assertEqual(
            cli._extract_parameter_labels(sample_json),
            ["linear[0].default.a", "linear[0].default.b"],
        )
        self.assertEqual(
            cli._extract_sample_parameters(sample_json, 0),
            [
                {"label": "linear[0].default.a", "value": 1.0},
                {"label": "linear[0].default.b", "value": -0.5},
            ],
        )

    def test_noise_estimation_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "noise_estimation.txt"
            path.write_text(
                "# sigma2_mode\testimated\n"
                "# estimated_sigma2\t0.01\n"
                "# replica_id\t3\n"
                "# min_free_energy\t-12.5\n"
                "sigma2\tinverse_temperature\tfree_energy\n"
                "0.02\t0.5\t-8.0\n"
                "0.01\t1.0\t-12.5\n",
                encoding="utf-8",
            )
            parsed = cli._read_noise_estimation(path)
            self.assertEqual(parsed["sigma2_mode"], "estimated")
            self.assertEqual(parsed["estimated_sigma2"], 0.01)
            self.assertEqual(parsed["replica_id"], 3)
            self.assertEqual(parsed["min_free_energy"], -12.5)
            self.assertEqual(cli._noise_record_at_beta_one(parsed["records"])["free_energy"], -12.5)


class InitSpectralTests(unittest.TestCase):
    def test_init_spectral_generates_v2_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "spectral"
            self.assertEqual(cli.main(["init", "spectral", str(target)]), 0)
            self.assertTrue((target / "src" / "main.cpp").exists())
            self.assertTrue((target / "src" / "target.hpp").exists())
            self.assertFalse((target / "src" / "prior.hpp").exists())
            self.assertFalse((target / "core").exists())
            self.assertTrue((target / "data" / "data.csv").exists())
            generated_config = json.loads((target / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(generated_config["project"]["model"], "spectral")
            self.assertEqual(generated_config["data"]["format"], "csv")
            self.assertEqual(generated_config["data"]["input_columns"], ["x"])
            self.assertIn("_comment_replica_num", generated_config["emc"])
            self.assertEqual(generated_config["emc"]["parallel_workers"], 0)
            self.assertEqual(generated_config["emc"]["likelihood_workers"], 1)
            self.assertTrue(generated_config["emc"]["progress"])
            self.assertEqual(generated_config["build"]["libs"], [])
            self.assertEqual(cli.main(["check", str(target / "config.json"), "--sources"]), 0)

    def test_init_force_replaces_v2_template_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "spectral"
            self.assertEqual(cli.main(["init", "spectral", str(target)]), 0)
            (target / "src" / "target.hpp").write_text("// stale\n", encoding="utf-8")

            self.assertEqual(cli.main(["init", "spectral", str(target), "--force"]), 0)
            self.assertIn("TargetFunction", (target / "src" / "target.hpp").read_text(encoding="utf-8"))
            self.assertFalse((target / "core").exists())

    @unittest.skipUnless(
        shutil.which("c++"),
        "requires c++",
    )
    def test_init_spectral_cpp_syntax(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "spectral"
            self.assertEqual(cli.main(["init", "spectral", str(target)]), 0)
            self.assertEqual(cli.main(["run", str(target / "config.json"), "--skip-build", "--skip-exec"]), 0)
            subprocess.run(
                [
                    "c++",
                    "-fsyntax-only",
                    "main.cpp",
                    "-I" + str(Path(__file__).resolve().parents[1] / "cpp" / "include"),
                    "-std=c++20",
                    "-pthread",
                ],
                cwd=target / "src",
                check=True,
            )

    def test_init_linear_generates_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "linear"
            self.assertEqual(cli.main(["init", "linear", str(target)]), 0)
            self.assertTrue((target / "src" / "main.cpp").exists())
            self.assertTrue((target / "src" / "target.hpp").exists())
            self.assertTrue((target / "data" / "data.csv").exists())
            generated_config = json.loads((target / "config.json").read_text(encoding="utf-8"))
            parameters = generated_config["model"]["models"][0]["parameters"]
            self.assertEqual(parameters[0]["prior"]["type"], "normal")
            self.assertIn("_comment_C_d", parameters[0])
            self.assertIn("_comment_sample_num", generated_config["emc"])
            self.assertIn("estimate_sigma2", generated_config["model"]["noise"])
            self.assertIn("sigma2_min", generated_config["model"]["noise"])
            self.assertNotIn("sigma2_max", generated_config["model"]["noise"])
            self.assertEqual(cli.main(["check", str(target / "config.json")]), 0)

    def test_init_spectral_generates_named_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "spectral"
            self.assertEqual(cli.main(["init", "spectral", str(target)]), 0)
            self.assertTrue((target / "src" / "main.cpp").exists())
            self.assertTrue((target / "src" / "target.hpp").exists())
            self.assertFalse((target / "src" / "prior.hpp").exists())
            self.assertTrue((target / "data" / "data.csv").exists())
            generated_config = json.loads((target / "config.json").read_text(encoding="utf-8"))
            model = generated_config["model"]["models"][0]
            self.assertEqual(model["basis_count"], 3)
            self.assertEqual(model["parameters"][0]["prior"]["type"], "gamma")
            self.assertEqual(model["parameters"][1]["prior"]["type"], "normal")
            self.assertEqual(cli.main(["check", str(target / "config.json")]), 0)

    def test_init_background_spectral_generates_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "background_spectral"
            self.assertEqual(cli.main(["init", "background-spectral", str(target)]), 0)
            self.assertTrue((target / "src" / "main.cpp").exists())
            self.assertTrue((target / "src" / "target.hpp").exists())
            self.assertFalse((target / "src" / "prior.hpp").exists())
            generated_config = json.loads((target / "config.json").read_text(encoding="utf-8"))
            models = generated_config["model"]["models"]
            self.assertEqual([model["name"] for model in models], ["linear_background", "spectral_peaks"])
            self.assertEqual(models[0]["basis_count"], 1)
            self.assertEqual(models[1]["basis_count"], 2)
            self.assertEqual(cli.main(["check", str(target / "config.json")]), 0)


class V2CoreTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_v2_linear_example_builds_and_runs(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "cpp" / "examples" / "linear_1d" / "main.cpp"
        include = root / "cpp" / "include"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            executable = work / "linear_1d"
            subprocess.run(
                ["c++", str(source), f"-I{include}", "-std=c++20", "-O2", "-pthread", "-o", str(executable)],
                check=True,
            )
            subprocess.run([str(executable)], cwd=work, check=True)
            sample = json.loads((work / "linear_1d_sample.json").read_text(encoding="utf-8"))
            self.assertEqual(sample["schema_version"], 4)
            self.assertEqual(len(sample["parameters"]), 2)
            self.assertEqual(sample["sample_count"], 10)
            self.assertEqual(sample["samples"]["format"], "columnar_v1")
            self.assertEqual(len(cli._extract_samples(sample)), 10)
            self.assertEqual(len(cli._extract_posterior(sample)), 10)

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_v2_spectral_basis_example_builds_and_runs(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "cpp" / "examples" / "spectral_basis" / "main.cpp"
        include = root / "cpp" / "include"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            executable = work / "spectral_basis"
            subprocess.run(
                ["c++", str(source), f"-I{include}", "-std=c++20", "-O2", "-pthread", "-o", str(executable)],
                check=True,
            )
            subprocess.run([str(executable)], cwd=work, check=True)
            sample = json.loads((work / "spectral_basis_sample.json").read_text(encoding="utf-8"))
            self.assertEqual(sample["schema_version"], 4)
            self.assertEqual(len(sample["parameters"]), 9)
            self.assertEqual(sample["sample_count"], 5)
            self.assertEqual(sample["samples"]["format"], "columnar_v1")
            self.assertEqual(len(cli._extract_samples(sample)), 5)

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_v2_linear_plus_spectral_example_builds_and_recovers_background(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "cpp" / "examples" / "linear_plus_spectral" / "main.cpp"
        include = root / "cpp" / "include"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            executable = work / "linear_plus_spectral"
            subprocess.run(
                ["c++", str(source), f"-I{include}", "-std=c++20", "-O2", "-pthread", "-o", str(executable)],
                check=True,
            )
            subprocess.run([str(executable)], cwd=work, check=True)
            sample = json.loads((work / "linear_plus_spectral_sample.json").read_text(encoding="utf-8"))
            posterior = cli._extract_posterior(sample)
            samples = cli._extract_samples(sample)
            map_index = max(range(len(posterior)), key=posterior.__getitem__)
            map_intercept, map_slope = samples[map_index][0], samples[map_index][1]

            self.assertEqual(sample["schema_version"], 4)
            self.assertEqual(len(sample["parameters"]), 8)
            self.assertEqual(sample["sample_count"], 600)
            self.assertEqual(sample["samples"]["format"], "columnar_v1")
            self.assertEqual(len(samples), 600)
            self.assertLess(abs(map_intercept - 0.25), 0.08)
            self.assertLess(abs(map_slope - 0.12), 0.08)
            labels = [parameter["label"] for parameter in sample["parameters"]]
            self.assertIn("linear_background[0].default.intercept", labels)
            self.assertIn("spectral_peaks[1].default.b", labels)

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_v2_synthetic_linear_map_is_close_to_truth(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "cpp" / "examples" / "linear_1d_synthetic" / "main.cpp"
        include = root / "cpp" / "include"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            executable = work / "linear_1d_synthetic"
            subprocess.run(
                ["c++", str(source), f"-I{include}", "-std=c++20", "-O2", "-pthread", "-o", str(executable)],
                check=True,
            )
            subprocess.run([str(executable)], cwd=work, check=True)
            sample = json.loads((work / "linear_1d_synthetic_sample.json").read_text(encoding="utf-8"))
            posterior = cli._extract_posterior(sample)
            samples = cli._extract_samples(sample)
            map_index = max(range(len(posterior)), key=posterior.__getitem__)
            map_intercept, map_slope = samples[map_index]

            self.assertEqual(len(samples), 800)
            self.assertLess(abs(map_intercept - 1.25), 0.08)
            self.assertLess(abs(map_slope - (-0.80)), 0.08)
            self.assertTrue((work / "linear_1d_synthetic_data.txt").exists())

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_v2_synthetic_linear_plot_outputs(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "cpp" / "examples" / "linear_1d_synthetic" / "main.cpp"
        include = root / "cpp" / "include"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            executable = work / "linear_1d_synthetic"
            subprocess.run(
                ["c++", str(source), f"-I{include}", "-std=c++20", "-O2", "-pthread", "-o", str(executable)],
                check=True,
            )
            subprocess.run([str(executable)], cwd=work, check=True)
            output_dir = work / "figures"
            self.assertEqual(
                cli.main([
                    "plot",
                    str(work / "linear_1d_synthetic_sample.json"),
                    "--output-dir",
                    str(output_dir),
                ]),
                0,
            )
            self.assertTrue((output_dir / "corner.png").exists() or (output_dir / "posterior.svg").exists())
            self.assertTrue((output_dir / "posterior_max.json").exists())

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_run_linear_project_map_is_close_to_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "linear"
            self.assertEqual(cli.main(["init", "linear", str(target)]), 0)
            self.assertEqual(cli.main(["run", str(target / "config.json")]), 0)

            generated_header = target / "src" / "generated_v2_config.hpp"
            sample_path = target / "result" / "sample.json"
            log_path = target / "result" / "log.txt"
            noise_path = target / "result" / "noise_estimation.txt"
            diagnostics_path = target / "result" / "diagnostics.tsv"
            diagnostics_warnings_path = target / "result" / "diagnostics_warnings.tsv"
            self.assertTrue(generated_header.exists())
            self.assertTrue(sample_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(noise_path.exists())
            self.assertTrue(diagnostics_path.exists())
            self.assertTrue(diagnostics_warnings_path.exists())

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("posterior_replica_id:", log_text)
            self.assertIn("posterior_inverse_temperature:", log_text)
            self.assertIn("posterior_sigma2:", log_text)
            self.assertIn("diagnostics_tsv: diagnostics.tsv", log_text)
            self.assertIn("diagnostics_warnings_tsv: diagnostics_warnings.tsv", log_text)
            self.assertIn("diagnostic_warning_count:", log_text)
            self.assertNotIn("mh_acceptance_rates:", log_text)
            self.assertNotIn("exchange_rates:", log_text)

            diagnostics_lines = diagnostics_path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(diagnostics_lines)
            self.assertEqual(
                diagnostics_lines[0].split("\t"),
                ["Inv Temp", "linear[0].default.intercept", "linear[0].default.slope", "Exchange %", "<Energy>"],
            )
            self.assertEqual(len(diagnostics_lines), 37)
            first_diagnostics = diagnostics_lines[1].split("\t")
            self.assertEqual(first_diagnostics[0], "*0.00e+00")
            self.assertEqual(first_diagnostics[1:3], ["100.00", "100.00"])
            self.assertNotEqual(first_diagnostics[3], "*****")
            self.assertGreater(float(first_diagnostics[3]), 0.0)
            self.assertIn("*****", diagnostics_lines[-1].split("\t"))
            diagnostics_warning_lines = diagnostics_warnings_path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(diagnostics_warning_lines)
            self.assertEqual(
                diagnostics_warning_lines[0].split("\t"),
                ["type", "replica_id", "next_replica_id", "inv_temp", "next_inv_temp", "target", "rate_percent", "warning"],
            )

            noise_lines = noise_path.read_text(encoding="utf-8").splitlines()
            estimated_line = next(line for line in noise_lines if line.startswith("# estimated_sigma2"))
            estimated_sigma2 = float(estimated_line.split("\t")[1])
            replica_line = next(line for line in noise_lines if line.startswith("# replica_id"))
            posterior_replica_id = int(replica_line.split("\t")[1])
            configured_sigma2_min = json.loads((target / "config.json").read_text(encoding="utf-8"))["model"]["noise"]["sigma2_min"]
            self.assertGreater(estimated_sigma2, configured_sigma2_min)
            self.assertIn("# sigma2_mode\testimated", noise_lines)
            self.assertIn("# replica_index_base\t0", noise_lines)
            self.assertTrue(any(line.startswith("# candidate_max_sigma2") for line in noise_lines))
            self.assertIn("sigma2\tinverse_temperature\tfree_energy", noise_lines)

            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            self.assertEqual(sample["posterior_replica_id"], posterior_replica_id)
            self.assertAlmostEqual(sample["posterior_sigma2"], estimated_sigma2)
            posterior = cli._extract_posterior(sample)
            samples = cli._extract_samples(sample)
            map_index = max(range(len(posterior)), key=posterior.__getitem__)
            map_intercept, map_slope = samples[map_index]

            self.assertEqual(sample["schema_version"], 4)
            self.assertEqual(sample["sample_count"], 800)
            self.assertEqual(sample["samples"]["format"], "columnar_v1")
            self.assertEqual(len(samples), 800)
            self.assertLess(abs(map_intercept - 1.25), 0.08)
            self.assertLess(abs(map_slope - (-0.80)), 0.08)

            self.assertEqual(cli.main(["plot", str(sample_path)]), 0)
            figure_dir = target / "result" / "figures"
            self.assertTrue((figure_dir / "corner.png").exists() or (figure_dir / "posterior.svg").exists())
            self.assertTrue((figure_dir / "posterior_max.json").exists())

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_diagnostics_marks_unscaled_temperature_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "linear"
            self.assertEqual(cli.main(["init", "linear", str(target)]), 0)
            config_path = target / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["emc"]["replica_num"] = 4
            config["emc"]["gamma"] = 100.0
            config["emc"]["sample_num"] = 8
            config["emc"]["burnin_num"] = 4
            write_json(config_path, config)

            self.assertEqual(cli.main(["run", str(config_path)]), 0)

            diagnostics_lines = (target / "result" / "diagnostics.tsv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(diagnostics_lines[1].split("\t")[0].startswith("*"))
            self.assertEqual(diagnostics_lines[1].split("\t")[1:3], ["100.00", "100.00"])
            self.assertTrue(diagnostics_lines[2].split("\t")[0].startswith("*"))
            self.assertFalse(diagnostics_lines[3].split("\t")[0].startswith("*"))
            self.assertFalse(diagnostics_lines[4].split("\t")[0].startswith("*"))

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_diagnostics_warnings_report_extreme_acceptance_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "linear"
            self.assertEqual(cli.main(["init", "linear", str(target)]), 0)
            config_path = target / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["emc"]["replica_num"] = 4
            config["emc"]["gamma"] = 10.0
            config["emc"]["sample_num"] = 20
            config["emc"]["burnin_num"] = 10
            config["emc"]["parallel_workers"] = 1
            config["emc"]["progress"] = False
            for parameter in config["model"]["models"][0]["parameters"]:
                parameter["C"] = 1e-12
            write_json(config_path, config)

            self.assertEqual(cli.main(["run", str(config_path)]), 0)

            warnings_path = target / "result" / "diagnostics_warnings.tsv"
            warning_lines = warnings_path.read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(warning_lines), 1)
            self.assertEqual(
                warning_lines[0].split("\t"),
                ["type", "replica_id", "next_replica_id", "inv_temp", "next_inv_temp", "target", "rate_percent", "warning"],
            )
            warning_rows = [line.split("\t") for line in warning_lines[1:]]
            self.assertTrue(any(row[0] == "mh_acceptance" for row in warning_rows))
            self.assertTrue(any(row[-1] == "above_high_threshold" for row in warning_rows))
            self.assertFalse(any(row[0] == "mh_acceptance" and row[1] == "0" for row in warning_rows))

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_tune_writes_tuned_config_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "linear"
            self.assertEqual(cli.main(["init", "linear", str(target)]), 0)
            config_path = target / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["emc"]["replica_num"] = 4
            config["emc"]["gamma"] = 1.2
            config["emc"]["sample_num"] = 80
            config["emc"]["burnin_num"] = 80
            config["emc"]["parallel_workers"] = 1
            config["emc"]["progress"] = False
            write_json(config_path, config)

            tuned_path = target / "config.tuned.json"
            self.assertEqual(
                cli.main([
                    "tune",
                    str(config_path),
                    "--sample-num",
                    "8",
                    "--burnin-num",
                    "4",
                    "--c-rounds",
                    "1",
                    "--d-rounds",
                    "1",
                    "--gamma-candidates",
                    "2",
                    "--max-replica-num",
                    "4",
                    "--output-config",
                    str(tuned_path),
                    "--quiet",
                ]),
                0,
            )

            tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
            self.assertEqual(tuned["emc"]["sample_num"], 80)
            self.assertEqual(tuned["emc"]["burnin_num"], 80)
            self.assertIn("gamma", tuned["emc"])
            for parameter in tuned["model"]["models"][0]["parameters"]:
                self.assertIn("C", parameter)
                self.assertIn("d", parameter)
                self.assertGreater(parameter["C"], 0.0)
                self.assertGreaterEqual(parameter["d"], 0.0)

            report_tsv = target / "result" / "tuning" / "tune_report.tsv"
            report_json = target / "result" / "tuning" / "tune_report.json"
            self.assertTrue(report_tsv.exists())
            self.assertTrue(report_json.exists())
            report_lines = report_tsv.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(report_lines), 2)
            self.assertIn("phase\tround\tlabel", report_lines[0])
            self.assertTrue(any("top_exchange=" in line for line in report_lines))
            self.assertTrue(any("cold_accept_min=" in line for line in report_lines))
            self.assertTrue(any(line.startswith("local_step\t") for line in report_lines))
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(Path(report["tuned_config"]).resolve(), tuned_path.resolve())
            self.assertIn("final", report)

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_run_spectral_project_builds_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "spectral"
            self.assertEqual(cli.main(["init", "spectral", str(target)]), 0)
            self.assertEqual(cli.main(["run", str(target / "config.json")]), 0)

            generated_header = target / "src" / "generated_v2_config.hpp"
            sample_path = target / "result" / "sample.json"
            log_path = target / "result" / "log.txt"
            diagnostics_path = target / "result" / "diagnostics.tsv"
            self.assertTrue(generated_header.exists())
            self.assertTrue(sample_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(diagnostics_path.exists())

            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            samples = cli._extract_samples(sample)
            posterior = cli._extract_posterior(sample)
            self.assertEqual(sample["schema_version"], 4)
            self.assertEqual(len(sample["parameters"]), 9)
            self.assertEqual(sample["sample_count"], 240)
            self.assertEqual(sample["samples"]["format"], "columnar_v1")
            self.assertEqual(len(samples), 240)
            self.assertEqual(len(posterior), 240)

            self.assertEqual(cli.main(["plot", str(sample_path)]), 0)
            figure_dir = target / "result" / "figures"
            self.assertTrue((figure_dir / "corner.png").exists() or (figure_dir / "posterior.svg").exists())
            self.assertTrue((figure_dir / "posterior_max.json").exists())

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_select_peaks_spectral_project_selects_three_peak_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "spectral"
            self.assertEqual(cli.main(["init", "spectral", str(target)]), 0)
            self.assertEqual(
                cli.main(["select-peaks", str(target / "config.json"), "--min", "2", "--max", "3"]),
                0,
            )

            summary_path = target / "result" / "model_selection" / "peak_count" / "peak_selection.json"
            table_path = target / "result" / "model_selection" / "peak_count" / "peak_selection.txt"
            svg_path = target / "result" / "model_selection" / "peak_count" / "peak_selection.svg"
            self.assertTrue(summary_path.exists())
            self.assertTrue(table_path.exists())
            self.assertTrue(svg_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["criterion"], "estimated-noise")
            self.assertEqual(summary["selected_peak_count"], 3)
            self.assertEqual([item["peak_count"] for item in summary["candidates"]], [2, 3])
            self.assertLess(summary["candidates"][1]["score"], summary["candidates"][0]["score"])

    @unittest.skipUnless(shutil.which("c++"), "requires c++")
    def test_run_background_spectral_project_outputs_selected_posterior_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "background_spectral"
            self.assertEqual(cli.main(["init", "background-spectral", str(target)]), 0)
            self.assertEqual(cli.main(["run", str(target / "config.json")]), 0)

            sample_path = target / "result" / "sample.json"
            log_path = target / "result" / "log.txt"
            generated_header = target / "src" / "generated_v2_config.hpp"
            noise_path = target / "result" / "noise_estimation.txt"
            diagnostics_path = target / "result" / "diagnostics.tsv"
            self.assertTrue(sample_path.exists())
            self.assertTrue(log_path.exists())
            self.assertTrue(generated_header.exists())
            self.assertTrue(noise_path.exists())
            self.assertTrue(diagnostics_path.exists())
            noise_lines = noise_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("sigma2\tinverse_temperature\tfree_energy", noise_lines)
            posterior_replica_id = int(next(line for line in noise_lines if line.startswith("# replica_id")).split("\t")[1])
            estimated_sigma2 = float(next(line for line in noise_lines if line.startswith("# estimated_sigma2")).split("\t")[1])

            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            posterior = cli._extract_posterior(sample)
            samples = cli._extract_samples(sample)

            self.assertEqual(sample["schema_version"], 4)
            self.assertEqual(len(sample["parameters"]), 8)
            self.assertEqual(sample["sample_count"], 600)
            self.assertEqual(sample["samples"]["format"], "columnar_v1")
            self.assertEqual(len(samples), 600)
            self.assertEqual(len(posterior), 600)
            self.assertEqual(sample["posterior_replica_id"], posterior_replica_id)
            self.assertAlmostEqual(sample["posterior_sigma2"], estimated_sigma2)

            self.assertEqual(cli.main(["plot", str(sample_path)]), 0)
            figure_dir = target / "result" / "figures"
            self.assertTrue((figure_dir / "corner.png").exists() or (figure_dir / "posterior.svg").exists())
            self.assertTrue((figure_dir / "posterior_max.json").exists())


@unittest.skip("The upstream repository's unrelated example gallery is intentionally not bundled here.")
class ExampleWorkflowTests(unittest.TestCase):
    def test_linear_1d_minimal_example_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        project = root / "examples" / "linear_1d_minimal"
        config = cli.load_config(project / "config.json")

        self.assertEqual(cli.validate_config(config, require_sources=True), [])
        self.assertEqual(config.data_format, "csv")
        self.assertEqual(config.input_columns, ["x"])
        self.assertEqual(config.output_columns, ["y"])
        self.assertFalse(config.estimate_sigma2)
        self.assertEqual(config.model_layout[0]["layers"][0]["parameters"], ["a", "b"])
        self.assertIn("a * x[0] + b", (project / "src" / "target.hpp").read_text(encoding="utf-8"))
        self.assertIn("TRUE_A = 0.80", (project / "generate_data.py").read_text(encoding="utf-8"))
        make_config = (project / "make_config.py").read_text(encoding="utf-8")
        self.assertIn("def build_config(", make_config)
        self.assertIn("def write_config(", make_config)
        self.assertIn("normal(\"a\", mean=0.0, sigma=3.0)", make_config)
        self.assertIn("ASSUMED_SIGMA2 = 0.0025", make_config)
        self.assertIn("SAMPLE_NUM = 600", make_config)
        self.assertIn("DEFAULT_EMC = {", make_config)
        self.assertIn("MAP  a", (project / "map_summary.py").read_text(encoding="utf-8"))

        notebook = json.loads((project / "linear_1d_minimal.ipynb").read_text(encoding="utf-8"))
        joined_sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertNotIn("run_python(", joined_sources)
        self.assertIn("true_a = 0.80", joined_sources)
        self.assertIn("rng.gauss(0.0, noise_sigma)", joined_sources)
        self.assertIn("## 2. 生成モデルを記述する", joined_sources)
        self.assertIn("target_source = r'''#pragma once", joined_sources)
        self.assertIn("target_path.write_text", joined_sources)
        self.assertIn("from make_config import normal, uniform, write_config", joined_sources)
        self.assertIn("write_config(", joined_sources)
        self.assertIn("parameters = [", joined_sources)
        self.assertIn("sample_num = 600", joined_sources)
        self.assertIn("run_cli(\"check\", \"config.json\", \"--sources\")", joined_sources)
        self.assertIn("run_cli(\"tune\", \"config.json\"", joined_sources)
        self.assertIn("run_cli(\"run\", \"config.tuned.json\")", joined_sources)
        self.assertIn("run_cli(\"plot\", \"result/sample.json\")", joined_sources)
        self.assertIn("estimate_sigma2 = False", joined_sources)
        self.assertIn("sample = json.loads(sample_path.read_text", joined_sources)
        self.assertIn("MAP  a", joined_sources)
        self.assertIn("## 8. MAP 推定線をデータへ重ねる", joined_sources)
        self.assertIn("MAP reconstruction RMSE", joined_sources)
        self.assertIn("map_curve =", joined_sources)
        self.assertIn("measurement and MAP reconstruction", joined_sources)
        readme = (project / "README.md").read_text(encoding="utf-8")
        self.assertIn("MAP推定線の重ね描き", readme)

    def test_spectral_minimal_example_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        project = root / "examples" / "spectral_minimal"
        config = cli.load_config(project / "config.json")

        self.assertEqual(cli.validate_config(config, require_sources=True), [])
        self.assertEqual(config.data_format, "csv")
        self.assertEqual(config.input_columns, ["x"])
        self.assertEqual(config.output_columns, ["y"])
        self.assertFalse(config.estimate_sigma2)
        self.assertEqual(config.model_layout[0]["name"], "spectral_peaks")
        self.assertEqual(config.model_layout[0]["basis_count"], 3)
        self.assertEqual(config.model_layout[0]["layers"][0]["parameters"], ["a", "mu", "b"])
        target_text = (project / "src" / "target.hpp").read_text(encoding="utf-8")
        self.assertIn("static thread_local std::vector<double> amplitudes", target_text)
        self.assertIn("half_curvatures[peak] = -0.5 * params.Value(0, peak, 2);", target_text)
        self.assertIn("std::exp(half_curvatures[peak] * diff * diff)", target_text)
        self.assertIn("params.Layout().Spec()", target_text)
        self.assertIn("spec.models[0].basis_count", target_text)

        make_config = (project / "make_config.py").read_text(encoding="utf-8")
        self.assertIn("def gamma(", make_config)
        self.assertIn("basis_count: int = BASIS_COUNT", make_config)
        self.assertIn("ESTIMATE_SIGMA2 = False", make_config)
        self.assertIn("DEFAULT_PARAMETER_TUNING", make_config)

        notebook = json.loads((project / "spectral_minimal.ipynb").read_text(encoding="utf-8"))
        joined_sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertNotIn("run_python(", joined_sources)
        self.assertIn("true_peaks = [", joined_sources)
        self.assertIn("rng.gauss(0.0, noise_sigma)", joined_sources)
        self.assertIn("## 2. 生成モデルを記述する", joined_sources)
        self.assertIn('target_source = "#pragma once', joined_sources)
        self.assertIn("target_path.write_text", joined_sources)
        self.assertIn("basis_count = len(true_peaks)", joined_sources)
        self.assertIn("gamma(\"a\", shape=2.0, scale=0.5)", joined_sources)
        self.assertIn("normal(\"mu\", mean=1.5, sigma=0.5)", joined_sources)
        self.assertIn("gamma(\"b\", shape=14.0, scale=10.0)", joined_sources)
        self.assertIn("write_config(", joined_sources)
        self.assertIn("run_cli(\"check\", \"config.json\", \"--sources\")", joined_sources)
        self.assertIn("run_cli(\"tune\", \"config.json\"", joined_sources)
        self.assertIn("run_cli(\"run\", \"config.tuned.json\")", joined_sources)
        self.assertIn("run_cli(\"plot\", \"result/sample.json\", \"--sort-peaks-by\", \"mu\")", joined_sources)
        self.assertIn("\"select-peaks\"", joined_sources)
        self.assertIn("\"--criterion\", \"fixed-noise\"", joined_sources)
        self.assertIn("config.model_selection.json", joined_sources)
        self.assertIn("selected_peak_count", joined_sources)
        self.assertIn("peak_position[0]", joined_sources)
        self.assertIn("map_sorted = sorted", joined_sources)
        self.assertIn("## 8. MAP 再構成曲線をデータへ重ねる", joined_sources)
        self.assertIn("MAP reconstruction RMSE", joined_sources)
        self.assertIn("map_curve =", joined_sources)
        self.assertIn("measurement and MAP reconstruction", joined_sources)
        self.assertNotIn("旧版", joined_sources)
        self.assertNotIn("spd_1", joined_sources)
        readme = (project / "README.md").read_text(encoding="utf-8")
        self.assertIn("select-peaks config.tuned.json --min 1 --max 5 --criterion fixed-noise", readme)
        self.assertIn("MAP再構成曲線の重ね描き", readme)
        self.assertNotIn("旧版", readme)
        self.assertNotIn("spd_1", readme)

    def test_background_spectral_minimal_example_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        project = root / "examples" / "background_spectral_minimal"
        config = cli.load_config(project / "config.json")

        self.assertEqual(cli.validate_config(config, require_sources=True), [])
        self.assertEqual(config.data_format, "csv")
        self.assertEqual(config.input_columns, ["x"])
        self.assertEqual(config.output_columns, ["y"])
        self.assertFalse(config.estimate_sigma2)
        self.assertEqual([model["name"] for model in config.model_layout], ["linear_background", "spectral_peaks"])
        self.assertEqual(config.model_layout[0]["basis_count"], 1)
        self.assertEqual(config.model_layout[1]["basis_count"], 2)
        self.assertEqual(config.model_layout[0]["layers"][0]["parameters"], ["intercept", "slope"])
        self.assertEqual(config.model_layout[1]["layers"][0]["parameters"], ["a", "mu", "b"])

        target_text = (project / "src" / "target.hpp").read_text(encoding="utf-8")
        self.assertIn("params.Value(0, 0, 0)", target_text)
        self.assertIn("static thread_local std::vector<double> amplitudes", target_text)
        self.assertIn("half_curvatures[peak] = -0.5 * params.Value(1, peak, 2);", target_text)
        self.assertIn("std::exp(half_curvatures[peak] * diff * diff)", target_text)
        self.assertIn("spec.models[1].basis_count", target_text)

        make_config = (project / "make_config.py").read_text(encoding="utf-8")
        self.assertIn("PEAK_BASIS_COUNT = 2", make_config)
        self.assertIn("LINEAR_PARAMETERS = [", make_config)
        self.assertIn("PEAK_PARAMETERS = [", make_config)
        self.assertIn("ESTIMATE_SIGMA2 = False", make_config)

        notebook = json.loads((project / "background_spectral_minimal.ipynb").read_text(encoding="utf-8"))
        joined_sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertIn("true_background = {", joined_sources)
        self.assertIn("true_peaks = [", joined_sources)
        self.assertIn("rng.gauss(0.0, noise_sigma)", joined_sources)
        self.assertIn("## 2. 生成モデルを記述する", joined_sources)
        self.assertIn("target_path.write_text", joined_sources)
        self.assertIn("linear_parameters = [", joined_sources)
        self.assertIn("peak_parameters = [", joined_sources)
        self.assertIn("write_config(", joined_sources)
        self.assertIn("run_cli(\"check\", \"config.json\", \"--sources\")", joined_sources)
        self.assertIn("run_cli(\"tune\", \"config.json\"", joined_sources)
        self.assertIn("run_cli(\"run\", \"config.tuned.json\")", joined_sources)
        self.assertIn("run_cli(\"plot\", \"result/sample.json\", \"--sort-peaks-by\", \"mu\")", joined_sources)
        self.assertIn("\"select-peaks\"", joined_sources)
        self.assertIn("\"--model\", \"spectral_peaks\"", joined_sources)
        self.assertIn("\"--criterion\", \"fixed-noise\"", joined_sources)
        self.assertIn("map_background", joined_sources)
        self.assertIn("## 8. MAP 再構成曲線をデータへ重ねる", joined_sources)
        self.assertIn("MAP reconstruction RMSE", joined_sources)
        self.assertIn("map_curve =", joined_sources)
        self.assertIn("measurement and MAP reconstruction", joined_sources)
        self.assertNotIn("旧版", joined_sources)
        readme = (project / "README.md").read_text(encoding="utf-8")
        self.assertIn("MAP再構成曲線の重ね描き", readme)

    def test_measurement_beta_prior_minimal_example_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        project = root / "examples" / "measurement_beta_prior_minimal"
        config = cli.load_config(project / "config.json")

        self.assertEqual(cli.validate_config(config, require_sources=True), [])
        self.assertEqual(config.data_format, "whitespace")
        self.assertFalse(config.data_header)
        self.assertEqual(config.input_dim, 1)
        self.assertEqual(config.output_dim, 1)
        self.assertTrue(config.estimate_sigma2)
        self.assertEqual(config.model_layout[0]["name"], "spectral_peaks")
        self.assertEqual(config.model_layout[0]["basis_count"], 4)
        self.assertEqual(config.model_layout[0]["layers"][0]["parameters"], ["a", "mu", "sigma"])

        target_text = (project / "src" / "target.hpp").read_text(encoding="utf-8")
        self.assertIn("KernelDigamma", target_text)
        self.assertIn("params.Value(0, peak, 2)", target_text)
        self.assertIn("spec.models[0].basis_count", target_text)

        make_config = (project / "make_config.py").read_text(encoding="utf-8")
        self.assertIn("def beta(", make_config)
        self.assertIn('"type": "beta"', make_config)
        self.assertIn("SIGMA2_MIN = 1.0e-6", make_config)
        self.assertIn("ESTIMATE_SIGMA2 = True", make_config)

        notebook = json.loads((project / "measurement_beta_prior_minimal.ipynb").read_text(encoding="utf-8"))
        joined_sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertIn('data_path = PROJECT_DIR / "data" / "data.txt"', joined_sources)
        self.assertIn("## 2. 生成モデルを記述する", joined_sources)
        self.assertIn("target_path.write_text", joined_sources)
        self.assertIn('beta("a", alpha=2.0, beta=2.0, lower=0.0, upper=0.6)', joined_sources)
        self.assertIn("estimate_sigma2 = True", joined_sources)
        self.assertIn('run_cli("check", "config.json", "--sources")', joined_sources)
        self.assertIn('run_cli("tune", "config.json"', joined_sources)
        self.assertIn('run_cli("run", "config.tuned.json")', joined_sources)
        self.assertIn('run_cli("plot", "result/sample.json", "--sort-peaks-by", "mu")', joined_sources)
        self.assertIn('"estimated-noise"', joined_sources)
        self.assertIn("posterior_sigma2", joined_sources)
        self.assertIn("\\alpha^2 F(\\omega)", joined_sources)
        self.assertIn("## 8. kNN 密度ベースの empirical mode を MAP と比べる", joined_sources)
        self.assertIn("empirical mode sample index", joined_sources)
        self.assertIn("same sample as MAP", joined_sources)
        self.assertIn("kNN candidate count", joined_sources)
        self.assertIn("MAP reconstruction RMSE", joined_sources)
        self.assertIn("map_curve", joined_sources)
        self.assertIn("selected_peak_count", joined_sources)

        readme = (project / "README.md").read_text(encoding="utf-8")
        self.assertIn("空白区切りの計測データ", readme)
        self.assertIn("区間付き Beta 事前分布", readme)
        self.assertIn("kNN 密度ベースの empirical mode", readme)
        self.assertIn("select-peaks config.tuned.json --min 2 --max 6 --criterion estimated-noise", readme)
        self.assertNotIn("旧版", readme)
        self.assertNotIn("spd_1", joined_sources)

    def test_btk_synthetic_minimal_example_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        project = root / "examples" / "btk_synthetic_minimal"
        config = cli.load_config(project / "config.json")

        self.assertEqual(cli.validate_config(config, require_sources=True), [])
        self.assertEqual(config.data_format, "csv")
        self.assertTrue(config.data_header)
        self.assertEqual(config.input_columns, ["V"])
        self.assertEqual(config.output_columns, ["G"])
        self.assertTrue(config.estimate_sigma2)
        self.assertAlmostEqual(config.sigma2_min, 1.0e-4)
        self.assertEqual(config.model_layout[0]["name"], "btk")
        self.assertEqual(config.model_layout[0]["basis_count"], 1)
        self.assertEqual(config.model_layout[0]["layers"][0]["parameters"], ["Delta", "T", "P", "Z"])

        target_text = (project / "src" / "target.hpp").read_text(encoding="utf-8")
        self.assertIn("SigmaN", target_text)
        self.assertIn("SigmaP", target_text)
        self.assertIn("RawBtkConductance", target_text)
        self.assertIn("Normalization", target_text)
        self.assertIn("GaussLegendreIntegrate", target_text)
        self.assertIn("CachedNormalization", target_text)
        self.assertIn("params.Value(0, 0, 3)", target_text)
        self.assertIn("kBoltzmannMevPerKelvin", target_text)

        make_config = (project / "make_config.py").read_text(encoding="utf-8")
        self.assertIn('uniform("Delta", lower=0.5, upper=1.5)', make_config)
        self.assertIn('uniform("T", lower=0.5, upper=4.0)', make_config)
        self.assertIn('uniform("P", lower=0.0, upper=0.8)', make_config)
        self.assertIn('uniform("Z", lower=0.0, upper=1.0)', make_config)
        self.assertIn("SIGMA2_MIN = 1.0e-4", make_config)
        self.assertIn("ESTIMATE_SIGMA2 = True", make_config)

        notebook = json.loads((project / "btk_synthetic_minimal.ipynb").read_text(encoding="utf-8"))
        joined_sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertIn("TRUE_DELTA = 1.0", joined_sources)
        self.assertIn("TRUE_T = 1.5", joined_sources)
        self.assertIn("TRUE_P = 0.35", joined_sources)
        self.assertIn("TRUE_Z = 0.15", joined_sources)
        self.assertIn("NOISE_SIGMA = 0.02", joined_sources)
        self.assertIn("import numpy as np", joined_sources)
        self.assertIn("from scipy.integrate import quad", joined_sources)
        self.assertIn('env["PYTHONPATH"] = str(REPO_ROOT)', joined_sources)
        self.assertIn("sys.path.insert(0, str(PROJECT_DIR))", joined_sources)
        self.assertIn("quad(func, a, b, limit=300)", joined_sources)
        self.assertIn("np.random.default_rng(SEED)", joined_sources)
        self.assertIn("rng.normal(0.0, NOISE_SIGMA", joined_sources)
        self.assertIn("## 2. 生成モデルを記述する", joined_sources)
        self.assertIn("target_path.write_text", joined_sources)
        self.assertIn('uniform("Delta", lower=0.5, upper=1.5)', joined_sources)
        self.assertIn("estimate_sigma2 = True", joined_sources)
        self.assertIn('run_cli("check", "config.json", "--sources")', joined_sources)
        self.assertIn('run_cli("tune", "config.json"', joined_sources)
        self.assertIn('run_cli("run", "config.tuned.json")', joined_sources)
        self.assertIn('run_cli("plot", "result/sample.json")', joined_sources)
        self.assertIn("MAP reconstruction RMSE", joined_sources)
        self.assertIn("BTK data and MAP reconstruction", joined_sources)
        self.assertNotIn("旧版", joined_sources)

        readme = (project / "README.md").read_text(encoding="utf-8")
        self.assertIn("Delta", readme)
        self.assertIn("MAP 再構成曲線", readme)
        self.assertIn("btk_synthetic_minimal.ipynb", readme)

    def test_joint_marginal_mode_demo_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        project = root / "examples" / "joint_marginal_mode_demo"

        notebook = json.loads((project / "joint_marginal_mode_demo.ipynb").read_text(encoding="utf-8"))
        joined_sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertIn("joint theoretical mode component", joined_sources)
        self.assertIn("x-marginal theoretical mode component", joined_sources)
        self.assertIn("sample_count = 30000", joined_sources)
        self.assertIn("empirical joint mode bin center", joined_sources)
        self.assertIn("empirical x-marginal mode bin center", joined_sources)
        self.assertIn("joint mode A", joined_sources)
        self.assertIn("x marginal after projection", joined_sources)

        readme = (project / "README.md").read_text(encoding="utf-8")
        self.assertIn("2次元の混合ガウス分布", readme)
        self.assertIn("2次元の joint density では A が最も高い", readme)
        self.assertIn("x` 周辺化", readme)

    def test_linear_1d_workflow_example_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        project = root / "examples" / "linear_1d_workflow"
        config = cli.load_config(project / "config.json")

        self.assertEqual(cli.validate_config(config, require_sources=True), [])
        self.assertIn("_comment", config.raw)
        self.assertEqual(config.data_format, "csv")
        self.assertEqual(config.input_columns, ["x"])
        self.assertEqual(config.output_columns, ["y"])
        self.assertTrue(config.progress)
        self.assertIn("TargetFunction", (project / "src" / "target.hpp").read_text(encoding="utf-8"))
        self.assertEqual(config.model_layout[0]["layers"][0]["parameters"], ["a", "b"])

        notebook = json.loads((project / "linear_1d_bayes_flow.ipynb").read_text(encoding="utf-8"))
        joined_sources = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertIn("run_cli(\"check\", \"config.json\", \"--sources\")", joined_sources)
        self.assertIn("run_cli(\"tune\", \"config.json\"", joined_sources)
        self.assertIn("run_cli(\"run\", \"config.tuned.json\")", joined_sources)
        self.assertIn("noise_estimation.txt", joined_sources)
        self.assertIn("params.Value(model_id, basis_id, parameter_id)", joined_sources)
        self.assertIn("TargetFunction returns the noise-free mean part", joined_sources)
        self.assertIn("y_i = a x_i + b", joined_sources)
        self.assertIn("TRUE_A = 0.80", joined_sources)
        self.assertIn("TRUE_B = -1.25", joined_sources)
        self.assertIn("The _comment entries are ordinary JSON keys", joined_sources)
        self.assertIn("sample.json stores labels once", joined_sources)
        self.assertIn("Numeric samples are columnar", joined_sources)
        self.assertIn("`replica_num`", joined_sources)
        self.assertIn("C / (data_size * beta)^d", joined_sources)
        self.assertIn("## 8. MAP 推定線をデータへ重ねる", joined_sources)
        self.assertIn("MAP reconstruction RMSE", joined_sources)
        self.assertIn("map_curve =", joined_sources)
        self.assertIn("measurement and MAP reconstruction", joined_sources)
        readme = (project / "README.md").read_text(encoding="utf-8")
        self.assertIn("MAP推定線の重ね描き", readme)


if __name__ == "__main__":
    unittest.main()
