import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINEAR_DIR = ROOT / "tutorials" / "01_linear_model"
XPS_DIR = ROOT / "tutorials" / "02_xps_spectral_decomposition"


class TutorialContractTests(unittest.TestCase):
    def test_linear_tutorial_matches_book_conditions(self):
        config = json.loads((LINEAR_DIR / "config.json").read_text(encoding="utf-8"))
        source = (LINEAR_DIR / "linear_model_bayesian_inference.py").read_text(encoding="utf-8")

        self.assertIn("N = 50", source)
        self.assertIn("A_TRUE = 0.8", source)
        self.assertIn("B_TRUE = 0.1", source)
        self.assertIn("SIGMA2_TRUE = 0.01", source)
        self.assertTrue(config["model"]["noise"]["estimate_sigma2"])
        priors = [parameter["prior"] for parameter in config["model"]["models"][0]["parameters"]]
        self.assertEqual(
            priors,
            [
                {"type": "uniform", "lower": -1.0, "upper": 1.0},
                {"type": "uniform", "lower": -1.0, "upper": 1.0},
            ],
        )

    def test_xps_tutorial_matches_book_conditions(self):
        config = json.loads((XPS_DIR / "config.json").read_text(encoding="utf-8"))
        source = (XPS_DIR / "xps_spectral_decomposition.py").read_text(encoding="utf-8")
        target = (XPS_DIR / "src" / "target.hpp").read_text(encoding="utf-8")

        for peak in (
            '{"A": 0.6, "mu": 1.20, "w": 0.10}',
            '{"A": 1.5, "mu": 1.45, "w": 0.08}',
            '{"A": 1.2, "mu": 1.70, "w": 0.07}',
        ):
            self.assertIn(peak, source)
        self.assertIn("N = 300", source)
        self.assertIn("SIGMA2 = 0.01", source)
        self.assertIn('"1",\n    "--max",\n    "6"', source)
        self.assertIn('"fixed-noise"', source)

        self.assertEqual(config["model"]["models"][0]["basis_count"], 3)
        self.assertEqual(
            config["model"]["noise"],
            {"type": "gaussian", "sigma2_min": 0.01, "estimate_sigma2": False},
        )
        self.assertIn("(energy - position) / width", target)
        self.assertIn("std::exp(-0.5 * scaled * scaled)", target)

    def test_published_notebooks_are_executed_and_clean(self):
        notebooks = (
            LINEAR_DIR / "linear_model_bayesian_inference.ipynb",
            XPS_DIR / "xps_spectral_decomposition.ipynb",
        )
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            counts = [cell["execution_count"] for cell in code_cells]
            errors = [
                output
                for cell in code_cells
                for output in cell.get("outputs", [])
                if output.get("output_type") == "error"
            ]

            self.assertEqual(counts, list(range(1, len(code_cells) + 1)), path)
            self.assertEqual(errors, [], path)
            notebook_text = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", notebook_text, path)
            output_text = json.dumps(
                [output for cell in code_cells for output in cell.get("outputs", [])],
                ensure_ascii=False,
            )
            self.assertNotIn("/var/folders/", output_text, path)
            self.assertNotIn("FigureCanvasAgg is non-interactive", output_text, path)

            embedded_images = [
                output
                for cell in code_cells
                for output in cell.get("outputs", [])
                if "image/png" in output.get("data", {}) or "image/svg+xml" in output.get("data", {})
            ]
            self.assertGreaterEqual(len(embedded_images), 4, path)


if __name__ == "__main__":
    unittest.main()
