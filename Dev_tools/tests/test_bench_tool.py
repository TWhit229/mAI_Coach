import sys
from pathlib import Path

# Add parent directory to path so we can import the module
sys.path.append(str(Path(__file__).parent.parent))

import unittest
from core.metrics import _dist, _landmark_xy as _lm_xy
from bench_dataset_tool import extract_example
from label_config import DEFAULT_TAGS as ALL_TAGS

class TestBenchTool(unittest.TestCase):
    def test_dist(self):
        self.assertEqual(_dist((0, 0), (3, 4)), 5.0)
        self.assertEqual(_dist((1, 1), (1, 1)), 0.0)
        self.assertEqual(_dist((-1, -1), (-4, -5)), 5.0)

    def test_lm_xy(self):
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0}]
        self.assertEqual(_lm_xy(landmarks, 0), (0.5, 0.5))
        self.assertIsNone(_lm_xy(landmarks, 1))  # Index out of bounds
        self.assertIsNone(_lm_xy({}, 0))  # Invalid input

    def test_extract_example_valid(self):
        # Construct a minimal valid rep
        mock_rep = {
            "metrics": {
                "tracking_quality": 0.9,
                "grip_ratio_median": 1.5,
            },
            "frames": [
                {
                    "pose_present": True,
                    "landmarks": [
                        {"x": 0, "y": 0} for _ in range(33)
                    ]
                }
            ],
            "tags": ["no_major_issues"]
        }
        
        # Update landmarks for 11, 12, 15, 16 to be valid
        mock_rep["frames"][0]["landmarks"][11] = {"x": 0.4, "y": 0.2, "z": 0}
        mock_rep["frames"][0]["landmarks"][12] = {"x": 0.6, "y": 0.2, "z": 0}
        mock_rep["frames"][0]["landmarks"][15] = {"x": 0.35, "y": 0.8, "z": 0}
        mock_rep["frames"][0]["landmarks"][16] = {"x": 0.65, "y": 0.8, "z": 0}

        result = extract_example(mock_rep, ALL_TAGS)
        self.assertIsNotNone(result)
        features, labels = result
        
        # Check features length (should be 12 based on the code)
        self.assertEqual(len(features), 12)
        # Check labels length matches ALL_TAGS
        self.assertEqual(len(labels), len(ALL_TAGS))
        # Check one-hot encoding
        if "no_major_issues" in ALL_TAGS:
            no_issues_idx = ALL_TAGS.index("no_major_issues")
            self.assertEqual(labels[no_issues_idx], 1)

    def test_extract_example_unreliable(self):
        mock_rep = {"tracking_unreliable": True}
        self.assertIsNone(extract_example(mock_rep, ALL_TAGS))

    def test_extract_example_bad_quality(self):
        mock_rep = {
            "metrics": {"tracking_quality": 0.4},
            "frames": [{"foo": "bar"}]
        }
        self.assertIsNone(extract_example(mock_rep, ALL_TAGS))

if __name__ == "__main__":
    unittest.main()
