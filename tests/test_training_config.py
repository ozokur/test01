import os
import tempfile
import unittest

from yolo_gui import (
    DEFAULT_PRETRAINED_MODEL,
    DetectedObject,
    InferenceStats,
    TrainingConfig,
    collect_box_detections,
    count_predictions_from_results,
    describe_cuda_support,
    generate_mock_training_configs,
    get_app_version,
    is_valid_weight_reference,
    list_image_files,
    parse_detection_count,
)


class TrainingConfigTests(unittest.TestCase):
    def test_build_command_includes_all_parameters(self):
        config = TrainingConfig(
            dataset_yaml="data.yaml",
            model_weights="yolov8n.pt",
            epochs=50,
            batch_size=8,
            image_size=640,
            project_name="my_project",
            task="detect",
        )

        command = config.build_command()

        self.assertIn("yolo", command)
        self.assertIn("task=detect", command)
        self.assertIn("mode=train", command)
        self.assertIn("data=data.yaml", command)
        self.assertIn("model=yolov8n.pt", command)
        self.assertIn("epochs=50", command)
        self.assertIn("batch=8", command)
        self.assertIn("imgsz=640", command)
        self.assertIn("project=my_project", command)

    def test_build_command_omits_empty_project(self):
        config = TrainingConfig(
            dataset_yaml="data.yaml",
            model_weights="yolov8n.pt",
            epochs=50,
            batch_size=8,
            image_size=640,
            project_name="",
            task="segment",
        )

        command = config.build_command()

        self.assertNotIn("project=", " ".join(command))
        self.assertIn("task=segment", command)


class DescribeCudaSupportTests(unittest.TestCase):
    def test_returns_message_when_cuda_not_available(self):
        class FakeCuda:
            def is_available(self):
                return False

        class FakeTorch:
            cuda = FakeCuda()

        message = describe_cuda_support(FakeTorch())

        self.assertEqual(message, "CUDA Support: Not available")

    def test_returns_device_names_when_available(self):
        class FakeCuda:
            def __init__(self):
                self._names = ["NVIDIA RTX 3090", "NVIDIA RTX 3080"]

            def is_available(self):
                return True

            def device_count(self):
                return len(self._names)

            def get_device_name(self, index):
                return self._names[index]

        class FakeTorch:
            cuda = FakeCuda()

        message = describe_cuda_support(FakeTorch())

        self.assertIn("CUDA Support: Available", message)
        self.assertIn("NVIDIA RTX 3080", message)

    def test_returns_message_when_cuda_attribute_missing(self):
        class FakeTorch:
            pass

        message = describe_cuda_support(FakeTorch())

        self.assertEqual(message, "CUDA Support: PyTorch without CUDA support")


class GenerateMockTrainingConfigsTests(unittest.TestCase):
    def test_creates_requested_number_of_configs(self):
        configs = generate_mock_training_configs()

        self.assertEqual(len(configs), 30)
        self.assertTrue(all(isinstance(cfg, TrainingConfig) for cfg in configs))

    def test_supports_custom_count(self):
        configs = generate_mock_training_configs(12)

        self.assertEqual(len(configs), 12)
        self.assertEqual(configs[0].dataset_yaml, "data/mock_dataset_0.yaml")
        self.assertEqual(configs[-1].project_name, "mock_project_11")

    def test_rejects_non_positive_count(self):
        with self.assertRaises(ValueError):
            generate_mock_training_configs(0)


class WeightReferenceTests(unittest.TestCase):
    def test_default_alias_is_valid(self):
        self.assertTrue(is_valid_weight_reference(DEFAULT_PRETRAINED_MODEL))

    def test_requires_existing_path_for_custom_files(self):
        self.assertFalse(is_valid_weight_reference("nonexistent/best.pt"))
        self.assertFalse(is_valid_weight_reference("custom.pt"))

        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "best.pt")
            with open(weights_path, "w", encoding="utf-8") as handle:
                handle.write("data")

            self.assertTrue(is_valid_weight_reference(weights_path))


class VersionTests(unittest.TestCase):
    def test_version_string_is_not_empty(self):
        version = get_app_version()

        self.assertIsInstance(version, str)
        self.assertTrue(version)


class InferenceStatsTests(unittest.TestCase):
    def test_describe_before_any_runs(self):
        stats = InferenceStats()

        self.assertEqual(stats.describe(), "Performance: No inferences run yet.")

    def test_record_updates_metrics(self):
        stats = InferenceStats()
        stats.record(0.5, 3)
        stats.record(0.2, 0)

        self.assertEqual(stats.total_images, 2)
        self.assertAlmostEqual(stats.total_time, 0.7)
        self.assertAlmostEqual(stats.average_duration, 0.35)
        self.assertEqual(stats.total_detections, 3)
        self.assertEqual(stats.images_with_detections, 1)
        self.assertIn("Runs=2", stats.describe())
        self.assertIn("Best=0.20s", stats.describe())
        self.assertIn("Detections: 0", stats.describe())
        self.assertIn("With Objects=1/2", stats.describe())

    def test_negative_duration_treated_as_zero(self):
        stats = InferenceStats()
        stats.record(-3.0)

        self.assertEqual(stats.total_images, 1)
        self.assertEqual(stats.total_time, 0.0)
        self.assertIn("Last=0.00s", stats.describe())
        self.assertIn("Detections: Unknown", stats.describe())
        self.assertIn("With Objects=0/1", stats.describe())


class ParseDetectionCountTests(unittest.TestCase):
    def test_extracts_box_pattern(self):
        output = "some logs... 5 boxes, Speed: 5.0ms"
        self.assertEqual(parse_detection_count(output), 5)

    def test_returns_none_when_missing(self):
        self.assertIsNone(parse_detection_count("no detection info here"))


class CountPredictionsFromResultsTests(unittest.TestCase):
    def test_counts_boxes_and_masks(self):
        class Boxes(list):
            pass

        class Masks(list):
            pass

        class Result:
            def __init__(self, boxes=None, masks=None):
                self.boxes = boxes
                self.masks = masks

        results = [
            Result(boxes=Boxes([1, 2, 3])),
            Result(masks=Masks([1, 2])),
            Result(boxes=Boxes()),
        ]

        self.assertEqual(count_predictions_from_results(results), 5)

    def test_handles_missing_data(self):
        class Result:
            def __init__(self):
                self.foo = "bar"

        self.assertIsNone(count_predictions_from_results([Result()]))


class CollectBoxDetectionsTests(unittest.TestCase):
    def test_extracts_boxes_with_labels(self):
        class FakeTensor:
            def __init__(self, data):
                self._data = data

            def tolist(self):
                return self._data

        class FakeBoxes:
            def __init__(self):
                self.xyxy = FakeTensor([[0, 1, 2, 3], [10, 11, 12, 13]])
                self.cls = FakeTensor([0, 1])
                self.conf = FakeTensor([0.9, 0.5])

        class FakeResult:
            def __init__(self):
                self.boxes = FakeBoxes()
                self.names = {0: "person", 1: "dog"}

        detections = collect_box_detections([FakeResult()])

        self.assertEqual(len(detections), 2)
        self.assertTrue(all(isinstance(det, DetectedObject) for det in detections))
        self.assertEqual(detections[0].label, "person")
        self.assertAlmostEqual(detections[0].xyxy[2], 2.0)
        self.assertAlmostEqual(detections[1].confidence, 0.5)

    def test_handles_missing_data_gracefully(self):
        class EmptyResult:
            def __init__(self):
                self.names = {}

        self.assertEqual(collect_box_detections([]), [])
        self.assertEqual(collect_box_detections([EmptyResult()]), [])


class ListImageFilesTests(unittest.TestCase):
    def test_filters_supported_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            supported = ["image1.jpg", "image2.PNG", "photo.tiff"]
            unsupported = ["notes.txt", "script.py", "archive.zip"]

            for name in supported + unsupported:
                path = os.path.join(temp_dir, name)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("test")

            result = list_image_files(temp_dir)

            expected = [
                os.path.join(temp_dir, "image1.jpg"),
                os.path.join(temp_dir, "image2.PNG"),
                os.path.join(temp_dir, "photo.tiff"),
            ]

            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
