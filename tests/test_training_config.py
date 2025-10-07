import unittest

from yolo_gui import TrainingConfig, describe_cuda_support


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


if __name__ == "__main__":
    unittest.main()
