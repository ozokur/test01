import unittest

from yolo_gui import TrainingConfig


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


if __name__ == "__main__":
    unittest.main()
