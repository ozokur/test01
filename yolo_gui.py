"""YOLO training GUI application.

This script provides a simple Tkinter-based GUI for configuring and launching
YOLO training jobs. Users can pick the dataset YAML file, pre-trained weights
and adjust training parameters such as the number of epochs, batch size, image
size and project name. Training is executed in a background thread using the
`yolo` command line interface, and stdout/stderr from the process is streamed
into the GUI log window.

Dependencies:
    * Python 3.9+
    * Tkinter (included with most Python installations)
    * The `ultralytics` package that provides the `yolo` CLI.

The GUI focuses on being beginner-friendly for quickly iterating on datasets
without needing to remember CLI flags.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import queue
import re
import shlex
import subprocess
import threading
import time
import tkinter as tk
from collections import Counter
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional


APP_VERSION = "1.5.0"


DEFAULT_PRETRAINED_MODEL = "yolov8n.pt"


KNOWN_WEIGHT_ALIASES = {
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
    "yolov8n-cls.pt",
    "yolov8s-cls.pt",
    "yolov8m-cls.pt",
    "yolov8l-cls.pt",
    "yolov8x-cls.pt",
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt",
}


def get_app_version() -> str:
    """Return the human-readable application version."""

    return APP_VERSION


def _load_torch_module() -> Optional[Any]:
    """Load the torch module if it is installed."""

    if importlib.util.find_spec("torch") is None:
        return None

    return importlib.import_module("torch")


TORCH_MODULE = _load_torch_module()


def _load_ultralytics_model_class() -> Optional[Any]:
    """Return the ``YOLO`` class from ultralytics if the package is installed."""

    spec = importlib.util.find_spec("ultralytics")
    if spec is None:
        return None

    module = importlib.import_module("ultralytics")
    return getattr(module, "YOLO", None)


YOLO_MODEL_CLASS = _load_ultralytics_model_class()


def _load_pil_components() -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Return Pillow's ``Image``, ``ImageTk`` and ``ImageDraw`` modules when available."""

    pillow_spec = importlib.util.find_spec("PIL")
    if pillow_spec is None:
        return None, None, None

    image_spec = importlib.util.find_spec("PIL.Image")
    imgtk_spec = importlib.util.find_spec("PIL.ImageTk")
    draw_spec = importlib.util.find_spec("PIL.ImageDraw")

    if image_spec is None or imgtk_spec is None or draw_spec is None:
        return None, None, None

    image_module = importlib.import_module("PIL.Image")
    imgtk_module = importlib.import_module("PIL.ImageTk")
    draw_module = importlib.import_module("PIL.ImageDraw")
    return image_module, imgtk_module, draw_module


PIL_IMAGE_MODULE, PIL_IMAGETK_MODULE, PIL_IMAGEDRAW_MODULE = _load_pil_components()


def describe_cuda_support(torch_module: Optional[Any] = None) -> str:
    """Return a human-readable description of CUDA availability."""

    module = torch_module if torch_module is not None else TORCH_MODULE

    if module is None:
        return "CUDA Support: PyTorch not installed"

    try:
        if not getattr(module, "cuda", None):
            return "CUDA Support: PyTorch without CUDA support"

        if not module.cuda.is_available():
            return "CUDA Support: Not available"

        device_count = module.cuda.device_count()
        if device_count == 0:
            return "CUDA Support: No CUDA devices detected"

        device_names = {
            module.cuda.get_device_name(index) for index in range(device_count)
        }
        devices_str = ", ".join(sorted(device_names))
        return f"CUDA Support: Available ({devices_str})"
    except Exception:
        return "CUDA Support: Unable to determine"


@dataclass
class TrainingConfig:
    """Stores YOLO training parameters."""

    dataset_yaml: str
    model_weights: str
    epochs: int
    batch_size: int
    image_size: int
    project_name: str
    task: str = "detect"

    def build_command(self) -> list[str]:
        """Build the CLI command for the provided configuration."""

        command = [
            "yolo",
            f"task={self.task}",
            "mode=train",
            f"data={self.dataset_yaml}",
            f"model={self.model_weights}",
            f"epochs={self.epochs}",
            f"batch={self.batch_size}",
            f"imgsz={self.image_size}",
        ]

        if self.project_name:
            command.append(f"project={self.project_name}")

        return command


def generate_mock_training_configs(count: int = 30) -> list[TrainingConfig]:
    """Create a list of mock ``TrainingConfig`` instances for experimentation.

    Parameters
    ----------
    count:
        Number of mock configurations to generate. Defaults to 30.

    Returns
    -------
    list[TrainingConfig]
        Generated configurations cycling through the supported YOLO tasks.

    Raises
    ------
    ValueError
        If ``count`` is not a positive integer.
    """

    if count <= 0:
        raise ValueError("count must be a positive integer")

    tasks = ("detect", "segment", "classify", "pose")
    configs: list[TrainingConfig] = []

    for index in range(count):
        configs.append(
            TrainingConfig(
                dataset_yaml=f"data/mock_dataset_{index}.yaml",
                model_weights=f"weights/mock_weights_{index}.pt",
                epochs=10 + index,
                batch_size=4 + (index % 8),
                image_size=416 + (index % 5) * 32,
                project_name=f"mock_project_{index}",
                task=tasks[index % len(tasks)],
            )
        )

    return configs


def is_valid_weight_reference(weights_path: str) -> bool:
    """Return ``True`` when the provided weights path or alias can be used."""

    if not weights_path:
        return False

    normalized = weights_path.strip()
    if not normalized:
        return False

    if os.path.exists(normalized):
        return True

    if os.path.dirname(normalized):
        return False

    return normalized in KNOWN_WEIGHT_ALIASES


def list_image_files(folder: str) -> list[str]:
    """Return supported image files inside ``folder`` sorted alphabetically."""

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
    files: list[str] = []
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in supported:
            files.append(path)
    return files


@dataclass
class InferenceStats:
    """Track timing and detection metrics for model inference runs."""

    total_images: int = 0
    total_time: float = 0.0
    last_duration: float = 0.0
    best_duration: float = field(default=float("inf"))
    total_detections: int = 0
    last_detections: Optional[int] = None
    images_with_detections: int = 0

    def record(self, duration: float, detections: Optional[int] = None) -> None:
        """Record a new inference duration and detection count."""

        safe_duration = max(duration, 0.0)
        self.total_images += 1
        self.total_time += safe_duration
        self.last_duration = safe_duration
        if safe_duration < self.best_duration:
            self.best_duration = safe_duration

        if detections is None:
            self.last_detections = None
            return

        safe_detections = max(detections, 0)
        self.last_detections = safe_detections
        self.total_detections += safe_detections
        if safe_detections > 0:
            self.images_with_detections += 1

    @property
    def average_duration(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.total_time / self.total_images

    def describe(self) -> str:
        if self.total_images == 0:
            return "Performance: No inferences run yet."

        best_value = (
            f"{self.best_duration:.2f}" if self.best_duration != float("inf") else "-"
        )
        detection_summary = "Detections: Unknown"
        if self.last_detections is not None:
            detection_summary = f"Detections: {self.last_detections}"

        detected_images = f"With Objects={self.images_with_detections}/{self.total_images}"
        if self.total_images == 0:
            detected_images = "With Objects=0/0"

        return (
            "Performance: "
            f"Runs={self.total_images} | "
            f"Last={self.last_duration:.2f}s | "
            f"Avg={self.average_duration:.2f}s | "
            f"Best={best_value}s | "
            f"{detection_summary} | "
            f"{detected_images}"
        )


@dataclass(frozen=True)
class DetectedObject:
    """Bounding box metadata extracted from YOLO inference results."""

    xyxy: tuple[float, float, float, float]
    label: str
    confidence: Optional[float] = None

    def formatted_label(self) -> str:
        if self.confidence is None:
            return self.label
        return f"{self.label} ({self.confidence:.2f})"


BOUNDING_BOX_COLORS = [
    "#FF6B6B",
    "#6BCB77",
    "#4D96FF",
    "#FFD93D",
    "#9B5DE5",
    "#FF8E00",
]


def _coerce_to_list(value: Any) -> list[Any]:
    """Best-effort conversion of array-like objects to a Python list."""

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            return []

    try:
        return list(value)
    except Exception:
        return []


def collect_box_detections(results: Any) -> list[DetectedObject]:
    """Extract bounding boxes, labels and confidences from YOLO results."""

    try:
        iterable = list(results)
    except TypeError:
        return []

    if not iterable:
        return []

    detections: list[DetectedObject] = []

    for result in iterable:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        xyxy_values = _coerce_to_list(getattr(boxes, "xyxy", None))
        cls_values = _coerce_to_list(getattr(boxes, "cls", None))
        conf_values = _coerce_to_list(getattr(boxes, "conf", None))

        names = getattr(result, "names", None)
        name_lookup = names if isinstance(names, dict) else {}

        for index, bbox in enumerate(xyxy_values):
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            cls_idx: Optional[int] = None
            if index < len(cls_values):
                try:
                    cls_idx = int(cls_values[index])
                except (TypeError, ValueError):
                    cls_idx = None

            confidence: Optional[float] = None
            if index < len(conf_values):
                try:
                    confidence = float(conf_values[index])
                except (TypeError, ValueError):
                    confidence = None

            label = (
                name_lookup.get(cls_idx, str(cls_idx))
                if cls_idx is not None
                else "unknown"
            )

            detections.append(
                DetectedObject(
                    xyxy=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    label=label,
                    confidence=confidence,
                )
            )

    return detections


def parse_detection_count(output: str) -> Optional[int]:
    """Try to extract the number of detections from CLI output."""

    normalized = output.lower()
    patterns = (
        r"boxes=([0-9]+)",
        r"([0-9]+)\s+boxes",
        r"detections=([0-9]+)",
        r"([0-9]+)\s+detections",
    )

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                continue

    return None


def count_predictions_from_results(results: Any) -> Optional[int]:
    """Estimate the number of predictions returned by ultralytics results."""

    if results is None:
        return None

    try:
        iterable = list(results)
    except TypeError:
        return None

    if not iterable:
        return 0

    total = 0
    has_data = False

    for result in iterable:
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            try:
                count = len(boxes)
            except TypeError:
                count = len(getattr(boxes, "data", []) or [])
            total += count
            has_data = True
            continue

        masks = getattr(result, "masks", None)
        if masks is not None:
            try:
                total += len(masks)
            except TypeError:
                total += len(getattr(masks, "data", []) or [])
            has_data = True
            continue

        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None:
            try:
                total += len(keypoints)
            except TypeError:
                total += len(getattr(keypoints, "data", []) or [])
            has_data = True
            continue

        probs = getattr(result, "probs", None)
        if probs is not None:
            has_data = True
            total += 1

    if not has_data:
        return None

    return total


class YOLOTrainerGUI:
    """Tkinter application for configuring and launching YOLO training."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"YOLO Trainer v{get_app_version()}")
        self.root.geometry("720x620")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.training_thread: Optional[threading.Thread] = None
        self.process: Optional[subprocess.Popen[str]] = None
        self.inference_thread: Optional[threading.Thread] = None

        self.test_images: list[str] = []
        self.current_image_index: int = -1
        self.inference_stats = InferenceStats()
        self.detections_by_image: dict[str, list[DetectedObject]] = {}
        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self._preview_photo: Optional[tk.PhotoImage] = None
        self._notified_missing_pillow = False
        self._notified_missing_overlay = False

        self._create_widgets()
        self._start_log_updater()

        if DEFAULT_PRETRAINED_MODEL in KNOWN_WEIGHT_ALIASES:
            self._append_log(
                "Using default YOLO weights alias "
                f"'{DEFAULT_PRETRAINED_MODEL}'. Select custom weights if needed."
            )

    def _create_widgets(self) -> None:
        padding = {"padx": 8, "pady": 4}

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Dataset selection row
        dataset_label = ttk.Label(main_frame, text="Dataset YAML:")
        dataset_label.grid(row=0, column=0, sticky=tk.W, **padding)

        self.dataset_var = tk.StringVar()
        dataset_entry = ttk.Entry(main_frame, textvariable=self.dataset_var, width=60)
        dataset_entry.grid(row=0, column=1, sticky=tk.EW, **padding)

        dataset_button = ttk.Button(
            main_frame, text="Select", command=self._select_dataset
        )
        dataset_button.grid(row=0, column=2, sticky=tk.W, **padding)

        # Model weights selection row
        weights_label = ttk.Label(main_frame, text="Model Weights:")
        weights_label.grid(row=1, column=0, sticky=tk.W, **padding)

        self.weights_var = tk.StringVar(value=DEFAULT_PRETRAINED_MODEL)
        weights_entry = ttk.Entry(main_frame, textvariable=self.weights_var, width=60)
        weights_entry.grid(row=1, column=1, sticky=tk.EW, **padding)

        weights_button = ttk.Button(
            main_frame, text="Select", command=self._select_weights
        )
        weights_button.grid(row=1, column=2, sticky=tk.W, **padding)

        # Training parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Training Parameters", padding=10)
        params_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, **padding)
        params_frame.columnconfigure(1, weight=1)

        self.epochs_var = tk.IntVar(value=100)
        self.batch_var = tk.IntVar(value=16)
        self.image_var = tk.IntVar(value=640)
        self.project_var = tk.StringVar(value="yolo-project")
        self.task_var = tk.StringVar(value="detect")

        self._add_labeled_entry(params_frame, "Epochs", self.epochs_var, 0)
        self._add_labeled_entry(params_frame, "Batch Size", self.batch_var, 1)
        self._add_labeled_entry(params_frame, "Image Size", self.image_var, 2)
        self._add_labeled_entry(
            params_frame, "Project Name", self.project_var, 3, entry_type="text"
        )

        task_label = ttk.Label(params_frame, text="Task")
        task_label.grid(row=4, column=0, sticky=tk.W, **padding)
        task_combo = ttk.Combobox(
            params_frame,
            textvariable=self.task_var,
            values=("detect", "segment", "classify", "pose"),
            state="readonly",
        )
        task_combo.grid(row=4, column=1, sticky=tk.EW, **padding)

        # Model testing frame
        test_frame = ttk.LabelFrame(main_frame, text="Model Testing", padding=10)
        test_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, **padding)
        test_frame.columnconfigure(1, weight=1)

        test_label = ttk.Label(test_frame, text="Image Folder:")
        test_label.grid(row=0, column=0, sticky=tk.W, **padding)

        self.test_folder_var = tk.StringVar()
        test_entry = ttk.Entry(test_frame, textvariable=self.test_folder_var, width=45)
        test_entry.grid(row=0, column=1, sticky=tk.EW, **padding)

        select_folder_btn = ttk.Button(
            test_frame, text="Select", command=self._select_test_folder
        )
        select_folder_btn.grid(row=0, column=2, sticky=tk.W, **padding)

        self.current_image_var = tk.StringVar(value="No image selected.")
        current_image_label = ttk.Label(
            test_frame, textvariable=self.current_image_var, foreground="#555555"
        )
        current_image_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, **padding)

        button_container = ttk.Frame(test_frame)
        button_container.grid(row=2, column=0, columnspan=3, sticky=tk.W, **padding)

        self.prev_button = ttk.Button(
            button_container,
            text="← Previous",
            command=self._show_previous_image,
            state=tk.DISABLED,
        )
        self.prev_button.pack(side=tk.LEFT, padx=4)

        self.solve_button = ttk.Button(
            button_container,
            text="Solve",
            command=self._solve_current_image,
            state=tk.DISABLED,
        )
        self.solve_button.pack(side=tk.LEFT, padx=4)

        self.next_button = ttk.Button(
            button_container,
            text="Next →",
            command=self._show_next_image,
            state=tk.DISABLED,
        )
        self.next_button.pack(side=tk.LEFT, padx=4)

        self.performance_var = tk.StringVar(value=self.inference_stats.describe())
        performance_label = ttk.Label(test_frame, textvariable=self.performance_var)
        performance_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, **padding)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, **padding)

        self.start_button = ttk.Button(
            control_frame, text="Start Training", command=self._start_training
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        stop_button = ttk.Button(control_frame, text="Stop", command=self._stop_training)
        stop_button.pack(side=tk.LEFT, padx=5)

        # CUDA status information
        self.cuda_status_var = tk.StringVar(value="CUDA Support: Checking...")
        cuda_label = ttk.Label(main_frame, textvariable=self.cuda_status_var)
        cuda_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, **padding)

        self._update_cuda_status()

        # Log output
        log_label = ttk.Label(main_frame, text="Training Log")
        log_label.grid(row=6, column=0, columnspan=3, sticky=tk.W, **padding)

        self.log_text = tk.Text(main_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=7, column=0, columnspan=3, sticky=tk.NSEW, **padding)

        scrollbar = ttk.Scrollbar(
            main_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        scrollbar.grid(row=7, column=3, sticky=tk.NS)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1)

    def _add_labeled_entry(
        self,
        frame: ttk.Frame,
        label_text: str,
        variable: tk.Variable,
        row: int,
        entry_type: str = "int",
    ) -> None:
        label = ttk.Label(frame, text=label_text)
        label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)

        if entry_type == "int":
            entry = ttk.Spinbox(
                frame,
                from_=1,
                to=1000,
                textvariable=variable,
                width=10,
            )
        else:
            entry = ttk.Entry(frame, textvariable=variable, width=25)

        entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)

    def _select_dataset(self) -> None:
        path = filedialog.askopenfilename(
            title="Select YOLO dataset YAML",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.dataset_var.set(path)

    def _select_weights(self) -> None:
        path = filedialog.askopenfilename(
            title="Select YOLO weights (.pt)",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.weights_var.set(path)

    def _select_test_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select image folder for testing")
        if not folder:
            return

        images = list_image_files(folder)
        if not images:
            messagebox.showinfo(
                "No images found",
                "The selected folder does not contain any supported image files.",
            )
            return

        self.test_folder_var.set(folder)
        self.test_images = images
        self.current_image_index = 0
        self.inference_stats = InferenceStats()
        self.detections_by_image = {}
        self._update_performance_label()
        self._update_test_controls()
        self._update_current_image_label()

    def _update_test_controls(self) -> None:
        has_images = bool(self.test_images)
        nav_state = tk.NORMAL if len(self.test_images) > 1 else tk.DISABLED

        self.solve_button.configure(state=tk.NORMAL if has_images else tk.DISABLED)
        self.prev_button.configure(state=nav_state)
        self.next_button.configure(state=nav_state)

    def _ensure_preview_window(self) -> None:
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.deiconify()
            return

        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("Image Preview")
        self.preview_window.geometry("960x720")
        self.preview_window.protocol("WM_DELETE_WINDOW", self._on_preview_close)

        self.preview_label = tk.Label(self.preview_window, anchor="center", bg="#202020")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def _on_preview_close(self) -> None:
        self._close_preview_window()

    def _close_preview_window(self) -> None:
        if self.preview_window is None:
            return

        if self.preview_window.winfo_exists():
            self.preview_window.destroy()

        self.preview_window = None
        self.preview_label = None
        self._preview_photo = None

    def _display_current_image(self, image_path: str) -> None:
        detections = self.detections_by_image.get(image_path)
        photo = self._create_photo_image(image_path, detections)
        if photo is None:
            self._close_preview_window()
            return

        self._ensure_preview_window()
        if not self.preview_window or not self.preview_label:
            return

        self.preview_window.title(f"Image Preview - {os.path.basename(image_path)}")
        self._preview_photo = photo
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def _create_photo_image(
        self, image_path: str, detections: Optional[list[DetectedObject]]
    ) -> Optional[tk.PhotoImage]:
        overlay_requested = bool(detections)

        if (
            overlay_requested
            and (
                PIL_IMAGE_MODULE is None
                or PIL_IMAGETK_MODULE is None
                or PIL_IMAGEDRAW_MODULE is None
            )
            and not self._notified_missing_overlay
        ):
            self._append_log(
                "Bounding boxes require Pillow. Install Pillow to view detection overlays."
            )
            self._notified_missing_overlay = True

        if PIL_IMAGE_MODULE is not None and PIL_IMAGETK_MODULE is not None:
            try:
                image = PIL_IMAGE_MODULE.open(image_path).convert("RGB")

                if overlay_requested and detections and PIL_IMAGEDRAW_MODULE is not None:
                    drawer = PIL_IMAGEDRAW_MODULE.Draw(image)
                    for detection in detections:
                        color = BOUNDING_BOX_COLORS[
                            hash(detection.label) % len(BOUNDING_BOX_COLORS)
                        ]

                        x1, y1, x2, y2 = detection.xyxy
                        box_points = [int(x1), int(y1), int(x2), int(y2)]
                        drawer.rectangle(box_points, outline=color, width=3)

                        label_text = detection.formatted_label()
                        if label_text:
                            padding = 4
                            text_bbox = None
                            if hasattr(drawer, "textbbox"):
                                try:
                                    text_bbox = drawer.textbbox((0, 0), label_text)
                                except Exception:
                                    text_bbox = None

                            if text_bbox is not None:
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]
                            else:
                                try:
                                    text_width, text_height = drawer.textsize(label_text)
                                except Exception:
                                    text_width, text_height = (0, 0)

                            background_top = max(int(y1) - text_height - 2 * padding, 0)
                            background = [
                                int(x1),
                                background_top,
                                int(x1) + text_width + 2 * padding,
                                background_top + text_height + 2 * padding,
                            ]
                            drawer.rectangle(background, fill=color)
                            text_position = (
                                int(x1) + padding,
                                background_top + padding,
                            )
                            try:
                                drawer.text(text_position, label_text, fill="#000000")
                            except Exception:
                                drawer.text(text_position, label_text)

                max_size = (960, 720)
                resampling_attr = getattr(PIL_IMAGE_MODULE, "Resampling", None)
                if resampling_attr is not None:
                    resample = resampling_attr.LANCZOS
                else:
                    resample = getattr(PIL_IMAGE_MODULE, "LANCZOS", None)
                    if resample is None:
                        resample = getattr(PIL_IMAGE_MODULE, "ANTIALIAS", None)

                if resample is not None:
                    image.thumbnail(max_size, resample=resample)
                else:
                    image.thumbnail(max_size)

                return PIL_IMAGETK_MODULE.PhotoImage(image=image)
            except Exception as exc:
                self._append_log(
                    "Preview error: Pillow could not load "
                    f"{os.path.basename(image_path)} ({exc}). Falling back to Tkinter loader."
                )

        if PIL_IMAGE_MODULE is None and not self._notified_missing_pillow:
            self._append_log(
                "Install Pillow to enable high-quality previews for additional image formats."
            )
            self._notified_missing_pillow = True

        try:
            return tk.PhotoImage(file=image_path)
        except Exception as exc:
            self._append_log(
                f"Preview error: Unable to display {os.path.basename(image_path)} ({exc})."
            )
            return None

    def _update_current_image_label(self) -> None:
        if not self.test_images or self.current_image_index < 0:
            self.current_image_var.set("No image selected.")
            self._close_preview_window()
            return

        image_path = self.test_images[self.current_image_index]
        self.current_image_var.set(
            f"Image {self.current_image_index + 1}/{len(self.test_images)}: {image_path}"
        )
        self._display_current_image(image_path)

    def _show_next_image(self) -> None:
        if not self.test_images:
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.test_images)
        self._update_current_image_label()

    def _show_previous_image(self) -> None:
        if not self.test_images:
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.test_images)
        self._update_current_image_label()

    def _solve_current_image(self) -> None:
        if not self.test_images:
            messagebox.showinfo("Model Testing", "Please select an image folder first.")
            return

        if self.inference_thread and self.inference_thread.is_alive():
            messagebox.showinfo(
                "Model Testing",
                "Inference is already running. Please wait for it to finish.",
            )
            return

        weights_path = self.weights_var.get().strip()
        if not weights_path:
            messagebox.showerror(
                "Model Testing", "Select model weights before running inference."
            )
            return
        if not is_valid_weight_reference(weights_path):
            messagebox.showerror(
                "Model Testing", "Model weights path does not exist."
            )
            return

        image_path = self.test_images[self.current_image_index]
        self.detections_by_image.pop(image_path, None)
        self._display_current_image(image_path)
        task = self.task_var.get().strip() or "detect"

        command = [
            "yolo",
            f"task={task}",
            "mode=predict",
            f"model={weights_path}",
            f"source={image_path}",
            "save=False",
        ]

        self._append_log(
            "Starting inference with command:\n" f"{shlex.join(command)}\n"
        )

        def run_inference() -> None:
            detection_count: Optional[int] = None
            start_time = time.perf_counter()

            try:
                if YOLO_MODEL_CLASS is not None:
                    try:
                        model = YOLO_MODEL_CLASS(weights_path)
                        results = model.predict(image_path, verbose=False)
                        duration = time.perf_counter() - start_time
                        detection_count = count_predictions_from_results(results)
                        detection_boxes = collect_box_detections(results)
                        if detection_count is None:
                            detection_count = len(detection_boxes)
                        self.detections_by_image[image_path] = detection_boxes
                        self.root.after(
                            0, lambda path=image_path: self._display_current_image(path)
                        )
                        self.inference_stats.record(duration, detection_count)
                        self.log_queue.put(
                            f"Inference completed in {duration:.2f}s for {os.path.basename(image_path)}."
                        )
                        self._log_detection_summary(detection_count, detection_boxes)
                        return
                    except Exception as exc:
                        self.log_queue.put(
                            f"Python inference failed ({exc}). Falling back to CLI execution."
                        )
                        start_time = time.perf_counter()

                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                duration = time.perf_counter() - start_time
                self.detections_by_image.pop(image_path, None)
                if result.stdout:
                    for line in result.stdout.splitlines():
                        self.log_queue.put(line)
                    detection_count = parse_detection_count(result.stdout)

                if result.returncode == 0:
                    self.inference_stats.record(duration, detection_count)
                    self.log_queue.put(
                        f"Inference completed in {duration:.2f}s for {os.path.basename(image_path)}."
                    )
                    self._log_detection_summary(detection_count, None)
                else:
                    self.log_queue.put(
                        f"Inference failed with exit code {result.returncode}."
                    )
            except FileNotFoundError:
                self.log_queue.put(
                    "The 'yolo' command was not found. Install ultralytics and ensure it is on your PATH."
                )
            except Exception as exc:
                self.log_queue.put(f"An unexpected inference error occurred: {exc}")
            finally:
                self.inference_thread = None
                self.root.after(0, self._update_performance_label)

        self.inference_thread = threading.Thread(target=run_inference, daemon=True)
        self.inference_thread.start()

    def _update_performance_label(self) -> None:
        self.performance_var.set(self.inference_stats.describe())

    def _log_detection_summary(
        self,
        detection_count: Optional[int],
        detections: Optional[list[DetectedObject]],
    ) -> None:
        if detection_count is None:
            self.log_queue.put(
                "Detections: Unable to determine. Review the output above for details."
            )
            return

        if detection_count == 0:
            self.log_queue.put("Detections: 0 (no objects detected).")
        elif detection_count == 1:
            self.log_queue.put("Detections: 1 object detected.")
        else:
            self.log_queue.put(f"Detections: {detection_count} objects detected.")

        if not detections:
            return

        class_counts = Counter(det.label for det in detections)
        if not class_counts:
            return

        breakdown = ", ".join(
            f"{label}: {class_counts[label]}" for label in sorted(class_counts)
        )
        self.log_queue.put(f"Detected classes → {breakdown}")

    def _validate_config(self) -> Optional[TrainingConfig]:
        dataset_yaml = self.dataset_var.get().strip()
        model_weights = self.weights_var.get().strip()

        if not dataset_yaml:
            messagebox.showerror("Validation Error", "Please select a dataset YAML file.")
            return None
        if not os.path.exists(dataset_yaml):
            messagebox.showerror("Validation Error", "Dataset YAML path does not exist.")
            return None

        if not model_weights:
            messagebox.showerror("Validation Error", "Please select model weights (.pt).")
            return None
        if not is_valid_weight_reference(model_weights):
            messagebox.showerror("Validation Error", "Model weights path does not exist.")
            return None

        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_var.get())
            image_size = int(self.image_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Validation Error", "Epoch, batch and image size must be integers.")
            return None

        if epochs <= 0 or batch_size <= 0 or image_size <= 0:
            messagebox.showerror(
                "Validation Error", "Epochs, batch size and image size must be positive."
            )
            return None

        project_name = self.project_var.get().strip()
        task = self.task_var.get().strip() or "detect"

        return TrainingConfig(
            dataset_yaml=dataset_yaml,
            model_weights=model_weights,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            project_name=project_name,
            task=task,
        )

    def _start_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Training", "A training process is already running.")
            return

        config = self._validate_config()
        if not config:
            return

        command = config.build_command()
        self._append_log(
            f"Starting training with command:\n{shlex.join(command)}\n"
        )

        self.start_button.state(["disabled"])

        def run_training() -> None:
            try:
                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.log_queue.put(line.rstrip())
                return_code = self.process.wait()
                if return_code == 0:
                    self.log_queue.put("Training completed successfully.")
                else:
                    self.log_queue.put(f"Training failed with exit code {return_code}.")
            except FileNotFoundError:
                self.log_queue.put(
                    "The 'yolo' command was not found. Install ultralytics with "
                    "`pip install ultralytics` and ensure it is on your PATH."
                )
            except Exception as exc:
                self.log_queue.put(f"An unexpected error occurred: {exc}")
            finally:
                self.process = None
                self.training_thread = None
                self.root.after(0, lambda: self.start_button.state(["!disabled"]))

        self.training_thread = threading.Thread(target=run_training, daemon=True)
        self.training_thread.start()

    def _stop_training(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self._append_log("Stop signal sent to training process.")
        else:
            messagebox.showinfo("Training", "No training process is currently running.")

    def _append_log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def _start_log_updater(self) -> None:
        def poll_queue() -> None:
            while True:
                try:
                    line = self.log_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._append_log(line)
            self.root.after(200, poll_queue)

        poll_queue()

    def _update_cuda_status(self) -> None:
        self.cuda_status_var.set(describe_cuda_support())


def main() -> None:
    root = tk.Tk()
    app = YOLOTrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
