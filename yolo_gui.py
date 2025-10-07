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
import os
import queue
import shlex
import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional


def _load_torch_module() -> Optional[Any]:
    """Load the torch module if it is installed."""

    if importlib.util.find_spec("torch") is None:
        return None

    return importlib.import_module("torch")


TORCH_MODULE = _load_torch_module()


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


class YOLOTrainerGUI:
    """Tkinter application for configuring and launching YOLO training."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YOLO Trainer")
        self.root.geometry("720x520")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.training_thread: Optional[threading.Thread] = None
        self.process: Optional[subprocess.Popen[str]] = None

        self._create_widgets()
        self._start_log_updater()

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

        self.weights_var = tk.StringVar()
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

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, **padding)

        self.start_button = ttk.Button(
            control_frame, text="Start Training", command=self._start_training
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        stop_button = ttk.Button(control_frame, text="Stop", command=self._stop_training)
        stop_button.pack(side=tk.LEFT, padx=5)

        # CUDA status information
        self.cuda_status_var = tk.StringVar(value="CUDA Support: Checking...")
        cuda_label = ttk.Label(main_frame, textvariable=self.cuda_status_var)
        cuda_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, **padding)

        self._update_cuda_status()

        # Log output
        log_label = ttk.Label(main_frame, text="Training Log")
        log_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, **padding)

        self.log_text = tk.Text(main_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=6, column=0, columnspan=3, sticky=tk.NSEW, **padding)

        scrollbar = ttk.Scrollbar(
            main_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        scrollbar.grid(row=6, column=3, sticky=tk.NS)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)

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
        if not os.path.exists(model_weights):
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
