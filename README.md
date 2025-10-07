# YOLO Training GUI

This repository provides a simple Tkinter-based desktop application for
configuring and launching YOLO training runs. Use it to pick a dataset YAML,
select pre-trained weights, and start training with a friendly interface.

## Requirements

- Python 3.9+
- Tkinter (included in most Python distributions)
- [`ultralytics`](https://pypi.org/project/ultralytics/) package providing the
  `yolo` command line interface

Install dependencies:

```bash
pip install ultralytics
```

## Usage

Run the GUI application:

```bash
python yolo_gui.py
```

1. Click **Select** next to "Dataset YAML" to choose your dataset configuration
   file.
2. Click **Select** next to "Model Weights" to pick the pre-trained `.pt`
   weights file.
3. Adjust epochs, batch size, image size, project name, and task as needed.
4. Press **Start Training** to launch the YOLO CLI in the background. Training
   logs stream into the window. Use **Stop** to terminate the run early.

> **Note:** Ensure that the `yolo` executable from the `ultralytics` package is
> available on your PATH before starting training.

## Testing

Run the unit tests to verify the command-building logic for the training
configuration helper:

```bash
python -m unittest discover -s tests
```
