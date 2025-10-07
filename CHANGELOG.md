# Changelog

All notable changes to this project will be documented in this file.

## [1.5.0] - 2024-05-21
### Added
- Bounding-box overlays with class labels when running model tests so detections
  are highlighted directly in the preview window.
- Log summaries that break detections down by class, making it easier to see
  which objects were found during inference.
- Helper utilities and unit tests for extracting structured detection metadata
  from Ultralytics results, plus GUI updates that refresh overlays after each
  run.

## [1.4.1] - 2024-05-21
### Added
- Default YOLOv8 weights alias pre-populated in the GUI so the app launches
  ready to train or test immediately.
- Validation helper and documentation updates that recognize common YOLO weight
  shortcuts, allowing Ultralytics to download them automatically when needed.

## [1.4.0] - 2024-05-21
### Added
- Automatic image preview window that opens while browsing and testing sample
  images so you can see exactly what the model is evaluating.
- Pillow integration (with graceful fallback) to resize previews and support
  more image formats when available.
- Documentation and requirements updates describing the preview workflow and
  optional dependency.

## [1.3.1] - 2024-05-21
### Added
- Detection summaries in the inference log so you can see when no objects were
  found versus how many were detected.
- Expanded performance banner metrics that highlight the most recent detection
  count and how many test images included objects.
- Helpers and tests for extracting detection counts from both CLI output and
  ultralytics Python results, plus automatic usage when running the GUI.

## [1.3.0] - 2024-05-21
### Added
- Model testing panel with folder selection, navigation controls, and a solve
  button for running YOLO predictions on individual images.
- Inference performance tracker that summarizes run counts alongside last,
  average, and best durations directly in the GUI.
- Utility helpers and tests for listing supported images and reporting
  inference metrics.
- README documentation covering the new evaluation workflow.

## [1.2.1] - 2024-05-21
### Added
- Consolidated quick-start command checklist covering virtual environment setup,
  dependency installation, GUI launch, and running the automated tests.

## [1.2.0] - 2024-05-21
### Added
- Scripted shortcut for creating a Python virtual environment before installing dependencies.
- Documentation updates covering the new helper and activation instructions.

## [1.1.0] - 2024-05-21
### Added
- Convenience scripts for installing dependencies and launching the GUI.
- Application version banner in the window title and a helper to retrieve it programmatically.
- Documented version history and shortcuts in the README.

## [1.0.0] - 2024-05-21
### Added
- Initial release of the YOLO training GUI with CUDA availability indicator and mock configuration helpers.
