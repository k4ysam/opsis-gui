# SurgicalNav

A standalone Python image-guided surgical navigation application — a full reimplementation of
[SlicerOpenNav](https://github.com/SlicerIGT/SlicerOpenNav) with no 3D Slicer dependency.

Run on any machine with Python 3.12+, VTK 9, and PySide6.

---

## Features

| Stage | Description |
|---|---|
| **Patients** | Browse DICOM directories, select a series, load into all four views |
| **Planning** | Threshold skin segmentation, paint-brush target segmentation, trajectory line placement, anatomical landmark placement |
| **Registration** | 9-step wizard: PLUS connect → pivot calibration → spin calibration → landmark registration → surface ICP refinement |
| **Navigation** | Real-time MPR slice scrolling, distance-to-target label, trajectory depth gauge, freeze toggle, coloured needle actor |

---

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r surgical_nav/requirements.txt

# 3. Run the application
python -m surgical_nav.main
```

The app opens with a **MockIGTLClient** active (synthetic circular pointer path at 10 Hz) so
all tracker-dependent features work without physical hardware.

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.12+ |
| vtk | 9.6.0 |
| PySide6 | 6.10.2 |
| SimpleITK | 2.5.3 |
| pydicom | 3.0.2 |
| numpy | ≥ 2.0 |
| scipy | ≥ 1.12 |

Optional (real tracker hardware):

```bash
pip install pyigtl
```

---

## Running Tests

```bash
pytest tests/ -v
```

244 tests covering every module. All tests run headless (`QT_QPA_PLATFORM=offscreen`) — no
display or GPU required.

---

## Project Structure

```
surgical_nav/
├── main.py                        # Entry point
├── requirements.txt
├── app/
│   ├── main_window.py             # QMainWindow + QStackedWidget + toolbar
│   └── scene_graph.py             # Singleton scene graph (transforms, volumes, fiducials)
├── workflow/
│   ├── base_page.py               # Abstract WorkflowPage
│   ├── patients_page.py           # Stage 0: DICOM loading
│   ├── planning_page.py           # Stage 1: segmentation + trajectory + landmarks
│   ├── registration_page.py       # Stage 2: 9-step calibration + registration wizard
│   └── navigation_page.py         # Stage 3: real-time MPR navigation
├── rendering/
│   ├── vtk_widget.py              # QVTKRenderWindowInteractor wrapper
│   ├── slice_viewer.py            # vtkImageReslice pipeline (axial/coronal/sagittal)
│   ├── volume_viewer.py           # vtkSmartVolumeMapper + surface + needle actors
│   ├── reslice_driver.py          # 20 Hz timer driving all slice viewers from tracker
│   ├── paint_brush.py             # Voxel painting in label image
│   └── layout_manager.py          # 2-up / 6-up layout switching
├── tracking/
│   ├── igtl_client.py             # Real pyigtl OpenIGTLink client (QThread)
│   ├── mock_igtl_client.py        # Synthetic circular-path tracker for development
│   ├── transform_store.py         # Thread-safe transform dict
│   └── plus_launcher.py           # PlusServer subprocess launcher
├── calibration/
│   ├── pivot_calibrator.py        # Sphere-fit least-squares pivot calibration
│   └── spin_calibrator.py         # PCA shaft-axis spin calibration
├── registration/
│   ├── landmark_registrar.py      # Umeyama SVD rigid landmark registration
│   └── surface_registrar.py       # vtkIterativeClosestPointTransform ICP
├── segmentation/
│   ├── threshold_segmenter.py     # SimpleITK threshold + morphology + connected component
│   └── surface_extractor.py       # vtkMarchingCubes → smoothed vtkPolyData
├── dicom/
│   ├── dicom_indexer.py           # pydicom series discovery
│   └── dicom_loader.py            # SimpleITK series → vtkImageData (LPS→RAS)
└── utils/
    ├── math_utils.py              # compose, invert_transform, rmse, LPS↔RAS
    └── vtk_utils.py               # numpy ↔ vtkImageData helpers
```

---

## Transform Chain

```
PointerToTracker          ← live from OpenIGTLink / PLUS
  └→ POINTER_CALIBRATION  ← pivot calibration result
      └→ IMAGE_REGISTRATION         ← landmark registration result
          └→ IMAGE_REGISTRATION_REFINEMENT  ← ICP surface refinement result
```

All transforms are 4×4 numpy arrays stored in the `SceneGraph` singleton as `TransformNode`
objects.

---

## Architecture Notes

**Thread model** — The `IGTLClient` / `MockIGTLClient` run in a `QThread` (worker-object
pattern) and emit Qt signals. All VTK `Render()` calls happen on the main thread via `QTimer`
— VTK is not thread-safe.

**Scene Graph** — A lightweight Python singleton with a parent-dict transform tree and an
observer/notification pattern. No MRML or Slicer dependency.

**VTK initialisation on Windows** — `QVTKRenderWindowInteractor.Initialize()` must be called
inside `showEvent()`. Both the stacked-widget panel and the viewer panel are initialised there.

**Coordinate system** — RAS throughout. SimpleITK returns LPS; the X and Y axes are flipped
at load time in `DICOMLoader`.
