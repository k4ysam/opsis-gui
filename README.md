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
| **LARK Capture** | Capture physical-space landmarks with a FALCON/LARK optical tracker; compute FRE/TRE against ground-truth `.tag` files; export validation CSV |

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

### Tracker backend selection

Set the `TRACKER` environment variable (or persist via `AppSettings.tracker_mode`):

| Value | Backend |
|---|---|
| `mock` *(default)* | Synthetic circular-path pointer — no hardware needed |
| `plus` | Real PLUS server via OpenIGTLink (port 18944) |
| `lark` | FALCON/LARK optical tracker via pyigtl (port 18995) |

```bash
# Connect to a running FALCON/LARK tracker
set TRACKER=lark
python -m surgical_nav.main
```

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

> **LARK/FALCON**: start `real_multicam.py` with `IGT=True` on port 18995, then launch the GUI with `TRACKER=lark`.

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
│   ├── navigation_page.py         # Stage 3: real-time MPR navigation
│   └── lark_capture_page.py       # Stage 5: LARK live capture + FRE/TRE validation
├── rendering/
│   ├── vtk_widget.py              # QVTKRenderWindowInteractor wrapper
│   ├── slice_viewer.py            # vtkImageReslice pipeline (axial/coronal/sagittal)
│   ├── volume_viewer.py           # vtkSmartVolumeMapper + surface + needle actors
│   ├── reslice_driver.py          # 20 Hz timer driving all slice viewers from tracker
│   ├── paint_brush.py             # Voxel painting in label image
│   └── layout_manager.py          # 2-up / 6-up layout switching
├── tracking/
│   ├── igtl_client.py             # Real pyigtl OpenIGTLink client (QThread); supports device name remapping
│   ├── lark_igtl_client.py        # IGTLClient pre-configured for FALCON/LARK (port 18995, PointerDevice→PointerToTracker)
│   ├── mock_igtl_client.py        # Synthetic circular-path tracker for development
│   ├── transform_store.py         # Thread-safe transform dict
│   └── plus_launcher.py           # PlusServer subprocess launcher
├── calibration/
│   ├── pivot_calibrator.py        # Sphere-fit least-squares pivot calibration
│   └── spin_calibrator.py         # PCA shaft-axis spin calibration
├── registration/
│   ├── landmark_registrar.py      # Umeyama SVD rigid landmark registration
│   ├── validation_engine.py       # FRE/TRE computation against ground-truth .tag files
│   └── surface_registrar.py       # vtkIterativeClosestPointTransform ICP
├── segmentation/
│   ├── threshold_segmenter.py     # SimpleITK threshold + morphology + connected component
│   └── surface_extractor.py       # vtkMarchingCubes → smoothed vtkPolyData
├── dicom/
│   ├── dicom_indexer.py           # pydicom series discovery
│   └── dicom_loader.py            # SimpleITK series → vtkImageData (LPS→RAS)
├── persistence/
│   ├── case_manager.py            # Case save/load (SceneGraph → JSON + .nrrd)
│   └── tag_file_io.py             # FALCON .tag and .xfm file I/O
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

## LARK Integration

The **LARK Capture** stage (Stage 5) connects to a running [FALCON](https://github.com/opsis-medical/FALCON) optical tracking system and lets the user:

1. See the live pointer position in real time
2. Press **Space** to capture physical-space landmark positions
3. Load a ground-truth `.tag` file (paired image-space / physical-space point sets)
4. Run Procrustes registration and compute **FRE** (fiducial registration error) and **TRE** (target registration error) per point
5. Export results as CSV

LARK Capture unlocks immediately after a patient is loaded — it does not require Planning or Registration.

**FALCON `.tag` file format** (one point-pair per line):
```
# x_image y_image z_image  x_physical y_physical z_physical
10.0 20.0 30.0  11.0 19.0 31.0
40.0 50.0 60.0  41.0 49.0 61.0
```

A sample fixture is at `tests/fixtures/sample_ground_truth.tag`.

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
