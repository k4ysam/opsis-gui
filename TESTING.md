# SurgicalNav — Manual Testing Guide

End-to-end walkthrough for testing every stage of the GUI by hand.

---

## Prerequisites

### 1. Run the application

```bash
cd "C:\Users\Samaksh\Documents\test\gui"
venv\Scripts\activate
python -m surgical_nav.main
```

The app opens with a **MockIGTLClient** already running — it simulates a tracked pointer
tracing a 50 mm circular path at 10 Hz. No physical hardware is needed to test any stage.

### 2. Get a test DICOM dataset

You need a folder of `.dcm` files from a single CT series. Free public options:

| Dataset | Link | Notes |
|---|---|---|
| **3D-IRCADb-01** | [ircad.fr/research/3dircadb](https://www.ircad.fr/research/3dircadb/3dircadb1/) | Free registration; clear liver/bone anatomy |
| **RIDER Lung CT** | [cancerimagingarchive.net](https://www.cancerimagingarchive.net/) → search "RIDER Lung CT" | Public, no login required |
| **OsiriX DICOM Library** | [osirix-viewer.com/resources/dicom-image-library](https://www.osirix-viewer.com/resources/dicom-image-library/) | Multiple phantoms available |
| **3D Slicer sample data** | Open 3D Slicer → Welcome → Download Sample Data → "CTChest" | If Slicer is already installed |

Any single-series CT folder works. Extract to somewhere accessible (e.g. `C:\TestData\CT\`).

---

## Stage 0 — Patients

**Goal:** Load a DICOM series into all four views and create a named case.

| # | Action | Expected result |
|---|---|---|
| 1 | Click **Browse…** | File dialog opens |
| 2 | Select the folder containing your `.dcm` files | Series table populates — patient name, series description, slice count |
| 3 | Click a row in the table | Row highlights; Load button enables |
| 4 | Type a case name in the text field (e.g. `TestCase01`) | |
| 5 | Click **Load** | Progress shown; all 4 viewers (3D + axial + coronal + sagittal) display the volume |
| 6 | Check the window title bar | Case name appears |
| 7 | Check status bar at the bottom | Stage complete message |
| 8 | Check the toolbar | **Planning** button unlocks |
| 9 | Verify auto-save | `~/OpenNav/Cases/TestCase01/metadata.json` and `volume.nii.gz` exist on disk |

---

## Stage 1 — Planning

The app auto-navigates here after loading. Work through Steps 1–4.

### Step 1 — Skin Segmentation

| # | Action | Expected result |
|---|---|---|
| 1 | Adjust **HU Low / HU High** sliders to bracket skin (try −200 to +500 for CT) | |
| 2 | Click **Segment Skin** | Progress bar runs; semi-transparent skin mesh appears in 3D view |
| 3 | Rotate 3D view (left-click drag) | Mesh wraps the body outline correctly |

### Step 2 — Target Segmentation

| # | Action | Expected result |
|---|---|---|
| 1 | Click **Step 2: Target** | Step 2 panel opens |
| 2 | Click a bright structure on any slice (bone or lesion) | Seed placed; ConnectedThreshold region grows |
| 3 | Target mesh appears in 3D view in a distinct colour | |

### Step 3 — Trajectory

| # | Action | Expected result |
|---|---|---|
| 1 | Click **Place Entry** then click a location on any slice | Entry point placed |
| 2 | Click **Place Target** then click a different slice location | Target point placed |
| 3 | Check 3D view | Yellow line connects Entry → Target |

### Step 4 — Landmarks (minimum 3 required)

| # | Action | Expected result |
|---|---|---|
| 1 | Click **Place Landmark** then click an identifiable anatomical point | Row added to landmark table |
| 2 | Repeat for 2 more distinct points | 3 rows in table |
| 3 | Check the status label | Shows ≥ 3 landmarks |
| 4 | **Complete Planning** button enables | |
| 5 | Click **Complete Planning** | Auto-saves; Registration unlocks; app navigates to Registration |

---

## Stage 2 — Registration (9-step wizard)

The MockIGTLClient is already running — the PLUS indicator in the top-right toolbar should be green.

### Steps 1–2 — PLUS Connect and Tool Verify

These are informational stubs. Click **Next** on each. Confirm the status lights show green (Pointer) and yellow/green (HeadFrame).

### Steps 3–4 — Pivot Calibration

| # | Action | Expected result |
|---|---|---|
| 1 | Click **Start Collection** | Progress bar begins filling automatically |
| 2 | Wait ~15 seconds | Bar reaches 150 samples (mock client feeds data continuously) |
| 3 | Click **Calibrate** | RMSE displayed — should be < 0.8 mm with mock data |
| 4 | Click **Accept** | `POINTER_CALIBRATION` stored; next step unlocks |

### Step 5 — Spin Calibration

Same flow as pivot: **Start Collection → wait → Calibrate → check RMSE < 1.0 mm → Accept**.

### Step 6 — Landmark Registration

| # | Action | Expected result |
|---|---|---|
| 1 | Left column shows your 3+ planning landmarks (image coordinates) | |
| 2 | For each row click **Collect Physical Point** | Physical coordinate fills in from mock pointer position |
| 3 | Once all pairs collected, click **Register** | RMSE shown — should be < 3.0 mm |
| 4 | Click **Accept** | `IMAGE_REGISTRATION` node stored |

### Steps 7–8 — Surface Trace and ICP

| # | Action | Expected result |
|---|---|---|
| 1 | Click **Start Surface Trace** | Collection mode active |
| 2 | Click 20–30 points on the skin surface in the slice views | Points accumulate in the table |
| 3 | Click **Run ICP** | Mean closest-point distance shown — should be < 3.0 mm |
| 4 | Click **Accept Registration** | `IMAGE_REGISTRATION_REFINEMENT` stored; Navigation unlocks |

### Step 9 — Verification

Review the final RMSE summary. Click **Go to Navigation**.

---

## Stage 3 — Navigation

| # | Action | Expected result |
|---|---|---|
| 1 | App auto-navigates here | All 4 views still show the volume |
| 2 | Watch **Distance to target** label | Updates at 10 Hz as mock pointer moves |
| 3 | Watch **Depth along trajectory** label | Shows depth / total length in mm |
| 4 | Observe needle actor in 3D view | Cylinder traces a circle; colour is green (SEEN) |
| 5 | Observe slice viewers | Axial / coronal / sagittal scroll in sync with pointer |
| 6 | Check 3D view | Yellow trajectory line (Entry → Target) is visible |
| 7 | Click **Freeze** | Slices stop scrolling; button reads "Unfreeze" |
| 8 | Click **Unfreeze** | Slices resume scrolling |

---

## Stage 4 — Landmark Manager

| # | Action | Expected result |
|---|---|---|
| 1 | Navigate to Landmark Manager via toolbar | Tabs for `PLANNING_LANDMARKS` and `TRAJECTORY_POINTS` |
| 2 | Click the `PLANNING_LANDMARKS` tab | Table shows label, X, Y, Z for each point |
| 3 | Select a row, click **Delete Selected** | Row removed; count in status label updates |
| 4 | Click **Export CSV…** | Save dialog opens; choose a path |
| 5 | Open the exported file in a text editor or Excel | Header row `label,x_mm,y_mm,z_mm` + one data row per point |
| 6 | Click **Refresh** | Table reloads from current scene state |

---

## Persistence Verification

Auto-save fires at the end of every stage. To verify what was written:

```
~/OpenNav/Cases/TestCase01/
    metadata.json
    volume.nii.gz
    transforms/
        POINTER_CALIBRATION.npy
        IMAGE_REGISTRATION.npy
        IMAGE_REGISTRATION_REFINEMENT.npy
    landmarks/
        PLANNING_LANDMARKS.json
        TRAJECTORY_POINTS.json
```

To verify a full round-trip from the Python REPL:

```python
from surgical_nav.app.scene_graph import SceneGraph
from surgical_nav.persistence.case_manager import CaseManager

SceneGraph.reset()
mgr = CaseManager()
print(mgr.list_cases())                         # ['TestCase01', ...]

meta = mgr.load_case("TestCase01", SceneGraph.instance())
print(meta)                                     # {case_name, created, modified, stage}

node = SceneGraph.instance().get_node("IMAGE_REGISTRATION")
print(node.matrix)                              # 4×4 numpy array
```

---

## Automated Test Suite

All 294 unit and integration tests run headless (no display needed):

```bash
pytest tests/ -v
```

Individual modules:

```bash
pytest tests/test_case_manager.py -v        # persistence
pytest tests/test_settings.py -v            # QSettings
pytest tests/test_landmark_manager_page.py -v
pytest tests/test_navigation_page.py -v
pytest tests/test_reslice_driver.py -v
pytest tests/test_registration_page.py -v
pytest tests/test_pivot_calibrator.py -v
pytest tests/test_landmark_registrar.py -v
pytest tests/test_surface_registrar.py -v
```

---

## Quick Smoke-Test Checklist

Work through this top-to-bottom after any significant code change:

```
[ ] App launches without errors or console tracebacks
[ ] DICOM series loads and all 4 views render the volume
[ ] Skin segmentation produces a visible mesh in 3D view
[ ] Trajectory Entry/Target line appears in 3D view
[ ] 3+ landmarks placed; Planning stage completes
[ ] Pivot calibration runs to 150 samples; RMSE < 0.8 mm
[ ] Landmark registration RMSE < 3.0 mm
[ ] Navigation: slices scroll with mock pointer motion
[ ] Navigation: Distance-to-target label updates live at ~10 Hz
[ ] Navigation: Freeze/Unfreeze works correctly
[ ] Landmark Manager: CSV export produces a valid file
[ ] ~/OpenNav/Cases/<name>/ contains expected files after each stage
[ ] pytest tests/ — 294 passed, 0 failed
```

---

## Known Limitations

| Item | Status |
|---|---|
| Real PLUS / optical tracker | Not tested — MockIGTLClient substitutes throughout |
| Physical landmark collection | Simulated by mock pointer position at time of click |
| "Load case" in the GUI | Not yet implemented — use the Python REPL snippet above |
| PyInstaller `.exe` build | Spec written (`surgical_nav.spec`) but build not yet verified |
| Slice click-to-place fiducials | Requires mouse events on VTK widget — not yet wired |
