# Surgical Nav — Task Tracker

## Phase 1: Foundation
- [x] Project structure + requirements.txt
- [ ] utils/math_utils.py + tests
- [ ] app/scene_graph.py + tests
- [ ] rendering/vtk_widget.py
- [ ] rendering/slice_viewer.py
- [ ] rendering/layout_manager.py
- [ ] app/main_window.py + smoke test

## Phase 2: DICOM + Patients Page
- [ ] dicom/dicom_indexer.py + tests
- [ ] dicom/dicom_loader.py + tests
- [ ] rendering/volume_viewer.py
- [ ] workflow/patients_page.py

## Phase 3: Segmentation + Planning
- [ ] segmentation/threshold_segmenter.py + tests
- [ ] segmentation/surface_extractor.py + tests
- [ ] rendering/paint_brush.py
- [ ] workflow/planning_page.py

## Phase 4: Tracking Infrastructure
- [ ] tracking/transform_store.py + tests
- [ ] tracking/igtl_client.py
- [ ] tracking/mock_igtl_client.py
- [ ] tracking/plus_launcher.py

## Phase 5: Calibration
- [ ] calibration/pivot_calibrator.py + tests
- [ ] calibration/spin_calibrator.py + tests

## Phase 6: Registration
- [ ] registration/landmark_registrar.py + tests
- [ ] registration/surface_registrar.py + tests

## Phase 7: Navigation Page
- [ ] rendering/reslice_driver.py
- [ ] workflow/navigation_page.py

## Phase 8: Persistence + Packaging
- [ ] persistence/case_manager.py + tests
- [ ] workflow/landmark_manager_page.py
- [ ] app/settings.py
- [ ] surgical_nav.spec (PyInstaller)
