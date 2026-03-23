# surgical_nav.spec — PyInstaller build spec
#
# Build with:
#   pyinstaller surgical_nav.spec
#
# Output: dist/surgical_nav/surgical_nav.exe  (--onedir)

import sys
from pathlib import Path
import vtkmodules

block_cipher = None

# ---------------------------------------------------------------------------
# Collect VTK data files (lookup tables, shaders, etc.)
# ---------------------------------------------------------------------------
vtk_data_dir = Path(vtkmodules.__file__).parent

vtk_datas = []
for suffix in ("*.py", "*.pyi", "*.pyd", "*.dll", "*.so"):
    for f in vtk_data_dir.rglob(suffix):
        rel = f.relative_to(vtk_data_dir.parent)
        vtk_datas.append((str(f), str(rel.parent)))

# ---------------------------------------------------------------------------
# Application assets
# ---------------------------------------------------------------------------
app_datas = [
    ("surgical_nav/assets", "surgical_nav/assets"),
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ["surgical_nav/main.py"],
    pathex=["."],
    binaries=[],
    datas=vtk_datas + app_datas,
    hiddenimports=[
        # VTK
        "vtkmodules",
        "vtkmodules.all",
        "vtkmodules.util",
        "vtkmodules.util.numpy_support",
        "vtkmodules.qt.QVTKRenderWindowInteractor",
        # PySide6 extras often missed
        "PySide6.QtSvg",
        "PySide6.QtPrintSupport",
        # SimpleITK
        "SimpleITK",
        # pydicom
        "pydicom",
        "pydicom.encoders",
        "pydicom.encoders.gdcm",
        # optional pyigtl
        "pyigtl",
        # numpy / scipy
        "numpy",
        "scipy",
        "scipy.spatial",
        "scipy.linalg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="surgical_nav",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # windowed app — no console
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="surgical_nav",
)
