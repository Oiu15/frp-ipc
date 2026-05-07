# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH)

a = Analysis(
    ["app.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[(str(ROOT / "config" / "default.yaml"), "config")],
    hiddenimports=[
        "numpy",
        "openpyxl",
        "pandas",
        "pymodbus",
        "pymodbus.client",
        "scipy",
        "serial",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["demo", "tests"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FRP-IPC",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FRP-IPC",
)
