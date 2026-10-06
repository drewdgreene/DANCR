# PyInstaller spec for DANCR. Build with: pyinstaller packaging/dancr.spec
# Produces one folder with two programs: DANCR (the window, no console) and
# dancr-cli (console: the command line and the MCP server, which need stdio).
import re
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

root = Path(SPECPATH).parent
version = re.search(r'__version__ = "([^"]+)"', (root / "dancr" / "__init__.py").read_text()).group(1)
# The frozen app has no .py files for the cache to hash, so the code fingerprint is computed here from the source
sys.path.insert(0, str(root))
from dancr.core.executor import CODE_FINGERPRINT
fingerprint = Path(workpath) / "fingerprint.txt"
fingerprint.parent.mkdir(parents=True, exist_ok=True)
fingerprint.write_text(CODE_FINGERPRINT)
datas = [(str(root / "dancr" / "assets"), "dancr/assets"), (str(root / "dancr" / "help"), "dancr/help"),
         (str(fingerprint), "dancr"), (str(root / "AGENTS.md"), ".")]
datas += collect_data_files("pyqtgraph", includes=["**/*.ui", "**/*.png", "**/*.svg"])
# Only the MCP server stack (mcp.cli needs the optional typer extra and must not be pulled in)
hidden = collect_submodules("dancr") + [
    "scipy.optimize", "scipy.special", "fastexcel", "xlsxwriter", "matplotlib.backends.backend_agg",
    "mcp.server.mcpserver", "mcp.server.stdio", "mcp_types"]
# Bundle every capability in the one install (no separate downloads): the geographic, database and
# scientific-array packages ship with the app, so a frozen DANCR never asks the person to add an extra.
extra_datas, extra_binaries, extra_hidden = [], [], []
for pkg in ("shapely", "pyproj", "pyogrio", "sqlalchemy", "psycopg", "psycopg_binary",
            "xarray", "netCDF4", "h5py", "pandas", "cftime", "httpx", "httpcore", "certifi"):
    try:
        d, b, h = collect_all(pkg)
        extra_datas += d
        extra_binaries += b
        extra_hidden += h
    except Exception as exc:                      # a package not installed in this build: skip it
        print(f"dancr.spec: could not collect {pkg}: {exc}")
datas += extra_datas
hidden += extra_hidden

a = Analysis(
    [str(root / "dancr" / "__main__.py")],
    pathex=[str(root)],
    binaries=extra_binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "pyarrow", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtQuick", "PySide6.QtQml", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
icon = str(root / "dancr" / "assets" / "icon.png") if sys.platform != "win32" else str(root / "packaging" / "icon.ico")
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DANCR", console=False, icon=icon)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="dancr-cli", console=True, icon=icon)
coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False, name="DANCR")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="DANCR.app", icon=str(root / "packaging" / "icon.icns"), bundle_identifier="org.dancr.DANCR",
                 info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": version, "CFBundleVersion": version})
