# PyInstaller spec for DANCR. Build with: pyinstaller packaging/dancr.spec
# Produces one folder with two programs: DANCR (the window, no console) and
# dancr-cli (console: the command line and the MCP server, which need stdio).
import re
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH).parent
version = re.search(r'__version__ = "([^"]+)"', (root / "dancr" / "__init__.py").read_text()).group(1)
datas = [(str(root / "dancr" / "assets"), "dancr/assets"),
         (str(root / "docs" / "help.md"), "docs"), (str(root / "docs" / "formulas.md"), "docs"),
         (str(root / "AGENTS.md"), ".")]
datas += collect_data_files("pyqtgraph", includes=["**/*.ui", "**/*.png", "**/*.svg"])
# Only the MCP server stack (mcp.cli needs the optional typer extra and must not be pulled in)
hidden = collect_submodules("dancr") + [
    "scipy.optimize", "scipy.special", "fastexcel", "xlsxwriter", "matplotlib.backends.backend_agg",
    "mcp.server.mcpserver", "mcp.server.stdio", "mcp_types"]

a = Analysis(
    [str(root / "dancr" / "__main__.py")],
    pathex=[str(root)],
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
