"""Make packaging/icon.ico and icon.icns from dancr/assets/icon.png (needs Pillow)."""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent
src = Image.open(root.parent / "dancr" / "assets" / "icon.png").convert("RGBA")
src.save(root / "icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
src.save(root / "icon.icns")
print("wrote icon.ico and icon.icns")
