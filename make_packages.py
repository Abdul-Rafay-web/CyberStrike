from pathlib import Path

PACKAGES = [
    "core",
    "agents",
    "search",
    "prob",
    "csp",
    "ml",
    "evo",
    "db",
    "runtime",
    "ui",
    "experiments",
    "utils",
    "tests",
]
KEEP_FOLDERS = ["data/maps", "results", "docs"]

root = Path(__file__).parent

for name in PACKAGES:
    folder = root / name
    folder.mkdir(exist_ok=True)
    (folder / "__init__.py").touch()

for name in KEEP_FOLDERS:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".gitkeep").touch()

print("done")
