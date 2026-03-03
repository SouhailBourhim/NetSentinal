# ═══════════════════════════════════════════════════
# FORCE MODULE RELOAD - Add this to your notebook
# ═══════════════════════════════════════════════════
import sys
import importlib
sys.path.append('..')

# Force reload modules to pick up changes
modules_to_reload = [
    'src.data.preprocessor',
    'src.features.engineer', 
    'src.data.splitter'
]

for module in modules_to_reload:
    if module in sys.modules:
        importlib.reload(sys.modules[module])
        print(f"Reloaded {module}")

from src.data.preprocessor import NetworkDataPreprocessor
from src.features.engineer import NetworkFeatureEngineer
from src.data.splitter import DataSplitter

print("✅ Modules loaded with force reload")