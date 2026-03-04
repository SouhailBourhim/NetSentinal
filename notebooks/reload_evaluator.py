# ═══════════════════════════════════════════════════
# FORCE RELOAD ROBUST EVALUATOR - Add this to your notebook
# ═══════════════════════════════════════════════════
import sys
import importlib
sys.path.append('..')

# Force reload the robust evaluator module to pick up changes
modules_to_reload = [
    'src.models.robust_evaluator',
    'src.models.base_model',
    'src.models.xgboost_model',
    'src.models.random_forest',
    'src.models.isolation_forest',
    'src.models.autoencoder'
]

for module in modules_to_reload:
    if module in sys.modules:
        importlib.reload(sys.modules[module])
        print(f"Reloaded {module}")

from src.models.robust_evaluator import RobustEvaluator

print("✅ RobustEvaluator module reloaded with latest changes")