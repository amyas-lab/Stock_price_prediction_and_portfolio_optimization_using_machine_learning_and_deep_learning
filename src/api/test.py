# ── Verify all API files created ─────────────────────────────
from pathlib import Path

ROOT_PATH = Path('/Users/cps/DL4AI-240166-project-1')

files_to_check = [
    ROOT_PATH / 'requirements_api.txt',
    ROOT_PATH / 'run_api.py',
    ROOT_PATH / 'test_api.py',
    ROOT_PATH / 'src' / 'api' / '__init__.py',
    ROOT_PATH / 'src' / 'api' / 'config.py',
    ROOT_PATH / 'src' / 'api' / 'models.py',
    ROOT_PATH / 'src' / 'api' / 'main.py',
]

print("API File Status:")
print("─" * 55)
all_exist = True
for f in files_to_check:
    exists = f.exists()
    status = '✓' if exists else '✗ MISSING'
    print(f"  {status} {f.relative_to(ROOT_PATH)}")
    if not exists:
        all_exist = False

print("─" * 55)
if all_exist:
    print("✓ All files present — ready to run API")
    print(f"\nStart server:")
    print(f"  cd {ROOT_PATH}")
    print(f"  python run_api.py")
else:
    print("✗ Some files missing — rerun creation cells")