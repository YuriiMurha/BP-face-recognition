import sys


def check_import(module_name):
    try:
        __import__(module_name)
        print(f"[OK] {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import {module_name}: {e}")
        return False


modules = [
    "dlib",
    "cv2",
    "tensorflow",
    "mtcnn",
    "face_recognition",
    "psycopg2",
    "numpy",
    "pandas",
]

print(f"Python version: {sys.version}")
print("-" * 40)

failed = []
for mod in modules:
    if not check_import(mod):
        failed.append(mod)

if failed:
    print("-" * 40)
    print(f"Failed modules: {', '.join(failed)}")
    sys.exit(1)
else:
    print("-" * 40)
    print("All core dependencies verified!")
    sys.exit(0)
