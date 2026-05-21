# scripts/diag.py
import ctypes, os, glob

lib_dir = r"C:\university\diplom\env\Lib\site-packages\torch\lib"
os.add_dll_directory(lib_dir)

# по очереди грузим всё, что есть в torch/lib, и смотрим, что упадёт первым
for path in sorted(glob.glob(os.path.join(lib_dir, "*.dll"))):
    name = os.path.basename(path)
    try:
        ctypes.WinDLL(path)
        print(f"OK    {name}")
    except OSError as e:
        print(f"FAIL  {name}  winerror={e.winerror}  {e.strerror}")