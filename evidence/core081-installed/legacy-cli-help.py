import runpy
import sys
from pathlib import Path
import torch

torch.set_num_threads(1);torch.set_num_interop_threads(1)
script=Path(sys.executable).parent/sys.argv[1]
sys.argv=[str(script),'--help']
try:runpy.run_path(str(script),run_name='__main__')
except SystemExit as exc:assert exc.code==0
assert not torch.cuda.is_initialized()
print('Installed console help contract passed; CUDA initialized: False')
