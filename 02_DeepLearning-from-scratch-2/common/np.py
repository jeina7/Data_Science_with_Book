from common.config import GPU

if GPU:
    import cupy as np
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    # np.add.at = np.scatter_add
else:
    import numpy as np
