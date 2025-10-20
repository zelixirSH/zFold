# 显式导入关键子模块，使它们成为包的一部分
from . import esm
from . import esm_latest
# 可以继续导入其他需要的子模块，例如：
from . import utils
from . import embeddings

# 可选：定义包的公开接口
__all__ = ['esm', 'esm_latest']  # 明确指定通过 from zfold.network import * 时会导入哪些模块
