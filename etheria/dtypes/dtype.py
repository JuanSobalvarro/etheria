from enum import Enum
from etheria._eth import _DType

class DType(Enum):
    FLOAT32 = _DType.FLOAT32
    FLOAT64 = _DType.FLOAT64
    INT16 = _DType.INT16
    INT32 = _DType.INT32
    INT64 = _DType.INT64
    BOOL = _DType.BOOL

    @staticmethod
    def from_core_dtype(core_dtype: _DType):
        try:
            return DType(core_dtype)
        except ValueError:
            return None
