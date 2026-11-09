"""Windows 启动兼容补丁。

PyQt5 的 Windows wheel 在 ``PyQt5/Qt5/bin`` 下自带一套 MSVC 14.26 运行库，
并在导入时把该目录加入 DLL 搜索路径。onnxruntime 1.28 需要系统里的新版
MSVC 运行库；如果 PyQt5 先导入，onnxruntime 会因 DLL 初始化失败而无法加载
（``ImportError: DLL load failed ... initialization routine failed``）。

在导入 PyQt5 之前预加载系统 MSVC 运行库，Windows 加载器会复用已加载的模块，
避免它被 PyQt5 自带的旧版本覆盖。非 Windows 平台直接跳过。
"""

import ctypes
import sys

_RUNTIME_DLLS = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
)


def preload_msvc_runtime():
    if sys.platform != "win32":
        return
    for name in _RUNTIME_DLLS:
        try:
            ctypes.WinDLL(name)
        except OSError:
            pass


preload_msvc_runtime()
