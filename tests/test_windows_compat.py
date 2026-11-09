import listen
import win_compat


class FakeStream:
    def start(self):
        pass

    def read(self, frames):
        raise RuntimeError("测试中不读取麦克风")

    def close(self):
        pass


def _patch_audio(monkeypatch):
    monkeypatch.setattr(listen.sd, "RawInputStream", lambda **kwargs: FakeStream())
    monkeypatch.setattr(listen.Listen, "start_threading", lambda self: None)


def test_whisper_cuda_init_failure_falls_back_to_cpu(monkeypatch):
    created = []

    class FakeWhisperModel:
        def __init__(self, name, device, compute_type, local_files_only=False):
            created.append((device, compute_type, local_files_only))
            if device == "cuda":
                raise RuntimeError("cublas64_12.dll not found")

    _patch_audio(monkeypatch)
    monkeypatch.setattr(listen.ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(listen, "WhisperModel", FakeWhisperModel)

    listen.Listen()

    assert created == [("cuda", "float16", True), ("cpu", "int8", True)]


def test_whisper_cache_miss_retries_online(monkeypatch):
    created = []

    class FakeWhisperModel:
        def __init__(self, name, device, compute_type, local_files_only=False):
            created.append((device, compute_type, local_files_only))
            if local_files_only:
                raise FileNotFoundError("not cached")

    _patch_audio(monkeypatch)
    monkeypatch.setattr(listen.ctranslate2, "get_cuda_device_count", lambda: 0)
    monkeypatch.setattr(listen, "WhisperModel", FakeWhisperModel)

    listen.Listen()

    assert created == [("cpu", "int8", True), ("cpu", "int8", False)]


def test_preload_msvc_runtime_is_noop_on_non_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(win_compat.sys, "platform", "linux")
    monkeypatch.setattr(win_compat.ctypes, "WinDLL", lambda name: calls.append(name), raising=False)

    win_compat.preload_msvc_runtime()

    assert calls == []
