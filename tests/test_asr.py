"""ASR 引擎层单测：工厂选择/回退、下载器、假 recognizer 的转写链。不加载真模型。"""

import tarfile

import numpy as np
import pytest

import asr as asr_mod
from asr import SenseVoiceEngine, create_asr


class FakeRecognizer:
    """sherpa OfflineRecognizer 最小替身：记录喂入音频，吐预设文本。"""

    def __init__(self, text="你好呀"):
        self.text = text
        self.fed = []

    def create_stream(self):
        return self

    def accept_waveform(self, sample_rate, samples):
        self.fed.append((sample_rate, np.asarray(samples)))

    def decode(self, stream):
        pass

    @property
    def result(self):
        class R:
            text = f"  {self.text}  "  # 带空白验证 strip

        return R()


# ---------- 工厂 ----------


def test_factory_defaults_to_sensevoice(monkeypatch):
    monkeypatch.setattr(asr_mod, "SenseVoiceEngine", lambda model_dir=None: "SV")
    assert create_asr() == "SV"
    assert create_asr({"ASR_BACKEND": "sensevoice"}) == "SV"


def test_factory_whisper_backend(monkeypatch):
    called = {}

    def fake_whisper():
        called["w"] = True
        return "W"

    monkeypatch.setattr(asr_mod, "WhisperEngine", fake_whisper)
    assert create_asr({"ASR_BACKEND": "whisper"}) == "W"
    assert called["w"] is True


def test_factory_falls_back_to_whisper(monkeypatch):
    """SenseVoice 不可用（缺依赖/模型）→ 回退 Whisper，不抛异常。"""

    def broken_sv(model_dir=None):
        raise ImportError("no sherpa-onnx")

    monkeypatch.setattr(asr_mod, "SenseVoiceEngine", broken_sv)
    monkeypatch.setattr(asr_mod, "WhisperEngine", lambda: "W")
    assert create_asr({"ASR_BACKEND": "sensevoice"}) == "W"


# ---------- SenseVoiceEngine ----------


def test_sensevoice_transcribe_pipeline():
    """转写链：建流 → 喂音频（含 0.3s 尾部补零）→ 解码 → strip 返回。"""
    fake = FakeRecognizer("今天天气不错。")
    engine = SenseVoiceEngine(recognizer=fake)
    audio = np.ones(1600, dtype=np.float32) * 0.5
    text = engine.transcribe(audio, sample_rate=16000)
    assert text == "今天天气不错。"
    sr, fed = fake.fed[0]
    assert sr == 16000
    assert len(fed) == 1600 + int(0.3 * 16000)  # 尾部补零长度


# ---------- 下载器 ----------


def test_download_skips_when_files_exist(tmp_path):
    (tmp_path / "model.int8.onnx").write_bytes(b"x")
    (tmp_path / "tokens.txt").write_bytes(b"y")
    assert asr_mod.download_sensevoice_model(tmp_path) is True


def test_download_failure_returns_false(tmp_path, monkeypatch):
    def broken_urlretrieve(url, path, reporthook=None):
        raise OSError("network down")

    monkeypatch.setattr(asr_mod.urllib.request, "urlretrieve", broken_urlretrieve)
    assert asr_mod.download_sensevoice_model(tmp_path) is False


def test_download_extracts_int8_and_tokens_only(tmp_path, monkeypatch):
    """下载后解包：只提取 model.int8.onnx 与 tokens.txt（展平），压缩包删除。"""
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "model.int8.onnx").write_bytes(b"int8")
    (inner / "model.onnx").write_bytes(b"fp32")  # 不要的大模型
    (inner / "tokens.txt").write_bytes(b"tok")
    tar_path = tmp_path / "fake.tar.bz2"
    with tarfile.open(tar_path, "w:bz2") as tar:
        tar.add(inner / "model.int8.onnx", arcname="sense-voice-2024/model.int8.onnx")
        tar.add(inner / "model.onnx", arcname="sense-voice-2024/model.onnx")
        tar.add(inner / "tokens.txt", arcname="sense-voice-2024/tokens.txt")

    target = tmp_path / "target"
    target.mkdir()

    def fake_urlretrieve(url, path, reporthook=None):
        # 模拟下载：把预制的 tar 拷到目标路径
        import shutil

        shutil.copy(tar_path, path)

    monkeypatch.setattr(asr_mod.urllib.request, "urlretrieve", fake_urlretrieve)
    assert asr_mod.download_sensevoice_model(target) is True
    assert (target / "model.int8.onnx").read_bytes() == b"int8"
    assert (target / "tokens.txt").read_bytes() == b"tok"
    assert not (target / "model.onnx").exists()  # fp32 大模型不提取
    assert not (target / "sense-voice.tar.bz2").exists()  # 压缩包已删


@pytest.mark.parametrize("missing", ["model", "tokens"])
def test_download_missing_one_file_fails(tmp_path, monkeypatch, missing):
    """两文件缺一 = 走下载路径；下载被断网桩拦住返回 False（不触发真实网络）。"""

    def broken(url, path, reporthook=None):
        raise OSError("network down")

    monkeypatch.setattr(asr_mod.urllib.request, "urlretrieve", broken)
    if missing != "model":
        (tmp_path / "model.int8.onnx").write_bytes(b"x")
    if missing != "tokens":
        (tmp_path / "tokens.txt").write_bytes(b"y")
    assert asr_mod.download_sensevoice_model(tmp_path) is False
