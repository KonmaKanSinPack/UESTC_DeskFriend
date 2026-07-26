import numpy as np

from listen import SileroVadOnnx


def test_vad_silence_scores_low():
    vad = SileroVadOnnx(16000)
    silence = np.zeros(512, dtype=np.float32)
    assert vad(silence) < 0.1


def test_vad_reset_restores_state():
    vad = SileroVadOnnx(16000)
    chunk = np.zeros(512, dtype=np.float32)
    vad(chunk)  # 状态被推进
    vad.reset()
    assert np.all(vad.state == 0)
    assert np.all(vad.context == 0)
