"""TTS 语音输出测试：抽象契约、Dummy 占位行为、打断位置记录、工厂装配、模型自动下载。"""

import asyncio

from tts import COSYVOICE2_HF_REPO, DEFAULT_MODEL_DIR, CosyVoice2TTS, DummyTTS, create_tts, ensure_cosyvoice_model


class TestDummyTTS:
    def test_speak_records_and_completes(self):
        tts = DummyTTS(sentence_gap=0)
        asyncio.run(tts.speak("你好呀。今天怎么样？"))
        assert tts.speak_calls == ["你好呀。今天怎么样？"]
        assert tts.busy is False
        assert tts.played_text == "你好呀。今天怎么样？"

    def test_interrupt_mid_speech_returns_played_prefix(self):
        """播放中打断 → 返回已播放文本前缀（打断位置记号）。"""
        tts = DummyTTS(sentence_gap=0.05)

        async def run():
            task = asyncio.create_task(tts.speak("第一句。第二句。第三句。"))
            await asyncio.sleep(0.08)  # 播完第一句、第二句播到一半
            prefix = await tts.interrupt()
            await task  # speak 的 finally 会收尾（busy=False）
            return prefix

        prefix = asyncio.run(run())
        assert tts.busy is False
        assert "第一句" in prefix  # 已播放部分含第一句

    def test_interrupt_when_idle_returns_empty(self):
        tts = DummyTTS()
        assert asyncio.run(tts.interrupt()) == ""

    def test_speak_empty_ignored(self):
        tts = DummyTTS()
        asyncio.run(tts.speak(""))
        assert tts.speak_calls == []


class TestCosyVoice2Stub:
    def test_construct_and_speak_no_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tts.ensure_cosyvoice_model", lambda d: tmp_path)  # 防测试触发真下载
        tts = CosyVoice2TTS(voice_ref="ref.wav", voice_ref_text="参考文本")
        asyncio.run(tts.speak("桃桃在呢"))
        assert tts.speak_calls == ["桃桃在呢"]
        assert tts.busy is False


class TestEnsureModel:
    def test_skips_download_when_model_exists(self, tmp_path):
        (tmp_path / "model.pt").touch()
        result = ensure_cosyvoice_model(str(tmp_path))
        assert result == tmp_path  # 不触发下载

    def test_downloads_when_missing(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda repo_id, local_dir: calls.append((repo_id, local_dir)),
        )
        ensure_cosyvoice_model(str(tmp_path))
        assert calls == [(COSYVOICE2_HF_REPO, str(tmp_path))]

    def test_default_dir_constant(self):
        assert DEFAULT_MODEL_DIR == "assets/models/CosyVoice2-0.5B"


class TestCreateTTS:
    def test_dummy_default(self):
        assert isinstance(create_tts({}), DummyTTS)

    def test_none_backend_uses_dummy(self):
        assert isinstance(create_tts({"TTS_BACKEND": "none"}), DummyTTS)

    def test_cosyvoice2_stub(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tts.ensure_cosyvoice_model", lambda d: tmp_path)  # 防测试触发真下载
        tts = create_tts({"TTS_BACKEND": "cosyvoice2", "TTS_VOICE_REF": "ref.wav"})
        assert isinstance(tts, CosyVoice2TTS)
        assert tts.voice_ref == "ref.wav"

    def test_unknown_backend_raises(self):
        try:
            create_tts({"TTS_BACKEND": "bogus"})
            assert False, "应抛 ValueError"
        except ValueError:
            pass
