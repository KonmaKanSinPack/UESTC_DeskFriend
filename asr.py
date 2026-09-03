"""语音识别（ASR）引擎层：与 tts.py 对称的纯工具层，供 listen.py（耳器官）装配。

- ASR 抽象 + 双实现 + 工厂：SenseVoice（本地 sherpa-onnx，默认）/ Whisper（回退）
- 引擎切换是耳器官内部细节：listen 对外接口（text_signal 等）不因引擎而变
- 不 import 任何器官/大脑（纯工具层，同 tts.py）

为什么换 SenseVoice（2026-09-03）：whisper-small 的中文识别弱（同音字/专有名词/
口语填充词，CPU int8 量化再损一档）。SenseVoice-Small 中文准确率显著更高、CPU 推理
更快（纯 onnxruntime 无 torch，与"依赖瘦身"决策一致），自带标点与 ITN（数字/日期
阿拉伯化）。Whisper 保留为回退，并补 initial_prompt 压繁体/翻译腔。
"""

import socket
import tarfile
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "assets" / "models" / "sense-voice"
# 模型源分两级（本机实测 2026-09-03）：
# 1) 首选逐文件直链：hf-mirror（国内可达，实测 939KB tokens/263MB 模型均可下）
#    → huggingface.co（海外机器）。仓库是官方 2024-07-17 转换的两文件直传，
#    免 tar 解包。
# 2) 兜底 GitHub release tar.bz2（含 fp32/int8 两版 onnx，只提取 int8 与词表）。
#    本机实测其资产主机 GET 超时（HEAD 却可达），故只作最后手段。
MODEL_REPO = "HatiSkoll28/sherpa-onnx-sense-voice-int8"
MODEL_FILE_URLS = {
    "model.int8.onnx": [
        f"https://hf-mirror.com/{MODEL_REPO}/resolve/main/model.int8.onnx",
        f"https://huggingface.co/{MODEL_REPO}/resolve/main/model.int8.onnx",
    ],
    "tokens.txt": [
        f"https://hf-mirror.com/{MODEL_REPO}/resolve/main/tokens.txt",
        f"https://huggingface.co/{MODEL_REPO}/resolve/main/tokens.txt",
    ],
}
MODEL_DOWNLOAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
)


@contextmanager
def _ipv4_only():
    """下载期间强制 IPv4 解析（用完恢复，不污染全局网络栈）。

    本机实测（2026-09-03）：IPv6 路由到模型 CDN 会被重置（WinError 10054），
    且无 UA 的请求被 CDN 403；「IPv4 + UA」组合 200 可达（curl 能下同 URL 正因
    它走 IPv4 回退 + 自带 UA）。作用域收窄到单次下载：应用其它网络（LLM/TTS/桥）
    不受影响；仅首次下载模型时会短暂经过本上下文。
    """
    orig = socket.getaddrinfo

    def _v4(*args, **kwargs):
        return [r for r in orig(*args, **kwargs) if r[0] == socket.AF_INET]

    socket.getaddrinfo = _v4
    try:
        yield
    finally:
        socket.getaddrinfo = orig


def _http_download(url, dest: Path) -> None:
    """带 UA + IPv4 的流式下载（失败抛异常，由调用方决定回退）。

    为什么不用 urlretrieve：hf-mirror 的 CDN 实测拒绝 Python-urllib 默认 UA
    （WinError 10054 / 403；同 URL curl 可下）。每 ~10% 打一次进度；
    Content-Length 缺失时跳过进度。
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (UESTC-DeskFriend)"})
    with _ipv4_only(), urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last_pct = 0, -1
        while True:
            block = resp.read(1 << 16)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                pct = done * 100 // total
                if pct != last_pct and pct % 10 == 0:
                    last_pct = pct
                    print(f"下载进度：{pct}%")


def _download_first_available(urls, target: Path) -> bool:
    """按序尝试直链下载单个文件到 target；任一成功即返回 True。

    失败时清掉半截文件再试下一源——下载中途断开会留下不完整的
    model.int8.onnx，若不清，下次启动会把它当完整模型加载然后神秘报错。
    """
    for url in urls:
        try:
            print(f"下载模型文件：{url}")
            _http_download(url, target)
            print("下载完成")
            return True
        except Exception as e:
            print(f"该源失败（{e}），尝试下一个...")
            target.unlink(missing_ok=True)
    return False


def _download_and_extract_tar(model_dir: Path, model_path: Path, tokens_path: Path) -> bool:
    """兜底路径：下载 GitHub release tar.bz2 并只提取 int8 模型与词表两文件。"""
    tar_path = model_dir / "sense-voice.tar.bz2"
    print(f"直链均不可用，尝试 GitHub release tar：{MODEL_DOWNLOAD_URL}")
    try:
        _http_download(MODEL_DOWNLOAD_URL, tar_path)
        print("下载完成，解压模型文件...")
    except Exception as e:
        print(f"tar 下载失败（{e}）")
        tar_path.unlink(missing_ok=True)
        return False
    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            for member in tar.getmembers():
                # 按 basename 匹配并展平到目标目录（tar 内带一层版本目录名）
                if Path(member.name).name in ("model.int8.onnx", "tokens.txt"):
                    member.name = Path(member.name).name
                    tar.extract(member, model_dir, filter="data")
    except Exception as e:
        print(f"模型解压失败（{e}）")
        return False
    finally:
        tar_path.unlink(missing_ok=True)  # 压缩包不留盘（~250MB）
    return model_path.exists() and tokens_path.exists()


def download_sensevoice_model(model_dir=DEFAULT_MODEL_DIR) -> bool:
    """确保 SenseVoice 模型文件存在，缺失则自动下载；成功返回 True。

    复现/原理：模型不进 git（~230MB，assets/models/ 已 gitignore）。首次启动检测
    两个文件（model.int8.onnx + tokens.txt）缺失 → 逐文件直链下载（hf-mirror 优先）
    → 直链全败再走 GitHub tar 兜底（只提取需要的两文件）。有缓存即完全离线
    （启动零下载原则不破坏）；任何失败返回 False，调用方回退 Whisper。
    """
    model_dir = Path(model_dir)
    model_path = model_dir / "model.int8.onnx"
    tokens_path = model_dir / "tokens.txt"
    if model_path.exists() and tokens_path.exists():
        return True

    model_dir.mkdir(parents=True, exist_ok=True)
    for name, urls in MODEL_FILE_URLS.items():
        target = model_dir / name
        if target.exists():
            continue
        if not _download_first_available(urls, target):
            return _download_and_extract_tar(model_dir, model_path, tokens_path)
    return True


class ASR(ABC):
    """语音识别引擎抽象：转写 16kHz float32 单声道音频，返回文本。

    同步接口——跑在耳的监听线程里（录音结束后转写，无并发场景），不加 async。
    """

    name = "abstract"

    @abstractmethod
    def transcribe(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> str: ...


class SenseVoiceEngine(ASR):
    """SenseVoice-Small（sherpa-onnx ONNX int8）：中文优先的多语识别（zh/en/ja/ko/yue）。

    use_itn=True 开逆文本正则（"一百二十三"→"123"、日期/单位归一）并输出标点。
    recognizer 可注入（测试替身）；生产路径先确保模型文件再创建识别器。
    """

    name = "sensevoice"

    def __init__(self, recognizer=None, model_dir=None, num_threads=2):
        if recognizer is None:
            import sherpa_onnx  # 惰性导入：未装依赖时让工厂好回退 whisper

            model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
            if not model_dir.is_absolute():
                model_dir = PROJECT_DIR / model_dir
            if not download_sensevoice_model(model_dir):
                raise RuntimeError("SenseVoice 模型文件不可用")
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_dir / "model.int8.onnx"),
                tokens=str(model_dir / "tokens.txt"),
                num_threads=num_threads,
                language="",  # 空串 = 多语自动检测（中文场景自动落 zh）
                use_itn=True,
            )
        self.recognizer = recognizer

    def transcribe(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
        # 尾部补 0.3s 静音：离线流式模型对"音频戛然而止"的最后一词易丢帧，
        # 补一段零值让它把尾部特征读完（sherpa 官方示例同款做法）
        pad = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, np.concatenate([audio_f32, pad]))
        self.recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()


class WhisperEngine(ASR):
    """faster-whisper（原 listen.py 逻辑迁入 + initial_prompt 强化）。

    initial_prompt="以下是普通话的句子。" 是社区验证的中文约束技巧：给模型一个
    简体中文书面语的"上文示范"，显著压低繁体输出、翻译腔与静音段幻觉。
    """

    name = "whisper"
    INITIAL_PROMPT = "以下是普通话的句子。"

    def __init__(self, model_size="small", model=None):
        if model is None:
            import ctranslate2
            from faster_whisper import WhisperModel

            # 有 CUDA 用 GPU，否则回退 CPU int8（量化换速度，精度损一档）
            if ctranslate2.get_cuda_device_count() > 0:
                device, compute_type = "cuda", "float16"
            else:
                device, compute_type = "cpu", "int8"
            print(f"Whisper 推理设备：{device} ({compute_type})")
            try:
                # 优先离线加载本地缓存：在线模式即使模型已缓存也会先连 HF 校验，
                # 网络不通时会卡死在 TCP 连接上（2026-08-10 踩过）
                model = WhisperModel(model_size, device=device, compute_type=compute_type, local_files_only=True)
            except Exception:
                print("本地未找到 Whisper 模型缓存，转为在线下载...")
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model = model

    def transcribe(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
        segments, _info = self.model.transcribe(
            audio_f32,
            beam_size=5,
            language="zh",
            initial_prompt=self.INITIAL_PROMPT,
        )
        return "".join(s.text for s in segments).strip()


def create_asr(config=None) -> ASR:
    """按 config 的 ASR_BACKEND 装配引擎；sensevoice 不可用时回退 whisper。

    ASR_BACKEND = "sensevoice"（默认）/ "whisper"；ASR_MODEL_DIR 可指定模型目录
    （相对路径相对项目根）。回退链与 vision 的截屏回退同哲学：主引擎缺席时
    功能降级可用，而不是启动失败。
    """
    cfg = config or {}
    backend = cfg.get("ASR_BACKEND", "sensevoice").lower()
    if backend == "sensevoice":
        try:
            engine = SenseVoiceEngine(model_dir=cfg.get("ASR_MODEL_DIR"))
            print("ASR 引擎：SenseVoice（本地 sherpa-onnx）")
            return engine
        except Exception as e:
            print(f"SenseVoice 初始化失败（{e}），回退 Whisper")
    return WhisperEngine()
