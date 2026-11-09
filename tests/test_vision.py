from PIL import Image

import vision


def _fake_grab(calls):
    def grab(**kwargs):
        calls.append(kwargs)
        return Image.new("RGB", (4, 4))

    return grab


def test_grab_screenshot_windows_uses_all_screens(monkeypatch):
    monkeypatch.setattr(vision.sys, "platform", "win32")
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    calls = []
    monkeypatch.setattr(vision.ImageGrab, "grab", _fake_grab(calls))

    image = vision.grab_screenshot()

    assert image.size == (4, 4)
    assert calls == [{"all_screens": True}]


def test_grab_screenshot_non_windows_uses_plain_grab(monkeypatch):
    monkeypatch.setattr(vision.sys, "platform", "linux")
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    calls = []
    monkeypatch.setattr(vision.ImageGrab, "grab", _fake_grab(calls))

    image = vision.grab_screenshot()

    assert image.size == (4, 4)
    assert calls == [{}]
