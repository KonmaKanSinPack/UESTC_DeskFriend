import copy
import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "pet.db"

# 记忆阈值的默认值在策略 owner（backends/openai.py 的 DEFAULT_*），经 config 的
# MEMORY_* 键覆盖——本模块纯机制（SQL 存取），不持有自己不用的策略数字


def sanitize_message(msg):
    """返回消息的落库副本:把 base64 图片等大体积内容替换为占位符,避免 pet.db 膨胀。"""
    m = copy.deepcopy(msg)
    content = m.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                part["image_url"] = {"url": "[image omitted]"}
    return m


def render_messages(msgs):
    """把消息列表渲染成纯文本,作为摘要 LLM 的输入。"""
    lines = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            # 多模态消息:只取文本部分,图片标注占位
            parts = [p.get("text", "") if p.get("type") == "text" else "[图片]" for p in content]
            content = " ".join(parts)
        if role == "user":
            lines.append(f"用户: {content}")
        elif role == "assistant":
            if content:
                lines.append(f"桃桃: {content}")
        elif role == "tool":
            lines.append(f"[工具 {m.get('name')}] {content}")
    return "\n".join(lines)


def revive_message(msg):
    """库 → 运行时的还原副本：把 sanitize 落库的图片占位符换成 API 安全的文本段。

    为什么必须有（2026-09-04 修复）：占位符 "[image omitted]" 留在 image_url.url
    里时，重启后恢复的上下文发给 API 会被当作 base64 解码，实测报
    500 convert_request_failed（illegal base64 data at input byte 0）——此后每一轮
    对话都带着这个坏消息，全部失败。判定规则：url 不以 "data:" 开头的图片段
    一律换成文本占位（真实内存图片恒为 data: URL，只会命中落库占位符）。
    """
    m = copy.deepcopy(msg)
    content = m.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "image_url"
                and not str(part.get("image_url", {}).get("url", "")).startswith("data:")
            ):
                parts.append({"type": "text", "text": "[历史截图，内容已省略]"})
            else:
                parts.append(part)
        m["content"] = parts
    return m


class MemoryStore:
    """对话记忆存储:完整消息历史(messages)+ 单行滚动摘要(summaries)。

    不变式:未压缩(summarized=0)的消息与 Brain 内存中的 context 一一对应、顺序一致。
    """

    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                msg_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                summarized INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                content TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def max_turn_id(self):
        row = self.conn.execute("SELECT COALESCE(MAX(turn_id), 0) FROM messages").fetchone()
        return row[0]

    def next_turn_id(self):
        return self.max_turn_id() + 1

    def add_message(self, turn_id, msg):
        clean = sanitize_message(msg)
        self.conn.execute(
            "INSERT INTO messages (turn_id, role, msg_json, created_at) VALUES (?, ?, ?, ?)",
            (turn_id, clean.get("role", ""), json.dumps(clean, ensure_ascii=False), time.time()),
        )
        self.conn.commit()

    def load_unsummarized(self):
        """加载全部未压缩消息（按写入顺序），用于启动时恢复上下文。

        返回前过 revive_message：落库的图片占位符若原样回到上下文，会在下一轮
        请求里被 API 当 base64 解码而 500（见 revive_message 注释）。
        """
        rows = self.conn.execute("SELECT msg_json FROM messages WHERE summarized = 0 ORDER BY id").fetchall()
        return [revive_message(json.loads(r[0])) for r in rows]

    def unsummarized_turn_count(self):
        row = self.conn.execute("SELECT COUNT(DISTINCT turn_id) FROM messages WHERE summarized = 0").fetchone()
        return row[0]

    def turns_to_summarize(self, keep_recent):
        """返回应并入摘要的最老轮次 (turn_ids, msgs),保留最近 keep_recent 轮不动。"""
        rows = self.conn.execute(
            "SELECT DISTINCT turn_id FROM messages WHERE summarized = 0 ORDER BY turn_id"
        ).fetchall()
        turn_ids = [r[0] for r in rows][:-keep_recent]
        if not turn_ids:
            return [], []
        placeholders = ",".join("?" * len(turn_ids))
        rows = self.conn.execute(
            f"SELECT msg_json FROM messages WHERE summarized = 0 AND turn_id IN ({placeholders}) ORDER BY id",
            turn_ids,
        ).fetchall()
        return turn_ids, [json.loads(r[0]) for r in rows]

    def mark_summarized(self, turn_ids):
        placeholders = ",".join("?" * len(turn_ids))
        self.conn.execute(f"UPDATE messages SET summarized = 1 WHERE turn_id IN ({placeholders})", turn_ids)
        self.conn.commit()

    def get_summary(self):
        row = self.conn.execute("SELECT content FROM summaries WHERE id = 1").fetchone()
        return row[0] if row else None

    def set_summary(self, content):
        self.conn.execute(
            "INSERT OR REPLACE INTO summaries (id, content, updated_at) VALUES (1, ?, ?)",
            (content, time.time()),
        )
        self.conn.commit()

    # ---- 事实型长期记忆（facts）----

    def get_facts(self):
        """返回 [(id, content), ...]，按写入顺序。"""
        return self.conn.execute("SELECT id, content FROM facts ORDER BY id").fetchall()

    def add_fact(self, content):
        now = time.time()
        self.conn.execute(
            "INSERT INTO facts (content, created_at, updated_at) VALUES (?, ?, ?)",
            (content, now, now),
        )
        self.conn.commit()

    def update_fact(self, fact_id, content):
        self.conn.execute(
            "UPDATE facts SET content = ?, updated_at = ? WHERE id = ?",
            (content, time.time(), fact_id),
        )
        self.conn.commit()

    def delete_fact(self, fact_id):
        self.conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        self.conn.commit()

    def replace_facts(self, contents):
        """整体替换 facts 表（用于超上限时的 LLM 合并压缩）。"""
        now = time.time()
        self.conn.execute("DELETE FROM facts")
        self.conn.executemany(
            "INSERT INTO facts (content, created_at, updated_at) VALUES (?, ?, ?)",
            [(c, now, now) for c in contents],
        )
        self.conn.commit()

    # ---- 抽取进度记录（meta）----

    def get_meta(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def get_turns_since(self, turn_id, limit):
        """返回 turn_id 之后最近 limit 轮的 (turn_ids, msgs)，用于批量事实抽取。"""
        rows = self.conn.execute(
            "SELECT DISTINCT turn_id FROM messages WHERE turn_id > ? ORDER BY turn_id DESC LIMIT ?",
            (turn_id, limit),
        ).fetchall()
        turn_ids = sorted(r[0] for r in rows)
        if not turn_ids:
            return [], []
        placeholders = ",".join("?" * len(turn_ids))
        rows = self.conn.execute(
            f"SELECT msg_json FROM messages WHERE turn_id IN ({placeholders}) ORDER BY id",
            turn_ids,
        ).fetchall()
        return turn_ids, [json.loads(r[0]) for r in rows]
