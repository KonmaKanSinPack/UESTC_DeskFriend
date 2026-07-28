import copy
import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "pet.db"

# 未压缩轮次超过 MAX_CONTEXT_TURNS 时触发摘要压缩,压缩后只保留最近 KEEP_RECENT_TURNS 轮
MAX_CONTEXT_TURNS = 30
KEEP_RECENT_TURNS = 10


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
                lines.append(f"糯糯: {content}")
        elif role == "tool":
            lines.append(f"[工具 {m.get('name')}] {content}")
    return "\n".join(lines)


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
        """加载全部未压缩消息(按写入顺序),用于启动时恢复上下文。"""
        rows = self.conn.execute("SELECT msg_json FROM messages WHERE summarized = 0 ORDER BY id").fetchall()
        return [json.loads(r[0]) for r in rows]

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
