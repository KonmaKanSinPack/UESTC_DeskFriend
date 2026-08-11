"""LLM 决策器：判定一条用户消息是否需要桌宠回应（should_reply 的统一实现）。

两个后端共用：
- openai 后端：ui 把 JUDGE_SYSTEM_PROMPT 放进 context，走通用 LLM 通道判定
- astrbot 后端：直接持有一个 LLMJudge（桌宠侧自调 LLM，判定不进 AstrBot 对话流，
  不污染桃桃记忆）

判定倾向：不确定时输出 true（宁可多回，不可漏听——原本地关键词规则实测漏判严重）。
"""

import re

JUDGE_SYSTEM_PROMPT = (
    "你是桌宠的消息过滤器，判断用户的话是否需要桌宠回应。\n"
    "规则：\n"
    "1. 直接对桌宠说的任何话（提问、指令、聊天、撒娇、抱怨）→ true\n"
    '2. 叫桌宠名字、说"看看我的屏幕"等指令 → true\n'
    "3. 只有明显与桌宠无关的背景谈话（在跟别人聊天、电视声音、自言自语碎片）→ false\n"
    "4. 不确定时输出 true（宁可多回，不可漏听）\n"
    "示例：\n"
    "用户的消息是：你在吗 → true\n"
    "用户的消息是：看看我的屏幕 → true\n"
    "用户的消息是：今天好累啊 → true\n"
    "用户的消息是：嗯嗯 → false\n"
    "用户的消息是：（电视里的台词）→ false\n"
    "只输出 true 或 false，不要输出其他内容。"
)


class LLMJudge:
    """LLM 判定器：client 可注入（测试用），生产路径由工厂按 config 构建。"""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    async def should_reply(self, user_text) -> bool:
        """判定用户消息是否需要回应；调用失败回退 true（宁可多回不可漏听）。"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户的消息是：{user_text}"},
                ],
                # 推理模型（如 MiniMax-M3）会先输出 <think> 思考块再给结论，
                # token 太短会被思考块截断导致判定失效（实测 max_tokens=8 必挂）
                max_tokens=200,
            )
            content = (response.choices[0].message.content or "").lower()
            # 剥掉 <think> 思考块（实测输出形如 "<think>…</think>\ntrue"）
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
            matches = re.findall(r"\b(true|false)\b", content)
            return matches[-1] == "true" if matches else False
        except Exception as e:
            print(f"判定器调用失败：{e}；默认回复（宁可多回不可漏听）")
            return True
