import requests
import re
import time
from typing import List

BASE_URL = "http://localhost:1234/v1"
MODEL = "qwen2.5-14b-instruct"

CALLS = 0
IN_TOKENS = 0
OUT_TOKENS = 0
TIME_MS = 0.0

# ----------------------- Model Call -----------------------

def _count_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def _count_message_tokens(messages) -> int:
    return sum(_count_tokens(m.get("content", "")) for m in messages)


def chat(messages, temperature=0.2, timeout=120):
    global CALLS, IN_TOKENS, OUT_TOKENS
    global TIME_MS
    CALLS += 1
    IN_TOKENS += _count_message_tokens(messages)
    t0 = time.perf_counter()
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    OUT_TOKENS += _count_tokens(content)
    TIME_MS += (time.perf_counter() - t0) * 1000.0
    return content


# ----------------------- Memory Utils -----------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def jaccard(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def importance_score(text: str) -> float:
    low = text.lower()
    score = 0.0
    for kw in ["must", "need", "require", "prefer", "goal", "constraint", "cannot", "never"]:
        if kw in low:
            score += 2.0
    if len(tokenize(text)) > 40:
        score += 1.0
    return score


def summarize(text: str, max_tokens: int) -> str:
    prompt = (
        f"请用中文结构化总结，长度不超过 {max_tokens} tokens。\n"
        f"- 只包含输入中明确出现的事实。\n"
        f"- 不得新增目标、计划、工具、供应商或步骤。\n"
        f"- 如果没有相关信息，请写“未提及”。\n"
        f"- 输出格式严格如下：\n"
        f"偏好: ...\n"
        f"目标: ...\n"
        f"限制: ...\n\n"
        f"{text}"
    )
    return chat([{"role": "user", "content": prompt}], temperature=0.1)


# ----------------------- Dual-Rate Memory -----------------------

class DualRateMemory:
    def __init__(self, fast_tokens=250, slow_tokens=300, slow_every=4, slow_importance=3.0, recent_keep=1):
        self.fast_tokens = fast_tokens
        self.slow_tokens = slow_tokens
        self.slow_every = slow_every
        self.slow_importance = slow_importance
        self.recent_keep = recent_keep
        self.fast = ""
        self.slow = ""
        self.recent: List[str] = []
        self.turn = 0

    def update(self, new_text: str):
        self.turn += 1
        self.fast = summarize(self.fast + "\n" + new_text, self.fast_tokens)

        do_slow = False
        if self.slow_every > 0 and (self.turn % self.slow_every == 0):
            do_slow = True
        if importance_score(new_text) >= self.slow_importance:
            do_slow = True

        if do_slow:
            self.slow = summarize(self.slow + "\n" + new_text + "\n" + self.fast, self.slow_tokens)

        if self.slow and jaccard(self.fast, self.slow) < 0.15:
            self.fast = summarize(self.slow + "\n" + self.fast, self.fast_tokens)

        self.recent.append(new_text)
        if len(self.recent) > self.recent_keep:
            self.recent.pop(0)

    def context(self) -> str:
        return (self.slow + "\n" + self.fast + "\n" + "\n".join(self.recent)).strip()


# ----------------------- Demo -----------------------

if __name__ == "__main__":
    mem = DualRateMemory()

    demo_inputs = [
        "I prefer short answers and practical steps.",
        "My goal is to build an AI agent backend service.",
        "I cannot use cloud-only solutions due to compliance.",
        "Let's plan a local deployment with LM Studio and Qwen2.5-14B.",
    ]

    for msg in demo_inputs:
        mem.update(msg)
        ctx = mem.context()
        reply = chat([
            {"role": "system", "content": "请勿编造细节，只能使用提供的上下文。"},
            {"role": "user", "content": f"上下文:\n{ctx}\n\n问题: 按如下格式总结（缺失写“未提及”）：\n偏好: ...\n目标: ...\n限制: ..."}
        ])
        print("Assistant:", reply)
        time.sleep(0.2)

    print("")
    print(f"cost: calls={CALLS} in_tokens={IN_TOKENS} out_tokens={OUT_TOKENS} time_ms={TIME_MS:.1f}")
