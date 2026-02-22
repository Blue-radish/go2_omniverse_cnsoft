#!/usr/bin/env python3
"""
ernie_llm_converter.py  ·  OCR text → /robot0/direction_cmd via ERNIE chat

* Step 1: local regex on the OCR text itself
* Step 2: ask ERNIE (plain chat, no tools) to answer with ONLY one of
          left / right / forward / back
"""

import os, re, rclpy
from rclpy.node        import Node
from std_msgs.msg      import String
from geometry_msgs.msg import Twist
from openai import OpenAI
from dotenv import load_dotenv

# ───────── ERNIE client ───────────────────────────────────────────
load_dotenv()   
                                           # ~/.ernie.env
client = OpenAI(
    api_key="",
    base_url="https://aistudio.baidu.com/llm/lmapi/v3",
)

# 1) local regex ---------------------------------------------------
local_patterns = {
    "left":    re.compile(r"(左|turn\s*left)",      re.I),
    "right":   re.compile(r"(右|turn\s*right)",     re.I),
    # "forward": re.compile(r"(前|go\s*forward)",     re.I),
    # "back":    re.compile(r"(后|back)",             re.I),
}

# 2) assistant regex for answer -----------------------------------
# ans_pat = re.compile(r"\b(left|right|forward|back)\b", re.I)
ans_pat = re.compile(r"\b(left|right)\b", re.I)

SYSTEM_PROMPT = (
    "You are a robot controller.  "
    "When the user gives a short phrase, respond with *one single word* "
    "chosen from: left, right, forward, back.\n"
    "Do NOT add any other text."
)

class ErnieConverter(Node):
    def __init__(self):
        super().__init__("ernie_llm_converter")
        self.sub = self.create_subscription(String, "/ocr_result", self._cb, 10)
        # 修改：发布方向指令而不是直接控制速度
        self.direction_pub = self.create_publisher(String, "/robot0/direction_cmd", 10)
        
        self.get_logger().info("ERNIE LLM converter ready")

    # ───── main OCR callback ──────────────────────────────────────
    def _cb(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        direction = None

        # ---- 1) local regex
        for d, pat in local_patterns.items():
            if pat.search(text):
                direction = d
                self.get_logger().info(f"[Regex] {text} → {direction}")
                break

        # ---- 2) ask LLM
        if direction is None:
            try:
                rep = client.chat.completions.create(
                    model="ernie-3.5-8k",          # any chat model is fine
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": text},
                    ],
                )
            except Exception as e:
                self.get_logger().error(f"ERNIE API error: {e}")
                return

            answer = (rep.choices[0].message.content or "").strip()
            m = ans_pat.search(answer)
            if m:
                direction = m.group(1).lower()
                self.get_logger().info(f"[LLM] {text} → {direction}")
            else:
                self.get_logger().info(f"LLM replied «{answer}» – no match")
                return

        # 修改：发布方向指令而不是直接控制速度
        dir_msg = String()
        dir_msg.data = direction
        self.direction_pub.publish(dir_msg)
        self.get_logger().info(f"发布方向指令: {direction}")

# ───────── entry point ────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ErnieConverter())
    rclpy.shutdown()

if __name__ == "__main__":
    main()