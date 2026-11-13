#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate deterministic chitchat and hard-negative intent rows."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import INTENT_DIR, normalize_question, write_json, write_jsonl


GENERAL_TOPICS = [
    "北京天气",
    "上海周末路线",
    "杭州西湖游玩",
    "数据库索引",
    "Python 列表推导",
    "Linux 文件权限",
    "Git 分支合并",
    "番茄炒蛋做法",
    "咖啡手冲比例",
    "跑步热身",
    "睡眠质量",
    "肩颈拉伸",
    "英语从句",
    "论文摘要",
    "PPT 配色",
    "摄影构图",
    "植物浇水",
    "宠物训练",
    "地铁换乘",
    "高铁改签规则",
    "租房合同",
    "个人所得税",
    "象棋开局",
    "电影叙事",
    "音乐节拍",
    "历史朝代",
    "地理经纬度",
    "物理惯性",
    "化学酸碱",
    "数学概率",
]
HARD_TOPICS = [
    "空气炸锅怎么清洗内胆",
    "运动鞋鞋带怎么系不容易散",
    "蓝牙耳机连接电脑没声音怎么办",
    "羽绒服自己在家怎么收纳",
    "咖啡豆研磨粗细怎么选",
    "机械键盘轴体有什么区别",
    "行李箱密码锁忘了怎么处理",
    "电动牙刷刷头怎么消毒",
    "猫粮开袋后怎么保存",
    "防晒霜用量怎么算",
    "扫地机器人地图怎么重置",
    "净水器滤芯原理是什么",
]
TEMPLATES = [
    "{topic}能简单解释一下吗？",
    "{topic}有什么注意事项？",
    "新手想了解{topic}，从哪里开始？",
    "{topic}有没有更生活化的说法？",
    "帮我整理一下{topic}的步骤。",
]
PREFIXES = [
    "",
    "用通俗的话说，",
    "给新手解释，",
    "帮我快速整理，",
    "按步骤说，",
    "从常见误区角度，",
    "给长辈讲的话，",
    "做笔记用，",
    "请举个例子，",
    "按重点归纳，",
    "从安全角度说，",
    "适合入门者的话，",
    "用三句话概括，",
    "按实践经验说，",
    "帮我列个清单，",
    "从原理角度讲，",
    "偏生活化一点，",
    "按排查思路说，",
]
ECOM_BLOCKLIST = ["订单", "退款", "退货", "优惠券", "购物车", "下单", "物流到哪", "发票", "价保", "客服工单"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--out", type=Path, default=INTENT_DIR / "chitchat.jsonl")
    parser.add_argument("--stats-out", type=Path, default=INTENT_DIR / "_aux" / "chitchat_stats.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    topics = GENERAL_TOPICS + HARD_TOPICS
    rows = []
    seen: set[str] = set()
    idx = 0
    blocked = 0
    max_attempts = max(args.count * 20, 1000)
    while len(rows) < args.count and idx < max_attempts:
        topic = topics[idx % len(topics)]
        tmpl = TEMPLATES[(idx // len(topics)) % len(TEMPLATES)]
        prefix = PREFIXES[(idx // (len(topics) * len(TEMPLATES))) % len(PREFIXES)]
        idx += 1
        q = prefix + tmpl.format(topic=topic)
        if any(word in q for word in ECOM_BLOCKLIST):
            blocked += 1
            continue
        key = normalize_question(q)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": f"chitchat_{len(rows):06d}", "问题": q, "类型": "闲聊类", "关键词": "", "source": "synthetic_chitchat"})

    if len(rows) < args.count:
        raise SystemExit(f"only generated {len(rows)} unique chitchat rows after {max_attempts} attempts")

    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    write_json(args.stats_out, {"total": len(rows), "blocked": blocked, "hard_topics": len(HARD_TOPICS)})
    print(f"chitchat rows={len(rows)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
