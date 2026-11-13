#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare SFT datasets for 100-tool intent recognition fine-tuning."""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (
    FINETUNE_DIR,
    INTENT_DIR,
    KEYWORD_MAP_PATH,
    TOOL_TRACE_DIR,
    load_jsonl,
    normalize_question,
    read_json,
    split_tools,
    tools_to_keywords,
    write_json,
    write_jsonl,
)


SYSTEM_PROMPT = (
    "你是电商导购意图识别模型。判断用户问题的意图类型并提取关键词。"
    "类型只能是“电商类”或“闲聊类”。"
    "电商类包括商品、订单、支付、物流、优惠、会员、购物车、售后、用户偏好、风控和人工客服等购物服务诉求。"
    "闲聊类是与购物决策或电商服务无关的日常、百科、写作、编程、旅行、健康、做菜、清洁等问题。"
    "只输出 JSON：{\"类型\":\"电商类|闲聊类\",\"关键词\":\"...\"}。"
)

GENERAL_NEGATIVES = [
    "北京今天适合骑车吗？",
    "Python 里列表和元组有什么区别？",
    "番茄炒蛋怎么做更嫩？",
    "跑步前怎么热身比较好？",
    "帮我写一段会议纪要模板。",
    "杭州两日游怎么安排？",
    "咖啡手冲水温一般多少？",
    "肩颈酸痛有哪些拉伸动作？",
    "Git rebase 和 merge 有什么区别？",
    "怎么给绿萝浇水？",
    "上海周末去哪散步？",
    "数据库索引为什么能加速？",
    "Linux 权限 rwx 怎么理解？",
    "英语定语从句怎么区分？",
    "论文摘要一般怎么写？",
    "摄影构图有哪些基础方法？",
    "地铁换乘怎么规划更省时间？",
    "租房合同要注意哪些条款？",
    "个人所得税专项扣除怎么理解？",
    "象棋开局有哪些常见思路？",
    "电影叙事结构怎么分析？",
    "音乐节拍怎么数？",
    "中国历史朝代顺序是什么？",
    "经纬度坐标怎么看？",
    "物理惯性怎么解释？",
    "酸碱中和反应是什么？",
    "数学概率入门怎么学？",
    "Excel VLOOKUP 怎么用？",
    "路由器信号差怎么排查？",
    "冰箱结霜是什么原因？",
    "空调制冷慢怎么处理？",
    "家庭收纳怎么分类？",
    "演讲开场怎么写？",
    "压力大时怎么放松？",
    "游泳换气怎么练？",
    "羽毛球发力有什么技巧？",
    "篮球投篮姿势怎么调整？",
    "围棋定式适合怎么入门？",
    "植物换盆要注意什么？",
    "护照办理流程是什么？",
]
HARD_NEGATIVES = [
    "空气炸锅内胆糊了怎么清洗？",
    "蓝牙耳机连接笔记本后声音断断续续怎么办？",
    "运动鞋鞋带怎么系更牢？",
    "羽绒服压缩收纳会不会影响保暖？",
    "猫粮开封后常温能放多久？",
    "机械键盘青轴和茶轴打字有什么不同？",
    "防晒霜 SPF 是什么意思？",
    "扫地机器人地图乱了怎么重新建图？",
    "电动牙刷刷头怎么消毒？",
    "行李箱密码锁忘了怎么打开？",
    "咖啡豆酸味太明显怎么调整冲法？",
    "智能手表运动数据怎么看？",
    "保温杯有异味怎么去掉？",
    "显卡温度太高怎么排查？",
    "婴儿纸尿裤尺码怎么按体重看？",
    "床垫太软会不会腰酸？",
    "净水器滤芯原理是什么？",
    "冲锋衣怎么洗不伤防水层？",
    "洗地机滚刷怎么清洁？",
    "乳胶枕发黄还能用吗？",
    "跑步机噪音大怎么处理？",
    "口红色号怎么根据肤色选？",
    "台灯色温对眼睛有什么影响？",
    "牛仔裤缩水怎么恢复？",
    "微波炉加热饭菜要不要盖盖子？",
    "车载充电器功率怎么看？",
    "牛奶开封后冷藏能放几天？",
    "儿童书包背负系统怎么调？",
]
HARD_POSITIVE_TEMPLATES = [
    ("我购物车里的东西差一点满减，帮我看看怎么凑单再算下最终要付多少。", "购物车，凑单，优惠计算"),
    ("这双鞋我不知道尺码，先按我的尺码档案推荐一下，再看对应颜色有没有库存。", "尺码档案，尺码推荐，库存"),
    ("订单显示快递好几天没动，帮我查物流，如果像丢件就看看怎么处理。", "物流，丢件处理"),
    ("刚买完发现降价了，帮我确认能不能价保，能的话就申请。", "价保，申请价保"),
    ("这个商品我担心不是正品，帮我查下店铺资质和正品保障。", "品牌店铺，正品保障"),
    ("售后单一直没回应，帮我看进度，不行就升级人工。", "售后进度，人工售后"),
]
NEGATIVE_TEMPLATES = [
    "{topic}能简单解释一下吗？",
    "{topic}有什么注意事项？",
    "新手想了解{topic}，从哪里开始？",
    "帮我整理一下{topic}的步骤。",
]
NEGATIVE_ANGLES = [
    "用通俗的话",
    "从原理角度",
    "按步骤",
    "列常见误区",
    "给新手",
    "用简短版本",
    "从实践角度",
    "按准备事项",
    "按排查思路",
    "按学习路径",
    "从安全角度",
    "按时间顺序",
    "按重点总结",
    "从生活场景",
    "按对比方式",
    "给我一个清单",
    "用问答方式",
    "按入门到进阶",
    "从常见问题",
    "按经验建议",
]
NEGATIVE_CONTEXTS = [
    "",
    "我在自学时想问，",
    "准备给同学解释，",
    "做笔记用，",
    "请用三句话概括，",
    "不要太专业，",
    "偏实用一点，",
    "按常见场景说，",
    "帮我快速判断，",
    "给长辈讲的话，",
]


def response(intent_type: str, keywords: str) -> str:
    return f'{{"类型":"{intent_type}","关键词":"{keywords}"}}'


def message_row(question: str, intent_type: str, keywords: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response(intent_type, keywords)},
        ]
    }


def make_negative_questions(topics: list[str], target: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for context in NEGATIVE_CONTEXTS:
        for angle in NEGATIVE_ANGLES:
            for tmpl in NEGATIVE_TEMPLATES:
                for topic in topics:
                    q = f"{context}{angle}，{tmpl.format(topic=topic)}"
                    if q not in seen:
                        seen.add(q)
                        out.append(q)
                    if len(out) >= target:
                        return out
    return out


def build_examples(train_rows: list[dict], keyword_map: dict[str, str], include_hard: bool) -> list[dict]:
    examples: list[dict] = []
    for idx, row in enumerate(train_rows):
        tools = split_tools(row.get("工具"))
        examples.append(
            {
                "id": f"ecom_{idx:06d}",
                "question": row["问题"],
                "type": "电商类",
                "keywords": tools_to_keywords(tools, keyword_map),
                "source": row.get("source", "tool_trace_train"),
                "tools": ", ".join(tools),
            }
        )
    general_target = max(len(GENERAL_NEGATIVES), len(train_rows) // 2)
    general_questions = make_negative_questions(GENERAL_NEGATIVES, general_target)
    for idx, q in enumerate(general_questions):
        examples.append({"id": f"neg_{idx:06d}", "question": q, "type": "闲聊类", "keywords": "", "source": "synthetic_general_negative", "tools": ""})
    if include_hard:
        hard_target = max(len(HARD_NEGATIVES), len(train_rows) // 2)
        hard_questions = make_negative_questions(HARD_NEGATIVES, hard_target)
        for idx, q in enumerate(hard_questions):
            examples.append({"id": f"hard_neg_{idx:06d}", "question": q, "type": "闲聊类", "keywords": "", "source": "synthetic_hard_negative", "tools": ""})
        for idx, (q, kw) in enumerate(HARD_POSITIVE_TEMPLATES):
            examples.append({"id": f"hard_pos_{idx:06d}", "question": q, "type": "电商类", "keywords": kw, "source": "synthetic_hard_positive", "tools": ""})
    return examples


def split_train_val(examples: list[dict], seed: int, val_ratio: float) -> tuple[list[dict], list[dict]]:
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_ratio))) if rows else 0
    return rows[n_val:], rows[:n_val]


def write_dataset(name: str, examples: list[dict], seed: int, val_ratio: float, out_dir: Path) -> dict:
    train, val = split_train_val(examples, seed, val_ratio)
    write_jsonl(out_dir / f"sft_{name}_train.jsonl", [message_row(x["question"], x["type"], x["keywords"]) for x in train])
    write_jsonl(out_dir / f"sft_{name}_val.jsonl", [message_row(x["question"], x["type"], x["keywords"]) for x in val])
    write_jsonl(out_dir / f"sft_{name}_meta_train.jsonl", train)
    write_jsonl(out_dir / f"sft_{name}_meta_val.jsonl", val)
    return {
        "name": name,
        "train": len(train),
        "val": len(val),
        "by_type_train": dict(Counter(x["type"] for x in train)),
        "by_source_train": dict(Counter(x["source"] for x in train)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trace", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--test-trace", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--intent-all", type=Path, default=INTENT_DIR / "intent_all.jsonl")
    parser.add_argument("--chitchat", type=Path, default=INTENT_DIR / "chitchat.jsonl")
    parser.add_argument("--keyword-map", type=Path, default=KEYWORD_MAP_PATH)
    parser.add_argument("--out-dir", type=Path, default=FINETUNE_DIR / "data")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_trace)
    keyword_map = read_json(args.keyword_map, {})
    base = build_examples(train_rows, keyword_map, include_hard=False)
    full = build_examples(train_rows, keyword_map, include_hard=True)

    ecom = [x for x in full if x["type"] == "电商类"]
    chat = [x for x in full if x["type"] == "闲聊类"]
    rng = random.Random(args.seed)
    rng.shuffle(ecom)
    rng.shuffle(chat)
    balanced = ecom + chat[: len(ecom)]
    hard = [x for x in full if x["source"].startswith("synthetic_hard")]

    summaries = [
        write_dataset("base", base, args.seed, args.val_ratio, args.out_dir),
        write_dataset("full", full, args.seed, args.val_ratio, args.out_dir),
        write_dataset("balanced", balanced, args.seed, args.val_ratio, args.out_dir),
        write_dataset("hard", hard, args.seed, args.val_ratio, args.out_dir),
    ]

    train_keys = {normalize_question(x["question"]) for x in full}
    frozen = load_jsonl(args.test_trace) + load_jsonl(args.intent_all) + load_jsonl(args.chitchat)
    frozen_keys = {normalize_question(row["问题"]) for row in frozen}
    overlap = sorted(train_keys & frozen_keys)
    audit = {
        "ok": len(overlap) == 0,
        "overlap_with_frozen": len(overlap),
        "overlap_examples": overlap[:20],
        "datasets": summaries,
        "empty_keyword_ecommerce": sum(1 for x in full if x["type"] == "电商类" and not x["keywords"]),
    }
    write_json(args.out_dir / "audit_sft_data.json", audit)
    print(f"sft datasets written -> {args.out_dir}, overlap_with_frozen={len(overlap)}")
    return 0 if audit["ok"] and audit["empty_keyword_ecommerce"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
