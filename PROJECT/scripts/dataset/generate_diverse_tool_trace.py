#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 100-tool traces with train/test expression-cluster isolation.

This generator deliberately does not import the legacy HARD_PHRASES bank. It
builds natural expression clusters from catalog capabilities, then separates
each tool's train-only and test-only clusters before composing traces.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_common import (
    CATALOG_PATH,
    TOOL_TRACE_DIR,
    catalog_by_id,
    domain_counts,
    group_by_domain,
    join_tools,
    load_catalog,
    normalize_question,
    order_tool_chain,
    split_tools,
    tool_counts,
    write_json,
    write_jsonl,
)
from dataset.expression_phrase_bank import phrase_bank, validate_phrase_bank as validate_seed_phrase_bank


PRODUCTS = [
    "蓝牙耳机",
    "运动鞋",
    "空气炸锅",
    "护肤套装",
    "儿童书包",
    "机械键盘",
    "咖啡豆",
    "羽绒服",
    "智能手表",
    "电动牙刷",
    "行李箱",
    "口红",
    "猫粮",
    "台灯",
    "保温杯",
    "显卡",
    "婴儿纸尿裤",
    "床垫",
    "净水器",
    "冲锋衣",
    "洗地机",
    "乳胶枕",
    "跑步机",
    "防晒霜",
    "登机箱",
    "牛奶",
    "车载充电器",
    "牛仔裤",
    "扫地机器人",
    "微波炉",
]
CITIES = ["上海", "北京", "杭州", "深圳", "广州", "成都", "南京", "武汉", "西安", "重庆"]
SPECS = ["黑色款", "白色款", "大容量", "轻便款", "标准版", "升级版", "家用款", "通勤款", "礼盒装"]
BUDGETS = ["200元内", "300元左右", "500元以内", "800元以内", "1000元左右", "性价比优先"]
CONTEXTS = [
    "我有点急",
    "给家里人买所以想稳一点",
    "准备今晚处理",
    "下手前想确认清楚",
    "刚收到东西",
    "担心弄错",
    "预算卡得比较紧",
    "大促快结束了",
    "客服让我先核对",
    "想少走弯路",
]
DETAILS = [
    "想明天前确认",
    "收件地点刚变过",
    "之前买过类似的",
    "看到价格有变化",
    "怕规则看错",
    "准备送生日礼物",
    "担心后面麻烦",
    "账号里资料可能不是最新",
]
TONES = ["麻烦", "尽快", "简单说下", "帮我一起", "我有点纠结", "请直接点", "先帮我", "顺手"]

SINGLE_TEMPLATES = [
    "{context}，我在看{city}这边的{product}{spec}，预算{budget}，{phrase}。",
    "给{product}{spec}做决定前，{detail}，{phrase}。",
    "{tone}确认下{city}这边{product}{spec}，{phrase}。",
    "这款{product}{spec}我有点拿不准，{phrase}。",
    "{detail}，跟{product}{spec}有关，{phrase}。",
    "下手前围绕{product}{spec}问一句：{phrase}，预算{budget}。",
    "我准备给{product}{spec}付款前，{phrase}，别绕太多。",
    "关于{city}收货的{product}{spec}，{phrase}。",
]
MULTI_TEMPLATES = [
    "{context}，我在{city}处理{product}{spec}这件事，{clauses}，{detail}。",
    "关于{product}{spec}，{clauses}；预算{budget}，麻烦按顺序处理。",
    "{detail}，现在围绕{product}{spec}有几件事：{clauses}。",
    "我想少来回沟通，{clauses}，对象是{city}这边的{product}{spec}。",
    "这单和{product}{spec}有关，{clauses}，最后给我一个明确结果。",
    "帮我把{product}{spec}的问题串起来：{clauses}。",
]
CONNECTOR_SETS = [
    ["先", "再", "另外", "然后", "最后", "顺带"],
    ["第一步", "接着", "同时", "必要时", "收尾时", "补充"],
    ["麻烦", "再帮我", "还要", "如果可以", "结束前", "顺手"],
    ["先别漏掉", "后面继续", "同一句里还想", "看完以后", "最后", "额外"],
]

SAFE_TERM_OVERRIDES = {
    "订单列表": "买过的记录清单",
    "订单详情": "这笔购买里的细节",
    "支付状态": "扣款有没有完成",
    "发票": "票据",
    "修改地址": "改寄送地点",
    "取消订单": "撤掉这笔购买",
    "账单明细": "费用拆分",
    "支付方式": "可用付款渠道",
    "重新支付": "重新拉起付款",
    "支付风控": "付钱被拦原因",
    "分期方案": "分开还款安排",
    "用券明细": "抵扣权益使用情况",
    "商品快照": "下手时页面留档",
    "确认收货": "提交已经拿到",
    "先用后付": "账期付款",
    "商品详情": "页面核心信息",
    "商品参数": "规格配置",
    "库存": "余量",
    "价格历史": "之前价位走势",
    "比价": "多方价差",
    "商品评价": "买家反馈",
    "评价摘要": "反馈归纳",
    "商品问答": "买前疑问回复",
    "正品保障": "真伪和授权",
    "保修": "修理保障",
    "尺码推荐": "合适尺寸建议",
    "相似推荐": "相近款建议",
    "替代商品": "可换选择",
    "品牌店铺": "品牌门店资质",
    "类目榜单": "品类排行",
    "新品": "新上架款",
    "到货提醒": "补上时叫我",
    "预售": "提前售卖规则",
    "活动": "促销玩法",
    "优惠券": "抵扣权益",
    "用券": "使用抵扣权益",
    "优惠计算": "最终应付估算",
    "会员等级": "会员档位",
    "会员价": "会员专属价",
    "积分抵扣": "点数抵钱",
    "积分余额": "点数剩余",
    "价保": "降价补差",
    "申请价保": "提交补差",
    "凑单": "补买跨门槛",
    "秒杀": "限时抢购",
    "活动预约": "促销资格预约",
    "赠品": "随单礼",
    "优惠方案": "省钱组合",
    "物流": "包裹轨迹",
    "配送时效": "送达预估",
    "配送方式": "送货选择",
    "预约配送": "约送达时段",
    "自提点": "取件位置",
    "修改配送": "切换送达办法",
    "合并发货": "一起寄出",
    "拆单发货": "分批寄出",
    "区域库存": "附近仓余量",
    "运费": "邮寄费用",
    "丢件处理": "包裹疑似丢失处理",
    "安装预约": "约上门装配",
    "退换规则": "退回更换限制",
    "退款申请": "发起返钱流程",
    "退款进度": "返款处理节点",
    "换货申请": "发起更换流程",
    "维修政策": "故障修理规则",
    "维修工单": "登记修理单",
    "售后进度": "处理单节点",
    "售后工单": "问题处理单",
    "补寄": "少件再寄",
    "退款金额": "可返金额",
    "售后凭证": "问题证明材料",
    "取消退款": "撤回返钱流程",
    "投诉渠道": "申诉入口",
    "人工售后": "真人处理",
    "质量问题": "质量争议",
    "用户画像": "账号偏好",
    "浏览记录": "看过的页面",
    "购买历史": "以前买过的记录",
    "收藏": "关注清单",
    "收货地址": "寄送地点",
    "默认地址": "首选寄送地点",
    "尺码档案": "身体尺寸资料",
    "更新尺码": "改身体尺寸资料",
    "个性推荐": "贴合偏好的推荐",
    "通知偏好": "消息订阅设置",
    "加入购物车": "放入待结算清单",
    "购物车": "待结算清单",
    "改购物车数量": "调整待结算数量",
    "移除购物车": "从待结算清单删掉",
    "稍后购买": "挪到以后再买",
    "结算预览": "付款前确认页",
    "提交订单": "生成待付款购买记录",
    "结算校验": "付款前可售检查",
    "异常订单": "购买风险异常",
    "敏感品类": "特殊类目",
    "限购": "购买数量限制",
    "账户风险": "账号限制原因",
    "人工客服": "真人客服单",
    "客服排队": "真人服务等待情况",
    "风险提示": "受限说明话术",
}
GENERIC_REPLACEMENTS = {
    "查询": "确认",
    "提交": "发起",
    "创建": "建立",
    "生成": "给出",
    "计算": "估算",
    "比较": "对比",
    "推荐": "给出建议",
    "返回": "给到",
    "订单": "购买记录",
    "商品": "这件东西",
    "购物车": "待结算清单",
    "优惠券": "抵扣权益",
    "售后": "后续处理",
    "退款": "返钱",
    "退货": "寄回",
    "物流": "包裹运输",
    "客服": "服务人员",
}


def catalog_term_index(catalog: list[dict[str, Any]]) -> dict[str, list[tuple[str, str, str]]]:
    out: dict[str, list[tuple[str, str, str]]] = {}
    for tool in catalog:
        terms: list[tuple[str, str, str]] = []
        for kind, field in [("tool_id", "tool_id"), ("name_zh", "name_zh"), ("keyword_label", "keyword_label")]:
            raw = str(tool.get(field, ""))
            normalized = normalize_question(raw)
            if normalized:
                terms.append((kind, raw, normalized))
        for phrase in tool.get("positive_phrases", []):
            normalized = normalize_question(str(phrase))
            if normalized:
                terms.append(("positive_phrase", str(phrase), normalized))
        out[tool["tool_id"]] = terms
    return out


def catalog_hits(text: str, terms: dict[str, list[tuple[str, str, str]]]) -> list[tuple[str, str, str]]:
    normalized = normalize_question(text)
    hits = []
    for tool_id, items in terms.items():
        for kind, raw, term in items:
            if term and term in normalized:
                hits.append((tool_id, kind, raw))
    return hits


def break_term(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text
    if text in SAFE_TERM_OVERRIDES:
        return SAFE_TERM_OVERRIDES[text]
    if len(text) <= 2:
        return text[0] + "相关" + text[1:]
    return text[:2] + "相关" + text[2:]


def safe_text(text: str, terms: dict[str, list[tuple[str, str, str]]]) -> str:
    out = text.strip(" 。；，、")
    raw_terms = sorted({raw for items in terms.values() for _, raw, _ in items if raw}, key=len, reverse=True)
    for raw in raw_terms:
        if raw in out:
            out = out.replace(raw, SAFE_TERM_OVERRIDES.get(raw, break_term(raw)))
    for raw, replacement in GENERIC_REPLACEMENTS.items():
        out = out.replace(raw, replacement)
    out = re.sub(r"\s+", "", out)
    out = re.sub(r"[。；;]+", "，", out).strip("，、。 ")
    return out


def description_fragments(tool: dict[str, Any], terms: dict[str, list[tuple[str, str, str]]]) -> list[str]:
    text = str(tool.get("description", "")).strip("。")
    pieces = [piece.strip(" ，、。") for piece in re.split(r"[，。；;]", text) if piece.strip(" ，、。")]
    if len(pieces) < 4:
        expanded = []
        for piece in pieces or [text]:
            parts = [x.strip(" ，、。") for x in re.split(r"[、]", piece) if x.strip(" ，、。")]
            expanded.extend(parts if len(parts) > 1 else [piece])
        pieces = expanded
    safe = []
    for piece in pieces:
        cleaned = safe_text(piece, terms)
        if cleaned and cleaned not in safe:
            safe.append(cleaned)
    if not safe:
        safe = [safe_text(str(tool.get("enhanced_desc", tool["tool_id"])), terms)]
    while len(safe) < 6:
        base = safe[len(safe) % len(safe)]
        suffix = ["这一项", "具体情况", "处理限制", "下一步", "用户侧结果", "时间节点"][len(safe) % 6]
        candidate = f"{base}的{suffix}"
        if candidate not in safe:
            safe.append(candidate)
    return safe


def phrase_variants(fragments: list[str], split: str, count: int) -> list[str]:
    if split == "train":
        templates = [
            "{a}这块帮我确认清楚",
            "我想知道{a}，别只给泛泛建议",
            "把{a}和{b}相关情况说明白",
            "{a}这一步现在该怎么处理",
            "围绕{a}给我一个可执行结果",
            "{a}如果有异常也一并说明",
            "先核对{a}，再告诉我结论",
            "{a}这件事不要漏掉限制条件",
        ]
        pool = fragments[: max(3, len(fragments) // 2)]
    elif split == "test":
        templates = [
            "不用看别的，{a}这一块帮我核实",
            "主要想确认{a}，顺带说明{b}",
            "{a}有没有问题，给我明确答复",
            "把{a}相关结果直接告诉我",
            "{a}这边如果卡住，说明原因",
            "按现在情况判断{a}是否可行",
        ]
        pool = fragments[max(1, len(fragments) // 2) :]
    elif split == "dpo_hard":
        templates = [
            "别只复述页面，我需要把{a}核到结果",
            "这一项绕一点说：先帮我弄明白{a}",
            "实际诉求落在{a}，同时把{b}对应边界交代清楚",
            "不用给功能名，按{a}这件具体事直接处理",
            "我真正关心的是{a}目前到底怎样",
            "把{a}从现状到可执行结果讲清楚",
            "先识别我说的{a}，别和相近业务混在一起",
            "有关{a}，请给出处理结果并说明{b}的限制",
        ]
        pool = fragments[1::2] + fragments[::2]
    else:
        raise ValueError(f"unknown expression split: {split}")
    if not pool:
        pool = fragments
    phrases = []
    for idx in range(count):
        tmpl = templates[idx % len(templates)]
        a = pool[idx % len(pool)]
        b = pool[(idx + 1) % len(pool)]
        phrase = tmpl.format(a=a, b=b).replace("，，", "，")
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def build_expression_clusters(
    catalog: list[dict[str, Any]],
    terms: dict[str, list[tuple[str, str, str]]],
    train_clusters_per_tool: int,
    test_clusters_per_tool: int,
    train_expression_profile: str = "train",
) -> dict[str, list[dict[str, Any]]]:
    seed_errors = validate_seed_phrase_bank(catalog)
    if seed_errors:
        preview = "; ".join(seed_errors[:10])
        raise SystemExit(f"expression phrase bank failed validation: {preview}")
    seed_bank = phrase_bank()
    clusters: dict[str, list[dict[str, Any]]] = {}
    for tool in catalog:
        fragments = description_fragments(tool, terms)
        rows = []
        split_counts = [
            ("train", train_clusters_per_tool),
            ("test", test_clusters_per_tool),
        ]
        if train_expression_profile not in {"train", "test"}:
            split_counts.append((train_expression_profile, train_clusters_per_tool))
        for split, count in split_counts:
            seed_phrases = seed_bank.get(tool["tool_id"], {}).get(split, [])
            generated_phrases = phrase_variants(fragments, split, count)
            candidates = []
            for phrase in seed_phrases + generated_phrases:
                cleaned = safe_text(phrase, terms)
                if cleaned and cleaned not in candidates:
                    candidates.append(cleaned)
                if len(candidates) >= count:
                    break
            for idx, phrase in enumerate(candidates):
                retry = 0
                fixed = phrase
                while catalog_hits(fixed, terms):
                    fixed = fixed + ["这边", "具体", "当前", "整体"][retry % 4]
                    fixed = safe_text(fixed, terms)
                    retry += 1
                    if retry > 20:
                        raise SystemExit(f"cannot remove catalog trigger from {tool['tool_id']} phrase={phrase}")
                rows.append(
                    {
                        "tool_id": tool["tool_id"],
                        "split": split,
                        "cluster_id": f"{tool['tool_id']}_{split}_{idx:02d}",
                        "phrase": fixed,
                        "summary": f"{split}-only expression cluster from GPT-5.5 seed bank plus capability fragments",
                        "capability_fragments": fragments,
                    }
                )
        clusters[tool["tool_id"]] = rows
    return clusters


def expression_for(clusters: dict[str, list[dict[str, Any]]], tool_id: str, split: str, idx: int) -> dict[str, Any]:
    rows = [row for row in clusters[tool_id] if row["split"] == split]
    if not rows:
        raise ValueError(f"no {split} expression clusters for {tool_id}")
    return rows[idx % len(rows)]


def sample_context(idx: int) -> dict[str, str]:
    return {
        "product": PRODUCTS[idx % len(PRODUCTS)],
        "city": CITIES[(idx // 3) % len(CITIES)],
        "spec": SPECS[(idx // 5) % len(SPECS)],
        "budget": BUDGETS[(idx // 7) % len(BUDGETS)],
        "context": CONTEXTS[(idx // 11) % len(CONTEXTS)],
        "detail": DETAILS[(idx // 13) % len(DETAILS)],
        "tone": TONES[(idx // 17) % len(TONES)],
    }


def make_single_question(phrase: str, idx: int, context_salt: int = 0) -> tuple[str, str]:
    style_idx = idx + context_salt
    template = SINGLE_TEMPLATES[style_idx % len(SINGLE_TEMPLATES)]
    return template.format(phrase=phrase, **sample_context(style_idx)), f"single_style_{style_idx % len(SINGLE_TEMPLATES)}"


def join_clauses(phrases: list[str], idx: int) -> str:
    connectors = CONNECTOR_SETS[idx % len(CONNECTOR_SETS)]
    parts = [f"{connectors[pos % len(connectors)]}{phrase}" for pos, phrase in enumerate(phrases)]
    if idx % 4 == 0:
        return "，".join(parts)
    if idx % 4 == 1:
        return "；".join(parts)
    if idx % 4 == 2:
        return "，".join(parts[:-1]) + ("，并且" + parts[-1] if parts else "")
    return "，".join(parts)


def make_multi_question(phrases: list[str], idx: int, context_salt: int = 0) -> tuple[str, str]:
    style_idx = idx + context_salt
    context = sample_context(style_idx)
    context["clauses"] = join_clauses(phrases, style_idx)
    template = MULTI_TEMPLATES[style_idx % len(MULTI_TEMPLATES)]
    return template.format(**context), f"multi_style_{style_idx % len(MULTI_TEMPLATES)}"


def combo_key(tools: list[str]) -> tuple[str, ...]:
    return tuple(sorted(tools))


def candidate_chain(
    catalog: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    idx: int,
    min_tools: int,
    max_tools: int,
    salt: int = 0,
) -> list[str]:
    anchor = catalog[(idx * 17 + salt * 31) % len(catalog)]
    target_len = min_tools + ((idx + salt) % (max_tools - min_tools + 1))
    chain = [tool for tool in anchor.get("depends_on", []) if tool in by_id] + [anchor["tool_id"]]
    grouped = group_by_domain(catalog)
    same_domain = [row["tool_id"] for row in grouped[anchor["domain"]] if row["tool_id"] not in chain]
    confusable = [tool for tool in anchor.get("confusable_tools", []) if tool in by_id and tool not in chain]
    cross_domain = [row["tool_id"] for row in catalog if row["domain"] != anchor["domain"] and row["tool_id"] not in chain]
    for pool_idx, pool in enumerate([confusable, same_domain, cross_domain]):
        if not pool:
            continue
        start = (idx * (pool_idx + 3) + salt) % len(pool)
        for tool_id in pool[start:] + pool[:start]:
            if len(chain) >= target_len:
                break
            if tool_id not in chain:
                chain.append(tool_id)
    return order_tool_chain(chain[:max_tools], by_id)


def make_row(
    trace_id: str,
    question: str,
    tools: list[str],
    split: str,
    source: str,
    by_id: dict[str, dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    template_family: str,
    combo_bucket: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expression_cluster_ids = {row["tool_id"]: row["cluster_id"] for row in cluster_rows}
    expression_phrases = {row["tool_id"]: row["phrase"] for row in cluster_rows}
    domains = sorted({by_id[tool]["domain"] for tool in tools})
    row = {
        "id": trace_id,
        "问题": question,
        "工具": join_tools(tools),
        "思考": "用户表达了电商服务诉求；按只读、判断、计算、写入的业务顺序选择对应工具。",
        "source": source,
        "domain": ",".join(domains),
        "tool_count": len(tools),
        "generation_split": split,
        "expression_cluster_ids": expression_cluster_ids,
        "template_family": template_family,
        "combo_seen_bucket": combo_bucket,
    }
    meta = {
        "trace_id": trace_id,
        "split": split,
        "query": question,
        "tools": tools,
        "tool_count": len(tools),
        "source": source,
        "expression_cluster_ids": expression_cluster_ids,
        "expression_phrases": expression_phrases,
        "tool_evidence": expression_phrases,
        "template_family": template_family,
        "combo_key": list(combo_key(tools)),
        "combo_seen_bucket": combo_bucket,
        "generator_model": "gpt-5.5_self_generated_expression_cluster_generator_v2",
    }
    return row, meta


def valid_question(question: str, tools: list[str], terms: dict[str, list[tuple[str, str, str]]]) -> bool:
    return len(set(tools)) == len(tools) and not catalog_hits(question, terms)


def generate_singles(
    catalog: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    terms: dict[str, list[tuple[str, str, str]]],
    clusters: dict[str, list[dict[str, Any]]],
    split: str,
    per_tool: int,
    seen_questions: set[str],
    context_salt: int = 0,
    expression_split: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for tool_index, tool in enumerate(catalog):
        produced = 0
        attempt = 0
        while produced < per_tool and attempt < per_tool * 120:
            idx = tool_index * 100000 + attempt
            cluster = expression_for(
                clusters, tool["tool_id"], expression_split or split, idx
            )
            question, family = make_single_question(cluster["phrase"], idx, context_salt)
            attempt += 1
            key = normalize_question(question)
            if key in seen_questions or not valid_question(question, [tool["tool_id"]], terms):
                continue
            seen_questions.add(key)
            trace_id = f"single_{split}_{tool_index:03d}_{produced:03d}"
            row, meta = make_row(
                trace_id,
                question,
                [tool["tool_id"]],
                split,
                f"diverse_cluster_single_{split}",
                by_id,
                [cluster],
                family,
                "single",
            )
            rows.append(row)
            meta_rows.append(meta)
            produced += 1
        if produced < per_tool:
            raise SystemExit(f"{tool['tool_id']} only generated {produced}/{per_tool} {split} single rows")
    return rows, meta_rows


def generate_multi_rows(
    catalog: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    terms: dict[str, list[tuple[str, str, str]]],
    clusters: dict[str, list[dict[str, Any]]],
    split: str,
    target: int,
    min_tools: int,
    max_tools: int,
    seen_questions: set[str],
    train_combos: set[tuple[str, ...]] | None = None,
    seen_combo_rows: list[list[str]] | None = None,
    seen_ratio: float = 0.0,
    context_salt: int = 0,
    expression_split: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[tuple[str, ...]]]:
    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    produced_combos: set[tuple[str, ...]] = set()
    target_seen = int(round(target * seen_ratio)) if split == "test" else 0
    attempt = 0
    while len(rows) < target and attempt < target * 500:
        want_seen = split == "test" and len(rows) < target_seen and seen_combo_rows
        if want_seen:
            tools = seen_combo_rows[(len(rows) + attempt) % len(seen_combo_rows)]
            bucket = "multi_seen_combo"
        else:
            tools = candidate_chain(catalog, by_id, attempt, min_tools, max_tools, salt=len(rows) + (11 if split == "test" else 0))
            if split == "test" and train_combos and combo_key(tools) in train_combos:
                attempt += 1
                continue
            bucket = "multi_unseen_combo" if split == "test" else "multi_train_combo"

        cluster_rows = [
            expression_for(
                clusters,
                tool_id,
                expression_split or split,
                attempt + pos * 37,
            )
            for pos, tool_id in enumerate(tools)
        ]
        question, family = make_multi_question([row["phrase"] for row in cluster_rows], attempt, context_salt)
        attempt += 1
        key = normalize_question(question)
        if key in seen_questions or not valid_question(question, tools, terms):
            continue
        seen_questions.add(key)
        produced_combos.add(combo_key(tools))
        trace_id = f"multi_{split}_{len(rows):05d}"
        row, meta = make_row(
            trace_id,
            question,
            tools,
            split,
            f"diverse_cluster_multi_{split}",
            by_id,
            cluster_rows,
            family,
            bucket,
        )
        rows.append(row)
        meta_rows.append(meta)
    if len(rows) < target:
        raise SystemExit(f"only generated {len(rows)}/{target} {split} multi rows")
    return rows, meta_rows, produced_combos


def stats_for_rows(rows: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "by_split": dict(sorted(Counter(row.get("generation_split", "") for row in rows).items())),
        "by_tool_count": dict(sorted(Counter(int(row.get("tool_count", 0)) for row in rows).items())),
        "by_tool": dict(sorted(tool_counts(rows).items())),
        "by_domain": dict(sorted(domain_counts(rows, by_id).items())),
        "by_combo_bucket": dict(sorted(Counter(row.get("combo_seen_bucket", "") for row in rows).items())),
        "duplicates_after_normalization": len(rows) - len({normalize_question(row["问题"]) for row in rows}),
    }


def split_stats(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    train_keys = {normalize_question(row["问题"]) for row in train_rows}
    test_keys = {normalize_question(row["问题"]) for row in test_rows}
    train_ids = {row["id"] for row in train_rows}
    test_ids = {row["id"] for row in test_rows}
    train_multi_combos = {combo_key(split_tools(row.get("工具"))) for row in train_rows if int(row.get("tool_count", 0)) > 1}
    test_multi = [row for row in test_rows if int(row.get("tool_count", 0)) > 1]
    seen_multi_rows = sum(1 for row in test_multi if combo_key(split_tools(row.get("工具"))) in train_multi_combos)
    return {
        "source": "diverse_expression_cluster_generator",
        "train": len(train_rows),
        "test": len(test_rows),
        "exact_normalized_query_overlap": len(train_keys & test_keys),
        "trace_id_overlap": len(train_ids & test_ids),
        "missing_train_tools": sorted(set(by_id) - set(tool_counts(train_rows))),
        "missing_test_tools": sorted(set(by_id) - set(tool_counts(test_rows))),
        "train_by_tool": dict(sorted(tool_counts(train_rows).items())),
        "test_by_tool": dict(sorted(tool_counts(test_rows).items())),
        "train_by_domain": dict(sorted(domain_counts(train_rows, by_id).items())),
        "test_by_domain": dict(sorted(domain_counts(test_rows, by_id).items())),
        "train_by_tool_count": dict(sorted(Counter(int(row.get("tool_count", 0)) for row in train_rows).items())),
        "test_by_tool_count": dict(sorted(Counter(int(row.get("tool_count", 0)) for row in test_rows).items())),
        "test_multi_rows_with_combo_seen_in_train": seen_multi_rows,
        "test_multi_rows_with_combo_seen_in_train_rate": round(seen_multi_rows / max(1, len(test_multi)), 6),
        "test_multi_rows_with_combo_unseen_in_train": len(test_multi) - seen_multi_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--single-out", type=Path, default=TOOL_TRACE_DIR / "single_intent.jsonl")
    parser.add_argument("--multi-out", type=Path, default=TOOL_TRACE_DIR / "multi_intent.jsonl")
    parser.add_argument("--train-out", type=Path, default=TOOL_TRACE_DIR / "train.jsonl")
    parser.add_argument("--test-out", type=Path, default=TOOL_TRACE_DIR / "test.jsonl")
    parser.add_argument("--single-stats-out", type=Path, default=TOOL_TRACE_DIR / "_aux" / "single_stats.json")
    parser.add_argument("--multi-stats-out", type=Path, default=TOOL_TRACE_DIR / "_aux" / "multi_stats.json")
    parser.add_argument("--split-stats-out", type=Path, default=TOOL_TRACE_DIR / "_aux" / "split_stats.json")
    parser.add_argument("--generation-stats-out", type=Path, default=TOOL_TRACE_DIR / "_aux" / "diverse_generation_stats.json")
    parser.add_argument("--compat-hard-stats-out", type=Path, default=TOOL_TRACE_DIR / "_aux" / "hard_generation_stats.json")
    parser.add_argument("--expression-clusters-out", type=Path, default=TOOL_TRACE_DIR / "_aux" / "expression_clusters.json")
    parser.add_argument("--generation-meta-out", type=Path, default=TOOL_TRACE_DIR / "_aux" / "generation_meta.jsonl")
    parser.add_argument("--single-train-per-tool", type=int, default=90)
    parser.add_argument("--single-test-per-tool", type=int, default=30)
    parser.add_argument("--multi-train-target", type=int, default=15000)
    parser.add_argument("--multi-test-target", type=int, default=3000)
    parser.add_argument("--test-seen-combo-ratio", type=float, default=0.5)
    parser.add_argument("--train-clusters-per-tool", type=int, default=8)
    parser.add_argument("--test-clusters-per-tool", type=int, default=6)
    parser.add_argument(
        "--train-expression-profile",
        choices=["train", "dpo_hard"],
        default="train",
        help=(
            "Expression cluster family for generated training rows. dpo_hard "
            "is disjoint from both canonical train and frozen test clusters."
        ),
    )
    parser.add_argument(
        "--train-context-salt",
        type=int,
        default=0,
        help="Offset for non-intent context/template sampling in train rows. Keep 0 to reproduce the current frozen split.",
    )
    parser.add_argument(
        "--test-context-salt",
        type=int,
        default=0,
        help="Offset for non-intent context/template sampling in test rows. Use a large nonzero value when creating a stricter context-isolated split.",
    )
    parser.add_argument("--min-tools", type=int, default=2)
    parser.add_argument("--max-tools", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    random.seed(args.seed)
    catalog = load_catalog(args.catalog)
    by_id = catalog_by_id(catalog)
    terms = catalog_term_index(catalog)
    clusters = build_expression_clusters(
        catalog,
        terms,
        args.train_clusters_per_tool,
        args.test_clusters_per_tool,
        args.train_expression_profile,
    )
    seen_questions: set[str] = set()

    train_single, train_single_meta = generate_singles(
        catalog,
        by_id,
        terms,
        clusters,
        "train",
        args.single_train_per_tool,
        seen_questions,
        context_salt=args.train_context_salt,
        expression_split=args.train_expression_profile,
    )
    test_single, test_single_meta = generate_singles(
        catalog,
        by_id,
        terms,
        clusters,
        "test",
        args.single_test_per_tool,
        seen_questions,
        context_salt=args.test_context_salt,
    )
    train_multi, train_multi_meta, train_combos = generate_multi_rows(
        catalog,
        by_id,
        terms,
        clusters,
        "train",
        args.multi_train_target,
        args.min_tools,
        args.max_tools,
        seen_questions,
        context_salt=args.train_context_salt,
        expression_split=args.train_expression_profile,
    )
    seen_combo_rows = [split_tools(row.get("工具")) for row in train_multi]
    test_multi, test_multi_meta, _ = generate_multi_rows(
        catalog,
        by_id,
        terms,
        clusters,
        "test",
        args.multi_test_target,
        args.min_tools,
        args.max_tools,
        seen_questions,
        train_combos=train_combos,
        seen_combo_rows=seen_combo_rows,
        seen_ratio=args.test_seen_combo_ratio,
        context_salt=args.test_context_salt,
    )

    train_rows = train_single + train_multi
    test_rows = test_single + test_multi
    single_rows = train_single + test_single
    multi_rows = train_multi + test_multi
    meta_rows = train_single_meta + test_single_meta + train_multi_meta + test_multi_meta

    write_jsonl(args.single_out, single_rows)
    write_jsonl(args.multi_out, multi_rows)
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.test_out, test_rows)
    write_jsonl(args.generation_meta_out, meta_rows)
    write_json(args.expression_clusters_out, clusters)

    single_stats = stats_for_rows(single_rows, by_id)
    multi_stats = stats_for_rows(multi_rows, by_id)
    split = split_stats(train_rows, test_rows, by_id)
    policy = {
        "generator": "gpt-5.5_self_generated_expression_cluster_generator_v2",
        "external_api_used": False,
        "note": "由当前 GPT-5.5 agent 编写的生成逻辑按 catalog 能力描述生成表达簇；未使用旧 HARD_PHRASES。",
        "train_expression_clusters": "每工具 train-only 能力片段表达簇。",
        "test_expression_clusters": "每工具 test-only 能力片段表达簇，使用不同描述片段和话语模板。",
        "single_train_per_tool": args.single_train_per_tool,
        "single_test_per_tool": args.single_test_per_tool,
        "multi_train_target": args.multi_train_target,
        "multi_test_target": args.multi_test_target,
        "test_seen_combo_ratio": args.test_seen_combo_ratio,
        "train_context_salt": args.train_context_salt,
        "test_context_salt": args.test_context_salt,
        "train_expression_profile": args.train_expression_profile,
        "seed": args.seed,
    }
    generation_stats = {
        "catalog_tools": len(catalog),
        "policy": policy,
        "single": single_stats,
        "multi": multi_stats,
        "split": split,
        "outputs": {
            "single": str(args.single_out),
            "multi": str(args.multi_out),
            "train": str(args.train_out),
            "test": str(args.test_out),
            "generation_meta": str(args.generation_meta_out),
            "expression_clusters": str(args.expression_clusters_out),
        },
    }
    write_json(args.single_stats_out, single_stats)
    write_json(args.multi_stats_out, multi_stats)
    write_json(args.split_stats_out, split)
    write_json(args.generation_stats_out, generation_stats)
    write_json(args.compat_hard_stats_out, generation_stats)

    print(
        "diverse trace dataset "
        f"single={len(single_rows)} multi={len(multi_rows)} train={len(train_rows)} test={len(test_rows)} "
        f"query_overlap={split['exact_normalized_query_overlap']} combo_seen_rate={split['test_multi_rows_with_combo_seen_in_train_rate']}"
    )
    return 0 if split["exact_normalized_query_overlap"] == 0 and split["trace_id_overlap"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
