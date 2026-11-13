"""高精度整轮复用识别与会话内来源轮排序。"""

from __future__ import annotations

import re
from collections.abc import Callable


TURN_REFERENCE = re.compile(r"第\s*(?:\d+|[一二三四五六七八九十]+)\s*轮")
HISTORY_ANCHOR = re.compile(
    r"刚才|刚刚|上一轮|上一步|上一条|之前|先前|前面|上次|早些时候|很久前|"
    r"最开始|一开始|最早|历史|那会儿|当时|曾经|"
    r"(?:那|这)(?:次|回|条|段|套|单|笔|件)(?:的)?|"
    r"(?:那些|这几件|那几件|这几项|那几项)|"
    r"提到|提过|说到|说过|聊到|聊过|问过|处理过|夹在.{1,40}中间"
)
WHOLE_SCOPE = re.compile(
    r"整套|整段|整个|全部|所有|一整套|完整|从头|"
    r"顺序.{0,8}(?:别|不)(?:变|改|动)|"
    r"内容.{0,8}(?:别|不)(?:变|改|动)|"
    r"一项.{0,4}(?:别|不)(?:漏|少)"
)
REPLAY_ACTION = re.compile(
    r"原样|照旧|照原样|照原来|按原来|按当时|按原顺序|"
    r"再来|再办|再做|再走|重做|重新(?:做|办|来|走|执行|处理)|重复|恢复|"
    r"继续(?:办|做|处理|弄|安排|查|看|记)?|接着(?:办|做|处理|弄|安排|查|看|记)?|"
    r"记(?:着|住|一下|上)|照办|按着办|"
    r"(?:按|照)(?:这个|那个).{0,10}(?:来|办|做|处理|安排|记)|"
    r"(?:都|一起)(?:办|做|处理|安排|看|查)|"
    r"(?:就)?(?:按|照)(?:刚才|前面)(?:说的|聊的|谈的)?"
)
STRONG_REPLAY = re.compile(r"原样|照旧|照原样|照原来|按原来|按原顺序")
CONVERSATION_ANCHOR = re.compile(
    r"第\s*(?:\d+|[一二三四五六七八九十]+)\s*轮|"
    r"刚才|刚刚|上一轮|上一步|上一条|之前|先前|前面|上次|最开始|一开始|"
    r"最早|历史|(?:那|这)(?:次|回|条|段|套)(?:的)?|"
    r"提到|提过|说到|说过|聊到|聊过|夹在.{1,40}中间"
)
REEXECUTION = re.compile(
    r"再来|再办|再做|再走|再查.{0,4}(?:遍|次|回)|重做|重新|重复|恢复"
)
EARLIEST = re.compile(r"最早|最开始|一开始|最前面|第一次")
LATEST = re.compile(r"最后一次|最后那次|最近一次|最晚那次")
SHORTEST = re.compile(r"最短|只说|只问|只看|只提|单独")
NATURAL_WHOLE_SCOPE = re.compile(
    r"(?:刚才|刚刚|前面|之前)(?:说|提|聊|谈)?(?:过)?的?"
    r"(?:那些|这几件|那几件|几件|几个|这几项|那几项|几项)"
    r"(?:事|事情|需求|问题|安排)?|"
    r"(?:那些|这几件|那几件|这几项|那几项)(?:事|事情|需求|问题|安排)?|"
    r"(?:刚才|刚刚|前面|之前)(?:说|提|聊|谈)?(?:过)?的?.{0,16}(?:都|一起)|"
    r"(?:刚才|刚刚|前面|之前).{0,28}(?:的事|这事|那事|这个问题|那个问题)"
)
IMPLICIT_CONFIRMATION = re.compile(
    r"^(?:(?:嗯|好|好的|行|可以|没问题|成)[，,、\s]*)?"
    r"(?:(?:那|就|那就|先)?(?:按|照)"
    r"(?:这个|这么|刚才(?:说的)?(?:那个|那样)?|你?刚刚说的(?:那个)?|前面说的)"
    r"(?:帮我)?(?:来|办|做|处理|安排|继续)|"
    r"(?:那|就|那就)?这么(?:办|做|处理|安排|改|提交|确认|预约|取消|下单|定(?:了|下来)?)|"
    r"(?:麻烦)?继续(?:帮我)?(?:办|做|处理|弄|安排))"
)
ASSENT_PREFIX = re.compile(r"^(?:嗯|好|好的|行|可以|没问题|成)(?:[，,、\s]|$)")
CONFIRMATION_DIRECTIVE = re.compile(
    r"(?:按|照)(?:这个|这么|刚才(?:说的)?(?:那个|那样)?|"
    r"你?刚刚说的(?:那个)?|前面说的).{0,24}?"
    r"(?:帮我)?(?:来|办|做|处理|安排|继续|改|删|用|提交|预约|确认)|"
    r"这么(?:办|做|处理|安排|改|提交|确认|预约|取消|下单|定(?:了|下来)?)"
)


_CLEAN_PATTERNS = (
    r"第\s*(?:\d+|[一二三四五六七八九十]+)\s*轮",
    r"顺序(?:和内容|内容)?都?(?:别|不要|不能)(?:变|改|动)",
    r"内容(?:和顺序|顺序)?都?(?:别|不要|不能)(?:变|改|动)",
    r"(?:别|不要)(?:改|漏|少)(?:任何)?(?:内容|步骤|一项|东西)?",
    r"(?:麻烦|劳驾|拜托|请|帮我|给我|替我)",
    r"(?:之前|先前|前面|上次|早些时候|很久前|最开始|一开始|最早|"
    r"历史(?:上|里|记录中|任务中)?|那会儿|当时|曾经|刚才|刚刚|"
    r"上一轮|上一步|上一条)",
    r"(?:提到|提过|说到|说过|聊到|聊过|问过|处理过|确认过)",
    r"(?:那些|这几件|那几件|几件|几个|这几项|那几项|几项)"
    r"(?:事|事情|需求|问题|安排)?",
    r"(?:整套|整段|整个|一整套|完整|全部|所有|从头)(?:的)?"
    r"(?:需求|诉求|任务|操作|步骤|核对)?",
    r"(?:那|这)(?:次|回|条|段|套|单|笔|件)(?:的)?",
    r"(?:需求|诉求|任务)(?:按|照)?",
    r"(?:按|照)?(?:原样|照旧|原来|当时|原顺序)",
    r"(?:重新|再)(?:来|办|做|走|查|处理|执行)(?:一|这|那)?(?:遍|次|回)?",
    r"(?:重做|重复|恢复)(?:一|这|那)?(?:遍|次|回)?",
    r"(?:一|这|那)(?:遍|次|回)",
    r"(?:还是|就按|就)",
    r"(?:继续|接着)(?:办|做|处理|弄|安排|查|看)?",
)
REFERENCE_BOILERPLATE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in _CLEAN_PATTERNS)
)
PUNCTUATION = re.compile(r"[，,。；;！!？?：:“”‘’（）()\s]+")
BETWEEN = re.compile(
    r"夹在(?P<left>[^，。]{1,36}?)(?:和|与)(?P<right>[^，。]{1,36}?)中间"
)
RELATIVE = re.compile(
    r"(?P<anchor>[^，。]{2,42}?)(?P<direction>后面|之后|前面)"
    r"(?P<target>[^，。]{0,42})"
)


def is_whole_turn_reference(query: str) -> bool:
    """只在历史锚点和整轮复用语义同时存在时触发。"""

    if is_implicit_confirmation(query):
        return True
    has_history = bool(TURN_REFERENCE.search(query) or HISTORY_ANCHOR.search(query))
    if not has_history or not REPLAY_ACTION.search(query):
        return False
    if WHOLE_SCOPE.search(query) or NATURAL_WHOLE_SCOPE.search(query):
        return True
    return bool(
        STRONG_REPLAY.search(query)
        and CONVERSATION_ANCHOR.search(query)
        and REEXECUTION.search(query)
    )


def is_implicit_confirmation(query: str) -> bool:
    """识别只能自然承接紧邻上一条的简短确认。"""

    value = query.strip()
    return bool(
        IMPLICIT_CONFIRMATION.search(value)
        or (
            len(value) <= 44
            and ASSENT_PREFIX.search(value)
            and CONFIRMATION_DIRECTIVE.search(value)
        )
    )


def reference_search_text(query: str) -> str:
    """删除复用元话语，保留用于定位来源轮的实体、事件和原句片段。"""

    value = query
    for _ in range(2):
        value = REFERENCE_BOILERPLATE.sub(" ", value)
    return PUNCTUATION.sub(" ", value).strip()


def _best_turn(
    query: str,
    documents: list[tuple[int, str]],
    score_documents: Callable[[str, list[str]], list[float]],
) -> int | None:
    query = reference_search_text(query)
    if not query or not documents:
        return None
    scores = score_documents(query, [text for _, text in documents])
    if not scores or max(scores) <= 0.0:
        return None
    return max(
        range(len(documents)),
        key=lambda index: (scores[index], documents[index][0]),
    )


def structural_reference_turn(
    query: str,
    documents: list[tuple[int, str]],
    score_documents: Callable[[str, list[str]], list[float]],
) -> int | None:
    """解析“夹在 A/B 中间”和“A 前后那条 B”这类可观察位置约束。"""

    if not documents:
        return None
    by_turn = {turn: text for turn, text in documents}
    between = BETWEEN.search(query)
    if between:
        left_index = _best_turn(between.group("left"), documents, score_documents)
        right_index = _best_turn(between.group("right"), documents, score_documents)
        if left_index is not None and right_index is not None:
            left = documents[left_index][0]
            right = documents[right_index][0]
            candidates = [turn for turn in by_turn if min(left, right) < turn < max(left, right)]
            if len(candidates) == 1:
                return candidates[0]

    for match in RELATIVE.finditer(query):
        anchor_text = reference_search_text(match.group("anchor"))
        target_text = reference_search_text(match.group("target"))
        if len(anchor_text) < 2:
            continue
        anchor_index = _best_turn(anchor_text, documents, score_documents)
        if anchor_index is None:
            continue
        anchor_turn = documents[anchor_index][0]
        direction = match.group("direction")
        if direction in {"后面", "之后"}:
            candidates = [(turn, text) for turn, text in documents if turn > anchor_turn]
            step = 1
        else:
            candidates = [(turn, text) for turn, text in documents if turn < anchor_turn]
            step = -1
        if not candidates:
            continue
        adjacent = anchor_turn + step
        if "紧接" in query and adjacent in by_turn:
            return adjacent
        if len(target_text) >= 2:
            target_index = _best_turn(target_text, candidates, score_documents)
            if target_index is not None:
                return candidates[target_index][0]
        if adjacent in by_turn:
            return adjacent
    return None


def temporally_rerank(
    query: str,
    documents: list[tuple[int, str]],
    scores: list[float],
    near_best_ratio: float = 0.70,
) -> list[int]:
    """在词面近似同分时落实最早/最后/最短约束，其余保持分数与近期优先。"""

    if len(documents) != len(scores):
        raise ValueError("documents and scores must align")
    if not documents:
        return []
    best = max(scores)
    near = [
        index
        for index, score in enumerate(scores)
        if score > 0.0 and score >= best * near_best_ratio
    ]
    if EARLIEST.search(query) and near:
        preferred = min(near, key=lambda index: documents[index][0])
    elif LATEST.search(query) and near:
        preferred = max(near, key=lambda index: documents[index][0])
    elif SHORTEST.search(query) and near:
        preferred = min(
            near,
            key=lambda index: (len(documents[index][1]), documents[index][0]),
        )
    else:
        preferred = max(
            range(len(documents)),
            key=lambda index: (scores[index], documents[index][0]),
        )
    rest = sorted(
        (index for index in range(len(documents)) if index != preferred),
        key=lambda index: (-scores[index], -documents[index][0]),
    )
    return [preferred, *rest]
