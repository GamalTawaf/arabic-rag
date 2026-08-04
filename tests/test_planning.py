"""Query planning tests.

The Arabic strings here are copied verbatim out of ``evals/data/eval_pairs.jsonl``
rather than invented, so a test failing means the planner broke on a question a
real annotator wrote. Two tests read the dataset itself and assert the aggregate
detector numbers — those are the regression guard on the lexicon: adding a marker
that misfires on MSA fails the suite instead of quietly costing precision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.constants import MAX_SEARCH_QUERIES, REGISTER_MARKERS
from app.data import Completion, Usage
from app.planning import (
    GULF_TO_MSA,
    LLMPlanner,
    NoopPlanner,
    PlannerUnavailable,
    RuleBasedPlanner,
    decompose,
    detect_register,
    get_planner,
    gulf_to_msa,
)
from app.planning.lexicon import _merge
from ingestion.normalize import normalize_query

DATASET = Path(__file__).resolve().parents[1] / "evals" / "data" / "eval_pairs.jsonl"

# Verbatim from the eval set.
GULF_Q = "شكثر ايام اجازه سنويه تجيني بعد ما اكمل سنه؟ وتزيد لو صار لي خمس سنين بالشركة؟"
GULF_TWO_PART = (
    "شكثر لازم يكونون العمال عندهم لين يجيبون ممرض دوام كامل، "
    "ومتى يصير لازم يفتحون عيادة فيها دكتور؟"
)
MSA_Q = "ما مدة الإخطار المطلوبة إذا رغب العامل في إنهاء عقد العمل غير محدد المدة؟"
MSA_SINGLE = "كم يوماً إجازة سنوية يستحق العامل بعد إتمام سنة من الخدمة؟"


def _dataset() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── lexicon ───────────────────────────────────────────────────────────────────
def test_lexicon_has_no_conflicting_keys():
    # Arrange: the same key in two category tables with two different values.
    # Act / Assert
    with pytest.raises(ValueError, match="conflicting lexicon entry"):
        _merge({"شكثر": "كم"}, {"شكثر": "متى"})


def test_lexicon_keys_are_normalized():
    """A key with a hamza or ta-marbuta could never be looked up: lookups normalize first."""
    unreachable = [key for key in GULF_TO_MSA if normalize_query(key) != key]
    assert unreachable == []


def test_lexicon_markers_are_normalized():
    unreachable = [marker for marker in REGISTER_MARKERS if normalize_query(marker) != marker]
    assert unreachable == []


def test_lexicon_values_are_not_themselves_keys():
    """What guarantees gulf_to_msa is idempotent — a rewrite must not be re-rewritten."""
    recursive = [
        (key, value)
        for key, value in GULF_TO_MSA.items()
        for token in normalize_query(value).split()
        if token in GULF_TO_MSA
    ]
    assert recursive == []


# ── detect_register ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "question",
    [
        GULF_Q,
        "وين ما يصير العمال يسوون اضراب؟ يعني في اي مجالات ممنوع منع بات؟",
        "الكفيل يعطيني راتبي كاش في يدي ولا لازم ينزله بالبنك؟",
        "انا لسا بفترة التجربة وابغى انتقل لشركة ثانيه، وش اسوي؟",
        "لو سكروا الشركة شوي عشان فيه خطر على العمال، يوقفون رواتبنا في هالمدة ولا لأ؟",
    ],
)
def test_detects_gulf_register_on_real_questions(question):
    assert detect_register(question) == "gulf"


@pytest.mark.parametrize(
    "question",
    [
        MSA_Q,
        MSA_SINGLE,
        "هل يجوز أن يسلمني صاحب العمل راتبي نقداً باليد؟",
        "ما الحد الأقصى لساعات العمل الأسبوعية وفقاً لقانون العمل القطري؟",
    ],
)
def test_does_not_flag_msa_questions_as_gulf(question):
    assert detect_register(question) == "msa"


def test_register_detection_over_the_whole_dataset():
    """Measured: 53/54 Gulf detected, 0/229 MSA misfires. Precision is the one that must hold."""
    pairs = _dataset()
    gulf = [p for p in pairs if p["dialect_tag"] == "gulf"]
    msa = [p for p in pairs if p["dialect_tag"] == "msa"]

    detected = sum(1 for p in gulf if detect_register(p["question"]) == "gulf")
    misfired = [p["id"] for p in msa if detect_register(p["question"]) == "gulf"]

    assert misfired == []  # a false positive means answering MSA in dialect
    assert detected >= 53, f"gulf recall regressed: {detected}/{len(gulf)}"


# ── gulf_to_msa ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("gulf", "expected_fragment"),
    [
        ("شكثر ياخذ وقت", "كم"),  # شكثر -> كم
        ("وش المطلوب مني", "ما"),  # وش -> ما
        ("منو اللي يدفع", "من الذي"),  # منو -> من, اللي -> الذي
        ("راتبي شلون بينزل", "كيف"),  # شلون -> كيف
        ("وين ما يصير الاضراب", "أين"),  # وين -> أين
        ("الكفيل يعطيني راتبي", "صاحب العمل"),  # الكفيل -> صاحب العمل
        ("الكفيل يعطيني راتبي", "أجري"),  # راتبي -> أجري
        ("يوقفون الشغل بالموقع", "العمل"),  # الشغل -> العمل
        ("صار لي خمس سنين", "سنوات"),  # سنين -> سنوات
        ("يوقفون رواتبنا في هالمدة", "المده"),  # هالـ clitic dropped
        ("وشكثر ياخذون وقت", "وكم"),  # the conjunction survives the rewrite
        ("ابغى اروح احج", "أريد"),  # أبغى -> أريد (hamza-insensitive lookup)
    ],
)
def test_gulf_to_msa_rewrites_documented_forms(gulf, expected_fragment):
    assert expected_fragment in gulf_to_msa(gulf)


def test_gulf_to_msa_leaves_msa_questions_untouched():
    for question in (MSA_Q, MSA_SINGLE, "ما الحد الأدنى للأجر في قانون العمل القطري؟"):
        assert gulf_to_msa(question) == question


def test_gulf_to_msa_does_not_touch_walaw():
    """ولو is MSA "even if", not و + لو — rewriting it to وإذا flips the meaning."""
    question = "هل يستطيع الموظف أن يشتغل لدى جهة أخرى ولو بدون مقابل؟"
    assert gulf_to_msa(question) == question


def test_gulf_to_msa_is_idempotent():
    once = gulf_to_msa(GULF_Q)
    assert gulf_to_msa(once) == once


def test_gulf_to_msa_preserves_punctuation_and_spacing():
    assert gulf_to_msa("شكثر؟  وش!") == "كم؟  ما!"


def test_gulf_to_msa_leaves_ambiguous_ma_alone():
    """ما is negation in Gulf and "what" in MSA; no token rule can tell them apart."""
    assert "ما" in gulf_to_msa("ما عندي عقد مكتوب مع الشركة")
    assert "لا" not in gulf_to_msa("ما عندي عقد مكتوب مع الشركة")


# ── decompose ─────────────────────────────────────────────────────────────────
def test_decompose_splits_a_real_two_part_question():
    parts = decompose(GULF_TWO_PART)
    assert len(parts) == 2
    assert parts[0].startswith("شكثر")
    assert parts[1].startswith("ومتى")


def test_decompose_leaves_single_questions_alone():
    for question in (MSA_SINGLE, "شكثر اجازة الولاده؟", MSA_Q):
        assert decompose(question) == [question]


def test_decompose_does_not_split_on_a_repeated_interrogative():
    """Same interrogative twice is one topic asked twice, not two sub-queries."""
    question = "كم ساعة عمل في الأسبوع وكم ساعة في اليوم؟"
    assert decompose(question) == [question]


def test_decompose_does_not_split_a_negated_clause():
    """"وما أخذ ولا ريال" is a statement; ما is excluded from the interrogative set."""
    question = "شكثر يوم اطفش من الدوام بدون عذر لين يطيروني وما اخذ ولا ريال اخر الخدمه؟"
    assert decompose(question) == [question]


def test_decompose_ignores_two_word_fragments():
    assert decompose("كم يوم؟ وكيف؟") == ["كم يوم؟ وكيف؟"]


# ── planners ──────────────────────────────────────────────────────────────────
async def test_noop_planner_searches_with_the_original_only():
    plan = await NoopPlanner().plan(GULF_Q)

    assert plan.search_queries == [GULF_Q]
    assert plan.rewritten is None
    assert plan.register == "gulf"  # register is still detected: it drives the answer
    assert plan.strategy == "noop"


async def test_rule_based_planner_searches_with_both_original_and_rewrite():
    plan = await RuleBasedPlanner().plan(GULF_Q)

    assert plan.search_queries[0] == GULF_Q  # the user's words are never dropped
    assert len(plan.search_queries) == 2
    assert plan.rewritten == plan.search_queries[1]
    assert "كم" in plan.rewritten
    assert plan.register == "gulf"
    assert plan.strategy == "rules"


async def test_rule_based_planner_passes_msa_through_unrewritten():
    plan = await RuleBasedPlanner().plan(MSA_Q)

    assert plan.search_queries == [MSA_Q]
    assert plan.rewritten is None
    assert plan.register == "msa"


async def test_rule_based_planner_does_not_decompose_by_default():
    """Measured: decomposition costs 4-5 points of recall@3 and adds nothing at k=10."""
    plan = await RuleBasedPlanner().plan(GULF_TWO_PART)

    assert len(plan.search_queries) == 2  # original + rewrite, no sub-queries


async def test_rule_based_planner_appends_sub_queries_when_enabled():
    plan = await RuleBasedPlanner(decompose_enabled=True).plan(GULF_TWO_PART)

    assert plan.search_queries[0] == GULF_TWO_PART
    assert len(plan.search_queries) == 4  # original + rewrite + two sub-queries
    assert plan.search_queries[-1].startswith("ومتى")


async def test_rule_based_planner_rewrites_the_sub_queries_too():
    plan = await RuleBasedPlanner(decompose_enabled=True).plan(GULF_TWO_PART)

    assert "شكثر" not in plan.search_queries[2]  # the Gulf interrogative is gone
    assert "كم" in plan.search_queries[2]


async def test_rule_based_planner_caps_the_query_count():
    three_part = (
        "شكثر اجازة الولاده؟ وشنو الشرط عشان استحقها؟ "
        "وشكثر لازم يبقى منها بعد ما اولد؟"
    )
    plan = await RuleBasedPlanner(decompose_enabled=True).plan(three_part)

    assert len(plan.search_queries) <= MAX_SEARCH_QUERIES


# ── LLM planner ───────────────────────────────────────────────────────────────
class _MockProvider:
    """Stands in for app.generation's Provider — the REAL signature.

    ``complete(system, messages, max_tokens) -> Completion``. An earlier version
    of this double took a single prompt string and returned a bare ``str``, a
    contract no shipped provider implements; it passed happily while LLMPlanner
    could not actually call any provider at all. A double that does not match the
    protocol tests nothing.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.systems: list[str] = []

    async def complete(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> Completion:
        self.systems.append(system)
        self.prompts.append("\n".join(message["content"] for message in messages))
        return Completion(
            text=self.reply,
            usage=Usage(input_tokens=10, output_tokens=20, cost_usd=0.0),
            provider="mock",
            model="mock-1",
        )


async def test_llm_planner_uses_the_provider_rewrite_and_sub_queries():
    provider = _MockProvider(
        json.dumps(
            {"rewritten": "كم يوم إجازة سنوية؟", "sub_queries": ["كم يوم إجازة؟", "هل تزيد بعد خمس سنوات؟"]},
            ensure_ascii=False,
        )
    )

    plan = await LLMPlanner(provider).plan(GULF_Q)

    assert GULF_Q in provider.prompts[0]
    assert plan.strategy == "llm"
    assert plan.register == "gulf"
    assert plan.rewritten == "كم يوم إجازة سنوية؟"
    assert plan.search_queries == [
        GULF_Q,
        "كم يوم إجازة سنوية؟",
        "كم يوم إجازة؟",
        "هل تزيد بعد خمس سنوات؟",
    ]


async def test_llm_planner_tolerates_prose_around_the_json():
    provider = _MockProvider('هذا هو الناتج:\n{"rewritten": "كم يوم؟", "sub_queries": []}\nانتهى')

    plan = await LLMPlanner(provider).plan(GULF_Q)

    assert plan.rewritten == "كم يوم؟"


async def test_llm_planner_falls_back_to_rules_on_unparseable_reply(caplog):
    plan = await LLMPlanner(_MockProvider("sorry, I cannot do that")).plan(GULF_Q)

    assert plan.strategy == "rules"  # visible in the Plan, not swallowed
    assert plan.search_queries[0] == GULF_Q
    assert "unusable" in caplog.text


def test_llm_planner_without_a_provider_errors_clearly(monkeypatch):
    """No API key on this machine: construction must explain itself, not traceback."""
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "", raising=False)
    monkeypatch.setattr("app.config.settings.google_api_key", "", raising=False)

    with pytest.raises(PlannerUnavailable) as excinfo:
        LLMPlanner()

    assert "rules" in str(excinfo.value)  # names the working alternative


def test_llm_planner_builds_the_configured_chain_when_a_key_is_present(monkeypatch):
    """Regression: ``get_provider()`` was called with no argument against
    ``get_provider(name)``, so construction ALWAYS raised PlannerUnavailable —
    and blamed a missing API key that was in fact set."""
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "sk-test", raising=False)
    monkeypatch.setattr("app.config.settings.providers", "anthropic", raising=False)

    planner = LLMPlanner()

    assert planner._provider.name == "failover:anthropic"


async def test_llm_planner_calls_the_provider_with_the_real_protocol():
    """system + messages + max_tokens, and it reads ``Completion.text``."""
    provider = _MockProvider('{"rewritten": "كم يوم؟", "sub_queries": []}')

    plan = await LLMPlanner(provider).plan(GULF_Q)

    assert provider.systems[0].startswith("أنت مساعد")
    assert GULF_Q in provider.prompts[0]
    assert plan.strategy == "llm"


# ── factory ───────────────────────────────────────────────────────────────────
def test_get_planner_defaults_to_rules():
    assert isinstance(get_planner(), RuleBasedPlanner)


def test_get_planner_builds_each_offline_strategy():
    assert isinstance(get_planner("noop"), NoopPlanner)
    assert isinstance(get_planner("rules"), RuleBasedPlanner)


def test_get_planner_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="unknown planning strategy"):
        get_planner("magic")


def test_get_planner_llm_is_unavailable_without_keys(monkeypatch):
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "", raising=False)
    monkeypatch.setattr("app.config.settings.google_api_key", "", raising=False)

    with pytest.raises(PlannerUnavailable):
        get_planner("llm")


def test_a_word_that_merely_ends_in_an_interrogative_is_not_one():
    """Regression: `token[1:]` stripped the first letter of EVERY token.

    حكم ("ruling") became كم ("how many") — and this is a labour-law corpus, so
    that word is everywhere. The phantom interrogative invented a second question
    and split the query into two searches, one of them nonsense.
    """
    assert decompose("ما حكم الفصل التعسفي في القانون؟") == [
        "ما حكم الفصل التعسفي في القانون؟"
    ]


def test_a_real_proclitic_conjunction_is_still_stripped():
    """و and ف genuinely glue onto the next word; فماذا is فـ + ماذا."""
    parts = decompose(
        "إذا استدعي الموظف في يوم راحته، فماذا يستحق؟ وهل يمكن تكرار ذلك؟"
    )
    assert len(parts) == 2


def test_an_ordinary_word_starting_with_sheen_is_not_a_clause_boundary():
    """Regression: the one-letter "ش" stem matched the start of any ش word.

    "الأجر وشروط العمل" — "pay and conditions of work" — split into two clauses
    because وشروط looks like و + ش.
    """
    assert decompose("هل الأجر وشروط العمل مذكورة في العقد؟") == [
        "هل الأجر وشروط العمل مذكورة في العقد؟"
    ]


def test_the_dataset_decomposition_rate_has_not_regressed():
    """The decomposition arm is a measured result; guard the aggregate.

    25, not 24: 24 was the intermediate count while the proclitic guard was too
    broad and swallowed the فـ in فماذا, which stopped msa-5-028 splitting. The
    guard was then narrowed to real Arabic proclitics and that question splits
    again — the case
    :func:`test_a_real_proclitic_conjunction_is_still_stripped` above pins
    directly. A bare total cannot say which question moved, so if this fails,
    diff the split ids before touching the number: 24 here once meant a genuine
    regression was passing.
    """
    pairs = _dataset()
    split = sum(1 for pair in pairs if len(decompose(pair["question"])) > 1)
    assert split == 25, f"decomposition rate moved: {split}/283"
