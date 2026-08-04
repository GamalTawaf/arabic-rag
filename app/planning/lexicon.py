"""Gulf (Khaleeji) → MSA lexicon, mined from the eval set — not from imagination.

Every entry below was found by tokenizing the 54 Gulf questions in
``evals/data/eval_pairs.jsonl`` and diffing their vocabulary against the 229 MSA
questions and against the 233 indexed corpus chunks. The parenthesised counts in
the comments are ``gulf`` = Gulf questions containing the token,
``msa`` = MSA questions containing it, ``corpus`` = corpus chunks containing the
MSA target. A mapping earns its place when the Gulf form is absent from the
corpus and the MSA form is frequent in it — that gap is the whole dialect
penalty, in one number per row.

**Keys are NORMALIZED tokens** (:func:`ingestion.normalize.normalize_query`:
hamza seats folded, ``ة→ه``, ``ى→ي``, digits ASCII), so a look-up succeeds
whichever way the user spelled the hamza. **Values are proper MSA orthography**,
because they are fed to the dense embedder and the corpus was embedded from raw
un-normalized text — writing ``اجر`` where the corpus says ``أجر`` would give
back part of what the rewrite just won.

Two things deliberately left out, because a wrong rewrite is worse than none:

* ``ما`` — negation in Gulf ("ما عندي عقد") *and* the MSA interrogative "what"
  ("ما هي مدة الإخطار"). It is the single most frequent token in both registers
  (gulf 18, msa 78). No token-level rule can tell the two apart, so it is left
  alone. ``مو`` / ``مب`` (the unambiguous Gulf negators) are mapped instead —
  neither occurs in this dataset, but they cost nothing and cannot misfire.
* ``حق`` — Gulf possessive ("حق العلاج" = the treatment's), but also the MSA noun
  "right", which is what the labour law is entirely about (gulf 1, msa 1). Left
  alone.

``شوي``, ``خلاص`` and ``يعني`` appear in :data:`REGISTER_MARKERS` but not in
:data:`GULF_TO_MSA`: they are unmistakable register signals with no single safe
MSA target ("شوي" is both "a little while" and "some of"). Detecting a register
is cheap to get wrong; rewriting is not.
"""

from __future__ import annotations

# ── interrogatives ────────────────────────────────────────────────────────────
# Corpus has none of these Gulf forms and all of the MSA ones.
_INTERROGATIVES: dict[str, str] = {
    "شكثر": "كم",  # (gulf 15, msa 0) — by far the strongest single marker
    "شلون": "كيف",  # (gulf 4, msa 0)
    "منو": "من",  # (gulf 4, msa 0; corpus من 409)
    "شنو": "ما",  # (gulf 3, msa 0)
    "وش": "ما",  # (gulf 3, msa 0)
    "وشو": "ما",  # unattested here, standard Khaleeji
    "وين": "أين",  # (gulf 1, msa 0)
    "ليش": "لماذا",  # unattested here, standard Khaleeji
    "كيفنا": "كما نشاء",  # (gulf 1) "على كيفنا" = as we please
}

# ── particles, conjunctions, relatives ────────────────────────────────────────
_PARTICLES: dict[str, str] = {
    "اللي": "الذي",  # (gulf 13, msa 0; corpus الذي 56)
    "عشان": "لأن",  # (gulf 7, msa 0) — also "in order to"; لأن is the commoner sense
    "علشان": "لأن",
    "ولا": "أم",  # (gulf 15, msa 0) — "ولا لا" = "أم لا"; see module note
    "لين": "حتى",  # (gulf 4, msa 0; corpus حتى 8)
    "لو": "إذا",  # (gulf 4, msa 2; corpus إذا 85, لو 0) — synonymous conditionals
    "بس": "فقط",  # (gulf 2, msa 0)
    "لسا": "ما زال",  # (gulf 2, msa 0)
    "احنا": "نحن",  # (gulf 1, msa 0)
    "جوه": "داخل",  # (gulf 1, msa 0)
    "برا": "خارج",  # (gulf 1, msa 0)
    "الحين": "الآن",  # (gulf 1, msa 0)
    "ببلاش": "مجانا",  # (gulf 1, msa 0)
    "مو": "ليس",  # unattested here; unambiguous Gulf negator
    "مب": "ليس",  # unattested here; unambiguous Gulf negator
}

# ── verbs and modals ──────────────────────────────────────────────────────────
# Only the inflections that actually occur are listed. This lexicon has no
# morphology: "أبغى" and "يبغى" are two separate rows on purpose.
_VERBS: dict[str, str] = {
    "ابغي": "أريد",  # (gulf 3 incl. "وابغى"; msa 0)
    "ابغا": "أريد",
    "ابي": "أريد",
    "يبغي": "يريد",
    "يبي": "يريد",
    "تبي": "تريد",  # (gulf 1)
    "بغيت": "أردت",
    "ودي": "أود",
    "اقدر": "أستطيع",  # (gulf 1)
    "تقدر": "تستطيع",
    "نقدر": "نستطيع",  # (gulf 1)
    "يقدر": "يستطيع",  # (gulf 7, msa 2)
    "يقدرون": "يستطيعون",  # (gulf 4 incl. "ويقدرون")
    "صار": "أصبح",  # (gulf 3, msa 0)
    "يصير": "يصبح",  # (gulf 4, msa 0)
    "اسوي": "أفعل",  # (gulf 1)
    "نسوي": "نفعل",  # (gulf 1)
    "يسوون": "يفعلون",  # (gulf 1)
    "سواها": "فعلها",  # (gulf 1)
    "يدش": "يدخل",  # (gulf 1)
    "اطفش": "أتغيب",  # (gulf 1) "أطفش من الدوام" = absent myself
    "يطردني": "ينهي خدمتي",  # (gulf 1; corpus إنهاء 14, الطرد 0)
    "يطردونه": "ينهون خدمته",  # (gulf 1)
    "يطيروني": "ينهون خدمتي",  # (gulf 1)
    "استلفت": "اقترضت",  # (gulf 1)
}

# ── nouns ─────────────────────────────────────────────────────────────────────
# The highest-value rows: the corpus simply does not contain the Gulf noun.
_NOUNS: dict[str, str] = {
    "راتب": "أجر",  # corpus أجر 40, راتب 0
    "راتبي": "أجري",  # (gulf 6)
    "الراتب": "الأجر",
    "رواتب": "أجور",
    "رواتبنا": "أجورنا",  # (gulf 1)
    "براتب": "بأجر",  # (gulf 3 incl. "وبراتب")
    "فلوس": "مال",  # (gulf 3, msa 0)
    "فلوسي": "مالي",  # (gulf 1)
    "فلوسه": "ماله",  # (gulf 1)
    "شغل": "عمل",  # (gulf 4, msa 0; corpus عمل 37, شغل 0)
    "الشغل": "العمل",  # (gulf 5, msa 0; corpus العمل 441)
    "بالشغل": "بالعمل",  # (gulf 1)
    "شغلي": "عملي",  # (gulf 1)
    "دوام": "ساعات عمل",  # (gulf 2; corpus ساعات 29, دوام 0)
    "الدوام": "ساعات العمل",  # (gulf 3)
    "كفيل": "صاحب عمل",  # corpus صاحب 169, كفيل 0
    "الكفيل": "صاحب العمل",  # (gulf 3, msa 0)
    "لكفيلي": "لصاحب عملي",  # (gulf 1)
    "دكتور": "طبيب",  # (gulf 1; corpus الطبيب 3, دكتور 0)
    "سنين": "سنوات",  # (gulf 2; corpus سنوات 10, سنين 0)
    "دريول": "سائق",
    "الدريول": "السائق",  # (gulf 2, msa 0)
    "خدامه": "عاملة منزلية",
    "الخدامه": "العاملة المنزلية",  # (gulf 1)
    "للخدامه": "للعاملة المنزلية",  # (gulf 1)
    "طياره": "طائرة",  # (gulf 1)
    "كاش": "نقدا",  # (gulf 1)
    "يهالي": "أطفالي",  # (gulf 1)
}


def _merge(*tables: dict[str, str]) -> dict[str, str]:
    """Merge the category tables, refusing to let one silently shadow another."""
    merged: dict[str, str] = {}
    for table in tables:
        for key, value in table.items():
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"conflicting lexicon entry {key!r}: {merged[key]!r} vs {value!r}"
                )
            merged[key] = value
    return merged


GULF_TO_MSA: dict[str, str] = _merge(_INTERROGATIVES, _PARTICLES, _VERBS, _NOUNS)

