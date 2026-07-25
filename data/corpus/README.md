# Corpus — Qatari labour legislation (Modern Standard Arabic)

Five public Qatari legal texts covering employment: the Labour Law and four instruments
that sit on top of it (domestic workers, minimum wage, heat-stress protection, labour
dispute committees). This is the retrieval corpus for the RAG service and the ground
truth the eval set is written against.

| doc_id | Articles | Characters |
|---|---:|---:|
| `qatar-labour-law-14-2004` | 164 | 66,794 |
| `qatar-domestic-workers-law-15-2017` | 24 | 8,370 |
| `qatar-minimum-wage-law-17-2020` | 8 | 2,631 |
| `qatar-labour-dispute-committees-decision-6-2018` | 20 | 4,890 |
| `qatar-heat-stress-decision-17-2021` | 10 | 4,359 |
| **total** | **226** | **87,044** |

Article counts are headings (`المادة N`) as published, including amended and inserted
articles (`مكرر`) and the four `مواد الإصدار` promulgation articles of Law 14/2004 —
which is why the Labour Law shows 164 headings for 148 numbered articles.

## Provenance

Every document was retrieved on **2026-07-24** from
[Al Meezan — Qatar Legal Portal](https://www.almeezan.qa), the legal database of the
Ministry of Justice, State of Qatar. Exact URLs, titles, licence text and article counts
are in `manifest.json`.

These are **public legal texts**, included here so that retrieval evaluation is
reproducible offline by anyone who clones the repo. Article 3 of Qatari Law No. 7 of 2002
on the Protection of Copyright and Neighbouring Rights excludes laws and official texts
from copyright protection. Al Meezan publishes them as *unofficial* consolidated copies
("الرجاء عدم اعتبار المادة المعروضة أعلاه رسمية") — so are these. **Do not rely on this
corpus for legal advice**; consult the Official Gazette (الجريدة الرسمية) for the
authoritative wording.

## Format

Plain UTF-8 text, one logical block per line, no HTML and no site navigation:

```
المادة 80
يحدد صاحب العمل موعد إجازة العامل السنوية حسب مقتضيات العمل، ...
ويجوز لصاحب العمل، بناءً على طلب كتابي من العامل أن يؤجل ...
```

Each document starts at its preamble (`نحن تميم بن حمد آل ثاني ...` / `مجلس الوزراء،`)
and ends at the last article. Article headings always start a line, so the chunker can
split on `^المادة` without any HTML parsing. Amendment annotations stay attached to the
heading (`المادة 145 - مكرر (اضيفت بموجب: مرسوم بقانون 18 / 2020)`) — they carry the
legislative history and are useful context for citations.

## Refreshing

```bash
python -m ingestion.fetch            # re-downloads into data/corpus/
python -m ingestion.fetch some/dir   # or somewhere else
```

The fetcher re-downloads every `manifest.json` entry that has a `source_url`, strips HTML
with a stdlib parser, and keeps only the text between that entry's `start_marker` and
`end_marker`. If Al Meezan changes its page layout a marker will stop matching and the
run fails loudly rather than writing a truncated document — update the markers, re-run,
and review the diff before committing. The committed `.txt` files are the offline source
of truth; nothing in the pipeline needs network access.
