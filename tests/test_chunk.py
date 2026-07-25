"""Unit tests for the article-aware Arabic chunker (pure logic, no DB)."""

from itertools import pairwise

from ingestion.chunk import TextChunk, chunk_document, split_articles

THREE_ARTICLES = """المادة (1)
تسري أحكام هذا القانون على أصحاب العمل والعمال ويحدد الوزير القواعد المنفذة له.

المادة (2)
لا يجوز الاتفاق على ما يخالف أحكام هذا القانون ولو كان الاتفاق سابقاً على العمل به.

المادة (3)
تكون مواعيد العمل وفترات الراحة على النحو الذي تحدده اللائحة التنفيذية.
"""

WITH_PREAMBLE = """قانون العمل القطري رقم (14) لسنة 2004
نحن حمد بن خليفة آل ثاني أمير دولة قطر، بعد الاطلاع على الدستور، قررنا القانون الآتي.

المادة (1)
يعمل بأحكام قانون العمل المرافق لهذا القانون.

المادة (2)
على جميع الجهات المختصة تنفيذ هذا القانون ونشره في الجريدة الرسمية.
"""


def test_three_articles_produce_one_chunk_each_in_order():
    # Arrange / Act
    chunks = chunk_document("qatar-labor-law", THREE_ARTICLES)

    # Assert
    assert [chunk.article for chunk in chunks] == ["1", "2", "3"]
    assert [chunk.seq for chunk in chunks] == [0, 0, 0]
    assert [chunk.id for chunk in chunks] == [
        "qatar-labor-law:1:0",
        "qatar-labor-law:2:0",
        "qatar-labor-law:3:0",
    ]
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert "أصحاب العمل والعمال" in chunks[0].text


def test_preamble_before_first_article_gets_article_none():
    # Arrange / Act
    chunks = chunk_document("qatar-labor-law", WITH_PREAMBLE)

    # Assert
    assert chunks[0].article is None
    assert chunks[0].id == "qatar-labor-law:p:0"
    assert "أمير دولة قطر" in chunks[0].text
    assert [chunk.article for chunk in chunks[1:]] == ["1", "2"]


def test_arabic_indic_article_number_is_converted_to_ascii():
    # Arrange
    text = "المادة ٥٧\nينتهي عقد العمل بانتهاء مدته أو بإتمام العمل المتفق عليه."

    # Act
    chunks = chunk_document("labor", text)

    # Assert
    assert [chunk.article for chunk in chunks] == ["57"]
    assert chunks[0].id == "labor:57:0"
    # The stored text keeps the original Arabic-Indic digits.
    assert "٥٧" in chunks[0].text


def test_parenthesised_article_number_is_extracted():
    # Arrange
    text = "المادة (12)\nيستحق العامل إجازة سنوية مدفوعة الأجر وفقاً لمدة خدمته."

    # Act
    articles = split_articles(text)

    # Assert
    assert [article for article, _ in articles] == ["12"]
    assert articles[0][1].startswith("المادة (12)")


def test_bare_maada_heading_without_definite_article_is_recognised():
    # Arrange
    text = "مادة 4\nيحظر تشغيل الأحداث في الأعمال الخطرة أو الضارة بالصحة."

    # Act
    chunks = chunk_document("regs", text)

    # Assert
    assert [chunk.article for chunk in chunks] == ["4"]


def test_long_article_splits_into_overlapping_chunks_without_mid_word_splits():
    # Arrange
    sentence = (
        "يلتزم صاحب العمل بأن يوفر للعامل وسائل الحماية والوقاية من الأخطار "
        "والأمراض التي قد تنشأ عن العمل وأن يتحمل نفقات العلاج اللازمة. "
    )
    body = sentence * 12
    text = f"المادة (٩٩)\n{body}"
    max_chars = 300
    overlap_chars = 60
    words = set((f"المادة (٩٩) {body}").split())

    # Act
    chunks = chunk_document("labor", text, max_chars=max_chars, overlap_chars=overlap_chars)

    # Assert
    assert len(chunks) > 1
    assert [chunk.seq for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.id for chunk in chunks] == [
        f"labor:99:{seq}" for seq in range(len(chunks))
    ]
    assert all(chunk.article == "99" for chunk in chunks)
    assert all(len(chunk.text) <= max_chars for chunk in chunks)
    # No chunk boundary lands inside a word.
    for chunk in chunks:
        assert all(word in words for word in chunk.text.split())
    # Consecutive chunks overlap: the tail of one reappears at the head of the next.
    for current, following in pairwise(chunks):
        assert following.text.split()[0] in current.text.split()


def test_chunk_ids_are_stable_across_identical_calls():
    # Arrange / Act
    first = chunk_document("qatar-labor-law", WITH_PREAMBLE)
    second = chunk_document("qatar-labor-law", WITH_PREAMBLE)

    # Assert
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first == second


def test_empty_document_returns_no_chunks():
    # Arrange / Act / Assert
    assert chunk_document("doc", "") == []
    assert chunk_document("doc", "   \n\t  ") == []
    assert split_articles("") == []


def test_document_without_article_headings_is_chunked_as_preamble():
    # Arrange
    text = "تعميم إداري بشأن مواعيد الدوام الرسمي خلال شهر رمضان المبارك."

    # Act
    chunks = chunk_document("circular", text)

    # Assert
    assert len(chunks) == 1
    assert chunks[0].article is None
    assert chunks[0].id == "circular:p:0"


def test_empty_article_bodies_are_dropped():
    # Arrange
    text = "المادة (1)\nيعمل بأحكام هذا القانون.\n\nالمادة (2)\n   \n"

    # Act
    chunks = chunk_document("doc", text)

    # Assert
    assert [chunk.article for chunk in chunks] == ["1"]
