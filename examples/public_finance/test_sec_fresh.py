"""Targeted tests for SEC holdout identity and template exclusion."""

import pytest

from sec_fresh import TemplateIndex, issuer, normalize


def test_cik_lockout_normalizes_padding():
    assert issuer(" 123 ") == issuer("0000000123")
    with pytest.raises(ValueError):
        issuer("123-X")


def test_template_index_excludes_numeric_and_punctuation_variants():
    index = TemplateIndex()
    index.add(normalize("The issuer amended its credit facility by $25 million."))
    assert index.contains(normalize("The ISSUER amended its credit facility by $90 million!"))
    assert not index.contains(normalize("The auditor resigned effective today."))


def test_template_index_excludes_truncated_repeated_excerpts():
    index = TemplateIndex()
    body = (
        "after review the registrant delivered written notice about a proposed change "
        "to its existing credit arrangements and confirmed that all obligations under "
        "the agreement would continue until the lender formally accepted the newly "
        "negotiated conditions and filed the related paperwork with the appropriate agency"
    )
    index.add(normalize(body))
    assert index.contains(normalize(" ".join(body.split()[:-4])))
    assert not index.contains(normalize("the newly elected board adopted a compensation plan after the meeting"))
