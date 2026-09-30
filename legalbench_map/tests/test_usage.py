"""The request log: what each request saves, and upgrading a database made
before it saved questions, documents and answers."""
from __future__ import annotations

import json
import sqlite3

from router.usage import Usage, document_id, new_request_id

DOC = "Licensee may audit the books of Licensor once per calendar year."
ANSWER = {"answer": "yes", "confidence": 1.0, "path": "pretier0", "answered_by": "Pre-Tier 0 (regex)", "ms": 1,
          "llm_calls": 0, "evidence": [{"text": DOC, "label": "yes", "p": 1.0}]}


def rows(db, sql, *args):
    return sqlite3.connect(db).execute(sql, args).fetchall()


def test_ids():
    assert new_request_id().startswith("req_") and new_request_id() != new_request_id()
    assert document_id(DOC) == document_id(DOC) != document_id(DOC + " ")
    assert document_id(DOC).startswith("doc_")


def test_log_saves_question_answer_and_document(tmp_path):
    db = tmp_path / "usage.db"
    u = Usage(db, daily_limit=5)
    rid = new_request_id()
    u.log("a@x.org", "ok", "jev", ANSWER, request_id=rid, question="Can we audit their books?", document=DOC)
    (question, doc_id, answer, confidence, evidence), = rows(
        db, "SELECT question, document_id, answer, confidence, evidence FROM requests WHERE id = ?", rid)
    assert (question, doc_id, answer, confidence) == ("Can we audit their books?", document_id(DOC), "yes", 1.0)
    assert json.loads(evidence) == ANSWER["evidence"]
    assert rows(db, "SELECT text FROM documents WHERE id = ?", doc_id) == [(DOC,)]


def test_a_document_is_stored_once(tmp_path):
    db = tmp_path / "usage.db"
    u = Usage(db, daily_limit=5)
    for q in ("Can we audit their books?", "How often?"):
        u.log("a@x.org", "ok", "jev", ANSWER, request_id=new_request_id(), question=q, document=DOC)
    assert rows(db, "SELECT COUNT(*) FROM documents") == [(1,)]
    assert rows(db, "SELECT COUNT(DISTINCT id), COUNT(*) FROM requests WHERE document_id = ?", document_id(DOC)) == [(2, 2)]


def test_a_refused_request_keeps_the_question_without_an_answer(tmp_path):
    db = tmp_path / "usage.db"
    Usage(db, daily_limit=5).log("a@x.org", "limited", "jev", request_id="req_1", question="Q?", document=DOC)
    assert rows(db, "SELECT status, question, document_id, answer, evidence FROM requests") == [
        ("limited", "Q?", document_id(DOC), None, None)]


def test_an_older_database_gains_the_new_columns_and_keeps_its_rows(tmp_path):
    db = tmp_path / "usage.db"
    old = sqlite3.connect(db)
    old.executescript("""
        CREATE TABLE requests (ts TEXT NOT NULL, day TEXT NOT NULL, email TEXT NOT NULL, status TEXT NOT NULL,
                               reader TEXT, path TEXT, answered_by TEXT, ms INTEGER, llm_calls INTEGER);
        INSERT INTO requests VALUES ('2026-09-30T07:36:10+00:00', '2026-09-30', 'a@x.org', 'ok', 'jev', 'tier0', 'Tier 0', 900, 0);
    """)
    old.commit()
    old.close()
    u = Usage(db, daily_limit=5)
    u.log("a@x.org", "ok", "jev", ANSWER, request_id="req_2", question="Q?", document=DOC)
    assert rows(db, "SELECT id, question, path, ms FROM requests ORDER BY ts, rowid") == [
        (None, None, "tier0", 900), ("req_2", "Q?", "pretier0", 1)]
    assert u.stats()["tiles"]["questions_total"] == 2
    Usage(db, daily_limit=5)  # opening it again changes nothing
