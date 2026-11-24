"""Offline checks for retries, voting, timeouts and strategy validation. Run: python -m pytest tests"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from llm_council import Council, CouncilConfig
from llm_council.engine import Response


def make_config(**kw):
    base = dict(OPENROUTER_API_KEY="sk-test", COUNCIL_MODELS="a,b,c", COUNCIL_MAX_RETRIES=4, COUNCIL_TIMEOUT=7)
    base.update(kw)
    return CouncilConfig(**base)


def reply(text):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def test_max_retries_reaches_client():
    assert Council(make_config()).judge_client.max_retries == 4


def test_unknown_strategy_rejected():
    with pytest.raises(ValidationError):
        make_config(COUNCIL_STRATEGY="majority")


def test_vote_ignores_case_and_punctuation():
    council = Council(make_config())
    answers = [Response("a", "Lyon"), Response("b", "Paris."), Response("c", " paris ")]
    assert council._vote(answers) == "Paris."


def test_failed_and_empty_members_are_dropped_and_synthesis_has_timeout():
    council = Council(make_config())
    create = AsyncMock(side_effect=[reply("one"), RuntimeError("boom"), reply(""), reply("final")])
    council.judge_client.chat.completions.create = create
    result = asyncio.run(council.query("q"))
    assert [r["model"] for r in result["individual_responses"]] == ["a"]
    assert result["final_answer"] == "final"
    assert create.call_args.kwargs["timeout"] == 7


def test_comma_separated_models_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("COUNCIL_MODELS", "x/one, y/two")
    assert CouncilConfig().council_models == ["x/one", "y/two"]


def test_judge_model_used_for_synthesis_and_defaults_to_first():
    for judge, expected in [("j/judge", "j/judge"), ("", "a")]:
        council = Council(make_config(COUNCIL_JUDGE_MODEL=judge))
        create = AsyncMock(side_effect=[reply("x"), reply("y"), reply("z"), reply("final")])
        council.judge_client.chat.completions.create = create
        asyncio.run(council.query("q"))
        assert create.call_args.kwargs["model"] == expected
