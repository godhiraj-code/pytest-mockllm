"""Tests for the LangChain integration."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pytest_mockllm.integrations.langchain import LangChainMock

pydantic = pytest.importorskip("pydantic")
BaseModel = pydantic.BaseModel


def test_provider_class_imported_before_fixture_is_intercepted(mock_langchain):
    """Module-level provider imports must not bypass the active fixture."""
    mock_langchain.add_response("Paris is the capital of France.")
    llm = ChatOpenAI(model="gpt-4o", api_key="fixture-only")
    prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")

    result = (prompt | llm).invoke({"country": "France"})

    assert result.content == "Paris is the capital of France."
    assert mock_langchain.call_count == 1


class HabitantsResponse(BaseModel):
    """Gets habitants number."""

    habitants: str


def _model_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return model.json()


def _model_data(model: BaseModel) -> dict[str, object]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_with_structured_output_parses_pydantic_model():
    response_model = HabitantsResponse(habitants="48 million")

    with LangChainMock() as mock_langchain:
        mock_langchain.add_response(content=_model_json(response_model))

        habitants = mock_langchain.model.with_structured_output(schema=HabitantsResponse).invoke(
            [{"role": "user", "content": "How many inhabitants does Spain have?"}]
        )

    assert isinstance(habitants, HabitantsResponse)
    assert _model_data(habitants) == _model_data(response_model)


def test_with_structured_output_parses_pydantic_model_async():
    response_model = HabitantsResponse(habitants="48 million")

    async def run_test() -> HabitantsResponse:
        with LangChainMock() as mock_langchain:
            mock_langchain.add_response(content=_model_json(response_model))

            return await mock_langchain.model.with_structured_output(
                schema=HabitantsResponse
            ).ainvoke([{"role": "user", "content": "How many inhabitants does Spain have?"}])

    habitants = asyncio.run(run_test())

    assert isinstance(habitants, HabitantsResponse)
    assert _model_data(habitants) == _model_data(response_model)


def test_with_structured_output_include_raw_uses_langchain_envelope():
    response_model = HabitantsResponse(habitants="48 million")

    with LangChainMock() as mock_langchain:
        mock_langchain.add_response(content=_model_json(response_model))

        result = mock_langchain.model.with_structured_output(
            schema=HabitantsResponse,
            include_raw=True,
        ).invoke([{"role": "user", "content": "How many inhabitants does Spain have?"}])

    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert result["raw"].content == _model_json(response_model)
    assert isinstance(result["parsed"], HabitantsResponse)
    assert result["parsing_error"] is None
