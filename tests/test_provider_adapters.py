import json
import sys
import types as pytypes
from types import SimpleNamespace

import pytest

from app import llm
from app.config import ModelConfig


class FakePart:
    @staticmethod
    def from_text(text):
        return SimpleNamespace(text=text, function_call=None, function_response=None)

    @staticmethod
    def from_function_call(name, args):
        return SimpleNamespace(
            text=None,
            function_call=SimpleNamespace(name=name, args=args),
            function_response=None,
        )

    @staticmethod
    def from_function_response(name, response):
        return SimpleNamespace(
            text=None,
            function_call=None,
            function_response=SimpleNamespace(name=name, response=response),
        )

    @staticmethod
    def from_bytes(data, mime_type):
        return SimpleNamespace(text=None, data=data, mime_type=mime_type)


class FakeContent:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts


class FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeGoogleTypes:
    Part = FakePart
    Content = FakeContent
    GenerateContentConfig = FakeConfig
    FunctionDeclaration = FakeConfig
    Tool = FakeConfig
    FunctionCallingConfig = FakeConfig
    ToolConfig = FakeConfig
    AutomaticFunctionCallingConfig = FakeConfig
    HttpOptions = FakeConfig


def install_fake_google(monkeypatch):
    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    genai.types = FakeGoogleTypes
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)


def test_gemini_content_translation_preserves_tool_turns(monkeypatch):
    install_fake_google(monkeypatch)
    contents, system = llm._gemini_contents(
        [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "Search this document."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": '{"query":"policy limit"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"matches":2}',
            },
        ],
        FakeGoogleTypes,
    )
    assert system == "Use tools when needed."
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[1].parts[0].function_call.name == "search_knowledge_base"
    assert contents[2].role == "tool"
    assert contents[2].parts[0].function_response.name == "search_knowledge_base"
    assert contents[2].parts[0].function_response.response == {"matches": 2}


def test_gemini_tools_and_forced_tool_choice_are_translated(monkeypatch):
    tools, tool_config = llm._gemini_tools(
        {
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search documents",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "search_knowledge_base"},
            },
        },
        FakeGoogleTypes,
    )
    assert len(tools) == 1
    assert tools[0].function_declarations[0].name == "search_knowledge_base"
    assert tool_config.function_calling_config.mode == "ANY"
    assert tool_config.function_calling_config.allowed_function_names == ["search_knowledge_base"]


def test_gemini_response_is_openai_chat_completion_compatible():
    call_part = SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name="lookup", args={"id": "A1"}),
    )
    response = SimpleNamespace(
        response_id="resp-1",
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=[SimpleNamespace(text="Found it."), call_part]),
            finish_reason="STOP",
        )],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=4,
            total_token_count=14,
        ),
    )
    result = llm._gemini_response(response, "gemini-3.5-flash")
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "Found it."
    call = result["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"id": "A1"}
    assert result["usage"]["total_tokens"] == 14


def test_openai_kwargs_use_the_session_model_not_client_supplied_model():
    model = ModelConfig(
        id="test",
        provider="openai",
        model_id="openai/gpt-4o-mini",
    )
    kwargs = llm._build_openai_kwargs(
        model,
        {
            "model": "attacker-selected-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10000,
        },
    )
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["max_completion_tokens"] == llm.settings.max_output_tokens
    assert "attacker-selected-model" not in kwargs.values()


def test_gemini_thought_signature_survives_tool_call_round_trip(monkeypatch):
    install_fake_google(monkeypatch)
    call_part = SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name="lookup", args={"id": "A1"}),
        thought_signature=b"opaque-gemini-signature",
    )
    response = SimpleNamespace(
        response_id="resp-signed",
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=[call_part]),
            finish_reason="STOP",
        )],
        usage_metadata=None,
    )
    result = llm._gemini_response(response, "gemini-3.5-flash")
    tool_call_id = result["choices"][0]["message"]["tool_calls"][0]["id"]
    assert ".gsig." in tool_call_id

    contents, _ = llm._gemini_contents(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id":"A1"}'},
                }],
            },
            {"role": "tool", "tool_call_id": tool_call_id, "content": '{"ok":true}'},
        ],
        FakeGoogleTypes,
    )
    restored_part = contents[0].parts[0]
    assert restored_part.thought_signature == b"opaque-gemini-signature"
    assert contents[1].parts[0].function_response.name == "lookup"


def test_anthropic_rejects_unsupported_json_object_response_format():
    model = ModelConfig(id="claude", provider="anthropic", model_id="claude-model")
    with pytest.raises(llm.ProviderRequestError):
        llm._build_openai_kwargs(model, {
            "messages": [{"role": "user", "content": "json please"}],
            "response_format": {"type": "json_object"},
        })


def test_anthropic_drops_unsupported_penalties_and_conflicting_top_p():
    model = ModelConfig(id="claude", provider="anthropic", model_id="claude-model")
    kwargs = llm._build_openai_kwargs(model, {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
    })
    assert kwargs["temperature"] == 0
    assert "top_p" not in kwargs
    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs


def test_gemini_stream_chunk_is_openai_compatible():
    function_part = SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name="lookup", args={"id": "A1"}),
        thought_signature=b"signed",
    )
    chunk = SimpleNamespace(candidates=[SimpleNamespace(
        content=SimpleNamespace(parts=[function_part]),
        finish_reason="STOP",
    )])
    payload = llm._gemini_chunk_payload(
        chunk,
        model="gemini-3.5-flash",
        stream_id="chatcmpl-test",
        created=123,
    )
    call = payload["choices"][0]["delta"]["tool_calls"][0]
    assert payload["object"] == "chat.completion.chunk"
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"id": "A1"}
    assert ".gsig." in call["id"]
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_openrouter_uses_openrouter_token_field():
    model = ModelConfig(
        id="openrouter",
        provider="openrouter",
        model_id="openai/gpt-oss-20b:free",
    )
    kwargs = llm._build_openai_kwargs(model, {
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 123,
    })
    assert kwargs["max_tokens"] == 123
    assert "max_completion_tokens" not in kwargs


def test_anthropic_new_models_only_receive_default_sampling():
    model = ModelConfig(id="claude", provider="anthropic", model_id="claude-opus-5-5")
    kwargs = llm._build_openai_kwargs(model, {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "top_p": 0.5,
    })
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
