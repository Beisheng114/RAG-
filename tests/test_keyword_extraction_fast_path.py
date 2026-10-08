from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lightrag_helpers import load_modules, make_hybrid


@pytest.fixture
def hybrid():
    return make_hybrid(load_modules())


def test_fast_path_does_not_call_llm(hybrid):
    hybrid._fallback_keyword_extraction = MagicMock(return_value=(["主机"], ["故障"]))
    with patch.object(hybrid.extract_query_keywords.__globals__["requests"], "post") as post:
        assert hybrid.extract_query_keywords("主机过热") == (["主机"], ["故障"])
        post.assert_not_called()


def test_relation_query_parses_json_and_timeout(hybrid):
    response = MagicMock()
    response.json.return_value = {"message": {"content": '<|think|>{ignore}</|think|>```json\n{"entity_keywords":["主机"],"topic_keywords":["故障原因"]}\n```'}}
    hybrid.config.llm_timeout_chat = 7
    with patch.object(hybrid.extract_query_keywords.__globals__["requests"], "post", return_value=response) as post:
        assert hybrid.extract_query_keywords("主机导致什么问题") == (["主机"], ["故障原因"])
        assert post.call_args.kwargs["timeout"] == 7


@pytest.mark.parametrize("content", ["", "not json", '{"entity_keywords":"wrong"}', '{"entity_keywords":[],"topic_keywords":[]}'])
def test_bad_response_falls_back(hybrid, content):
    response = MagicMock()
    response.json.return_value = {"message": {"content": content}}
    fallback = hybrid._fallback_keyword_extraction("未知关系导致异常")
    with patch.object(hybrid.extract_query_keywords.__globals__["requests"], "post", return_value=response):
        assert hybrid.extract_query_keywords("未知关系导致异常") == fallback


def test_openai_compatible_retry_and_disabled_fast_path(hybrid):
    hybrid.llm_provider = "vllm"
    hybrid.config.lightrag_llm_keyword_fast_path = False
    call = hybrid.llm_client.chat.completions.create
    call.side_effect = [TimeoutError(), SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='{"entity_keywords":["主机","主机"],"topic_keywords":["维护"]}'))])]
    assert hybrid.extract_query_keywords("主机", retry_count=2) == (["主机"], ["维护"])
    assert call.call_args.kwargs["timeout"] == hybrid.config.llm_timeout_chat
