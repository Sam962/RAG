"""Ragas 0.4 imports VertexAI from langchain-community; stub it so evaluate can load."""

from __future__ import annotations

import sys
from types import ModuleType


class _UnusedVertexModel:
    pass


def patch_langchain_vertexai() -> None:
    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        stub = ModuleType(name)
        stub.ChatVertexAI = _UnusedVertexModel
        sys.modules[name] = stub
    try:
        import langchain_community.llms as llms
    except Exception:
        return
    if not hasattr(llms, "VertexAI"):
        llms.VertexAI = _UnusedVertexModel
