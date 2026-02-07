"""
This file encapsulates the interaction with LiteLLM.  No LiteLLM specific code
should exist outside this file.

The `interact` function is the main function to interact with the LLM. If
deprecating the use of LiteLLM, just the interact function in this file needs to
be implemented.

LiteLLM is being used as a means of convenience so as to allow not having to
deal with multiple providers individually. While this convenience is
appreciated, LiteLLM's code and docs quality, at the time of writing, are not up
to the mark.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import requests
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import otto
from otto.llm.format import get_messages
from otto.llm.types import (
	Content,
	ContentChunk,
	InteractReturn,
	Query,
	ReasoningEffort,
	Session,
	SessionItem,
	TextContent,
	TextContentChunk,
	ThinkingContent,
	ToolUseContent,
	ToolUseContentChunk,
	ToolUseDelta,
)
from otto.llm.utils import (
	DEFAULT_MODEL,
	DEFAULT_REASONING_BUDGET_MAP,
	MAX_RETRIES,
	get_agent_item,
	get_key,
	get_provider,
	get_sequence,
	get_session,
	to_content,
	update_session,
)

if TYPE_CHECKING:
	from collections.abc import Generator
	from datetime import datetime

	from litellm import CustomStreamWrapper
	from litellm.types.utils import ModelResponseStream


logger = otto.logger("otto.llm.litellm", "ERROR")


class StreamReturn(NamedTuple):
	chunks: list[ContentChunk]
	latency: float


class InteractReturnTuple(NamedTuple):
	response: InteractReturn | None
	reason: None | str


def interact(
	# query should be None only if session is provided with a call call update
	query: Query | None = None,
	*,
	session: Session | None = None,
	model: str | None = None,
	system: str | None = None,
	tools: list[dict] | None = None,
	reasoning_effort: ReasoningEffort | None = None,
) -> Generator[ContentChunk, None, InteractReturnTuple]:
	"""
	Interacts with an LLM using LiteLLM, handling conversation history,
	streaming responses, and tool usage.

	This function is a generator that yields response chunks and returns a final
	named tuple with the complete interaction response.

	Maintaining state:

	1. The `interact` function maintains state using a `Session` object. If no session
		object is provided, a new one is created.
	2. The provided session object should not contain the query. An item for the query is
		created and added to the session object.
	3. The `interact` generator's return value contains an updated session object on success.
	4. The returned session object will contain the user query and the agent response.
	5. To maintain history the returned object should be provided to the `interact` function
		upon each call.
	6. The returned `Session` object may be updated in certain cases, such
		as if appending a tool use response.

	Args:
		query: The user's input query to start a new conversation or continue an
			existing one.
		session: The existing conversation history (Session object) WITHOUT the query.
			Users query is added to the Session and an updated Session is returned.
			Required if `query` is not provided.
		model: The specific LiteLLM model identifier (e.g., "openai/gpt-4o").
			Overrides `model` if provided. If neither is provided, derived from `DEFAULT_LLM`.
		system: An optional system prompt to guide the LLM's behavior.
		tools: An optional list of tools (functions) the LLM can use.
		reasoning_effort: An optional reasoning effort to use if the model supports it.

	Yields:
		ContentChunk: Chunks of the response from the LLM as they are generated.

	Returns:
		InteractReturnTuple:
			- On success, `InteractReturnTuple.response` contains an `InteractReturn`
			  object with the generated agent response item, the updated session,
			  and a list of content chunks from the stream. `InteractReturnTuple.reason` is `None`.
			- On failure (e.g., API key issue), `InteractReturnTuple.response` is `None` and
			  and `InteractReturnTuple.reason` contains reason for failure.
	"""
	import litellm

	assert query is not None or session is not None, (
		"session (with tool result) is required if query is not provided"
	)

	content = None if query is None else to_content(query)
	model = model or DEFAULT_MODEL

	# Creates a new session if session is None, else uses a copy
	update = get_session(content, session)
	session_id = update["id"]

	if reason := _set_key(model):
		return InteractReturnTuple(None, reason)

	item = get_agent_item(model)
	item["meta"]["start_time"] = time.time()

	# Required cause of LiteLLM's spaghetti design
	done = threading.Event()

	def success_callback(
		kwargs: dict,  # kwargs to completion
		completion_response: dict,  # response from completion
		_start_time: datetime,
		_end_time: datetime,
	):
		usage = completion_response.get("usage", {})
		logger.debug(
			{
				"message": "callback called",
				"id": item["id"],
				"usage": usage.get("completion_tokens"),
				"end_reason": completion_response.get("choices", [{}])[0].get("finish_reason"),
			}
		)
		if usage.get("completion_tokens") is None or usage.get("prompt_tokens") is None:
			return  # Test based heuristic on when callback is final

		if (end_reason := _get_end_reason(completion_response)) is None:
			return  # Success callback is called multiple times for each chunk

		item["meta"]["input_tokens"] = usage.get("prompt_tokens", 0)
		item["meta"]["output_tokens"] = usage.get("completion_tokens", 0)
		item["meta"]["cost"] = kwargs.get("standard_logging_object", {}).get("response_cost", None)
		item["content"] = _get_content(completion_response)
		item["meta"]["end_reason"] = end_reason
		logger.debug({"message": "callback done set", "id": item["id"]})
		done.set()

	litellm.success_callback = [success_callback]

	items = get_sequence(update)
	messages, last_id = get_messages(
		items,
		system,
		preserve_thinking=_should_preserve_thinking(model),
	)

	think = {}
	if reasoning_effort and reasoning_effort != "None":
		think["reasoning_effort"] = reasoning_effort.lower()  # litellm expects "low", "medium", "high"

	if (
		reasoning_effort
		and reasoning_effort != "None"
		and (model.startswith("anthropic") or model.startswith("gemini"))
	):
		think["thinking"] = _get_thinking(reasoning_effort)

	# Inject native Google Search for Gemini models
	if model.startswith("gemini"):
		if tools is None:
			tools = []
		# If a web_search tool is present, replace with Gemini native grounding.
		has_web_search = any(t.get("function", {}).get("name") == "web_search" for t in tools)
		if has_web_search:
			tools = [t for t in tools if t.get("function", {}).get("name") != "web_search"]
			tools.append({"google_search_retrieval": {}})
		# Otherwise ensure a default public Google search tool is present
		elif not any("google_search" in str(t) for t in tools):
			# LiteLLM format for Gemini Google Search (Public Grounding)
			# Note: We use 'google_search' not 'google_search_retrieval' (Vertex Enterprise)
			tools.append({"google_search": {}})

	# Inject native Web Search for OpenAI models
	# (Disabled: OpenAI Chat Completions does not support 'web_search' tool type)
	# if model.startswith("openai") or model.startswith("gpt-"):
	# 	if tools is None:
	# 		tools = []
	# 	if not any(t.get("type") == "web_search" for t in tools if isinstance(t, dict)):
	# 		tools.append({"type": "web_search"})
	
	if model.startswith("gpt-") or model.startswith("openai"):
		# Route to Custom Responses API Adapter
		# This uses the new `web_search` tool type supported by GPT-4o on the Responses API
		
		# We use a custom generator here that mimics the litellm generator
		response_generator = _openai_responses_adapter(
			model=model,
			messages=messages,
			item=item,
			session_id=session_id,
            tools=tools or []
		)
		
		chunks = []
		try:
			for chunk in response_generator:
				yield chunk
				if item["meta"]["time_to_first_chunk"] == 0:
					item["meta"]["time_to_first_chunk"] = time.time() - item["meta"]["start_time"]
				chunks.append(chunk) # Collect chunks for final return
		except Exception as e:
			otto.log_error("openai_responses_adapter error", model=model, error=str(e))
			# Fallback or raise? We raise for now as this is the primary path.
			raise e

		logger.debug({"message": "responses adapter completed", "id": item["id"]})
		
		# Update session and return
		logger.debug({"message": "updating session", "id": item["id"]})
		update_session(update, last_id, item)
		item["meta"]["end_time"] = time.time()
		response = InteractReturn(item=item, update=update, chunks=chunks)
		return InteractReturnTuple(response, None)

	logger.debug({"message": "calling litellm.completion", "id": item["id"]})
	completion = _completions(
		model=model,
		messages=messages,
		tools=tools,
		stream=True,
		**think,
		# Drops incompatible params (doesn't seem to always work)
		drop_params=True,
	)

	# litellm.completion not typed well enough; this shouldn't throw when stream=True
	assert isinstance(completion, litellm.CustomStreamWrapper), "sanity check"

	chunks = []

	# Iterates over the completion stream and returns a list of ContentChunk
	stream_generator = _stream(completion, item, session_id)
	while True:
		try:
			yield next(stream_generator)
			if item["meta"]["time_to_first_chunk"] == 0:
				item["meta"]["time_to_first_chunk"] = time.time() - item["meta"]["start_time"]
		except StopIteration as e:
			ret = cast("StreamReturn", e.value)
			chunks = ret.chunks
			item["meta"]["inter_chunk_latency"] = ret.latency
			break

	logger.debug(
		{
			"message": "stream completed",
			"id": item["id"],
			"chunks": len(chunks),
		}
	)
	done.wait()

	logger.debug(
		{
			"message": "success callback called",
			"content_len": len(item["content"]),
			"thinking_content": [c for c in item["content"] if c["type"] == "thinking"],
		}
	)

	logger.debug({"message": "updating session", "id": item["id"]})
	# Update the update session with the item
	update_session(update, last_id, item)

	item["meta"]["end_time"] = time.time()

	response = InteractReturn(item=item, update=update, chunks=chunks)
	return InteractReturnTuple(response, None)


def _get_content(completion_response: dict) -> list[Content]:
	message = completion_response.get("choices", [{}])[0].get("message", {})
	return [
		*_get_thinking_content(message),
		*_get_text_content(message),
		*_get_tool_use_content(message),
	]


def _get_thinking_content(message: dict[str, Any]) -> list[ThinkingContent]:
	thinking_blocks = message.get("thinking_blocks") or []
	reasoning_content = message.get("reasoning_content")

	if not (thinking_blocks or reasoning_content):
		return []

	if reasoning_content and not thinking_blocks:
		assert isinstance(reasoning_content, str), "sanity check"
		return [
			ThinkingContent(
				type="thinking",
				text=reasoning_content,
				signature=None,
			)
		]

	# Thinking blocks are set by litellm only when model is an anthropic one we
	# don't directly use reasoning_content cause anthropic sends signatures with
	# thinking which needs to be sent back in the next request.
	blocks: list[ThinkingContent] = []
	for block in thinking_blocks:
		text = block.get("thinking")
		if not text:
			continue

		blocks.append(
			ThinkingContent(
				type="thinking",
				text=text,
				signature=block.get("signature"),
			)
		)

	return blocks


def _get_text_content(message: dict[str, Any]) -> list[TextContent]:
	if text := message.get("content"):
		return [TextContent(type="text", text=text)]

	return []


def _get_tool_use_content(message: dict[str, Any]) -> list[ToolUseContent]:
	content: list[ToolUseContent] = []
	if not message.get("tool_calls"):
		return content

	for tool_call in message.get("tool_calls", []):
		func = tool_call.get("function")
		content.append(
			ToolUseContent(
				type="tool_use",
				id=tool_call.get("id"),
				name=func.get("name"),
				_args=None,
				args=json.loads(func.get("arguments")),
				override=None,
				status="pending",
				result=None,
				start_time=0,
				end_time=0,
				stdout=None,
				stderr=None,
			)
		)
	return content


def _ensure_tool_responses_for_each_call(messages: list[dict]) -> list[dict]:
	"""
	Ensure every assistant message with tool_calls is followed by one tool message
	per tool_call_id. OpenAI (and compatible APIs) require this; add placeholders
	for any missing tool_call_id so the request is accepted.
	"""
	result = []
	i = 0
	while i < len(messages):
		msg = messages[i]
		if msg.get("role") == "assistant" and msg.get("tool_calls"):
			result.append(msg)
			tool_calls = msg["tool_calls"]
			ids_in_order = [tc.get("id") for tc in tool_calls if tc.get("id")]
			j = i + 1
			while j < len(messages) and messages[j].get("role") == "tool":
				j += 1
			id_to_tool_msg = {}
			for m in messages[i + 1 : j]:
				tid = m.get("tool_call_id")
				if tid:
					id_to_tool_msg[tid] = m
			for tid in ids_in_order:
				if tid in id_to_tool_msg:
					result.append(id_to_tool_msg[tid])
				else:
					result.append({"role": "tool", "tool_call_id": tid, "content": ""})
			i = j
			continue
		result.append(msg)
		i += 1
	return result


def _completions(**kwargs):
	"""Wrapper around litellm.completion that retries on rate limit errors."""
	retries = 0
	import litellm

	messages = kwargs.get("messages")
	if messages:
		kwargs = dict(kwargs)
		kwargs["messages"] = _ensure_tool_responses_for_each_call(messages)

	while True:
		try:
			return litellm.completion(**kwargs)
		except Exception as e:  # type: ignore
			if retries >= MAX_RETRIES or (
				"request would exceed the rate limit" not in str(e) and "Overloaded" not in str(e)
			):
				e.add_note("number of retries: " + str(retries))
				otto.log_error("litellm_completion error", model=kwargs.get("model"))
				raise e

			# Anthropic rate limit is set on a per minute basis
			delay = min(random.randint(0, 5 * (2**retries)), 60)
			time.sleep(delay)

			retries += 1


def _get_inter_chunk_latency(timestamps: list[float]) -> float:
	"""Average time between consecutive stream chunks (seconds)."""
	if len(timestamps) < 2:
		return 0.0
	diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
	return sum(diffs) / len(diffs)


def _stream(
	completion: CustomStreamWrapper, item: SessionItem, session_id: str | None
) -> Generator[ContentChunk, None, StreamReturn]:
	"""
	Iterates over the completion stream iterable, publishes the chunks, collates
	them into a list and if a signature is found, updates the item's content.

	returns a list of ContentChunk
	"""
	timestamps: list[float] = []
	chunks: list[ContentChunk] = []

	yield TextContentChunk(
		type="system",
		message="start",
		content="",
		item_id=item["id"],
		session_id=session_id or "",
	)

	try:
		for chunk in completion:
			timestamps.append(time.time())
			ccs = _stream_chunk(chunk, item["id"], session_id)
			for cc in ccs:
				logger.debug({"id": item["id"], "content": cc["content"]})
				chunks.append(cc)
				yield cc
	except Exception as e:
		yield TextContentChunk(
			type="system",
			message="error",
			content=str(e),
			item_id=item["id"],
			session_id=session_id or "",
		)
		raise e

	yield TextContentChunk(
		type="system",
		message="end",
		content="",
		item_id=item["id"],
		session_id=session_id or "",
	)

	return StreamReturn(chunks, _get_inter_chunk_latency(timestamps))


def _stream_chunk(chunk: ModelResponseStream, item_id: str, session_id: str | None) -> list[ContentChunk]:
	"""
	publish using user, session_id, item_id
	"""
	delta = chunk.choices[0].delta
	if hasattr(chunk, "choices") and len(chunk.choices) > 0:
		c0 = chunk.choices[0]
		
		# Handle Gemini Grounding Metadata (Citations)
		if hasattr(c0, "grounding_metadata") and c0.grounding_metadata:
			try:
				gm = c0.grounding_metadata
				chunks_info = gm.get("groundingChunks", [])
				# supports = gm.get("groundingSupports", []) 
				# (Inline citation logic omitted for streaming simplicity)
				
				if chunks_info:
					sources_text = "\n\n### Sources:\n"
					for idx, ch in enumerate(chunks_info):
						if "web" in ch:
							uri = ch["web"].get("uri")
							title = ch["web"].get("title", "Link")
							sources_text += f"{idx+1}. [{title}]({uri})\n"
					
					# Yield sources as a text chunk
					ccs.append(TextContentChunk(
						type="text",
						message="content",
						content=sources_text,
						item_id=item_id,
						session_id=session_id or "",
					))
			except Exception as e:
				logger.error(f"Error parsing grounding metadata: {e}")

	logger.debug(
		{
			"message": "_stream_chunk",
			"id": item_id,
			"session_id": session_id,
			"delta": delta.to_json(indent=None),
		}
	)

	if hasattr(delta, "reasoning_content") and delta.reasoning_content:
		cc = TextContentChunk(
			type="thinking",
			message="content",
			content=delta.reasoning_content,
			item_id=item_id,
			session_id=session_id or "",
		)
		return [cc]

	if hasattr(delta, "content") and delta.content:
		cc = TextContentChunk(
			type="text",
			message="content",
			content=delta.content,
			item_id=item_id,
			session_id=session_id or "",
		)
		return [cc]

	if not hasattr(delta, "tool_calls") or not delta.tool_calls:
		return []

	ccs = []
	for tool_call in delta.tool_calls:
		if not hasattr(tool_call, "function"):
			continue

		cc = ToolUseContentChunk(
			type="tool_use",
			message="content",
			content=ToolUseDelta(
				id=tool_call.id,
				name=tool_call.function.name,
				args=tool_call.function.arguments,
			),
			item_id=item_id,
			session_id=session_id or "",
		)

	return ccs


def _get_end_reason(completion_response: dict):
	comp = completion_response.get("choices", [{}])[0]
	if comp.get("finish_reason") == "tool_use" or comp.get("finish_reason") == "tool_calls":
		return "tool_use"

	if comp.get("finish_reason") == "stop":
		return "turn_end"

	return None


def _set_key(model: str) -> str | None:
	provider = get_provider(model)
	if not provider:
		return f"Model {model} not supported"

	key, value = get_key(provider)
	if not key:
		return f"Model {model} not supported"

	if not value:
		return f"API key {key} not set"

	os.environ[key] = value
	return None


def _should_preserve_thinking(model: str):
	# reference: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking?q=thinking#preserving-thinking-blocks
	return "sonnet" in model or "opus" in model


def _get_thinking(thinking_effort: ReasoningEffort | None):
	# Used by Gemini and Anthropic models
	if thinking_effort is None:
		return None

	return {"type": "enabled", "budget_tokens": DEFAULT_REASONING_BUDGET_MAP[thinking_effort]}


	if not diffs:
		return 0.0
	return sum(diffs) / len(diffs)


def _openai_responses_adapter(
	model: str,
	messages: list[dict],
	item: SessionItem,
	session_id: str | None,
    tools: list[dict]
) -> Generator[ContentChunk, None, None]:
	"""
	Custom adapter to call OpenAI v1/responses API directly.
	Bypasses LiteLLM to support 'web_search' tool.
	"""
	import os
	
	# Start System Chunk
	yield TextContentChunk(
		type="system",
		message="start", # Revert to 'start' too
		content="",
		item_id=item["id"],
		session_id=session_id or "",
	)
	
	# Determine if we should inject web_search
	# Logic: If 'tools' contains a tool named "search", "web_search", etc., we enable native search.
	enable_search = False
	final_tools = []
		
	search_triggers = ["search", "web_search", "google_search"]
	
	if tools:
		for t in tools:
			# Check for function tool
			func_name = t.get("function", {}).get("name", "")
			if func_name in search_triggers:
				enable_search = True
				# Do NOT include the local dummy/alias tool in the API call
				# This prevents the model from trying to call 'function:search' instead of using native search
				continue 
			final_tools.append(t)
	
	# If we found a search trigger, inject the NATIVE tool
	if enable_search:
		final_tools.append({"type": "web_search"})

	api_key = os.environ.get("OPENAI_API_KEY") # _set_key should have verified this
	if not api_key:
		raise Exception("OpenAI API Key not set")

    # Map LiteLLM/Otto messages to OpenAI API format
    # Simple pass-through mostly works, but check roles.
    # We strip 'name' if empty to avoid 400s
	cleaned_messages = []
	for m in messages:
		msg = {k: v for k, v in m.items() if v is not None}
		if "name" in msg and not msg["name"]:
			del msg["name"]
		cleaned_messages.append(msg)

    # Last message is usually user input, but 'input' field in Responses API 
    # might be separate?
    # Docs: "input": "user query", "messages": [history]
    # But usually 'messages' array handles it.
    # The example showed: "input": "..."
    # If using 'messages', we can probably omit 'input' or convert last user message to input?
    # OpenAI Responses API docs are scarce in my context, but standard Agents usually take 'messages'.
    # User's example used "input": "..." and NO "messages".
    # BUT we need history.
    # I will try sending `messages` + `input` (last message content).
    # Or just `model` + `tools` + `messages` (if supported).
    
    # User example:
    # "model": "gpt-5", "tools": [...], "input": "..."
    
    # We will assume 'input' is required and extract the LAST user message content.
    # If history exists, how do we pass it? 
    # Beta endpoints often differ. I will assume `messages` + `input` implies history.
    # Or strict "input-only" if it's stateless? The user script didn't show history.
    # I'll try to pass `messages` (excluding last) and `input` (last).
	
	input_text = ""
	history = []
	if cleaned_messages:
		last_msg = cleaned_messages[-1]
		if last_msg.get("role") == "user":
			input_text = last_msg.get("content", "")
			history = cleaned_messages[:-1]
		else:
			# If last is not user (rare), just send all as input? Unlikely.
			history = cleaned_messages
	
	payload = {
		"model": model.replace("openai/", ""), # strip provider prefix
		"tools": final_tools,
		"input": input_text
	}
	
	# If history is supported, add it. If not, this might fail or ignore it.
	# I will add it as valid context if the API accepts it.
	# NOTE: If v1/responses is STRICTLY "single turn with tools", we lose context.
	# But typically these new endpoints accept 'messages'.
	# I will try to pass 'messages' if history exists.
	# payload["messages"] = history # Uncomment if API supports it. User example didn't show it.
	
	headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}"
	}
	
	url = "https://api.openai.com/v1/responses"
	
	try:
		# Non-streaming call (Simpler parsing)
		resp = requests.post(url, headers=headers, json=payload, timeout=60)
		
		if resp.status_code != 200:
			raise Exception(f"OpenAI Responses API Error ({resp.status_code}): {resp.text}")
			
		data = resp.json()
		# Expecting list of events: [{type: web_search_call, ...}, {type: message, ...}]
		
		full_text = ""
		sources_text = ""
		
		for event in data:
			if not isinstance(event, dict): continue
			
			evt_type = event.get("type")
			
			if evt_type == "web_search_call":
				# Maybe log this?
				pass
				
			elif evt_type == "message":
				# Parse content
				content_list = event.get("content", [])
				for part in content_list:
					if part.get("type") == "output_text":
						text_part = part.get("text", "")
						annotations = part.get("annotations", [])
						
						# Process citations
						if annotations:
							# Sort by index reversed to insert? Or just append sources?
							# User wants "Sources" section appended.
							# Or inline? User code example did inline.
							# But here I'm reconstructing `ContentChunk`.
							# I will collect unique citations for the Source block.
							valid_links = []
							for ann in annotations:
								if ann.get("type") == "url_citation":
									valid_links.append((ann.get("url"), ann.get("title")))
							
							if valid_links:
								sources_text = "\n\n### Sources:\n"
								for idx, (uri, title) in enumerate(valid_links):
									sources_text += f"{idx+1}. [{title or 'Link'}]({uri})\n"
						
						full_text += text_part
		
		# Yield Text Chunk
		if full_text:
			yield TextContentChunk(
				type="text",
				message="content",
				content=full_text,
				item_id=item["id"],
				session_id=session_id or "",
			)
			
		# Yield Sources Chunk
		if sources_text:
			yield TextContentChunk(
				type="text",
				message="content",
				content=sources_text,
				item_id=item["id"],
				session_id=session_id or "",
			)

	except Exception as e:
		yield TextContentChunk(
			type="system",
			message="error",
			content=str(e),
			item_id=item["id"],
			session_id=session_id or "",
		)
		raise e

	# End System Chunk
	yield TextContentChunk(
		type="system",
		message="end",
		content="",
		item_id=item["id"],
		session_id=session_id or "",
	)
