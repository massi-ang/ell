from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ell.provider import APICallResult, Provider
from ell.types import Message, ContentBlock, ToolCall
from ell.types._lstr import _lstr
import json
from ell.configurator import config, register_provider
from ell.types.message import LMP
from ell.util.serialization import serialize_image
from io import BytesIO

try:
    import boto3

    class BedrockProvider(Provider):

        # XXX: This content block conversion etc might need to happen on a per model basis for providers like groq etc. We will think about this at a future date.
        @staticmethod
        def content_block_to_bedrock_converse_format(
            content_block: ContentBlock,
        ) -> Dict[str, Any]:
            if content_block.image:
                buffer = BytesIO()
                content_block.image.save(buffer, format="PNG")
                base64_image = buffer.getvalue()
                return {"image": {"format": "png", "source": {"bytes": base64_image}}}
            elif content_block.text:
                return {"text": content_block.text}
            elif content_block.parsed:
                return {"text": content_block.parsed.model_dump_json()}
            else:
                return None

        @staticmethod
        def message_to_bedrock_format(message: Message) -> Dict[str, Any]:
            converse_message = {
                "role": message.role,
                "content": list(
                    filter(
                        None,
                        [
                            BedrockProvider.content_block_to_bedrock_converse_format(c)
                            for c in message.content
                        ],
                    )
                ),
            }
            if message.tool_calls:
                try:
                    converse_message["content"].extend(
                        [
                            {
                                "toolUse": {
                                    "input": json.dumps(tool_call.params.model_dump()),
                                    "name": tool_call.tool.__name__,
                                    "toolUseId": tool_call.tool_call_id,
                                }
                            }
                            for tool_call in message.tool_calls
                        ]
                    )
                except TypeError as e:
                    print(
                        f"Error serializing tool calls: {e}. Did you fully type your @ell.tool decorated functions?"
                    )
                    raise
                converse_message["content"] = (
                    None  # Set content to null when there are tool calls
                )

            if message.tool_results:
                converse_message["content"].append(
                    {
                        "content": {"text": message.tool_results[0].result[0].text},
                        "toolUseId": message.tool_results[0].tool_call_id,
                    }
                )
                assert (
                    len(message.tool_results[0].result) == 1
                ), "Tool result should only have one content block"
                assert (
                    message.tool_results[0].result[0].type == "text"
                ), "Tool result should only have one text content block"
            return converse_message

        @classmethod
        def call_model(
            cls,
            client: Any,
            model: str,
            messages: List[Message],
            api_params: Dict[str, Any],
            tools: Optional[list[LMP]] = None,
        ) -> APICallResult:
            final_call_params = api_params.copy()
            converse_messages = [
                cls.message_to_bedrock_format(message)
                for message in messages
                if message.role != "system"
            ]
            final_call_params["system"] = [
                {"text": message.content[0].text}
                for message in messages
                if message.role == "system"
            ]
            actual_n = api_params.get("n", 1)
            final_call_params["modelId"] = model
            final_call_params["messages"] = converse_messages
            final_call_params.pop("n", None)
            final_call_params.pop("stream", None)
            final_call_params.pop("stream_options", None)

            if model == "o1-preview" or model == "o1-mini":
                # Ensure no system messages are present
                assert (
                    len(converse_messages.system) == 0
                ), "System messages are not allowed for o1-preview or o1-mini models"

                response = client.converse(**final_call_params)
                final_call_params.pop("stream", None)
                final_call_params.pop("stream_options", None)

            elif final_call_params.get("response_format"):
                final_call_params.pop("stream", None)
                final_call_params.pop("stream_options", None)
                response = client.converse(**final_call_params)
            else:
                # Tools not working with structured API
                if tools:
                    final_call_params["tool_choice"] = "auto"
                    final_call_params["tools"] = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.__name__,
                                "description": tool.__doc__,
                                "parameters": tool.__ell_params_model__.model_json_schema(),
                            },
                        }
                        for tool in tools
                    ]
                    final_call_params.pop("stream", None)
                    final_call_params.pop("stream_options", None)

                response = client.converse(**final_call_params)

            return APICallResult(
                response=response,
                actual_streaming=False,
                actual_n=actual_n,
                final_call_params=final_call_params,
            )

        @classmethod
        def process_response(
            cls,
            call_result: APICallResult,
            _invocation_origin: str,
            logger: Optional[Any] = None,
            tools: Optional[List[LMP]] = None,
        ) -> Tuple[List[Message], Dict[str, Any]]:
            choices_progress = defaultdict(list)
            api_params = call_result.final_call_params
            metadata = {}
            # XXX: Remove logger and refactor this API

            if not call_result.actual_streaming:
                response = [call_result.response]
            else:
                response = call_result.response

            print(response)
            tracked_results = []
            for chunk in response:
                usage = {}
                usage["prompt_tokens"] = chunk["usage"].get("inputTokens", 0)
                usage["completion_tokens"] = chunk["usage"].get("outputTokens", 0)
                usage["total_tokens"] = chunk["usage"].get("totalTokens", 0)

                metadata = {"usage": usage}
                print(metadata)
                content = []

                if call_result.actual_streaming:
                    text_content = "".join(
                        (choice.delta.content or "" for choice in choice_deltas)
                    )
                    if text_content:
                        content.append(
                            ContentBlock(
                                text=_lstr(
                                    content=text_content,
                                    _origin_trace=_invocation_origin,
                                )
                            )
                        )

                    # Determine the role for streaming responses, defaulting to 'assistant' if not provided
                    streamed_role = next(
                        (
                            choice.delta.role
                            for choice in choice_deltas
                            if choice.delta.role
                        ),
                        "assistant",
                    )
                else:

                    content.append(
                        ContentBlock(
                            text=_lstr(
                                content=chunk["output"]["message"]["content"][0][
                                    "text"
                                ],
                                _origin_trace=_invocation_origin,
                            )
                        )
                    )

                if False:
                    assert (
                        tools is not None
                    ), "Tools not provided, yet tool calls in response. Did you manually specify a tool spec without using ell.tool?"
                    for tool_call in choice.tool_calls:
                        matching_tool = next(
                            (
                                tool
                                for tool in tools
                                if tool.__name__ == tool_call.function.name
                            ),
                            None,
                        )
                        if matching_tool:
                            params = matching_tool.__ell_params_model__(
                                **json.loads(tool_call.function.arguments)
                            )
                            content.append(
                                ContentBlock(
                                    tool_call=ToolCall(
                                        tool=matching_tool,
                                        tool_call_id=_lstr(
                                            tool_call.id,
                                            _origin_trace=_invocation_origin,
                                        ),
                                        params=params,
                                    )
                                )
                            )

                tracked_results.append(
                    Message(
                        role=(
                            chunk["output"]["message"]["role"]
                            if not call_result.actual_streaming
                            else streamed_role
                        ),
                        content=content,
                    )
                )
            return tracked_results, metadata

        @classmethod
        def supports_streaming(cls) -> bool:
            return True

        @classmethod
        def get_client_type(cls) -> Type:
            client = boto3.client("bedrock-runtime")
            return type(client)

    register_provider(BedrockProvider)
except ImportError:
    pass
