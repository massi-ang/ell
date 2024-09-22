from ell.configurator import config
import boto3

import logging
import colorama

logger = logging.getLogger(__name__)


def register(client):
    """
    Register Bedrock models with the provided client.

    This function takes an Boto3 client and registers various Bedrock models
    with the global configuration. It allows the system to use these models
    for different AI tasks.

    Args:
        client (boto3.Client): An instance of the BedrockRuntime client to be used
                                for model registration.

    Note:
        The function doesn't return anything but updates the global
        configuration with the registered models.
    """
    model_data = [
        ("anthropic.claude-3-haiku-20240307-v1:0", "bedrock"),
    ]
    for model_id, owned_by in model_data:
        config.register_model(model_id, client)


default_client = None
try:
    default_client = boto3.client("bedrock-runtime")
except:
    pass

register(default_client)
config.default_client = default_client
