<div align="center">
  <a href="https://www.netmind.ai/">
    <img alt="netmind.ai" height="100px" src="https://netmindai.blob.core.windows.net/netmind/METMINDd6ab6f8d9e6ef91b80a19b0c089a7f13.svg">
  </a>
</div>

***


# NetMind Python API library
[![PyPI version](https://img.shields.io/pypi/v/netmind.svg)](https://pypi.org/project/netmind/)
[![PyPI version](https://img.shields.io/pypi/l/netmind.svg)](https://pypi.org/project/netmind/)
[![X](https://img.shields.io/badge/X-@NetMindAi-1DA1F2?logo=twitter&style=flat)](https://x.com/NetMindAi)
[![Facebook](https://img.shields.io/badge/Facebook-@netmindai-1877F2?logo=facebook&logoColor=white&style=flat)](https://www.facebook.com/netmindai)
[![Telegram](https://img.shields.io/badge/Telegram-@NetmindAI-2CA5E0?logo=telegram&logoColor=white&style=flat)](https://t.me/NetmindAI)

The [NetMind Python API Library](https://pypi.org/project/netmind/) is the official Python client for [NetMind](https://www.netmind.ai/)'s API platform, providing a convenient way for interacting with the REST APIs and enables easy integrations with Python 3.10+ applications with easy to use synchronous and asynchronous clients.


## 📚 Table of Contents

- [Installation](#installation)
    - [Setting up API Key](#setting-up-api-key)
        - [Setting environment variable](#setting-environment-variable)
        - [Using the client](#using-the-client)
- [Usage – Python Client](#usage--python-client)
    - [Chat Completions](#chat-completions)
        - [Streaming](#streaming)
        - [Async usage](#async-usage)
    - [Embeddings](#embeddings)
        - [Async usage](#async-usage-1)
    - [Files](#files)
        - [Async usage](#async-usage-2)
- [Usage – CLI](#usage--cli)

## Installation

To install NetMind Python Library from PyPI, simply run:

```shell
pip install --upgrade netmind
```

### Setting up API Key

> You will need to create an account with [NetMind.ai](https://www.netmind.ai) to obtain a NetMind API Key.

Once logged in to the NetMind Playground, you can find available API keys in [Dashboard](https://www.netmind.ai/user/dashboard).

#### Setting environment variable

```shell
export NETMIND_API_KEY=<your_netmind_api_key>
```

#### Using the client

```python
from netmind import NetMind


client = NetMind(api_key="your_netmind_api_key")
```

This repo contains both a Python Library and a CLI. We'll demonstrate how to use both below.

## Usage – Python Client

### Chat Completions

```python
from netmind import NetMind


client = NetMind()


# Simple text message
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "system", "content": "Act like you are a helpful assistant."},
        {"role": "user", "content": "Hi there!"},
    ],
    max_tokens = 512
)
print(response.choices[0].message.content)

# Multi-modal message with text and image
response = client.chat.completions.create(
    model="doubao/Doubao-1.5-vision-pro",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"
                }
            }
        ]
    }]
)
print(response.choices[0].message.content)
```

The chat completions API supports three types of content:
- Plain text messages using the `content` field directly
- Multi-modal messages with images using `type: "image_url"`


When using multi-modal content, the `content` field becomes an array of content objects, each with its own type and corresponding data.

#### Streaming

```python
from netmind import NetMind


client = NetMind()
stream = client.chat.completions.create(
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    messages=[
        {"role": "system", "content": "Act like you are a helpful assistant."},
        {"role": "user", "content": "Hi there!"},
    ],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

#### Async usage

```python
import asyncio
from netmind import AsyncNetMind


async_client = AsyncNetMind()


async def async_chat_completion():
    response = await async_client.chat.completions.create(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages=[
            {"role": "system", "content": "Act like you are a helpful assistant."},
            {"role": "user", "content": "Hi there!"},
        ]
    )
    print(response.choices[0].message.content)


asyncio.run(async_chat_completion())
```

### Embeddings

```python
from netmind import NetMind

client = NetMind()


response = client.embeddings.create(
    model="nvidia/NV-Embed-v2",
    input=["Hello world", "NetMind is awesome!"]
)
print(len(response.data[0].embedding))
```

#### Async usage

```python
import asyncio
from netmind import AsyncNetMind


async_client = AsyncNetMind()


async def async_embeddings():
    response = await async_client.embeddings.create(
        model="nvidia/NV-Embed-v2",
        input=["Hello world", "NetMind is awesome!"]
    )
    print(len(response.data[0].embedding))
asyncio.run(async_embeddings())
```

### Files

```python
from netmind import NetMind
from netmind.types.files import FilePurpose


client = NetMind()


# Upload a file
file_response = client.files.create(
    file="path/to/your/file.jsonl",
    purpose=FilePurpose.fine_tune
)
print(f"File uploaded with ID: {file_response.id}")

# List files
files = client.files.list()
print("files found:", len(files.data))


file_id = "your_file_id_here"
# Retrieve a file
file = client.files.retrieve(file_id)  
print(file)

# Retrieve download url for a file
download_url = client.files.retrieve_url(file_id)
print("Download URL:", download_url.presigned_url)


# Delete a file
client.files.delete(file_id)
```
#### Async usage
```python
import asyncio
from netmind import AsyncNetMind
from netmind.types.files import FilePurpose


async_client = AsyncNetMind()


async def async_file_operations():
    # Upload a file
    file_response = await async_client.files.create(
        file="path/to/your/file.jsonl",
        purpose=FilePurpose.fine_tune
    )
    print(f"File uploaded with ID: {file_response.id}")

    # List files
    files = await async_client.files.list()
    print("files found:", len(files.data))

    file_id = "your_file_id_here"
    # Retrieve a file
    file = await async_client.files.retrieve(file_id)  
    print(file)

    # Retrieve download url for a file
    download_url = await async_client.files.retrieve_url(file_id)
    print("Download URL:", download_url.presigned_url)

    # Delete a file
    await async_client.files.delete(file_id)

asyncio.run(async_file_operations())
```



## Usage – CLI
coming soon

