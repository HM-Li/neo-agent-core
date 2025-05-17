# Neo

An extensible framework for creating LLM applications that work across multiple providers.

Neo offers a consistent interface for different language models, allowing your applications to switch seamlessly between providers. The framework prioritizes customization, expandability, and the ability to use multiple LLM services together effectively.

## Installation

```bash
uv pip install -e .
```

## Getting Started

Check out our example notebooks to learn how to use neo:
- [LLM Interaction Management: Thread](./examples/content-context-thread.ipynb) - Learn how to manage conversation history with `threads`
- [Getting Started with Neo](./examples/agentic/neo.ipynb) - Basic introduction to using `neo`
- [Working with Different Models](./examples/models) - Lower level api for model providers

For more examples, see the [examples directory](./examples/).

## Development

Synchronize all dependencies including extras:

```bash
uv sync --all-extras
```

## Environment Variables

### Model Service Endpoint API Keys
- OpenAI: `OPENAI_API_KEY`
- XAI: `XAI_API_KEY`
- Google: `GEMINI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`

### Logging
Configure webhook logging:
```bash
export NEO_LOGGER_WEBHOOK_URL="your_webhook_url"
```