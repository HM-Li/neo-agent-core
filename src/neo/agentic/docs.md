# Noridoc: Agentic

Path: @/src/neo/agentic

### Overview

The agentic module is the orchestration layer for Neo's task-based agent system. It contains the model registry that manages all supported AI models across multiple providers (OpenAI, Anthropic, Google, XAI), the Neo class that executes tasks in DAG order with tool support, and support infrastructure for managing tasks, instructions, and tool registries.

### How it fits into the larger codebase

The agentic module serves as the core runtime orchestration layer for Neo. It depends on @/src/neo/models/providers for all model implementations, @/src/neo/tools for tool execution, @/src/neo/types for content types and modalities, and @/src/neo/contexts for conversation threading. Callers outside this module (notebooks, scripts, applications) interact with Neo through the Neo class to execute task DAGs. The model registry is the critical integration point between Neo's task execution engine and the underlying model providers - when a task specifies a model name (e.g., "gpt-5-pro"), the registry instantiates the correct model class with the appropriate parameters and modality support. Task dependencies, tool execution, and error handling are coordinated through the Neo class, which uses both the ModelRegistry and ToolRegistry.

### Core Implementation

The ModelRegistry class uses a Singleton metaclass to ensure a single global registry instance. Models are registered at module load time via a static list that maps model names (strings) to model classes and their supported input modalities. The `create_model()` method instantiates models with full parameter forwarding including thinking mode, tool support, custom API keys, structured response models, and response formatting options (JSON mode, boolean responses). Each registered model includes an input_modalities field that declares what content types it can accept (TEXT, IMAGE, AUDIO, STRUCTURED, DOCUMENT). The Neo class maintains a DAG of Task objects, respects task dependencies, and coordinates model execution with tool execution rounds and error handling. Tasks can inherit default model configurations from Neo's initialization, and the ModelRegistry's fuzzy matching feature allows approximate model name resolution if exact matches fail.

### Things to Know

The model registry pattern requires every model to implement the BaseChatModel interface from @/src/neo/models/providers/base.py, ensuring signature compatibility for the `create_model()` factory method. The model_registry.py file contains both the class definition and the static registration list - this dual responsibility means model additions require changes in two places (class instantiation logic and the models list). OpenAI models are registered through both OpenAIResponseModel and OpenAICompleteModel classes depending on the specific model's capabilities (the audio preview model uses OpenAICompleteModel while most others use OpenAIResponseModel). Input modalities are carefully curated per model - for example, gpt-5-pro supports TEXT, IMAGE, STRUCTURED, and DOCUMENT modalities, while o3-mini only supports TEXT. The `create_model()` method passes all initialization parameters directly to model constructors, so adding new model types may require updating this signature if they support new parameters. Task execution through Neo respects max_tool_execution_rounds to prevent infinite loops, and the default_user_input parameter ensures conversations can continue when a model response expects human input.

Created and maintained by Nori.
