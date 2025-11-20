# Test Coverage Gaps - Detailed Code Locations

## Critical Test Gaps - Provider Integrations

### 1. OpenAI Provider - UNTESTED

#### Implementation Locations:
- **Config:** `src/rainrag/config.py:62-77` (OpenAIConfig class)
- **Query Module:** 
  - `src/rainrag/query.py:39, 52` (client initialization)
  - `src/rainrag/query.py:186-197` (embed_query with OpenAI)
  - `src/rainrag/query.py:343-357` (generate_answer with OpenAI)
- **API Module:**
  - `src/rainrag/api.py:246` (health check)
  - `src/rainrag/api.py:252-253` (LLM provider detection)
  - `src/rainrag/api.py:267-268` (embedding model display)

#### Actual Code (query.py:186-197):
```python
elif self.config.embedding.provider == "openai":
    # Use OpenAI API embeddings
    logger.debug(f"Embedding query using OpenAI API: {query[:100]}...")
    try:
        response = self.openai_client.embeddings.create(
            model=self.config.openai.embedding_model,
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embeddings with OpenAI API: {e}")
        raise RuntimeError(f"OpenAI embeddings API error: {e}") from e
```

#### Actual Code (query.py:343-357):
```python
elif self.config.llm.provider == "openai":
    logger.info("Generating answer using OpenAI API...")
    try:
        response = self.openai_client.chat.completions.create(
            model=self.config.openai.model_name,
            messages=messages,
            max_tokens=self.config.openai.max_tokens,
            temperature=self.config.openai.temperature,
        )
        answer = response.choices[0].message.content.strip()
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Failed to generate answer with OpenAI API: {e}")
        raise RuntimeError(f"OpenAI API error: {e}") from e
```

#### Test Gaps:
- ❌ No tests in `tests/unit/test_query.py` for OpenAI
- ❌ No embedding provider tests with OpenAI
- ❌ No error handling tests
- ❌ No integration tests with OpenAI

---

### 2. Claude (Anthropic) Provider - UNTESTED

#### Implementation Locations:
- **Config:** `src/rainrag/config.py:79-90` (ClaudeConfig class)
- **Query Module:**
  - `src/rainrag/query.py:40, 58` (client initialization)
  - `src/rainrag/query.py:359-383` (generate_answer with Claude)
- **API Module:**
  - `src/rainrag/api.py:254-255` (LLM provider detection)

#### Actual Code (query.py:359-383):
```python
elif self.config.llm.provider == "claude":
    logger.info("Generating answer using Claude API...")
    try:
        # Extract system message and user messages for Claude API
        system_message = ""
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append(msg)

        response = self.claude_client.messages.create(
            model=self.config.claude.model_name,
            max_tokens=self.config.claude.max_tokens,
            temperature=self.config.claude.temperature,
            system=system_message,
            messages=claude_messages,
        )
        answer = response.content[0].text.strip()
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Failed to generate answer with Claude API: {e}")
        raise RuntimeError(f"Claude API error: {e}") from e
```

#### Test Gaps:
- ❌ System message extraction logic not tested
- ❌ Message format conversion not tested
- ❌ No error handling tests
- ❌ No tests for different Claude models
- ❌ No integration tests

---

### 3. Google Gemini Provider - UNTESTED

#### Implementation Locations:
- **Config:** `src/rainrag/config.py:92-108` (GeminiConfig class)
- **Query Module:**
  - `src/rainrag/query.py:41, 64` (client initialization via genai.configure())
  - `src/rainrag/query.py:199-211` (embed_query with Gemini)
  - `src/rainrag/query.py:385-422` (generate_answer with Gemini)
- **API Module:**
  - `src/rainrag/api.py:256-257` (LLM provider detection)
  - `src/rainrag/api.py:269-270` (embedding model display)

#### Actual Code (query.py:199-211):
```python
elif self.config.embedding.provider == "gemini":
    # Use Gemini API embeddings
    logger.debug(f"Embedding query using Gemini API: {query[:100]}...")
    try:
        result = genai.embed_content(
            model=self.config.gemini.embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Failed to generate embeddings with Gemini API: {e}")
        raise RuntimeError(f"Gemini embeddings API error: {e}") from e
```

#### Actual Code (query.py:385-422):
```python
elif self.config.llm.provider == "gemini":
    logger.info("Generating answer using Gemini API...")
    try:
        # Convert messages to Gemini format
        model = genai.GenerativeModel(self.config.gemini.model_name)

        # Extract system message and build conversation
        system_instruction = ""
        conversation_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                conversation_parts.append(msg["content"])
            elif msg["role"] == "assistant":
                # Gemini doesn't use explicit assistant messages in the same way
                # For now, we'll skip assistant messages or handle them differently
                pass

        # Combine system instruction with user message
        if system_instruction:
            prompt = f"{system_instruction}\n\n{conversation_parts[-1]}"
        else:
            prompt = conversation_parts[-1]

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self.config.gemini.max_tokens,
                temperature=self.config.gemini.temperature,
            )
        )
        answer = response.text.strip()
        logger.info("Answer generated successfully")
        return answer
    except Exception as e:
        logger.error(f"Failed to generate answer with Gemini API: {e}")
        raise RuntimeError(f"Gemini API error: {e}") from e
```

#### Test Gaps:
- ❌ genai.embed_content() not tested
- ❌ genai.GenerativeModel() not tested
- ❌ Message format conversion not tested
- ❌ task_type parameter handling not tested
- ❌ Error handling not tested
- ❌ No integration tests

---

### 4. Mistral API Embeddings - UNTESTED

#### Implementation Location:
- `src/rainrag/query.py:157-168` (embed_query with Mistral)

#### Actual Code:
```python
if self.config.embedding.provider == "mistral":
    # Use Mistral API embeddings
    logger.debug(f"Embedding query using Mistral API: {query[:100]}...")
    try:
        response = self.mistral_client.embeddings.create(
            model="mistral-embed",
            inputs=[query]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embeddings with Mistral API: {e}")
        raise RuntimeError(f"Mistral embeddings API error: {e}") from e
```

#### Test Gaps:
- ❌ No tests for mistral_client.embeddings.create()
- ❌ No error handling tests
- ❌ No integration tests

---

## CLI Module - Completely Untested

### Implementation Location:
- `src/rainrag/cli.py` (~150 lines)

### Commands Not Tested:
1. `ingest` - parses VTT files (lines 61-91)
2. `embed` - generates embeddings (lines 93-133)
3. `index` - creates vector index (lines 135-149+)
4. `query` - runs query (not shown in excerpt)
5. `serve` - starts API server (not shown in excerpt)

### Test Gaps:
- ❌ No unit tests for any CLI command
- ❌ No argument parsing tests
- ❌ No error message validation
- ❌ No help text verification
- ❌ No integration tests with CLI

---

## API Module - Incomplete Testing

### Health Endpoint (test_api.py) - Line 234-285

#### Current Implementation:
```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Tests the basic structure but not all provider combinations
```

#### What's NOT Tested:
- ❌ All provider combinations
- ❌ Each provider's specific health status
- ❌ Error conditions in health check

### Query Endpoint (test_api.py) - Line 289-360+

#### Current Implementation:
Uses RAGQueryEngine.query() which calls:
- embed_query() - NOT tested with OpenAI, Claude, Gemini
- build_prompt() - language-specific Russian only partially tested
- generate_answer() - NOT tested with OpenAI, Claude, Gemini

#### What's NOT Tested:
- ❌ /query endpoint with OpenAI provider
- ❌ /query endpoint with Claude provider
- ❌ /query endpoint with Gemini provider
- ❌ Different embedding providers
- ❌ Language parameter variations
- ❌ Error scenarios

### Authentication (test_api.py) - Line 207-232

#### Current Implementation:
```python
def verify_auth_token(authorization: Optional[str] = Header(None)) -> bool:
    """Verify authentication token if configured."""
    # Lines 207-232 in api.py
```

#### What's NOT Tested:
- ❌ verify_auth_token() function not tested
- ❌ Bearer token validation not tested
- ❌ Missing authorization header handling not tested
- ❌ Invalid token handling not tested
- ❌ Query endpoint authentication not tested

---

## Embedding Provider Coverage Matrix

### Current Test Status:

#### test_embed.py (11 tests):
- ✅ EmbeddingCache (4 tests)
  - Cache creation, save/load, exists, nonexistent
- ✅ Embedder with local provider (7 tests)
  - Model loading, document loading, generation, caching
- ❌ **OpenAI embeddings** - 0 tests
- ❌ **Mistral embeddings** - 0 tests
- ❌ **Gemini embeddings** - 0 tests

#### test_query.py (48 tests):
- ✅ Local embedding with E5 prefix (1 test)
  - test_embed_query_with_e5_prefix
- ❌ **OpenAI embeddings** - 0 tests
- ❌ **Mistral embeddings** - 0 tests
- ❌ **Gemini embeddings** - 0 tests

---

## LLM Provider Coverage Matrix

### Current Test Status in test_query.py:

#### vLLM (✅ WELL TESTED - 35+ tests):
- Chat template detection (Mistral, Gemma, GPT, generic)
- Prompt formatting for all templates
- Chat completions API
- Completions API
- Error handling and timeouts

#### Mistral (⚠️ MINIMALLY TESTED - 0 real API tests):
- Configuration tested in test_config.py
- No generate_answer tests with Mistral API

#### OpenAI (❌ NOT TESTED - 0 tests):
- No generate_answer tests
- No chat completions.create() tests
- No error handling

#### Claude (❌ NOT TESTED - 0 tests):
- No message.create() tests
- No system message extraction tests

#### Gemini (❌ NOT TESTED - 0 tests):
- No GenerativeModel tests
- No generate_content() tests
- No message conversion tests

---

## Recommended Test Additions

### Add to test_query.py:

#### OpenAI Tests (12 tests):
```python
class TestOpenAIEmbedding:
    def test_openai_embed_query()
    def test_openai_embed_error_handling()
    def test_openai_embed_rate_limit()
    def test_openai_embed_invalid_key()

class TestOpenAIGeneration:
    def test_openai_generate_answer()
    def test_openai_generate_error_handling()
    def test_openai_generate_timeout()
    def test_openai_generate_invalid_response()
    def test_openai_generate_different_models()
    def test_openai_query_pipeline()
    def test_openai_query_with_language()
    def test_openai_query_error_recovery()
```

#### Claude Tests (12 tests):
```python
class TestClaudeGeneration:
    def test_claude_generate_answer()
    def test_claude_system_message_extraction()
    def test_claude_message_conversion()
    def test_claude_generate_error_handling()
    def test_claude_generate_timeout()
    def test_claude_invalid_response()
    def test_claude_different_models()
    def test_claude_query_pipeline()
    def test_claude_query_with_language()
    def test_claude_message_format()
    def test_claude_streaming()
    def test_claude_error_recovery()
```

#### Gemini Tests (14 tests):
```python
class TestGeminiEmbedding:
    def test_gemini_embed_query()
    def test_gemini_embed_task_type()
    def test_gemini_embed_error_handling()

class TestGeminiGeneration:
    def test_gemini_generate_answer()
    def test_gemini_generative_model()
    def test_gemini_message_conversion()
    def test_gemini_system_instruction()
    def test_gemini_generate_config()
    def test_gemini_generate_error_handling()
    def test_gemini_generate_timeout()
    def test_gemini_invalid_response()
    def test_gemini_different_models()
    def test_gemini_query_pipeline()
    def test_gemini_query_with_language()
```

#### Mistral API Tests (8 tests):
```python
class TestMistralEmbedding:
    def test_mistral_embed_query()
    def test_mistral_embed_error_handling()
    def test_mistral_embed_timeout()
    def test_mistral_embed_invalid_key()

class TestMistralGeneration:
    def test_mistral_api_generate_answer()
    def test_mistral_api_error_handling()
    def test_mistral_api_timeout()
    def test_mistral_api_response_format()
```

### Add to test_api.py (10-12 tests):

```python
# Authentication tests
def test_health_openai_provider()
def test_health_claude_provider()
def test_health_gemini_provider()
def test_query_openai_provider()
def test_query_claude_provider()
def test_query_gemini_provider()
def test_query_different_embedding_providers()
def test_verify_auth_token_valid()
def test_verify_auth_token_invalid()
def test_verify_auth_token_missing()
def test_query_auth_required()
```

### Add test_cli.py (20 tests):

```python
class TestCLIIngest:
    def test_ingest_command()
    def test_ingest_with_config_option()
    def test_ingest_file_not_found()

class TestCLIEmbed:
    def test_embed_command()
    def test_embed_force_regenerate()
    def test_embed_with_openai()
    def test_embed_with_gemini()

class TestCLIIndex:
    def test_index_command()
    def test_index_recreate()

class TestCLIQuery:
    def test_query_command()
    def test_query_with_language()

class TestCLIServe:
    def test_serve_command()
    def test_serve_with_port()
```

### Add test_pipeline_providers.py (10 tests):

```python
class TestPipelineProviders:
    def test_pipeline_with_openai()
    def test_pipeline_with_claude()
    def test_pipeline_with_gemini()
    def test_pipeline_with_openai_embeddings()
    def test_pipeline_with_gemini_embeddings()
    def test_pipeline_provider_switching()
    def test_pipeline_error_handling()
    def test_pipeline_end_to_end_openai()
    def test_pipeline_end_to_end_claude()
    def test_pipeline_end_to_end_gemini()
```

---

## Total Estimated New Tests: 76-92 tests

This would bring total from 121 to ~200+ tests and increase overall coverage to 80%+ with provider coverage at 85%+.

