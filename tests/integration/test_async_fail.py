import pytest
import openai

@pytest.mark.asyncio
async def test_async_openai_fail(mock_openai):
    mock_openai.add_response("Hello from async!")
    
    client = openai.AsyncOpenAI(api_key="fake")
    print(f"DEBUG: client is {client}")
    # This should fail if the mock returns a sync object instead of a coroutine
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hi"}]
    )
    
    assert response.choices[0].message.content == "Hello from async!"
