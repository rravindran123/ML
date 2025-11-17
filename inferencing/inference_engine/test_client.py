import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from server import app
import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager

@pytest.fixture(scope="session")
def test_app():
    """Create a test app."""
    print("Setting up test app...")
    return app

@pytest.fixture
def client(test_app):
    """Create a test client with the test app."""
    return TestClient(test_app)


def test_basic_generate(client):
    """Test the /basic_generate route for correct server-client interaction."""
    response = client.post(
        "/test_basic_generate",
        json={"prompt": "Hello, I am"}
    )

    # Basic status check
    assert response.status_code == 200

    # Parse JSON response
    data = response.json()

    # Validate fields
    assert "generated_text" in data
    assert isinstance(data["generated_text"], str)
    assert data["generated_text"].startswith("Hello, I am")
    assert "generated response" in data["generated_text"]

def test_generate(client):
   response = client.post(
       "/basic_generate",
       json={"prompt": "I like cakes and cookies"}
   )
   assert response.status_code == 200
   data = response.json()
   assert "generated_text" in data
   assert isinstance(data["generated_text"], str)
   assert len(data["generated_text"]) > 0

def test_generate_batch(client):
   # Test with multiple prompts
   test_prompts = [
       "Hello, I am",
       "The weather is",
       "I want to",
       "The best way to",
       "The most efficient way to"
   ]
   response = client.post(
       "/generate",
       json={"prompts": test_prompts}
   )

@pytest.fixture(scope="session", autouse=True)
def teardown():
    yield
    # Ensure proper cleanup after all tests
    from server import cleanup
    cleanup()

# def test_generate(client):
#     response = client.post(
#         "/basic_generate",
#         json={"prompt": "Hello, I am"}
#     )
#     assert response.status_code == 200
#     data = response.json()
#     assert "generated_text" in data
#     assert isinstance(data["generated_text"], str)
#     assert len(data["generated_text"]) > 0


# @pytest_asyncio.fixture
# async def async_client(test_app, event_loop):
#     """Create an async test client with the test app."""
#     print("Creating async client...")
#     async with AsyncClient(app=test_app, base_url="http://test") as ac:
#         yield ac
@pytest_asyncio.fixture
async def async_client(test_app):
    """Create an async test client with the test app."""
    print("Creating async client...")
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    print("Creating event loop...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    print("Closing event loop...")
    loop.close()

@pytest.mark.asyncio
async def test_generate_stream(async_client):
    print("Starting test_generate_stream...")
    
    prompt = "Hello, I am"
    # Create a streaming request with stream=True
    async with async_client.stream("POST", "/generate_stream", json={"prompt": prompt}) as response:
        assert response.status_code == 200
        
        # Collect all tokens
        tokens = []
        sequence_ids = set()
        
        # Process the response stream directly
        async for chunk in response.aiter_bytes():
            if chunk:
                # Convert bytes to string and split by newlines
                lines = chunk.decode().split('\n')
                for line in lines:
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        token = data["token"]
                        print(f"Received token: {token}")  # Debug print
                        tokens.append(token)
                        sequence_ids.add(data["sequence_id"])
        
        # Verify we got some tokens
        assert len(tokens) > 0
        # Verify we got a sequence ID
        assert len(sequence_ids) == 1
        # Verify tokens form a coherent text
        generated_text = "".join(tokens)
        assert len(generated_text) > 0
        print(f"Final generated text: {prompt} {generated_text}")