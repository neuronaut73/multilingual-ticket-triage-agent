"""
Minimal HTTP client for the local Ollama API.

OllamaClient:
  - Sends a text prompt to the Ollama /api/generate endpoint.
  - Requests JSON-mode output (format="json").
  - Returns the raw response string.
  - No OpenAI, Anthropic, Instructor, or PydanticAI dependency.
"""

import requests


class OllamaClient:
    """
    Wraps the Ollama /api/generate endpoint.

    The caller is responsible for parsing and validating the returned string.
    Using format="json" asks Ollama to constrain its output to valid JSON,
    but the caller must still handle malformed or schema-invalid responses.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        temperature: float,
        timeout_seconds: int,
    ) -> None:
        self.base_url        = base_url.rstrip("/")
        self.model_name      = model_name
        self.temperature     = temperature
        self.timeout_seconds = timeout_seconds

    def generate_json(self, prompt: str) -> str:
        """
        Send prompt to Ollama and return the model's raw response text.

        Raises requests.HTTPError  on non-2xx HTTP responses.
        Raises requests.Timeout    if Ollama does not respond in time.
        Raises requests.ConnectionError if Ollama is not reachable.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model":  self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return data["response"]
