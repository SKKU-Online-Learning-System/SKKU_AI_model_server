"""Live check against official Transformers pooling; run in llm_runtime's uv env.

Checks the terminal token contract as well as the resulting embedding, so a
serving configuration that silently changes the vector space fails loudly.
"""
import os

from dotenv import load_dotenv
import httpx
import torch
from transformers import AutoProcessor, Qwen3VLModel


def main():
    load_dotenv()
    model_id = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
    base = f"http://127.0.0.1:{os.getenv('EMBEDDING_PORT', '8003')}"
    key = os.getenv("MODEL_SERVER_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    torch.set_num_threads(4)
    model = Qwen3VLModel.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
        {"role": "user", "content": [{"type": "text", "text": "Causal mask: row 3 sees columns 0–3."}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], return_tensors="pt")
    payload = {"model": model_id, "messages": messages,
               "add_generation_prompt": True, "add_special_tokens": True}
    with httpx.Client(base_url=base, headers=headers, timeout=120) as client:
        tokens = client.post("/tokenize", json=payload)
        tokens.raise_for_status()
        assert tokens.json()["tokens"] == inputs["input_ids"][0].tolist()
        response = client.post("/v1/embeddings", json=payload)
        response.raise_for_status()
    with torch.inference_mode():
        expected = model(**inputs).last_hidden_state[0, -1].float()
    actual = torch.tensor(response.json()["data"][0]["embedding"])
    similarity = torch.nn.functional.cosine_similarity(expected, actual, dim=0).item()
    assert similarity > 0.99, f"Embedding mismatch: cosine={similarity}"
    print(f"Official Transformers/vLLM parity: cosine={similarity:.6f}")


if __name__ == "__main__":
    main()
