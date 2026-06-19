Cascade Router ⚡️

Predictive Multi-Model AI Routing Infrastructure.

Cascade is a bare-metal C++ proxy that intercepts OpenAI SDK traffic and dynamically routes prompts to the most cost-effective model (e.g., gpt-4o-mini vs gpt-4o) based on semantic complexity. It reduces enterprise LLM bills by up to 75% while adding less than 5ms of latency.

The Problem

Enterprises default to hardcoding API calls to expensive frontier models because they lack the runtime infrastructure to confidently trust smaller, cheaper models. Existing Python-based routers (like LiteLLM) or SaaS platforms introduce 65ms - 200ms of latency, breaking agentic workflows and streaming UIs.

The Cascade Solution

Cascade moves routing intelligence ahead of the inference lifecycle and pushes it down to the metal.

Drop-in Replacement: Change 1 line of code in your OpenAI SDK (base_url = "http://localhost:8000/v1").

Predictive Intelligence: A highly distilled Logistic Regression classifier runs on top of a 384-dimensional WordPiece embedding space to predict $P(\text{success})$ for smaller models.

Zero-Overhead: Written in C++ utilizing SIMD JSON parsing and INT8 ONNX Runtime matrix multiplication. The entire routing decision executes in ~4.6 milliseconds.

🚀 Quick Start (Docker)

Run the pre-compiled Ubuntu container. It automatically loads the v0.1 routing weights and exposes the OpenAI-compatible proxy on port 8000.

# 1. Clone the repository
git clone [https://github.com/AmirMohaddesi/cascade-router.git](https://github.com/AmirMohaddesi/cascade-router.git)
cd cascade-router

# 2. Start the proxy
docker compose up -d


Test the Router:
Point your standard curl or Python OpenAI SDK to the proxy. The router will intercept it, rewrite the model to the cheapest capable tier, forward it securely to OpenAI, and return the response with a custom latency header.

curl -s -D - -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-openai-api-key" \
  -d '{"model":"cascade-auto","messages":[{"role":"user","content":"Explain recursion in one sentence."}]}'


Look for X-Cascade-Latency in the returned headers to verify the sub-5ms routing speed.

🧠 Architecture & The v0.1 "Brain"

This repository includes the complete open-source C++ proxy and a foundational router_weights.json brain (v0.1).

The v0.1 weights were trained using our automated LLM-as-a-Judge pipeline on an 800+ prompt enterprise dataset, successfully identifying the failure boundaries between GPT-4o and GPT-4o-mini with 67.8% baseline accuracy, yielding a 75% pass rate for the smaller model.

For Enterprise: The included Python pipeline (src/) allows organizations to ingest their own historical prompt logs, evaluate them, and train highly-calibrated, domain-specific routing weights tailored to their proprietary use cases.

📊 Benchmarks

Because Cascade avoids the Python Global Interpreter Lock (GIL) and external network hops, it is the only semantic router capable of scaling to thousands of concurrent requests without bottlenecking upstream applications.

Architecture

Implementation

Latency Overhead

Cascade Router

Bare-Metal C++, INT8 ONNX

~4.6 ms

Python Proxy

LiteLLM, FastAPI

~65.0 ms

SaaS Router

External API Network Hop

~180.0 ms

LLM-as-a-Judge

Evaluating via API

~850.0 ms

See the Technical Whitepaper for full methodology and latency charts.

License

The core C++ proxy and v0.1 routing weights are released under the MIT License.