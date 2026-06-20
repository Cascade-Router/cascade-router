# Cascade Router ⚡️

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Docker Hub](https://img.shields.io/badge/docker-ready-brightgreen.svg)]()
[![Latency](https://img.shields.io/badge/latency-<5ms-orange.svg)]()
[![Version](https://img.shields.io/badge/version-0.1.0_Beta-purple.svg)]()

**Predictive Multi-Model AI Routing Infrastructure.**

Cascade is a high-performance, bare-metal C++ proxy that intercepts OpenAI SDK traffic and dynamically routes prompts to the most cost-effective model (e.g., `gpt-4o-mini` vs `gpt-4o`). Powered by a highly distilled local embedding classifier, Cascade reduces enterprise LLM inference bills by up to **75%** while introducing **less than 5ms** of latency.

## ⚡️ The Cascade Architecture

```mermaid
graph TD
    A[User App / Python SDK] -->|HTTP POST| B(Cascade C++ Proxy)
    B --> C{ONNX Distilled Classifier}
    C -->|> 90% Confidence| D[gpt-4o-mini]
    C -->|< 90% Confidence| E[gpt-4o]
    D --> F[Runtime Output Validator]
    F -->|Validation Fail| E
    F -->|Validation Pass| G[Return Response]
    E --> G
```

1. **Transparent Integration:** A single-line `base_url` change in your OpenAI SDK (`http://localhost:8000/v1`).

---

## 🛑 The Industry Bottleneck
Enterprises default to hardcoding API calls to expensive frontier models out of fear of hallucinations or degraded output. Existing dynamic routing solutions—whether Python-based proxies (like LiteLLM) or third-party SaaS platforms—introduce 65ms to 200ms of latency. This unacceptable overhead breaks real-time agentic workflows, streaming UIs, and high-throughput data pipelines.

## 🚀 Quick Start
Deploy the pre-compiled, production-ready Ubuntu container. It automatically loads the v0.1 routing weights and exposes an OpenAI-compatible proxy.

```bash
# 1. Clone the repository
git clone https://github.com/AmirMohaddesi/cascade-router.git
cd cascade-router

# 2. Start the proxy in detached mode
docker compose up -d
```

## Verification & Testing
Point your standard curl or Python OpenAI SDK to the proxy. The router will seamlessly intercept the payload, rewrite the model target to the cheapest capable tier, execute the upstream request, and return the response.

```bash
curl -s -D - -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-openai-api-key" \
  -d '{"model":"cascade-auto","messages":[{"role":"user","content":"Explain recursion in one sentence."}]}'
```

Look for `X-Cascade-Latency` in the returned headers to verify the sub-5ms routing speed.

## 🧠 The v0.1 Foundation Model
This repository includes the open-source C++ proxy alongside our foundational `router_weights.json` (v0.1). Trained via an automated LLM-as-a-Judge pipeline on an 800+ prompt enterprise dataset, the v0.1 weights successfully map the failure boundaries between GPT-4o and GPT-4o-mini. It achieved a **67.8%** baseline accuracy, unlocking a **75%** pass rate for the smaller model without degrading response quality.

## 📊 Performance Benchmarks
By eliminating the Python Global Interpreter Lock (GIL) and bypassing external network hops, Cascade is the only semantic router capable of scaling to thousands of concurrent requests without bottlenecking upstream applications.

| Architecture | Implementation | Latency Overhead | Scaling Bottleneck |
|---|---|---|---|
| **Cascade Router** | Bare-Metal C++, INT8 ONNX | **~4.6 ms** | Negligible |
| Python Proxy | LiteLLM, FastAPI | ~65.0 ms | Python GIL, GC |
| SaaS Router | External API Network Hop | ~180.0 ms | TLS Handshake |
| LLM-as-a-Judge | Zero-Shot API Evaluation | ~850.0 ms | LLM TTFT |

For deep technical methodology, refer to the [Cascade Whitepaper](docs/whitepaper.md).

## License

The core C++ proxy and v0.1 routing weights are released under the MIT License.
