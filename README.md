Cascade is a hyper-optimized, intelligent AI routing proxy built in C++ (io_uring). It dynamically intercepts AI API requests, evaluates the semantic complexity of the prompt in under 5 milliseconds, and routes it to the most cost-effective LLM capable of answering it.Enterprises use Cascade to reduce their OpenAI and Anthropic inference bills by 60% to 70% without sacrificing output quality or introducing system-halting latency.🚀 Why Cascade?Static model routing is bleeding enterprise budgets. Hardcoding every API call to a frontier model (like gpt-4o or claude-3.5-sonnet) wastes capital on simple extraction and classification tasks.Python-based routers attempt to solve this, but introduce 50ms - 200ms of latency per request—a fatal bottleneck for concurrent Agentic Workflows.Cascade solves both:Predictive Routing: Uses a highly distilled, local ONNX embedding classifier to predict prompt complexity, not brittle heuristics.Bare-Metal Speed: Extracts embeddings, evaluates confidence thresholds, and routes the network request with < 5ms overhead.Zero-Friction: Drop it in as a 1-line base URL replacement in your existing OpenAI SDKs.🧠 How it WorksCascade operates ahead of the inference lifecycle.Intercept & Truncate: Captures the incoming API request and truncates the payload to the first 16 tokens to isolate semantic intent.Feature Extraction: Extracts a 384-dimensional dense embedding (via all-MiniLM-L6-v2 ONNX) and scalar metadata (syntactical density).Probability Calibration: A lightweight Logistic Regression model outputs $P(\text{success} \mid \text{model}_i)$.Optimistic Route: The prompt is dispatched to the lowest-cost model clearing the enterprise quality threshold ($\theta$).Progressive Escalation: If the small model fails structural validation, the request is instantly escalated to a frontier model within the same atomic network transaction.graph TD
    A[User Prompt] --> B(Cascade C++ Proxy)
    B --> C{Tiny Classifier}
    C -->|High Confidence| D[Small Model 8B]
    C -->|Medium Confidence| E[Medium Model 70B]
    C -->|Low Confidence| F[Frontier Model]
    D --> G{Validator}
    G -->|Pass| H[Response]
    G -->|Fail| F
⚡ BenchmarksCascade is designed to eliminate kernel context-switching and heap allocation bottlenecks.Event Loop: Linux io_uring Proactor pattern.Memory: Zero-copy std::pmr monotonic arenas.Inference: INT8 Quantized ONNX Runtime pinned to ORT_SEQUENTIAL.MetricTargetActual (Cascade v0.1)Routing Latency (16-token)< 5.0 ms4.02 msMemory Footprint (per 1k conns)< 10 MB8.4 MBModeled Cost Reduction> 60%68.2%💻 Quick StartCascade acts as a transparent proxy. You do not need to rewrite your application logic.1. Start the Local Server (Docker)docker run -p 8000:8000 -v ./models:/app/models cascaderouter/proxy:latest
2. Point your application to CascadeJust change the base_url. Cascade handles the rest.Python (OpenAI SDK)from openai import OpenAI

# Previously: client = OpenAI(api_key="sk-...")
client = OpenAI(
    base_url="http://localhost:8000/v1", # Point to Cascade
    api_key="sk-..."                     # Cascade passes this upstream securely
)

response = client.chat.completions.create(
    model="cascade-auto", # Tells Cascade to route dynamically
    messages=[{"role": "user", "content": "Extract the emails from this text..."}]
)
🏢 Enterprise LicenseThe core proxy and open-source models are free under the MIT License.Organizations deploying Cascade at scale can purchase an Enterprise License, which unlocks:Custom Model Weights: Fine-tune the Tiny Routing Classifier on your proprietary internal prompts.SOC2 Audit Logging: Complete, un-redacted telemetry of routing decisions.Single Sign-On (SSO): Control which engineering teams have access to the frontier fallback models.Contact founders@cascade-router.com for Enterprise inquiries.