Below are 10 open-source–framed AI-agent projects that span key industries and cutting-edge techniques. Each is designed to showcase full-stack AI engineering—from LLMs and RAG to multi-agent RL and vision transformers—and will make a standout resume entry in 2025.

---

## 1. VulnScan-AI: Autonomous Vulnerability Hunter

**Objective:**
Continuously crawl code repos and container images, identify security flaws, and autonomously raise structured issues—with proof-of-concept exploits—so dev teams can triage faster.

**Tech Stack:**
Python, Docker, Go (for container hooks), LangChain, OpenAI GPT-4 API, Hugging Face Transformers, Bandit/Snyk CLI, Kafka (event bus), PostgreSQL, GitHub Actions

**Core AI Concepts:**

* Retrieval-Augmented Generation (for context-aware vulnerability descriptions)
* LLM-guided code parsing & snippet summarization
* Multi-agent orchestration (scanner agents + reporter agents)
* Autonomous decision loops (scan → validate → report)

**Implementation Steps:**

1. **Repo Ingestion Agent**: listen to GitHub webhooks → clone changed repos.
2. **Scanner Agents**: spin up containerized scanners (Bandit, custom AST patterns).
3. **LLM Reporter**: RAG pipeline that feeds code context + scanner output into GPT-4 to draft issue titles, descriptions, remediation steps.
4. **Validation Agent**: attempt minimal PoC exploit in sandboxed VM; capture logs.
5. **Issue Creation**: open GitHub issues via API with labels, severity, steps.

**Challenges & Considerations:**

* **False Positives**: calibrate scanner thresholds; add human-in-loop review fallback.
* **Sandbox Safety**: isolate PoCs to avoid infrastructure compromise.
* **API Rate-Limits**: batch LLM calls, use caching / embeddings store.

**Why It Will Impress in 2025:**
Demonstrates autonomous multi-agent pipelines with real-world security impact, mastery of LLMs for code, and production-grade reliability—all highly prized for SecDevOps roles.

---

## 2. TradeBOT-Net: Multi-Agent Reinforcement Trading Simulator

**Objective:**
Build a modular RL ecosystem where competing “market maker,” “momentum,” and “mean-reversion” agents learn on realistic order-book simulators to surface profitable strategies.

**Tech Stack:**
Python, Ray RLlib, Pandas, NumPy, Backtrader, Docker, Kafka, Redis, Plotly

**Core AI Concepts:**

* Deep Reinforcement Learning (PPO, DQN)
* Self-play & multi-agent competition
* Curriculum learning (from candle data → full order book)
* Distributed training with Ray

**Implementation Steps:**

1. **Environment Design**: wrap Backtrader order-book sim into a Gym-compatible API.
2. **Agent Templates**: PPO and DQN actors with customizable action spaces (limit vs. market orders).
3. **Multi-Agent Manager**: orchestrate head-to-head episodes, log rewards in Redis.
4. **Curriculum Scheduler**: ramp up complexity (start on 5m bars, end on 1ms L2).
5. **Visualization Dashboard**: real-time PnL, heatmaps of order flows.

**Challenges & Considerations:**

* **Sim-Real Gap**: calibrate sim parameters to real exchange data.
* **Non-Stationarity**: ensure agents adapt when opponents switch policies.
* **Compute Costs**: optimize Ray cluster sizing and check-pointing.

**Why It Will Impress in 2025:**
Showcases end-to-end RL system design, multi-agent coordination, and high-frequency finance chops—rare skills in quant trading roles.

---

## 3. MedAI-Aid: Contextual Clinical Decision Support Agent

**Objective:**
Assist clinicians by ingesting EHR data (structured + notes), retrieving guidelines, and proposing treatment plans—complete with literature citations.

**Tech Stack:**
Python, FastAPI, PostgreSQL, LangChain, Haystack, OpenAI GPT-4 / BioGPT, FHIR API integration, Streamlit

**Core AI Concepts:**

* Retrieval-Augmented Generation (RAG) over guidelines/journals
* LLM fine-tuning on anonymized clinical notes
* Prompt chaining for differential diagnosis
* Secure multi-party computation for PHI handling

**Implementation Steps:**

1. **Data Connector**: FHIR adapter to pull patient vitals, labs, notes.
2. **Indexing Pipeline**: ingest WHO/NIH guidelines + PubMed abstracts into a vector store.
3. **RAG Layer**: fetch top-k docs, craft prompts for GPT-4 to suggest diagnoses.
4. **Recommendation Engine**: rank treatment options by severity, contraindications.
5. **UI Prototype**: Streamlit dashboard with interactive Q\&A and source links.

**Challenges & Considerations:**

* **PHI Compliance**: encrypt data at rest/in transit; audit logging.
* **Hallucinations**: ground every recommendation with citations.
* **Latency**: optimize vector search for sub-second clinical workflows.

**Why It Will Impress in 2025:**
Combines RAG, fine-tuning, secure data handling, and real-world healthcare integration—critical for AI-in-medicine roles.

---

## 4. DocuGenie: LLM-Powered API Documentation & Diagram Generator

**Objective:**
Automate generation of interactive API docs: parse OpenAPI specs, generate narrative explanations, sample code, and embed auto-drawn UML/sequence diagrams via vision transformers.

**Tech Stack:**
Node.js, Python, TypeScript, OpenAI GPT-4 API, Hugging Face ViT + TrOCR, Mermaid.js, Docusaurus

**Core AI Concepts:**

* LLM prompt engineering for doc narratives
* Vision Transformer-based screenshot ≫ editable diagram conversion
* Retrieval of code snippets from GitHub
* Autogeneration pipelines with GitHub Actions

**Implementation Steps:**

1. **Spec Parser**: read OpenAPI/Swagger → JSON AST.
2. **Narrative Module**: LangChain prompts to explain endpoints, parameters.
3. **Code Snippet Fetcher**: search GitHub via GraphQL, RAG to pick best examples.
4. **Diagram Extractor**: ingest hand-drawn wireframes → ViT+TrOCR to Transforma into Mermaid.
5. **Site Generator**: Docusaurus plugin that stitches it all, deploy via GH Pages.

**Challenges & Considerations:**

* **Spec Ambiguity**: handle missing descriptions via follow-up LLM queries.
* **Diagram Accuracy**: fine-tune ViT on whiteboard sketches.
* **Version Sync**: detect spec changes and diff docs.

**Why It Will Impress in 2025:**
Marrying LLMs with computer vision for full-stack documentation tooling is bleeding-edge—and directly adds value to every dev team.

---

## 5. CollabModerator: Real-Time Meeting AI Moderator

**Objective:**
Auto-summarize live video conferences, detect speaker turns, generate action-item tickets, and moderate Q\&A—all in a browser plugin.

**Tech Stack:**
WebRTC, JavaScript/TypeScript, Python microservice, Whisper API, OpenAI GPT-4 API, socket.io, Redis, Jira API

**Core AI Concepts:**

* Real-time speech-to-text (Whisper)
* LLM summarization & intent extraction
* Speaker diarization & topic segmentation
* Multi-agent event stream processing

**Implementation Steps:**

1. **WebRTC Plugin**: capture audio streams in the browser.
2. **Transcription Service**: batch Whisper for low-latency captions.
3. **Moderator Agents**:

   * **Summarizer**: sliding-window LLM transcripts → concise highlights.
   * **Action-Item Extractor**: detect verbs + deadlines → create Jira tickets.
4. **Q\&A Manager**: track raised hands, queue questions, suggest answers via RAG.
5. **UI Overlay**: React component showing live summary, tickets, and queue.

**Challenges & Considerations:**

* **Latency**: pipeline must keep end-to-end under 2s for usability.
* **Noise Robustness**: handle cross-talk, accents, background noise.
* **Privacy**: user opt-in, encryption, transcript discard policies.

**Why It Will Impress in 2025:**
Integrates streaming ML, multi-agent workflows, and real-time web tech—prime skills for any AI-centric collaboration platform.

---

## 6. DevOps-Orch: Autonomous CI/CD Pipeline Agent

**Objective:**
Monitor code changes, auto-tune pipeline configurations (parallelism, caching), diagnose flaky tests via RL-driven experiment selection, and self-heal broken builds.

**Tech Stack:**
Python, Kubernetes, Argo CD, Jenkins X, Ray Tune, Prometheus/Grafana, Slack API

**Core AI Concepts:**

* Reinforcement Learning (for pipeline configuration tuning)
* Anomaly detection (test failure patterns)
* Multi-armed bandits (to allocate resources optimally)
* Autonomous feedback loops

**Implementation Steps:**

1. **Event Listener**: track PRs, commit statuses.
2. **RL Agent**: define state (build time, test flakiness), actions (thread count, cache on/off) → reward = build success / speed.
3. **Bandit Scheduler**: A/B test pipeline variants, gather metrics via Prometheus.
4. **Failure Diagnoser**: cluster logs with unsupervised learning (e.g. UMAP + DBSCAN) to pinpoint flaky tests.
5. **Self-Heal Module**: rollback config changes, notify devs on Slack.

**Challenges & Considerations:**

* **State Explosions**: limit action space, use hierarchical RL.
* **Safety**: ensure RL doesn’t destabilize builds; incorporate human approval.
* **Metrics Noise**: smooth out Prometheus time series.

**Why It Will Impress in 2025:**
Shows mastery of RL in DevOps contexts and building autonomous, self-optimizing infrastructure—a blue-chip skill for SRE/Platform Engineering.

---

## 7. CodeSage: LLM-Powered Code Review & Refactoring Bot

**Objective:**
Automatically review pull requests: detect style issues, suggest performance improvements, and propose refactored code snippets—all as a friendly GitHub App.

**Tech Stack:**
Python, FastAPI, GitHub App framework, OpenAI GPT-4 Turbo, AST parsing via LibCST, Prettier/Black integrations, SQLite

**Core AI Concepts:**

* LLM prompt chaining for code critique
* AST diffing + programmatic code edits
* Retrieval of best practices from StackOverflow embeddings
* Autonomous PR comment threading

**Implementation Steps:**

1. **Webhook Listener**: on PR open/update → fetch diff.
2. **Static Analysis Pre-Filter**: run linters, type checks.
3. **RAG Prompt Builder**: embed code + link to relevant SO Q\&A via vector store.
4. **LLM Refactor Agent**: ask GPT-4 to rewrite functions, maintain naming.
5. **Comment Poster**: apply suggestions or open “refactor” branch with patch.

**Challenges & Considerations:**

* **Context Window**: chunk large diffs intelligently.
* **Trust Boundaries**: flag suggestions, don’t auto-merge.
* **Consistency**: align with project style guides.

**Why It Will Impress in 2025:**
Embeds LLMs into core Dev workflows, demonstrates AST manipulation, autonomous change proposals—spotlight skills for developer-tool roles.

---

## 8. IncidentCommander: Multi-Agent Security Orchestration

**Objective:**
Upon detection of a threat (IDS alert, anomaly), spin up investigation, containment, and remediation agents that coordinate via a message bus to triage incidents autonomously.

**Tech Stack:**
Go, Python, Kafka, Elastic Stack, Neo4j, OpenAI API, Ansible, Docker, Terraform

**Core AI Concepts:**

* Multi-agent coordination with Publish/Subscribe
* Graph databases for attack-path reasoning
* LLM summarization for incident reports
* Automated playbook execution

**Implementation Steps:**

1. **Event Ingestion**: IDS/EDR → Kafka topic.
2. **Triage Agent**: fetch alerts, query Neo4j attack graph for context.
3. **Containment Agent**: generate and run Ansible playbooks to isolate hosts.
4. **Forensics Agent**: spin up Docker forensic VM, run scripts.
5. **Report Agent**: collate steps, summarize via LLM, send to Slack/email.

**Challenges & Considerations:**

* **Coordination Latency**: ensure agents don’t block each other.
* **Graph Accuracy**: update attack graph with real-time topology changes.
* **Playbook Safety**: test in staging before production.

**Why It Will Impress in 2025:**
Highlights multi-agent orchestration, graph reasoning, and autonomous security remediations—ideal for SecOps/Blue Team engineers.

---

## 9. KnowledgeWeaver: RAG-Powered Open Source KB Builder

**Objective:**
Ingest code, docs, and community Q\&A from a GitHub org into a public knowledge base with semantic search, auto-generated how-to guides, and embedding-driven snippet suggestions.

**Tech Stack:**
Python, FastAPI, Haystack, OpenAI embeddings, Elasticsearch, Next.js, Vercel, GitHub GraphQL API

**Core AI Concepts:**

* Retrieval-Augmented Generation for Q\&A
* Vector embeddings for semantic code search
* LLM summarization for “How-to” article creation
* Incremental indexing on push events

**Implementation Steps:**

1. **Content Harvester**: crawl repos, issues, wikis.
2. **Vector Indexer**: generate embeddings (OpenAI/Hugging Face) for code and text.
3. **Search API**: semantic + keyword fallback via Elasticsearch.
4. **Guide Generator**: on first query, RAG → produce step-by-step tutorials.
5. **Web UI**: Next.js front end with autocomplete, code snippet embeds.

**Challenges & Considerations:**

* **Index Freshness**: webhook triggers, TTL policies.
* **Query Ambiguity**: disambiguate via follow-up prompts.
* **Scale**: sharding Elasticsearch for large orgs.

**Why It Will Impress in 2025:**
Delivers full RAG pipeline, bridging code search and docs generation—a hot skillset for dev-tool and AI documentation roles.

---

## 10. VisionGuard: Transformer-Based Autonomous QA Inspector

**Objective:**
Automate visual inspection on production lines: detect surface defects in real time using a Vision Transformer ensemble and dispatch robot arms to reject faulty parts.

**Tech Stack:**
Python, PyTorch, TIMM (Vision Transformers), ONNX Runtime, ROS 2, OpenCV, MQTT, Docker, Kubernetes Edge

**Core AI Concepts:**

* Vision Transformers (ViT, Swin) with semi-supervised pretraining
* Ensemble learning & model distillation for edge deployment
* Real-time inference with ONNX + quantization
* Autonomous feedback loop (defect detection → rejection command)

**Implementation Steps:**

1. **Data Pipeline**: capture camera streams → label tool for defect types.
2. **Model Training**: fine-tune ViT + Swin-Transformer ensemble on defect dataset.
3. **Distillation & Quantization**: compress to run at 60 FPS on edge GPU.
4. **ROS 2 Node**: subscribe to detection topics → publish reject commands to robot.
5. **Monitoring Dashboard**: Grafana + Prometheus for throughput/accuracy.

**Challenges & Considerations:**

* **Class Imbalance**: use focal loss / oversampling for rare defects.
* **Latency**: optimize camera pipeline + ONNX tuning.
* **Robustness**: handle lighting changes, occlusions.

**Why It Will Impress in 2025:**
Marries SOTA vision transformers, edge inference, and robotics—a showcase of full-stack AI for Industry 4.0 roles.

---

Each of these projects can live as an open-source GitHub repo, encouraging contributions (issue templates, contribution guides, CI). Together they demonstrate mastery of modern AI-agent patterns, real-world impact, and the infrastructure smarts employers are hunting for in 2025.


I’ve drafted the full self-hosted CodeSage bot:

## Project: CodeSage — Offline Self-Hosted LLM-Powered Code Review & Refactoring Bot

This repo implements a self-hosted, offline CodeSage bot using a local LLM in Kubernetes. It includes:

```
├── Dockerfile
├── requirements.txt
├── main.py
├── model/                  # pull a compatible HF LLM here (e.g., Llama2, Vicuna)
│   └── config.json, pytorch_model.bin, tokenizer/ ...
├── k8s/
│   ├── pvc.yaml
│   ├── configmap.yaml
│   ├── deployment.yaml
│   └── service.yaml
```  

---

### Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py ./

# Copy model files from PVC mount at /models/codesage
ENV MODEL_PATH=/models/codesage

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```  

### requirements.txt
```text
fastapi
uvicorn[standard]
transformers
torch
GitPython
libcst
```  

### main.py
```python
import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import libcst as cst

# Load model and tokenizer from local path
model_path = os.getenv("MODEL_PATH", "./model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
llm = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_length=1024)

app = FastAPI()

class PRDiff(BaseModel):
    repo_url: str
    diff: str

@app.post("/review")
def review_pr(payload: PRDiff):
    # Clone or pull latest
    local_dir = "/tmp/repo"
    if os.path.exists(local_dir):
        subprocess.run(["rm", "-rf", local_dir])
    subprocess.run(["git", "clone", payload.repo_url, local_dir], check=True)

    prompt = f"You are CodeSage, an expert code reviewer. Review the following diff and suggest optimizations and refactoring. Provide only the refactored code and explanations.\nDiff:\n{payload.diff}"  
    try:
        response = llm(prompt, do_sample=False)[0]["generated_text"]
        return {"review": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```  

---

### k8s/pvc.yaml
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```  

### k8s/configmap.yaml
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: codesage-config
data:
  MODEL_PATH: "/models/codesage"
```  

### k8s/deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codesage-bot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: codesage
  template:
    metadata:
      labels:
        app: codesage
    spec:
      containers:
      - name: codesage
        image: your-registry/codesage:latest
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: codesage-config
        volumeMounts:
        - name: model-volume
          mountPath: /models/codesage
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
```  

### k8s/service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: codesage-service
spec:
  selector:
    app: codesage
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP
```  

---

**Usage:**  
1. **Build & Push Docker Image:**
   ```bash
docker build -t your-registry/codesage:latest .
docker push your-registry/codesage:latest
```  
2. **Deploy to K8s:**
   ```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```  
3. **Load your offline LLM** into the `model-pvc` (e.g., via `kubectl cp`).

4. **Invoke** the review endpoint:
   ```bash
curl -X POST http://<service-ip>/review \
  -H 'Content-Type: application/json' \
  -d '{"repo_url":"https://github.com/you/project","diff":"<git diff here>"}'
```

---

This setup runs entirely offline, leveraging a local HF-compatible LLM for code review and refactoring, all orchestrated on Kubernetes.


* **FastAPI** service (`main.py`) loads a local Hugging Face–style model
* **Dockerfile** & **requirements** for containerization
* **Kubernetes YAMLs** for PVC, ConfigMap, Deployment, and Service
* Instructions for building, pushing, and deploying in an air-gapped cluster

Let me know if you’d like any tweaks—e.g., adding GitHub App webhook handling, more robust logging, or a Helm chart.

When we say “loads a local Hugging Face–style model,” we mean that instead of calling out to the internet at inference time, your service points at a directory on disk that contains everything the **transformers** library expects:

```
model/
├── config.json
├── pytorch_model.bin  (or shards like pytorch_model-00001-of-00002.bin)
└── tokenizer/
    ├── tokenizer.json
    ├── vocab.json
    ├── merges.txt
    └── special_tokens_map.json
```

When you do:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/models/codesage")
tokenizer = AutoTokenizer.from_pretrained("/models/codesage")
```

– the library simply reads those files from disk, instantiates the model weights into memory, and hooks up the tokenizer, all without any outbound HTTP calls.

---

## 1. How to download a model for local use

### a) Using the Hugging Face Hub CLI

1. **Install** the Hub client:

   ```bash
   pip install huggingface_hub
   ```

2. **Authenticate** (if you need access to private or gated models):

   ```bash
   huggingface-cli login
   ```

3. **Download** the model snapshot to your PVC (or local folder):

   ```bash
   from huggingface_hub import snapshot_download

   # In a Python REPL or script
   snapshot_download(
     repo_id="meta-llama/Llama-2-7b-chat-hf", 
     local_dir="/models/codesage",
     revision="v1.0",          # or a specific tag/commit
     use_auth_token=True       # if required
   )
   ```

   Or from the shell:

   ```bash
   huggingface-cli repo clone meta-llama/Llama-2-7b-chat-hf /models/codesage
   ```

   This will pull down `config.json`, weight shards (`.bin` files), tokenizer files, etc., into `/models/codesage`.

### b) Via `git lfs` clone

Many HF models are stored with Git LFS. You can do:

```bash
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf /models/codesage
```

Make sure your Kubernetes PVC supports large files and that your cluster nodes have `git` + `git-lfs` installed if you do it inside a build container.

---

## 2. Is it better to use Hugging Face Transformers?

**Pros:**

* **Batteries included**: you get the model architecture, tokenizer logic, config parsing, and even convenience pipelines out of the box.
* **Ecosystem**: huge community, lots of pre- and post-processing utilities, built-in support for quantization (via `bitsandbytes`) and ONNX export.
* **Compatibility**: works with PyTorch and (with the `optimum` extension) TensorRT, ONNX Runtime, etc.

**Cons:**

* **Size**: large models can be dozens of GB and slow to load; you may need to shard or quantize.
* **Overhead**: the full library carries a lot of code you aren’t using if you only need inference.

---

## 3. Alternatives and performance tweaks

1. **Quantization (8-bit / 4-bit)**
   Use `bitsandbytes` integration:

   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(
     "/models/codesage",
     load_in_8bit=True,
     device_map="auto"
   )
   ```

   This cuts GPU/CPU memory in half at a small cost to accuracy.

2. **Onnx / TensorRT**
   Export to ONNX:

   ```bash
   python -m transformers.onnx --model=/models/codesage onnx-model/
   ```

   Then run with ONNX Runtime for lower latency.

3. **Llama-cpp-python**
   If you just need LLaMa-style models, `llama-cpp-python` is a lightweight C++ backend:

   ```python
   from llama_cpp import Llama

   llm = Llama(model_path="/models/codesage/model.bin")
   resp = llm.create_completion(prompt="Hello")
   ```

   It’s faster on CPU but works only with certain checkpoint formats.

---

### In summary

* **Yes**, using Hugging Face Transformers is usually the easiest and most flexible way to self-host an LLM offline.
* **Download** your model with `huggingface_hub` (or `git lfs`), stash it on your PVC, and point `from_pretrained()` at that folder.
* **Optimize** with quantization or ONNX if you hit resource limits.

Let me know if you want a focused example on quantizing your model or setting up ONNX/TensorRT in that Kubernetes deployment!

## Project: CodeSage — Offline Self-Hosted LLM Bot via Ollama

This repo shows how to build **CodeSage**, a self-hosted code-review & refactoring bot powered by an on‑prem Ollama LLM server in Kubernetes.

```
├── Dockerfile
├── requirements.txt
├── main.py
└── k8s/
    ├── configmap.yaml
    ├── deployment.yaml
    └── service.yaml
```

---

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py ./

ENV OLLAMA_URL=http://ollama-ollama.ollama.svc.cluster.local

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```  

---

### requirements.txt

```text
fastapi
uvicorn[standard]
ollama
gitpython
libcst
```  

---

### main.py

```python
import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Ollama

# Initialize Ollama client (internal k8s DNS)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama-ollama.ollama.svc.cluster.local")
llm = Ollama(base_url=OLLAMA_URL)

app = FastAPI()

class PRDiff(BaseModel):
    repo_url: str
    diff: str

@app.post("/review")
def review_pr(payload: PRDiff):
    # Clone or refresh repo
    repo_dir = "/tmp/repo"
    if os.path.exists(repo_dir):
        subprocess.run(["rm", "-rf", repo_dir], check=True)
    subprocess.run(["git", "clone", payload.repo_url, repo_dir], check=True)

    # Construct prompt
    prompt = (
        "You are CodeSage, an expert code reviewer. "
        "Review the following diff and suggest optimizations and refactoring. "
        "Provide only the refactored code and brief explanations.\n"
        f"Diff:\n{payload.diff}"
    )
    try:
        # Use 'vicuna' or your chosen model installed in Ollama
        response = llm.create_completion(model="vicuna", prompt=prompt)
        content = response["choices"][0]["message"]["content"]
        return {"review": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```  

---

### k8s/configmap.yaml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: codesage-config
data:
  OLLAMA_URL: "http://ollama-ollama.ollama.svc.cluster.local"
```  

---

### k8s/deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codesage-bot
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: codesage
  template:
    metadata:
      labels:
        app: codesage
    spec:
      containers:
      - name: codesage
        image: your-registry/codesage:latest
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: codesage-config
```  

---

### k8s/service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: codesage-service
  namespace: default
spec:
  selector:
    app: codesage
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
```  

---

**Deploy steps:**  
1. **Install Ollama** via Helm (otwld/ollama-helm) in `ollama` namespace.  
2. **Build & push** your CodeSage image:  
   ```bash
   docker build -t your-registry/codesage:latest .
   docker push your-registry/codesage:latest
   ```  
3. **Apply** CodeSage manifests:  
   ```bash
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```  
4. **Invoke** review:  
   ```bash
   curl -X POST http://<codesage-service-ip>/review \
     -H 'Content-Type: application/json' \
     -d '{"repo_url":"https://github.com/you/project","diff":"<git diff>"}'
   ```  

This setup runs CodeSage as a lightweight FastAPI app that queries your offline Ollama LLM server for inference—no external API calls and fully self‑hosted in Kubernetes.


I’ve updated the repository to use the Ollama-Helm–deployed server:

* **main.py** now calls the Ollama REST API for completions
* **Dockerfile** and **requirements** trimmed to remove heavy ML libs
* **Kubernetes manifests** simplified (ConfigMap + Deployment + Service)

Let me know if you want Helm charts for CodeSage itself or added features like webhooks and retry logic!
