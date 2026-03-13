# 🧾 Invoice Processor v6.5 – Multi-Agent Invoice Classification System

A sophisticated invoice processing application using **3 specialized AI agents** with weighted voting consensus for maximum accuracy in invoice classification and data extraction.

![Version](https://img.shields.io/badge/version-6.5-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture Overview](#-architecture-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Agent System](#-agent-system)
- [Consensus Engine](#-consensus-engine)
- [Security Features](#-security-features)
- [Performance Optimization](#-performance-optimization)
- [Troubleshooting](#-troubleshooting)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)

---

## ✨ Features

### Core Capabilities

- **🤖 Multi-Agent AI System**: 3 specialized agents working in concert
  - **Classifier Agent**: Binary invoice/non-invoice classification
  - **Extractor Agent**: Structured data extraction (vendor, customer, amounts, dates)
  - **Anomaly Detector**: Detects non-invoice documents (CVs, contracts, certificates, etc.)

- **🎯 Weighted Voting Consensus**: Intelligent decision-making with agent agreement analysis
  - Auto-accept: High confidence invoices (≥70%)
  - Human review: Uncertain cases (50-70%)
  - Auto-reject: Clear non-invoices (<50%)

- **📝 Markdown-First Architecture (v7.0+)**
  - Agents communicate exclusively via structured Markdown tables
  - No JSON injection vulnerabilities
  - Master Instruction framework for validation

- **🔒 Security Features**
  - Input file validation (MIME type + size checks)
  - LLM output sanitization (anti-injection)
  - Optional data encryption (Fernet/AES-128)
  - Audit logging for compliance
  - Configuration integrity verification (SHA-256)

- **🖼️ Hybrid OCR Support**
  - **RapidOCR** (PaddleOCR via ONNX Runtime) for scanned documents
  - Direct PDF text extraction for digital documents (100% accuracy)
  - OCR hallucination detection and correction
  - Spatial layout reconstruction

- **🎨 Modern GUI**
  - Dark theme interface (CustomTkinter)
  - Real-time progress tracking
  - Memory/VRAM optimization controls
  - Interactive invoice filtering and selection

- **⚙️ External Configuration**
  - YAML-based rules (`config/rules.yaml`)
  - Tunable keywords and patterns without code changes
  - Multi-language support (Czech + English)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GUI Layer (GUI)                          │
│                    invoice_gui_v6.py                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Core Engine (core/)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ OCRExtractor │  │ MultiAgent   │  │ FileDiscovery│          │
│  │              │  │ Processor    │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Invoice      │  │ Invoice      │  │ Markdown     │          │
│  │ Filter Agent │  │ Organizer    │  │ Utils        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Agent Workflow (agent_workflow/)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Classifier   │  │ Extractor    │  │ Anomaly      │          │
│  │ Agent        │  │ Agent        │  │ Detector     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Consensus    │  │ OCR          │  │ Base Agent   │          │
│  │ Engine       │  │ Validator    │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ Agent Memory │  │ LangGraph    │                             │
│  │              │  │ Workflow     │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Ollama       │  │ RapidOCR     │  │ PyMuPDF      │          │
│  │ (LLM)        │  │ (OCR)        │  │ (PDF)        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **File Discovery** → Find supported files in source directory
2. **OCR Extraction** → Extract text (digital or OCR) with bounding boxes
3. **Markdown Enrichment** → Format as structured Markdown table + Master Instruction
4. **Agent Analysis** (Sequential with Feedback Loop):
   - **Classifier** → Invoice classification
   - **Anomaly Detector** → Verify classifier, detect non-invoice patterns
   - **Extractor** → Data extraction (skipped if classifier rejects + no refutation)
5. **Consensus Engine** → Weighted voting + decision making
6. **Result Organization** → Move/filter/sort invoices

---

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **Ollama** (for LLM inference)
  - Download from: https://ollama.ai
  - Required model: `llama3.2` (default) or `llama3.1:8b`
  ```bash
  ollama pull llama3.2
  ```

### Step-by-Step Installation

1. **Clone or download the repository**

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install OCR dependencies (optional, for scanned documents)**
   ```bash
   pip install rapidocr_onnxruntime opencv-python numpy
   ```

4. **Install Tesseract OCR (alternative OCR engine)**
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **Linux**: `sudo apt install tesseract-ocr tesseract-ocr-ces`
   - **macOS**: `brew install tesseract tesseract-lang`

5. **Verify installation**
   ```bash
   python invoice_gui_v6.py
   ```

---

## 🚀 Quick Start

### GUI Mode (Recommended)

```bash
python invoice_gui_v6.py
```

1. Select **source directory** (containing invoices)
2. Select **target directory** (for organized invoices)
3. Click **▶️ Spustit analýzu** (Start Analysis)
4. Review results and select invoices to process
5. Click **📂 Přesunout označené** (Move Selected)

### CLI Mode (Advanced)

```python
from pathlib import Path
from core import OCRExtractor, MultiAgentProcessor, process_invoice

# Initialize components
extractor = OCRExtractor()
processor = MultiAgentProcessor()

# Process single invoice
invoice = process_invoice(Path("invoice.pdf"), extractor, processor)

print(f"Is invoice: {invoice.is_invoice}")
print(f"Confidence: {invoice.confidence:.0%}")
print(f"Sender: {invoice.sender_name}")
print(f"Amount: {invoice.total_amount} {invoice.currency}")
```

### Batch Processing

```python
from pathlib import Path
from core import OCRExtractor, MultiAgentProcessor, FileDiscovery

extractor = OCRExtractor()
processor = MultiAgentProcessor()
discovery = FileDiscovery(Path("./invoices"))

# Find all supported files
files = discovery.find_files()

# Process each file
for file_path in files:
    invoice = process_invoice(file_path, extractor, processor)
    
    if invoice.is_invoice:
        print(f"✓ {file_path.name}: {invoice.sender_name} - {invoice.total_amount}")
    else:
        print(f"✗ {file_path.name}: Not an invoice")
```

---

## 📖 Usage

### Supported File Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | `.pdf` | Digital text + OCR fallback |
| JPEG | `.jpg`, `.jpeg` | OCR only |
| PNG | `.png` | OCR only |
| TIFF | `.tiff`, `.tif` | OCR only |
| BMP | `.bmp` | OCR only |
| GIF | `.gif` | OCR only |

### Memory and Performance Settings

The GUI provides real-time control over:

| Setting | Description | Recommended |
|---------|-------------|-------------|
| **OCR RAM (MB)** | Memory limit for OCR processing | 2048 MB |
| **Ollama VRAM (GB)** | VRAM allocation for LLM | 4-8 GB |
| **GC Threshold (%)** | Garbage collection trigger | 60% |
| **Profile** | Pre-set configurations | Balanced |

**Profiles:**
- **Conservative**: Minimal resource usage (1GB OCR, 2GB VRAM)
- **Balanced**: Default settings (2GB OCR, 4GB VRAM)
- **Aggressive**: Higher performance (4GB OCR, 8GB VRAM)
- **Maximum**: Maximum resources (8GB OCR, 12GB VRAM)

### Filtering and Sorting

**Filter by:**
- Sender name
- Recipient name
- Issue date
- Invoice number
- Decision type (auto-accept, human review, auto-reject)

**Sort by:**
- Issue date
- Due date
- Sender name
- Recipient name
- Amount

---

## ⚙️ Configuration

### Rules Configuration (`config/rules.yaml`)

All classification and extraction rules are defined in YAML format:

```yaml
classifier:
  # Immediate rejection keywords
  strong_negative_indicators:
    - 'plná moc'
    - 'životopis'
    - 'certifikát'
  
  # Context-dependent indicators
  context_negative_indicators:
    smlouva:
      - 'fakturujeme za'
      - 'faktura za'
  
  # Fallback keywords
  fallback:
    positive_keywords:
      - 'faktura'
      - 'invoice'
    negative_keywords:
      - 'upomínka'
      - 'smlouva'

consensus_engine:
  invoice_keywords_cs:
    - 'faktura'
    - 'daňový doklad'
    - 'IČO'
    - 'DIČ'
  
  invoice_keywords_en:
    - 'invoice'
    - 'tax document'
    - 'VAT'

extractor_agent:
  patterns:
    date: r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})'
    amount: r'(\d{1,3}(?:[\s\.,]\d{3})*(?:[\s\.,]\d{1,2})?)'
    ico: r'(?:ičo|ič)\s*[:.]?\s*(\d{6,10})'
    dic: r'(?:dič|vat\s*id)\s*[:.]?\s*([a-zA-Z]{2}\d{8,12})'

anomaly_detection:
  cv_resume:
    keywords:
      - 'životopis'
      - 'vzdělání'
      - 'praxe'
    threshold: 1
    veto: true
  
  certificate:
    keywords:
      - 'certifikát'
      - 'osvědčení'
    threshold: 1
    veto: true
```

### Privacy Mode

The application runs Ollama in **complete isolation** on localhost:

```python
# Automatically activated on startup
_set_ollama_privacy_mode()
```

This ensures:
- No external connections
- All processing stays local
- Complete data privacy

---

## 🤖 Agent System

### 1. Classifier Agent

**Purpose**: Binary invoice/non-invoice classification

**Input**: Structured Markdown table + Master Instruction

**Output**:
```markdown
## 📊 Document Classification
**Is Invoice:** ✅ YES
**Confidence:** 0.85

## 🔍 Elements
| Element | Present |
|---------|----------|
| Document Type Identified | ✅ |
| Supplier and Buyer Present | ✅ |
| Total Amount Present | ✅ |
| Date Present | ✅ |
| Payment Instructions Present | ✅ |

## 📋 Extracted Values
| Field | Value |
|------|---------|
| Document Type | Faktura |
| Supplier | ABC s.r.o. |
| Customer | Jan Novák |
| Amount | 1500 |
| Date | 26. 2. 2026 |

## 🧠 Reasoning
[Brief analysis based on Master Instruction]
```

**Key Features**:
- Strong negative indicators (immediate rejection)
- Context-dependent indicators
- Fallback keyword matching
- Anti-hallucination sanity checks

---

### 2. Extractor Agent

**Purpose**: Structured data extraction from invoices

**Input**: Markdown table + Classifier hints + Master Instruction

**Output**:
```markdown
## 📋 Extracted Data
| Field | Value |
|------|---------|
| Invoice Number | 2023001 |
| Vendor Name | Example Corp s.r.o. |
| Vendor Address | Test Street 123, 11000 Prague |
| Customer Name | John Doe |
| Customer Address | Nice Ave 45, 60200 Brno |
| Issue Date | 2023-10-25 |
| Due Date | 2023-11-08 |
| Total Amount | 1500.50 |
| Currency | CZK |
| VAT Amount | 315.10 |
| Base Amount | 1185.40 |
| Bank Account | 123456789/0100 |
| Variable Symbol | 2023001 |

**Completeness Score:** 0.95

## ⚠️ Validation Errors
- [Missing customer IČO]

## 🧠 Reasoning & Location
Vendor found in top block [vlevo]...
```

**Key Features**:
- Regex-based extraction (configurable)
- Spatial layout analysis (bounding boxes)
- Classifier feedback integration
- Validation error reporting

---

### 3. Anomaly Detector Agent

**Purpose**: Detect non-invoice documents

**Detects**:
- CVs and resumes
- Certificates and diplomas
- Contracts and agreements
- Reminders and payment demands
- Offers and quotes
- Inquiries and requests
- Internal documents
- Research reports
- Logistics reports (pallet lists, cargo manifests)

**Output**:
```markdown
## 🚨 Anomaly Detection
**Is Anomaly:** ❌ NO
**Anomaly Type:** null
**Confidence:** 0.95
**Refutes Classifier (Invoice Found):** ✅ YES

## 🔑 Detected Keywords
- none

## ⚠️ Flags
- none

## 🧠 Reasoning
Document contains invoice keywords (faktura, částka, IČO)
despite classifier rejection. Refuting classifier decision.
```

**Key Features**:
- Rule-based detection (YAML configuration)
- AI-powered validation
- Classifier refutation capability
- Keyword validation against actual text

---

## 🎯 Consensus Engine

### Decision Logic

The Consensus Engine combines agent results using **weighted voting**:

| Agent | Weight | Role |
|-------|--------|------|
| Classifier | 0.4 | Primary decision maker |
| Extractor | 0.3 | Data completeness validator |
| Anomaly | 0.3 | Non-invoice pattern detector |

### Decision Thresholds

| Decision Type | Confidence | Action |
|--------------|------------|--------|
| **Auto-Accept** | ≥ 70% | Automatically approve |
| **Human Review** | 50-70% | Flag for manual review |
| **Auto-Reject** | < 50% | Automatically reject |

### Special Rules

1. **Anomaly Veto**: High-confidence anomaly detection (≥85%) can override other agents
2. **Classifier Veto**: High-confidence rejection (≥90%) with incomplete extraction → reject
3. **CV/Resume Detection**: Automatic rejection regardless of other scores
4. **Research Report Detection**: Automatic rejection if detected
5. **Feedback Loop**: Anomaly agent can refute classifier rejection if invoice elements found

### Consensus Output

```python
{
    'is_invoice': True,
    'confidence': 0.85,
    'decision_type': 'auto_accept',
    'weighted_score': 0.82,
    'agent_scores': {
        'classifier': 0.90,
        'extractor': 0.75,
        'anomaly': 0.80
    },
    'agent_agreement': {
        'full_agreement': False,
        'majority_agreement': True,
        'agreement_count': 2,
        'total_agents': 3
    },
    'reasoning': 'Classifier very confident (90%) + 5/5 elements present',
    'extracted_data': {
        'vendor_name': 'ABC s.r.o.',
        'customer_name': 'Jan Novák',
        'total_amount': 1500.50,
        # ...
    }
}
```

---

## 🔒 Security Features

### 1. Input File Validation

```python
from core.security import validate_input_file

is_valid, reason = validate_input_file(Path("invoice.pdf"))
```

**Checks**:
- File existence and type
- File size limits (default: 50 MB)
- Allowed extensions
- MIME type validation (python-magic)
- Magic header verification (fallback)

### 2. LLM Output Sanitization

```python
from core.security import sanitize_llm_output

safe_reasoning = sanitize_llm_output(raw_llm_output)
```

**Protects against**:
- JSON injection
- Code block injection
- System prompt leakage
- HTML/script injection
- Data exfiltration (length limits)

### 3. Data Encryption (Optional)

```python
from core.security import get_encryptor

encryptor = get_encryptor()
encrypted = encryptor.encrypt_text("sensitive data")
decrypted = encryptor.decrypt_text(encrypted)
```

**Features**:
- Fernet encryption (AES-128-CBC + HMAC)
- Auto-generated key (`.security_key`)
- File encryption support

### 4. Audit Logging

```python
from core.security import get_audit_logger

audit = get_audit_logger()
audit.log_access("raw_text_logs/invoice_001.md", "read")
audit.log_classification("invoice_001.pdf", "auto_accept", 0.95)
```

**Logs**:
- Data access events
- Classification decisions
- Security events

### 5. Configuration Integrity

```python
from core.security import verify_config_integrity

is_valid, message = verify_config_integrity(Path("config/rules.yaml"))
```

**Features**:
- SHA-256 hash verification
- Tamper detection
- Auto-initialization on first run

---

## ⚡ Performance Optimization

### Memory Management

```python
# Monitor memory usage
from core import get_memory_usage, log_memory_state

memory = get_memory_usage()
print(f"RSS: {memory['rss_mb']:.1f}MB ({memory['percent']:.1f}%)")

log_memory_state("before_processing")
```

### VRAM Optimization

The application automatically calculates optimal `num_ctx` based on VRAM:

```python
from core import calculate_num_ctx

# Calculate context window size
num_ctx = calculate_num_ctx(vram_gb=4)  # Returns ~4096
```

### Batch Processing

```python
# Process files in batches to manage memory
BATCH_SIZE = 1  # Process one file at a time
MEMORY_CHECK_EVERY = 3  # Check memory every 3 files
GC_EVERY = 5  # Force garbage collection every 5 files
```

### OCR Optimization

```python
# Adjust OCR memory limit
extractor = OCRExtractor(ocr_memory_mb=2048)

# Adjust zoom level for OCR
OCR_ZOOM = 2.0  # Higher = better accuracy, slower
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Ollama not running"

**Solution**:
```bash
# Start Ollama manually
ollama serve

# Or let the app auto-start it
# (Check Windows Task Manager for ollama.exe)
```

#### 2. "Model not found"

**Solution**:
```bash
# Download required model
ollama pull llama3.2

# Or change model in GUI settings
```

#### 3. "RapidOCR not available"

**Solution**:
```bash
pip install rapidocr_onnxruntime opencv-python numpy
```

#### 4. "High memory usage"

**Solutions**:
- Lower OCR RAM in GUI settings (e.g., 1024 MB)
- Lower Ollama VRAM (e.g., 2 GB)
- Use "Conservative" profile
- Enable more frequent garbage collection

#### 5. "Slow processing"

**Solutions**:
- Increase VRAM for larger context window
- Use digital PDFs instead of scans (no OCR needed)
- Reduce OCR zoom level
- Process fewer files in batch

#### 6. "False rejections"

**Solutions**:
- Check `config/rules.yaml` for overly aggressive negative keywords
- Adjust anomaly detection thresholds
- Review classifier strong negative indicators
- Check agent reasoning in logs (`gui_output.log`)

### Log Files

- **GUI logs**: `gui_output.log`
- **Audit logs**: `audit.log`
- **Raw text logs**: `raw_text_logs/`

---

## 📚 API Reference

### Core Classes

#### `OCRExtractor`

```python
from core import OCRExtractor

extractor = OCRExtractor(ocr_memory_mb=2048)

# Extract text from file
text, is_digital, text_blocks = extractor.extract_text(Path("invoice.pdf"))

# Pre-filter before AI processing
classification, confidence, reason = extractor.pre_filter(text)
```

#### `MultiAgentProcessor`

```python
from core import MultiAgentProcessor

processor = MultiAgentProcessor(vram_limit_gb=4, num_ctx=4096)

# Analyze document
invoice = processor.analyze_document(
    markdown_input=markdown_text,
    file_path=Path("invoice.pdf"),
    text_blocks=text_blocks,
    master_instruction=master_instruction
)

# Get statistics
stats = processor.get_statistics()
```

#### `Invoice`

```python
from core import Invoice

# Invoice attributes
invoice.source_path       # Path to source file
invoice.sender_name       # Vendor name
invoice.sender_address    # Vendor address
invoice.recipient_name    # Customer name
invoice.recipient_address # Customer address
invoice.issue_date        # Issue date (YYYY-MM-DD)
invoice.due_date          # Due date (YYYY-MM-DD)
invoice.total_amount      # Total amount
invoice.currency          # Currency code
invoice.invoice_number    # Invoice number
invoice.is_invoice        # Boolean: is invoice?
invoice.confidence        # Confidence score (0-1)
invoice.decision_type     # auto_accept, human_review, auto_reject
invoice.requires_review   # Boolean: needs manual review?
```

### Agent Classes

#### `ClassifierAgent`

```python
from agent_workflow import ClassifierAgent

agent = ClassifierAgent(model="llama3.2", timeout=30, num_ctx=4096)

result = agent.analyze(
    markdown_input=markdown_text,
    text_blocks=text_blocks,
    master_instruction=master_instruction
)
```

#### `ExtractorAgent`

```python
from agent_workflow import ExtractorAgent

agent = ExtractorAgent(model="llama3.2", timeout=60, num_ctx=4096)

result = agent.analyze(
    markdown_input=markdown_text,
    metadata=classifier_metadata,
    master_instruction=master_instruction
)
```

#### `AnomalyDetectorAgent`

```python
from agent_workflow import AnomalyDetectorAgent

agent = AnomalyDetectorAgent(model="llama3.2", timeout=30, num_ctx=4096)

result = agent.analyze(
    markdown_input=markdown_text,
    metadata=classifier_info,
    master_instruction=master_instruction
)
```

#### `ConsensusEngine`

```python
from agent_workflow import ConsensusEngine

engine = ConsensusEngine(
    threshold_accept=0.7,
    threshold_review=0.5,
    anomaly_veto_threshold=0.85
)

consensus = engine.calculate_consensus(
    classifier_result=classifier_result,
    extractor_result=extractor_result,
    anomaly_result=anomaly_result,
    raw_text=markdown_text
)
```

### Configuration

```python
from config_loader import get_config, reload_config

# Get configuration
config = get_config()

# Access specific sections
invoice_keywords = config.get_all_invoice_keywords()
extractor_patterns = config.get_extractor_patterns()
anomaly_rules = config.get_anomaly_rules()

# Reload after config file changes
reload_config()
```

### Security

```python
from core.security import (
    get_encryptor,
    get_audit_logger,
    validate_input_file,
    sanitize_llm_output,
    verify_config_integrity,
    cleanup_old_files
)

# Validate input
is_valid, reason = validate_input_file(Path("invoice.pdf"))

# Sanitize LLM output
safe_text = sanitize_llm_output(raw_output)

# Encrypt data
encryptor = get_encryptor()
encrypted = encryptor.encrypt_text("sensitive")

# Audit logging
audit = get_audit_logger()
audit.log_classification("invoice.pdf", "auto_accept", 0.95)

# Verify config integrity
is_valid, message = verify_config_integrity(Path("config/rules.yaml"))

# Cleanup old files
deleted = cleanup_old_files(Path("raw_text_logs"), max_age_days=30)
```

---

## 📁 Project Structure

```
projekt třídění faktur/
├── invoice_gui_v6.py          # Main GUI application
├── config_loader.py            # YAML configuration loader
├── requirements.txt            # Python dependencies
├── .memory_settings.json       # Memory/VRAM settings
├── gui_output.log              # GUI runtime logs
├── audit.log                   # Security audit logs
│
├── config/
│   └── rules.yaml              # Classification/extraction rules
│
├── core/
│   ├── __init__.py             # Core module exports
│   ├── engine.py               # Main processing engine
│   ├── markdown_utils.py       # Markdown parsing utilities
│   └── security.py             # Security features
│
├── agent_workflow/
│   ├── __init__.py             # Agent module exports
│   ├── base_agent.py           # Base agent class
│   ├── classifier_agent.py     # Invoice classification
│   ├── extractor_agent.py      # Data extraction
│   ├── anomaly_agent.py        # Anomaly detection
│   ├── consensus_engine.py     # Weighted voting
│   ├── ocr_validator.py        # OCR text validation
│   ├── agent_memory.py         # Shared memory system
│   └── langgraph_workflow.py   # LangChain orchestration
│
└── raw_text_logs/              # Debug logs (plaintext markdown)
    └── [timestamp]_[filename].md
```

---

## 📊 Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Processing Speed** | 30-180 sec/file | Depends on document complexity |
| **Digital PDF** | ~30 sec | No OCR needed |
| **Scanned PDF** | ~90-180 sec | OCR processing required |
| **Accuracy** | ~95% | On clean, standard invoices |
| **Memory Usage** | 2-4 GB | Default settings |
| **VRAM Usage** | 2-8 GB | Model-dependent |

---

## 🔄 Version History

### v6.5 (Current)
- ✅ Markdown-first agent communication
- ✅ Master Instruction framework
- ✅ Shared agent memory system
- ✅ Enhanced security sanitization
- ✅ Privacy mode for Ollama
- ✅ Memory/VRAM optimization GUI

### v7.0+ Features
- ✅ Complete JSON elimination (Markdown-only)
- ✅ Spatial layout reconstruction
- ✅ Feedback loop (Anomaly → Extractor)
- ✅ Agent reasoning comparison
- ✅ Consistency validation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Write tests for new features

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Ollama** - Local LLM inference
- **RapidOCR** - OCR engine (PaddleOCR via ONNX)
- **PyMuPDF** - PDF text extraction
- **CustomTkinter** - Modern GUI framework
- **LangChain/LangGraph** - Workflow orchestration

---

## 📞 Support

For issues, questions, or suggestions:
- Check the **Troubleshooting** section
- Review logs in `gui_output.log`
- Examine agent reasoning in `raw_text_logs/`

---

**Built with ❤️ for efficient invoice processing**
