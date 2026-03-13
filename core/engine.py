#!/usr/bin/env python3
"""
Core Invoice Processing Engine

Business logic for invoice processing separated from GUI.
Supports OCR extraction, multi-agent analysis, and invoice organization.

Usage (CLI example):
    from core import OCRExtractor, MultiAgentProcessor, process_invoice
    
    extractor = OCRExtractor()
    processor = MultiAgentProcessor()
    invoice = process_invoice(Path("invoice.pdf"), extractor, processor)
"""

import os
import re
import json
import shutil
import logging
import gc
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Security module
try:
    from core.security import (
        get_encryptor, get_audit_logger, validate_input_file,
        sanitize_llm_output, cleanup_old_files
    )
    HAS_SECURITY = True
except ImportError:
    try:
        from .security import (
            get_encryptor, get_audit_logger, validate_input_file,
            sanitize_llm_output, cleanup_old_files
        )
        HAS_SECURITY = True
    except ImportError:
        HAS_SECURITY = False
        logging.warning("⚠️ Security module not available")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import agentů
try:
    from agent_workflow import (
        ClassifierAgent,
        ExtractorAgent,
        AnomalyDetectorAgent,
        ConsensusEngine,
        OCRTextValidator,
        validate_ocr_text,
    )
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    logging.warning(f"⚠️ Agent workflow not available: {e}")

# ─────────────────────────────────────────────
# Konfigurační konstanty
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"}
TEXT_MODEL = "llama3.2"
TEXT_MODEL_BETTER = "llama3.1:8b"
LOG_LEVEL = logging.INFO

# Detekce skenů
MIN_ZNAKU_PRO_DIGITALNI = 30  # Minimální počet znaků pro digitální text (ne sken)

# AI parametry pro deterministický výstup
AI_TEMPERATURE = 0.0
AI_TOP_P = 0.1

# Uložení surového textu pro debugging
SAVE_RAW_TEXT = True
RAW_TEXT_DIR = Path("raw_text_logs")

# Timeout configuration
REQUEST_TIMEOUT = 120
EXTRACTOR_TIMEOUT = 150

# Memory management
MAX_MEMORY_PERCENT = 60
DEBUG_MEMORY = True
BATCH_SIZE = 1
MEMORY_CHECK_EVERY = 3
GC_EVERY = 5

# OCR konstanty - RapidOCR (PaddleOCR přes ONNX Runtime)
USE_RAPIDOCR = True
OCR_ZOOM = 2.0  # Zoom pro OCR skenů

# Agent thresholds
THRESHOLD_ACCEPT = 0.7
THRESHOLD_REVIEW = 0.5
ANOMALY_VETO_THRESHOLD = 0.85
MIN_REALISTIC_AMOUNT = 1
CHECK_AMOUNT_REALISTIC = False

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Pomocné funkce pro monitoring paměti
# ─────────────────────────────────────────────
def get_memory_usage() -> dict:
    """Get current process memory usage."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }
    return {"rss_mb": 0, "vms_mb": 0, "percent": 0}


def log_memory_state(context: str = ""):
    """Log current memory state."""
    if not DEBUG_MEMORY:
        return
    mem = get_memory_usage()
    if mem["rss_mb"] > 0:
        logger.debug(f"💾 PAMĚŤ [{context}]: RSS={mem['rss_mb']:.1f}MB ({mem['percent']:.1f}%)")


# ─────────────────────────────────────────────
# Datová třída pro fakturu
# ─────────────────────────────────────────────
@dataclass
class Invoice:
    """Represents an invoice with extracted data and agent results."""
    
    source_path: Path
    sender_name: str = "Neznamy_odesilatel"
    sender_address: str = ""
    recipient_name: str = "Neznamy_prijemce"
    recipient_address: str = ""
    issue_date: str = "0000-00-00"
    due_date: str = "0000-00-00"
    total_amount: str = ""
    currency: str = ""
    invoice_number: str = ""
    is_invoice: bool = False
    confidence: float = 0.0
    raw_json: dict = field(default_factory=dict)
    
    # Agent results
    agent_results: dict = field(default_factory=dict)
    consensus_result: dict = field(default_factory=dict)
    decision_type: str = "pending"  # pending, auto_accept, human_review, auto_reject
    requires_review: bool = False

    @property
    def original_stem(self) -> str:
        stem = self.source_path.stem
        return _sanitize_filename(stem)

    @property
    def suffix(self) -> str:
        return self.source_path.suffix.lower()

    def matches_filter(self, filter_key: str, filter_value: str) -> bool:
        """Check if invoice matches filter criteria."""
        if not filter_key or not filter_value:
            return True
        value = getattr(self, filter_key, "")
        if not value:
            return False
        return filter_value.lower() in str(value).lower()


# ─────────────────────────────────────────────
# Pomocné funkce
# ─────────────────────────────────────────────
def _sanitize_filename(name: str) -> str:
    """Sanitize filename by removing special characters and diacritics."""
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = name.replace("á", "a").replace("é", "e").replace("í", "i")
    name = name.replace("ó", "o").replace("ú", "u").replace("ý", "y")
    name = name.replace("č", "c").replace("ď", "d").replace("ě", "e")
    name = name.replace("ň", "n").replace("ř", "r").replace("š", "s")
    name = name.replace("ť", "t").replace("ů", "u").replace("ž", "z")
    return name[:60]


def _clean_invoice_value(value: str) -> str:
    """Clean invoice field value (vendor/customer name) while preserving readability."""
    if not value:
        return value
    # Remove only dangerous characters for filenames, keep spaces and diacritics
    value = re.sub(r'[\\/:*?"<>|]', '', value)
    # Strip leading/trailing whitespace
    value = value.strip()
    # Limit length but keep full words
    if len(value) > 100:
        value = value[:97] + "..."
    return value


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON from text (supports code blocks and raw JSON)."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def save_raw_text(file_path: Path, markdown_input: str, is_digital: bool, invoice_result=None, text_blocks=None, master_instruction: str = None):
    """
    Uloží strukturovaná Markdown data, se kterými aplikace skutečně pracuje.

    ARCHITEKTURA: Agenti NEDOSTÁVAJÍ surový text! Proto ukládáme pouze:
    1. Markdown tabulku (strukturovaná data z text_blocks)
    2. Master Instruction (referenční rámec pro validaci)
    3. Spatial layout (rekonstrukce rozložení)

    Args:
        file_path: Path to source file
        markdown_input: Strukturovaná Markdown data (tabulka + layout)
        is_digital: True if text was extracted digitally from PDF (not OCR)
        invoice_result: Optional Invoice object with extracted data
        text_blocks: Optional list of text blocks with bbox positions
        master_instruction: Pevná instrukce pro validaci agentů
    """
    try:
        RAW_TEXT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_filename = f"{timestamp}_{file_path.stem}.md"
        raw_path = RAW_TEXT_DIR / raw_filename

        # Prepare invoice data for markdown table
        invoice_data = {}
        if invoice_result:
            invoice_data = {
                'Je faktura': '✅ ANO' if invoice_result.is_invoice else '❌ NE',
                'Confidence': f"{invoice_result.confidence:.0%}",
                'Rozhodnutí': invoice_result.decision_type,
                'Dodavatel': invoice_result.sender_name or '—',
                'Odběratel': invoice_result.recipient_name or '—',
                'Datum vystavení': invoice_result.issue_date or '—',
                'Datum splatnosti': invoice_result.due_date or '—',
                'Částka': f"{invoice_result.total_amount} {invoice_result.currency}".strip() or '—',
                'Číslo faktury': invoice_result.invoice_number or '—',
            }

        # Build the full markdown content in memory
        lines = []
        lines.append("# 📄 Analýza dokumentu\n\n")
        lines.append(f"**Soubor:** `{file_path.name}`\n\n")

        # Metadata table
        lines.append("## ℹ️ Metadata\n\n")
        lines.append("| Vlastnost | Hodnota |\n")
        lines.append("|-----------|---------|\n")
        lines.append(f"| Čas extrakce | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n")
        lines.append(f"| Typ textu | {'Digitální z PDF' if is_digital else 'OCR (sken)'} |\n")
        lines.append(f"| Počet textových bloků | {len(text_blocks):,} |\n" if text_blocks else "| Počet textových bloků | 0 |\n")
        lines.append(f"| Počet znaků (Markdown) | {len(markdown_input):,} |\n")
        lines.append("\n")

        # Invoice results table (if available)
        if invoice_result:
            lines.append("## 🏷️ Výsledky analýzy\n\n")
            lines.append("| Kategorie | Hodnota |\n")
            lines.append("|-----------|---------|\n")
            for key, value in invoice_data.items():
                lines.append(f"| {key} | {value} |\n")
            lines.append("\n")

            # Agent consensus details
            if invoice_result.consensus_result:
                consensus = invoice_result.consensus_result
                lines.append("### 🤖 Consensus Engine\n\n")
                lines.append("| Metric | Value |\n")
                lines.append("|--------|-------|\n")
                lines.append(f"| Weighted Score | {consensus.get('weighted_score', 0):.2f} |\n")
                lines.append(f"| Decision Type | {consensus.get('decision_type', 'N/A')} |\n")

                agent_scores = consensus.get('agent_scores', {})
                if agent_scores:
                    lines.append("\n### Agent Scores\n\n")
                    lines.append("| Agent | Score |\n")
                    lines.append("|-------|-------|\n")
                    for agent, score in agent_scores.items():
                        lines.append(f"| {agent.title()} | {score:.2f} |\n")
                lines.append("\n")

            # REASONING SECTION (Crucial for debugging) — SANITIZED
            lines.append("## 🧠 Reasoning & Agent Outputs\n\n")
            if hasattr(invoice_result, 'agent_results') and invoice_result.agent_results:
                for agent_name, result in invoice_result.agent_results.items():
                    lines.append(f"### 🤖 Agent: {agent_name.title()}\n")
                    lines.append(f"**Confidence:** {result.get('confidence', 0):.0%}\n\n")
                    if result.get('reasoning'):
                        reasoning = result.get('reasoning')
                        # Sanitize reasoning before writing to disk
                        if HAS_SECURITY:
                            reasoning = sanitize_llm_output(reasoning)
                        lines.append("#### 🧠 Reasoning:\n")
                        lines.append(f"{reasoning}\n\n")
            else:
                lines.append("*Žádná podrobná data od agentů nejsou k dispozici.*\n\n")

        # ⚠️ HLAVNÍ DATA: Strukturovaná Markdown data (s kterými agenti pracují)
        lines.append("## 📋 Strukturovaná data dokumentu (Markdown Input pro Agenty)\n\n")
        lines.append("Toto jsou JEDINÁ data, která dostávají agenti ke zpracování.\n\n")
        lines.append(markdown_input)
        lines.append("\n\n")

        # Master Instruction (pokud je k dispozici)
        if master_instruction:
            lines.append("## 🎯 Master Instruction (Referenční rámec)\n\n")
            lines.append("Tuto instrukci agenti používají pro validaci svých rozhodnutí.\n\n")
            lines.append(master_instruction)
            lines.append("\n\n")

        full_content = "".join(lines)

        # ⚠️ ŠIFROVÁNÍ ODSTRANĚNO - Všechny logy se ukládají jako plaintext
        # Důvod: Šifrování způsobovalo problémy s čitelností logů a komplikovalo debugging
        # Pokud potřebujete šifrování, můžete ho znovu povolit, ale pro vývoj a ladění
        # je plaintext nezbytný pro rychlou analýzu problémů.
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        logger.debug(f"  📝 Uložen markdown report: {raw_path}")

        # Audit log
        if HAS_SECURITY:
            audit = get_audit_logger()
            audit.log_access(str(raw_path), "write_plaintext")

    except Exception as e:
        logger.warning(f"  ⚠️ Nepodařilo se uložit markdown report: {e}")


# ─────────────────────────────────────────────
# Modul: OCR Text Extractor
# ─────────────────────────────────────────────
class OCRExtractor:
    """
    Extrahuje text z PDF a obrázků pomocí RapidOCR (PaddleOCR přes ONNX Runtime).

    Hybridní přístup:
    - PDF: Nejprve přímá extrakce textu (100% přesnost), pokud není dostupný, použije se OCR fallback
    - Obrázky: Vždy RapidOCR
    
    Uses external configuration from config/rules.yaml for keywords.
    Edit the YAML file to tune filtering rules without modifying this code.
    """

    def __init__(self, ocr_memory_mb: int = 2048):
        """
        Initialize OCRExtractor.

        Args:
            ocr_memory_mb: OCR memory limit in MB (affects ONNX Runtime threads)
        """
        # Load keywords from external config
        try:
            from pathlib import Path
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config_loader import get_config
            config = get_config()
            self.invoice_keywords = config.get_core_invoice_keywords()
            self.non_invoice_keywords = config.get_core_non_invoice_keywords()
        except Exception as e:
            # Fallback to built-in defaults
            self.invoice_keywords = [
                "faktura", "invoice", "daňový doklad", "receipt", "bill",
                "faktúra", "účtenka", "paragon"
            ]
            self.non_invoice_keywords = [
                "upomínka", "reminder", "výzva", "notice",
                "smlouva", "contract", "dohoda", "agreement",
                "nabídka", "offer", "objednávka", "order",
                "poznámka", "note", "zápis", "minutes",
                "vizitka", "business card", "reklama", "advertisement",
                "leták", "flyer", "prospekt", "catalog"
            ]
        
        self.fitz = None
        self.rapidocr = None
        self.cv2 = None
        self.np = None
        self.ocr_memory_mb = ocr_memory_mb
        self._init_libraries()

    def _init_libraries(self):
        """Načte potřebné knihovny pro RapidOCR s nastavením paměti."""
        try:
            import fitz
            self.fitz = fitz
            logger.debug("PyMuPDF načten")
        except ImportError as e:
            logger.warning(f"PyMuPDF není nainstalován: {e}")

        try:
            from rapidocr_onnxruntime import RapidOCR
            logger.info(f"🔍 Inicializuji RapidOCR (PaddleOCR modely přes ONNX Runtime, paměť: {self.ocr_memory_mb}MB)...")

            import onnxruntime

            # Vypočítat počet vláken based on memory
            num_threads = max(1, min(8, self.ocr_memory_mb // 512))

            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)
            os.environ["ONNXRUNTIME_NUM_THREADS"] = str(num_threads)

            logger.info(f"  → ONNX Runtime: {num_threads} vláken (z {self.ocr_memory_mb}MB)")

            self.rapidocr = RapidOCR()
            logger.info("✓ RapidOCR připraven")
        except ImportError as e:
            logger.error(f"RapidOCR není nainstalován: {e}")
            logger.error("💡 Instalace: pip install rapidocr_onnxruntime")
            return

        try:
            import cv2
            self.cv2 = cv2
            logger.debug("OpenCV načten")
        except ImportError as e:
            logger.warning(f"OpenCV není nainstalováno: {e}")

        try:
            import numpy as np
            self.np = np
            logger.debug("NumPy načten")
        except ImportError as e:
            logger.warning(f"NumPy není nainstalováno: {e}")

    def extract_text(self, file_path: Path) -> tuple:
        """
        Extrahuje text ze souboru (PDF nebo obrázek).

        Args:
            file_path: Cesta k souboru

        Returns:
            Tuple (text, is_digital, text_blocks):
                - text: Extrahovaný text nebo None
                - is_digital: True pokud byl použit digitální text z PDF (100% přesnost),
                              False pokud byl použit OCR
                - text_blocks: List of dict with text and position info (bbox)
        """
        if self.rapidocr is None:
            logger.error("RapidOCR není inicializován")
            return (None, False, [])

        # 🔒 SECURITY: Validate input file before processing
        if HAS_SECURITY:
            is_valid, reason = validate_input_file(file_path)
            if not is_valid:
                logger.warning(f"  🛡️ File validation failed: {reason} — {file_path.name}")
                audit = get_audit_logger()
                audit.log_security_event(
                    "input_validation_failed",
                    f"{file_path.name}: {reason}"
                )
                return (None, False, [])

        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self._extract_from_pdf(file_path)
        elif ext in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"}:
            text, blocks = self._extract_from_image(file_path)
            return (text, False, blocks)

        return (None, False, [])

    def _extract_from_pdf(self, file_path: Path) -> tuple:
        """
        Hybridní extrakce textu z PDF.

        Returns:
            Tuple (text, is_digital, text_blocks):
                - text: Extrahovaný text
                - is_digital: True pro digitální PDF, False pro OCR
                - text_blocks: List of dict with text and bbox positions
        """
        if self.fitz is None:
            logger.warning("PyMuPDF není dostupný")
            return (None, False, [])

        try:
            with self.fitz.open(str(file_path)) as doc:
                if len(doc) == 0:
                    return (None, False, [])

                if len(doc) > 1:
                    logger.info(f"  📄 Vícestránkové PDF ({len(doc)} stran) - analyzuji POUZE první stranu.")

                page = doc[0]
                
                # KROK 1: Zkusíme přímou extrakci textu s pozicemi (digitální PDF)
                page_text = page.get_text("text").strip()
                
                if len(page_text) >= MIN_ZNAKU_PRO_DIGITALNI:
                    logger.info(f"  📄 Strana 1/{len(doc)}: ✓ Digitální text ({len(page_text)} znaků) - OCR není potřeba")
                    
                    # Extrahovat bloky s pozicemi
                    text_blocks = []
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        if block.get("type") == 0:  # Text block
                            for line in block.get("lines", []):
                                for span in line.get("spans", []):
                                    text = span.get("text", "").strip()
                                    if text:
                                        # bbox je tuple (x0, y0, x1, y1), ne dict!
                                        bbox_tuple = span.get("bbox", (0, 0, 0, 0))
                                        text_blocks.append({
                                            "text": text,
                                            "bbox": {
                                                "x0": float(bbox_tuple[0]) if len(bbox_tuple) > 0 else 0,
                                                "y0": float(bbox_tuple[1]) if len(bbox_tuple) > 1 else 0,
                                                "x1": float(bbox_tuple[2]) if len(bbox_tuple) > 2 else 0,
                                                "y1": float(bbox_tuple[3]) if len(bbox_tuple) > 3 else 0
                                            },
                                            "font": span.get("font", ""),
                                            "size": span.get("size", 0)
                                        })
                    
                    return (page_text, True, text_blocks)

                # KROK 2: Stránka je pravděpodobně sken - použijeme OCR fallback
                logger.info(f"  📄 Strana 1/{len(doc)}: 🖼️ Sken detekován, spouštím RapidOCR...")
                ocr_text, ocr_blocks = self._ocr_pdf_page(page)
                return (ocr_text, False, ocr_blocks)

        except Exception as e:
            logger.warning(f"Chyba při čtení PDF: {e}")
            return (None, False, [])

    def _ocr_pdf_page(self, page) -> tuple:
        """
        Provede OCR na stránce PDF pomocí RapidOCR.
        
        Returns:
            Tuple (text, text_blocks):
                - text: Extrahovaný text
                - text_blocks: List of dict with text and bbox positions
        """
        if self.rapidocr is None or self.fitz is None or self.cv2 is None or self.np is None:
            logger.warning("RapidOCR nebo závislosti nejsou dostupné")
            return (None, [])

        try:
            mat = self.fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
            pix = page.get_pixmap(matrix=mat)

            img = self.np.frombuffer(pix.samples, dtype=self.np.uint8).reshape(pix.h, pix.w, pix.n)

            if pix.n == 4:
                img = self.cv2.cvtColor(img, self.cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img = self.cv2.cvtColor(img, self.cv2.COLOR_GRAY2BGR)

            result, _ = self.rapidocr(img)

            if result:
                extracted_text = "\n".join([item[1] for item in result])

                # Extrahovat bounding box informace z RapidOCR výsledků
                # RapidOCR vrací: [[x0, y0, x1, y1], text, confidence] nebo [[4 corner points], text, confidence]
                text_blocks = []
                for item in result:
                    if len(item) >= 2:
                        bbox_coords = item[0]
                        text = item[1]
                        confidence = item[2] if len(item) > 2 else 1.0

                        if text.strip():
                            # Normalizovat bbox_coords na flat list [x0, y0, x1, y1]
                            # Může být: [x0, y0, x1, y1] nebo [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
                            try:
                                if len(bbox_coords) == 4 and all(isinstance(p, list) for p in bbox_coords):
                                    # 4 corner points - převést na [x0, y0, x1, y1]
                                    x0, y0 = bbox_coords[0]
                                    x1, _ = bbox_coords[1]
                                    _, y1 = bbox_coords[2]
                                    flat_coords = [float(x0), float(y0), float(x1), float(y1)]
                                elif len(bbox_coords) >= 4:
                                    flat_coords = [float(c) for c in list(bbox_coords)[:4]]
                                else:
                                    flat_coords = [0.0, 0.0, 0.0, 0.0]
                            except (TypeError, ValueError, IndexError):
                                flat_coords = [0.0, 0.0, 0.0, 0.0]

                            # Robust: always ensure exactly 4 valid floats
                            if len(flat_coords) < 4 or not all(isinstance(c, (int, float)) for c in flat_coords[:4]):
                                flat_coords = [0.0, 0.0, 0.0, 0.0]

                            text_blocks.append({
                                "text": text.strip(),
                                "bbox": {
                                    "x0": flat_coords[0] / OCR_ZOOM,
                                    "y0": flat_coords[1] / OCR_ZOOM,
                                    "x1": flat_coords[2] / OCR_ZOOM,
                                    "y1": flat_coords[3] / OCR_ZOOM
                                },
                                "confidence": confidence,
                                "source": "ocr"
                            })

                return (extracted_text if extracted_text.strip() else None, text_blocks)

            return (None, [])

        except Exception as e:
            logger.warning(f"Chyba OCR PDF: {e}")
            return (None, [])

    def _extract_from_image(self, file_path: Path) -> tuple:
        """
        Extrahuje text z obrázku pomocí RapidOCR.
        
        Returns:
            Tuple (text, text_blocks):
                - text: Extrahovaný text
                - text_blocks: List of dict with text and bbox positions
        """
        if self.rapidocr is None or self.cv2 is None:
            logger.warning("RapidOCR nebo OpenCV nejsou dostupné")
            return (None, [])

        try:
            # Windows fix: cv2.imread() selhává s ne-ASCII znaky v cestě
            # Použijeme open() + np.frombuffer() + cv2.imdecode() pro správnou Unicode podporu
            with open(file_path, 'rb') as f:
                img_bytes = f.read()
            img_array = self.np.frombuffer(img_bytes, self.np.uint8)
            img = self.cv2.imdecode(img_array, self.cv2.IMREAD_COLOR)

            if img is None:
                logger.warning(f"Nelze načíst obrázek: {file_path}")
                return (None, [])

            result, _ = self.rapidocr(img)

            if result:
                extracted_text = "\n".join([item[1] for item in result])

                # Extrahovat bounding box informace z RapidOCR výsledků
                text_blocks = []
                for item in result:
                    if len(item) >= 2:
                        bbox_coords = item[0]
                        text = item[1]
                        confidence = item[2] if len(item) > 2 else 1.0

                        if text.strip():
                            # Normalizovat bbox_coords na flat list [x0, y0, x1, y1]
                            # Může být: [x0, y0, x1, y1] nebo [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
                            if len(bbox_coords) == 4 and all(isinstance(p, list) for p in bbox_coords):
                                # 4 corner points - převést na [x0, y0, x1, y1]
                                x0, y0 = bbox_coords[0]
                                x1, _ = bbox_coords[1]
                                _, y1 = bbox_coords[2]
                                flat_coords = [x0, y0, x1, y1]
                            elif len(bbox_coords) >= 4:
                                flat_coords = list(bbox_coords)
                            else:
                                flat_coords = [0.0, 0.0, 0.0, 0.0]

                            text_blocks.append({
                                "text": text.strip(),
                                "bbox": {
                                    "x0": flat_coords[0],
                                    "y0": flat_coords[1],
                                    "x1": flat_coords[2],
                                    "y1": flat_coords[3]
                                },
                                "confidence": confidence,
                                "source": "ocr"
                            })

                return (extracted_text if extracted_text.strip() else None, text_blocks)

            return (None, [])

        except Exception as e:
            logger.warning(f"Chyba OCR obrázku: {e}")
            return (None, [])

    def pre_filter(self, text: str) -> tuple:
        """
        Rychlý pravidlový filtr před AI analýzou.
        
        Returns:
            Tuple (classification, confidence, reason)
            classification: 'reject', 'uncertain', 'likely'
        """
        if not text or len(text.strip()) < 50:
            return ("reject", 0.95, "prázdný dokument")

        text_lower = text.lower()
        score = 0

        explicit_titles = ["faktura", "invoice", "daňový doklad", "zálohová faktura", "proforma"]
        has_explicit_title = any(title in text_lower for title in explicit_titles)

        positive = [
            ("dodavatel", 2), ("odběratel", 2), ("ičo", 2), ("dič", 2),
            ("celkem k úhradě", 3), ("datum splatnosti", 2), ("variabilní symbol", 2),
            ("faktura č", 2), ("částka", 1), ("kč", 1), ("czk", 1),
        ]

        for keyword, points in positive:
            if keyword in text_lower:
                score += points

        negative = [
            ("upomínka", -5), ("smlouva", -4), ("nabídka", -3),
            ("objednávka", -2), ("životopis", -4), ("certifikát", -4),
        ]

        for keyword, penalty in negative:
            if keyword in text_lower:
                score += penalty

        if score <= -3:
            return ("reject", min(0.95, 0.7 + abs(score) * 0.05), f"negativní skóre ({score})")

        if has_explicit_title and score >= 6:
            return ("likely", min(0.9, 0.6 + score * 0.03), f"pozitivní skóre ({score})")

        return ("uncertain", 0.5, f"nejistý případ (score={score})")


# ─────────────────────────────────────────────
# Modul: Multi-Agent Processor
# ─────────────────────────────────────────────
class MultiAgentProcessor:
    """
    Hlavní processor s multi-agent workflow.
    Spouští 3 agenty paralelně a kombinuje výsledky.
    """

    def __init__(self, vram_limit_gb: int = 4, num_ctx: int = 4096):
        """
        Initialize Multi-Agent Processor.

        Args:
            vram_limit_gb: VRAM limit for Ollama models (GB)
            num_ctx: Context window size for LLM
        """
        if not AGENTS_AVAILABLE:
            raise ImportError("Agent workflow module not available")

        self.classifier = ClassifierAgent(
            model=TEXT_MODEL,
            timeout=REQUEST_TIMEOUT,
            vram_limit_gb=vram_limit_gb,
            num_ctx=num_ctx
        )
        self.extractor = ExtractorAgent(
            model=TEXT_MODEL,
            timeout=EXTRACTOR_TIMEOUT,
            vram_limit_gb=vram_limit_gb,
            num_ctx=num_ctx
        )
        self.anomaly = AnomalyDetectorAgent(
            model=TEXT_MODEL,
            timeout=REQUEST_TIMEOUT,
            vram_limit_gb=vram_limit_gb,
            num_ctx=num_ctx
        )
        self.consensus = ConsensusEngine(
            threshold_accept=THRESHOLD_ACCEPT,
            threshold_review=THRESHOLD_REVIEW,
            anomaly_veto_threshold=ANOMALY_VETO_THRESHOLD
        )
        self.ocr_validator = OCRTextValidator(language="auto")

        self.stats = {
            "total": 0,
            "invoices": 0,
            "non_invoices": 0,
            "human_review": 0,
            "errors": 0,
        }

    def analyze_document(self, markdown_input: str, file_path: Path, text_blocks: list = None, master_instruction: Optional[str] = None) -> Optional[Invoice]:
        """
        Analyzuje dokument pomocí 3 agentů SEKVENČNĚ (s předáním informací).

        ARCHITEKTURA: Agenti NEDOSTÁVAJÍ surový text! Pracují pouze se:
        1. Markdown tabulkou (strukturovaná data z text_blocks)
        2. Master Instruction (pevný referenční rámec pro validaci)

        Args:
            markdown_input: Strukturovaná Markdown data (tabulka + layout)
            file_path: Cesta k souboru
            text_blocks: List textových bloků s bbox pozicemi (volitelné)
            master_instruction: Pevná instrukce pro validaci agentů
        """
        if not markdown_input or len(markdown_input.strip()) < 50:
            logger.debug(f"  ⚠️ Prázdný dokument: {file_path.name}")
            return None

        start_time = time.time()
        logger.debug(f"  🚀 Start analýzy: {file_path.name}")

        invoice = Invoice(source_path=file_path)

        try:
            # KROK 1: Classifier - pracuje POUZE s Markdown + Master Instruction
            logger.debug(f"  ⏳ Classifier...")
            try:
                classifier_result = self.classifier.analyze(
                    markdown_input,
                    metadata=None,
                    text_blocks=text_blocks,
                    master_instruction=master_instruction
                )
                logger.debug(f"  ✓ Classifier hotovo za {time.time() - start_time:.1f}s")
                logger.debug(f"    Výsledek: {'FAKTURA' if classifier_result.get('is_invoice') else 'NENÍ FAKTURA'} ({classifier_result.get('confidence', 0):.0%})")
            except Exception as e:
                logger.error(f"  ✗ Classifier ERROR: {e}")
                classifier_result = {"is_invoice": False, "confidence": 0.0, "reasoning": "Error"}

            # KROK 2: Anomaly Detector (Spouštíme před extraktorem kvůli případnému vyvrácení)
            logger.debug(f"  ⏳ Anomaly Detector...")
            try:
                anomaly_metadata = {
                    "text_blocks": text_blocks,
                    "classifier_reasoning": classifier_result.get("reasoning", ""),
                    "classifier_is_invoice": classifier_result.get("is_invoice", False)
                } if text_blocks else {
                    "classifier_reasoning": classifier_result.get("reasoning", ""),
                    "classifier_is_invoice": classifier_result.get("is_invoice", False)
                }
                anomaly_result = self.anomaly.analyze(
                    markdown_input,
                    anomaly_metadata,
                    master_instruction=master_instruction
                )
                logger.debug(f"  ✓ Anomaly hotovo za {time.time() - start_time:.1f}s")
            except TimeoutError:
                logger.error(f"  ⏰ Anomaly TIMEOUT")
                anomaly_result = {"is_anomaly": False, "confidence": 0.0, "refutes_classifier": False}
            except Exception as e:
                logger.error(f"  ✗ Anomaly ERROR: {e}")
                anomaly_result = {"is_anomaly": False, "confidence": 0.0, "refutes_classifier": False}

            # KROK 3: Extractor - se short-circuit a Feedback Loop logikou
            classifier_is_inv = classifier_result.get("is_invoice", False)
            refutes = anomaly_result.get("refutes_classifier", False)

            if not classifier_is_inv and not refutes:
                logger.info(f"  ⏭️ Extractor přeskočen: Classifier zamítl dokument a Anomaly to nevyvrátila.")
                extractor_result = {"completeness_score": 0.0, "validation_errors": ["Classifier rejected document"]}
            else:
                if not classifier_is_inv and refutes:
                    logger.warning(f"  🔄 FEEDBACK LOOP: Anomaly agent vyvrátil zamítnutí Classifieru! (Nalezena faktura). Spouštím Extractor a neutralizuji Classifier.")
                    # Upravíme výsledek Classifieru aby systém nezamítl správnou fakturu
                    classifier_result["is_invoice"] = True
                    classifier_result["confidence"] = 0.5  # Neutrální confidence
                    classifier_result["reasoning"] = classifier_result.get("reasoning", "") + " | (Vyvráceno Anomaly Agentem - nalezeny prvky faktury)"

                logger.debug(f"  ⏳ Extractor (s feedback loop logikou)...")
                try:
                    classifier_metadata = {
                        "classifier_reasoning": classifier_result.get("reasoning", ""),
                        "classifier_elements": classifier_result.get("elements_present", {}),
                        "classifier_is_invoice": classifier_result.get("is_invoice", False),
                        "classifier_confidence": classifier_result.get("confidence", 0),
                        "extracted_values": classifier_result.get("extracted_values", {}),
                        "text_blocks": text_blocks
                    }

                    extractor_result = self.extractor.analyze(
                        markdown_input,
                        classifier_metadata,
                        master_instruction=master_instruction
                    )
                    logger.debug(f"  ✓ Extractor hotovo za {time.time() - start_time:.1f}s")
                    logger.debug(f"    Completeness: {extractor_result.get('completeness_score', 0):.0%}")
                    if extractor_result.get("validation_errors"):
                        logger.debug(f"    Validation errors: {extractor_result['validation_errors'][:3]}")
                except TimeoutError:
                    logger.error(f"  ⏰ Extractor TIMEOUT")
                    extractor_result = {"completeness_score": 0.0, "validation_errors": ["Timeout"]}
                except Exception as e:
                    logger.error(f"  ✗ Extractor ERROR: {e}")
                    extractor_result = {"completeness_score": 0.0, "validation_errors": ["Error"]}

            # Uložení výsledků agentů
            invoice.agent_results = {
                "classifier": classifier_result,
                "extractor": extractor_result,
                "anomaly": anomaly_result,
            }

            # Výpočet konsenzu
            try:
                consensus_result = self.consensus.calculate_consensus(
                    classifier_result,
                    extractor_result,
                    anomaly_result,
                    raw_text=markdown_input  # Použijeme Markdown input (jediný text který máme)
                )

                if consensus_result.get("decision_type") == "retry":
                    logger.warning(f"  ⚠️ RETRY požadavek pro {file_path.name}: {consensus_result.get('reasoning')}")
                    consensus_result["is_invoice"] = None
                    consensus_result["decision_type"] = "human_review"
                    consensus_result["retry_info"] = consensus_result.get("keyword_analysis", {})
                    logger.info(f"  🔍 {file_path.name} přesunut do human review kvůli chybějícím invoice keywords")
            except Exception as consensus_error:
                logger.error(f"  Consensus error: {consensus_error}")
                clf_is_inv = classifier_result.get("is_invoice", False)
                clf_conf = classifier_result.get("confidence", 0)
                consensus_result = {
                    "is_invoice": clf_is_inv,
                    "confidence": clf_conf,
                    "decision_type": "auto_reject" if not clf_is_inv else "auto_accept",
                    "reasoning": f"Fallback due to consensus error: {consensus_error}",
                    "extracted_data": {}
                }

            invoice.consensus_result = consensus_result

            is_invoice = consensus_result.get("is_invoice")
            confidence = consensus_result.get("confidence", 0)
            decision_type = consensus_result.get("decision_type", "unknown")

            invoice.is_invoice = is_invoice if is_invoice is not None else False
            invoice.confidence = confidence
            invoice.decision_type = decision_type
            invoice.requires_review = (decision_type == "human_review")

            extracted = consensus_result.get("extracted_data", {})

            if extracted.get("vendor_name"):
                invoice.sender_name = _clean_invoice_value(extracted["vendor_name"])
            if extracted.get("vendor_address"):
                invoice.sender_address = _clean_invoice_value(extracted["vendor_address"])
            if extracted.get("customer_name"):
                invoice.recipient_name = _clean_invoice_value(extracted["customer_name"])
            if extracted.get("customer_address"):
                invoice.recipient_address = _clean_invoice_value(extracted["customer_address"])
            if extracted.get("issue_date"):
                invoice.issue_date = extracted["issue_date"]
            if extracted.get("due_date"):
                invoice.due_date = extracted["due_date"]
            if extracted.get("total_amount"):
                invoice.total_amount = str(extracted["total_amount"])
            if extracted.get("currency"):
                invoice.currency = extracted["currency"]
            if extracted.get("invoice_number"):
                invoice.invoice_number = _clean_invoice_value(extracted["invoice_number"])

            elapsed = time.time() - start_time

            if elapsed > 180:
                logger.warning(f"  ⚠️ Analýza trvala velmi dlouho: {elapsed:.1f}s")

            self.stats["total"] += 1

            if is_invoice is True:
                self.stats["invoices"] += 1
                status = "FAKTURA"
            elif is_invoice is False:
                self.stats["non_invoices"] += 1
                status = "NENÍ FAKTURA"
            else:
                self.stats["human_review"] += 1
                status = "REVIEW"

            logger.info(f"  ✓ {file_path.name}: {status} ({confidence:.0%}, {elapsed:.1f}s)")
            logger.debug(f"    Reasoning: {consensus_result.get('reasoning', 'N/A')[:80]}")

            return invoice

        except Exception as e:
            logger.error(f"  ✗ Chyba analýzy {file_path.name}: {e}")
            self.stats["errors"] += 1

            invoice.is_invoice = False
            invoice.confidence = 0.0
            invoice.decision_type = "error"
            invoice.raw_json = {"error": str(e)}
            return invoice

    def get_statistics(self) -> dict:
        """Vrátí statistiky zpracování."""
        return self.stats.copy()


# ─────────────────────────────────────────────
# Modul: Rychlé vyhledávání souborů
# ─────────────────────────────────────────────
class FileDiscovery:
    """Discovers supported invoice files in a directory."""
    
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir

    def find_files(self) -> List[Path]:
        """Find all supported files in source directory."""
        found: List[Path] = []
        logger.info(f"Prohledávám složku: {self.source_dir}")

        try:
            for path in self.source_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    found.append(path)
        except PermissionError as e:
            logger.warning(f"Přístup odepřen: {e}")

        logger.info(f"Nalezeno {len(found)} souborů ke zpracování.")
        return found


# ─────────────────────────────────────────────
# Modul: Filtrování faktur
# ─────────────────────────────────────────────
class InvoiceFilterAgent:
    """Filters invoices based on various criteria."""
    
    def __init__(self, invoices: List[Invoice]):
        self.invoices = invoices

    def filter_by_custom(self, filter_key: str, filter_value: str) -> List[Invoice]:
        """Filter invoices by custom field."""
        if not filter_key or not filter_value:
            return [inv for inv in self.invoices if inv.is_invoice]

        filtered = []
        for inv in self.invoices:
            if not inv.is_invoice:
                continue
            if inv.matches_filter(filter_key, filter_value):
                filtered.append(inv)

        return filtered

    def filter_by_decision_type(self, decision_type: str) -> List[Invoice]:
        """Filter invoices by decision type."""
        return [inv for inv in self.invoices if inv.decision_type == decision_type]

    def get_unique_values(self, field_name: str) -> List[str]:
        """Get unique values for a field."""
        values = set()
        for inv in self.invoices:
            if inv.is_invoice:
                val = getattr(inv, field_name, "")
                if val:
                    values.add(val)
        return sorted(values)


# ─────────────────────────────────────────────
# Modul: Třídění a přesun
# ─────────────────────────────────────────────
class InvoiceOrganizer:
    """Organizes and moves invoices to target directory."""
    
    SORT_OPTIONS = {
        "sender_name": "Odesílatel",
        "recipient_name": "Příjemce",
        "issue_date": "Datum_vystaveni",
        "due_date": "Datum_splatnosti",
        "total_amount": "Castka",
    }

    def __init__(self, invoices: List[Invoice], target_dir: Path):
        self.invoices = invoices
        self.target_dir = target_dir

    def sort_invoices(self, sort_key: str) -> List[Invoice]:
        """Sort invoices by key."""
        return sorted(
            self.invoices,
            key=lambda inv: getattr(inv, sort_key) or "ZZZZ"
        )

    def _build_new_name(self, index: int, invoice: Invoice, sort_key: str) -> str:
        """Build new filename for invoice."""
        criterion_value = _sanitize_filename(getattr(invoice, sort_key) or "nezname")
        seq = str(index).zfill(3)
        return f"{seq}_{criterion_value}_{invoice.original_stem}{invoice.suffix}"

    def process(self, sort_key: str) -> tuple:
        """
        Process and move invoices.
        
        Returns:
            Tuple (success_count, error_count)
        """
        if not self.invoices:
            return 0, 0

        sorted_invoices = self.sort_invoices(sort_key)

        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Nelze vytvořit cílovou složku: {e}")
            return 0, len(sorted_invoices)

        errors = 0
        success = 0

        for i, invoice in enumerate(sorted_invoices, start=1):
            if not invoice.is_invoice:
                logger.warning(f"⚠️ Přeskok '{invoice.source_path.name}': Není faktura")
                errors += 1
                continue

            new_name = self._build_new_name(i, invoice, sort_key)
            dest_path = self.target_dir / new_name

            if dest_path.exists():
                stem = dest_path.stem
                dest_path = self.target_dir / f"{stem}_dup{dest_path.suffix}"

            try:
                shutil.move(str(invoice.source_path), str(dest_path))
                logger.info(f"✓ Přesunuto: '{invoice.source_path.name}' → '{dest_path.name}'")
                success += 1
            except Exception as e:
                errors += 1
                logger.error(f"Chyba při přesunu '{invoice.source_path.name}': {e}")

        return success, errors


# ─────────────────────────────────────────────
# Pomocná funkce pro AI model
# ─────────────────────────────────────────────
def sort_text_blocks_smartly(text_blocks: List[Dict]) -> List[Dict]:
    """
    Seřadí textové bloky podle Y souřadnic (řádky) a následně podle X souřadnic (sloupce).
    Implementuje logiku z uprava.py pro detekci sloupců.
    """
    if not text_blocks:
        return []

    items = []
    for block in text_blocks:
        bbox = block.get('bbox', {})
        # Bezpečné načtení souřadnic (v6.7 style)
        x0, y0 = bbox.get('x0', 0), bbox.get('y0', 0)
        x1, y1 = bbox.get('x1', 0), bbox.get('y1', 0)
        
        if isinstance(x0, list): x0 = x0[0] if x0 else 0
        if isinstance(y0, list): y0 = y0[0] if y0 else 0
        if isinstance(x1, list): x1 = x1[0] if x1 else 0
        if isinstance(y1, list): y1 = y1[0] if y1 else 0

        items.append({
            'text': block.get('text', ''),
            'x_min': float(x0),
            'x_max': float(x1),
            'x_center': (float(x0) + float(x1)) / 2,
            'y_min': float(y0),
            'y_max': float(y1),
            'y_center': (float(y0) + float(y1)) / 2
        })

    # Najdeme rozsah X souřadnic pro detekci sloupců
    all_x = [i['x_center'] for i in items]
    min_x, max_x = min(all_x), max(all_x)
    page_width = max_x - min_x
    stred_stranky = (min_x + max_x) / 2

    # Threshold pro detekci mezery mezi sloupci (15% šířky)
    threshold = page_width * 0.15
    tolerance_y = 10
    radky = []

    # Seskupení do řádků
    for item in sorted(items, key=lambda x: (x['y_center'], x['x_min'])):
        nalezen_radek = False
        for radek in radky:
            if abs(item['y_center'] - radek['y_center']) <= tolerance_y:
                radek['items'].append(item)
                radek['y_center'] = sum(i['y_center'] for i in radek['items']) / len(radek['items'])
                nalezen_radek = True
                break
        if not nalezen_radek:
            radky.append({'y_center': item['y_center'], 'items': [item]})

    # Pro každý řádek rozdělíme na sloupce
    vysledek = []
    for radek in sorted(radky, key=lambda r: r['y_center']):
        polozky = sorted(radek['items'], key=lambda i: i['x_min'])

        sloupce_v_radku = []
        if polozky:
            current_sloupec = [polozky[0]]
            for i in range(1, len(polozky)):
                if polozky[i]['x_min'] - polozky[i-1]['x_max'] > threshold:
                    sloupce_v_radku.append(current_sloupec)
                    current_sloupec = []
                current_sloupec.append(polozky[i])
            if current_sloupec:
                sloupce_v_radku.append(current_sloupec)

        for sloupec in sloupce_v_radku:
            text = ' '.join(p['text'] for p in sloupec)
            x_pozice = 'vlevo' if sum(p['x_center'] for p in sloupec) / len(sloupec) < stred_stranky else 'vpravo'
            vysledek.append({
                'text': text,
                'sloupec': x_pozice,
                'y_center': radek['y_center']
            })

    return vysledek


def format_blocks_to_markdown_table(text_blocks: List[Dict]) -> str:
    """
    Převede textové bloky do vylepšené Markdown tabulky (podle uprava.py).
    Místo souřadnic obsahuje informaci o pozici (vlevo/vpravo).
    """
    serazene = sort_text_blocks_smartly(text_blocks)
    if not serazene:
        return ""
        
    md = "\n\n### 📋 Strukturovaná data dokumentu\n"
    md += "| Cislo | Pozice | Text |\n"
    md += "|-------|--------|------|\n"
    
    for i, blok in enumerate(serazene[:200], 1): # Limit context
        text_clean = blok['text'].replace('|', '\\|').replace('\n', ' ')
        md += f"| {i} | {blok['sloupec']} | {text_clean} |\n"
        
    return md


def generate_master_instruction(text_blocks: List[Dict]) -> str:
    """
    Generuje 'PLNY TEXT PODLE SOURADNIC' - master instrukci pro agenty.
    """
    serazene = sort_text_blocks_smartly(text_blocks)
    if not serazene:
        return ""
        
    output = "\n\n### 🎯 PLNY TEXT PODLE SOURADNIC (Master Instruction)\n\n"
    for blok in serazene:
        output += f"[{blok['sloupec']}] {blok['text']}\n"
        
    return output


def format_blocks_to_markdown(text_blocks: List[Dict]) -> str:
    """
    Zpětně kompatibilní wrapper, který nyní vrací oba nové formáty.
    """
    if not text_blocks:
        return ""
        
    table = format_blocks_to_markdown_table(text_blocks)
    master = generate_master_instruction(text_blocks)
    
    return f"{table}\n{master}"


# ─────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────
def process_invoice(
    file_path: Path,
    ocr_extractor: OCRExtractor,
    agent_processor: MultiAgentProcessor,
    save_raw: bool = SAVE_RAW_TEXT,
    master_instruction: Optional[str] = None
) -> Optional[Invoice]:
    """
    Process a single invoice file.

    This is the main high-level API function for processing invoices.
    It handles OCR extraction, validation, and multi-agent analysis.

    Args:
        file_path: Path to the invoice file
        ocr_extractor: OCRExtractor instance
        agent_processor: MultiAgentProcessor instance
        save_raw: Whether to save raw extracted text for debugging

    Returns:
        Invoice object with extracted data, or None if processing failed
    """
    logger.info(f"📄 Zpracovávám: {file_path.name}")

    # OCR extrakce - nyní vrací i text_blocks s pozicemi
    ocr_text, is_digital_text, text_blocks = ocr_extractor.extract_text(file_path)

    if not ocr_text:
        logger.debug(f"  ⚠️ Nepodařilo se extrahovat text: {file_path.name} (automaticky zamítnuto)")
        return Invoice(
            source_path=file_path,
            is_invoice=False,
            confidence=0.95,
            decision_type="auto_reject",
            raw_json={"error": "No text extracted", "reason": "Empty or unreadable file"}
        )

    # Save raw text for debugging (initial extraction) with positions
    # ⚠️ Ukládáme pouze strukturovaná Markdown data (žádný surový text!)
    if save_raw:
        # Předběžné uložení před zpracováním - pouze text_blocks
        from core.engine import format_blocks_to_markdown_table, generate_master_instruction
        md_table_preview = format_blocks_to_markdown_table(text_blocks) if text_blocks else ""
        master_preview = generate_master_instruction(text_blocks) if text_blocks else ""
        save_raw_text(file_path, md_table_preview, is_digital_text, invoice_result=None, text_blocks=text_blocks, master_instruction=master_preview)

    # OCR Text Validator - POUZE pro OCR text
    final_blocks = text_blocks
    if not is_digital_text:
        logger.debug(f"  🔍 Validace OCR textu: {file_path.name}")
        try:
            # Předáme přímo text_blocks místo ocr_text pro zachování struktury
            ocr_text, validation_result = validate_ocr_text(text_blocks, language="auto")
            final_blocks = validation_result.get("valid_blocks", text_blocks)

            if validation_result["hallucinated_words"] > 0:
                logger.info(f"  ✓ OCR Validator: {validation_result['valid_words']}/{validation_result['original_words']} slov validní, "
                            f"{validation_result['corrected_words']} opraveno, "
                            f"⚠️ {validation_result['hallucinated_words']} halucinací odstraněno")
            else:
                logger.debug(f"  ✓ OCR Validator: {validation_result['valid_words']}/{validation_result['original_words']} slov validní")

        except Exception as validation_error:
            logger.warning(f"  ⚠️ OCR Validator selhal: {validation_error} - používám původní text")

    # Pravidlový předběžný filtr
    pre_class, pre_conf, pre_reason = ocr_extractor.pre_filter(ocr_text)

    if pre_class == "reject" and pre_conf > 0.9:
        logger.debug(f"  ⏭️ Přeskočeno (pravidlový filtr): {file_path.name}")
        return Invoice(
            source_path=file_path,
            is_invoice=False,
            confidence=pre_conf,
            decision_type="auto_reject",
            raw_json={"pre_filter": True, "reason": pre_reason}
        )

    # v6.7: Použijeme VALIDNÍ bloky pro Markdown tabulku
    # (funkce již importovány výše pro save_raw_text)
    md_table = format_blocks_to_markdown_table(final_blocks)

    # Pokud master_instruction není předáno zvenčí, vygenerujeme ho z bloků
    if master_instruction is None:
        master_instruction = generate_master_instruction(final_blocks)

    # v6.7: Přidáme také rekonstrukci fyzického layoutu (text na řádcích s mezerami)
    spatial_layout = ""
    try:
        # Převedeme bbox na x, y pro BaseAgent nářadí
        mapped_blocks = []
        for b in final_blocks:
            bbox = b.get('bbox', {})
            mapped_blocks.append({
                'text': b.get('text', ''),
                'x': float(bbox.get('x0', 0)),
                'y': float(bbox.get('y0', 0))
            })
        spatial_layout = "\n\n### 📜 Vizualizace dokumentu (Reconstructed Layout)\n"
        spatial_layout += "Toto je simulovaný vzhled stránky. Použij ho pro pochopení struktury.\n"
        spatial_layout += "```\n"
        spatial_layout += agent_processor.classifier.reconstruct_spatial_layout(mapped_blocks)
        spatial_layout += "\n```\n"
    except Exception as e:
        logger.warning(f"  ⚠️ Chyba při rekonstrukci layoutu: {e}")

    # ⚠️ ARCHITEKTURA: Agenti dostanou POUZE strukturovaná data (Markdown + Master Instruction)
    # ŽÁDNÝ surový text se agentům nepředává!
    markdown_input = f"{md_table}\n\n{spatial_layout}".strip()

    # Multi-agent analýza - předáme POUZE Markdown data + Master Instruction
    logger.debug(f"  🔄 Spouštím agenty pro: {file_path.name}")
    invoice = agent_processor.analyze_document(markdown_input, file_path, text_blocks=final_blocks, master_instruction=master_instruction)

    if invoice is None:
        logger.warning(f"  ⚠️ analyze_document vrátil None pro: {file_path.name}")
        return Invoice(
            source_path=file_path,
            is_invoice=False,
            confidence=0.0,
            decision_type="error",
            raw_json={"error": "analyze_document returned None"}
        )

    # Save final markdown report with invoice results and positions
    # ⚠️ Ukládáme pouze strukturovaná Markdown data (žádný surový text!)
    if save_raw:
        save_raw_text(file_path, markdown_input, is_digital_text, invoice_result=invoice, text_blocks=final_blocks, master_instruction=master_instruction)

    return invoice


# ─────────────────────────────────────────────
# Ollama utility functions
# ─────────────────────────────────────────────

# 🔒 PRIVACY SETTINGS - Force Ollama to run in maximum privacy mode
# These environment variables ensure Ollama operates in complete isolation
def _set_ollama_privacy_mode():
    """
    Set Ollama environment variables for maximum privacy and local-only operation.

    This ensures:
    - Ollama binds ONLY to localhost (127.0.0.1)
    - No external connections accepted
    - No telemetry or external API calls
    - Debug mode disabled in production
    """
    import os

    # Force localhost binding ONLY - no external access
    if "OLLAMA_HOST" not in os.environ:
        os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
        logger.debug("🔒 Privacy: OLLAMA_HOST set to 127.0.0.1 (localhost only)")

    # Block all CORS origins except localhost
    if "OLLAMA_ORIGINS" not in os.environ:
        os.environ["OLLAMA_ORIGINS"] = "http://localhost:*,http://127.0.0.1:*,app://*"
        logger.debug("🔒 Privacy: OLLAMA_ORIGINS restricted to localhost")

    # Disable debug mode (unless explicitly enabled)
    if "OLLAMA_DEBUG" not in os.environ:
        os.environ["OLLAMA_DEBUG"] = "false"
        logger.debug("🔒 Privacy: OLLAMA_DEBUG disabled")

    # Disable telemetry completely
    if "OLLAMA_NOPROMPT" not in os.environ:
        os.environ["OLLAMA_NOPROMPT"] = "1"
        logger.debug("🔒 Privacy: OLLAMA_NOPROMPT enabled")

    # Force no HTTPS redirects (keep everything local)
    if "OLLAMA_INSECURE" not in os.environ:
        os.environ["OLLAMA_INSECURE"] = "1"
        logger.debug("🔒 Privacy: OLLAMA_INSECURE enabled (local only)")


def check_ollama_running() -> bool:
    """Check if Ollama is running on port 11434."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("127.0.0.1", 11434))
        sock.close()
        if result == 0:
            logger.info("✓ Ollama běží na portu 11434")
            return True
        return False
    except Exception as e:
        logger.debug(f"Ollama check error: {e}")
        return False


def try_start_ollama() -> bool:
    """
    Try to start Ollama server with MAXIMUM PRIVACY settings.

    Privacy mode ensures:
    - Ollama binds ONLY to localhost (127.0.0.1)
    - No external connections accepted
    - All data stays on local machine
    """
    # 🔒 PRIVACY: Set privacy mode BEFORE starting Ollama
    _set_ollama_privacy_mode()

    ollama_paths = [
        r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe",
        r"C:\Program Files\Ollama\ollama.exe",
        r"%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe",
    ]

    ollama_exe = None
    for path_template in ollama_paths:
        path = os.path.expandvars(path_template)
        if os.path.isfile(path):
            ollama_exe = path
            break

    if not ollama_exe:
        ollama_exe = shutil.which("ollama")

    if not ollama_exe:
        logger.warning("⚠️ Ollama executable nenalezen")
        return False

    try:
        logger.info(f"🚀 Spouštím Ollama (privacy mode): {ollama_exe}")
        logger.info("🔒 Privacy: Ollama poběží VÝHRADNĚ na localhostu - žádná externí připojení!")

        # Start Ollama with privacy environment variables
        env = os.environ.copy()
        # Already set by _set_ollama_privacy_mode() but ensure they're passed
        env["OLLAMA_HOST"] = "127.0.0.1:11434"
        env["OLLAMA_ORIGINS"] = "http://localhost:*,http://127.0.0.1:*,app://*"
        env["OLLAMA_DEBUG"] = "false"
        env["OLLAMA_NOPROMPT"] = "1"

        subprocess.Popen(
            [ollama_exe, "serve"],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env  # Pass privacy-configured environment
        )

        for i in range(20):
            time.sleep(0.5)
            if check_ollama_running():
                logger.info("✓ Ollama úspěšně spuštěna v privacy mode")
                return True

        logger.warning("⚠️ Ollama se nespustila včas")
        return False

    except Exception as e:
        logger.error(f"Chyba při spuštění Ollama: {e}")
        return False


def check_ollama_model(model_name: str = "llama3.2") -> bool:
    """Check if Ollama model is available."""
    try:
        import ollama
        models = ollama.list()
        for m in models.get("models", []):
            if model_name in m.get("name", ""):
                return True
        return False
    except Exception as e:
        logger.debug(f"Model check error: {e}")
        return False


def calculate_num_ctx(vram_gb: int) -> int:
    """
    Calculate optimal context window size based on VRAM.
    
    llama3.2 needs approximately:
    - 2GB for model weights
    - ~1GB per 1024 tokens of context
    
    Args:
        vram_gb: Available VRAM in GB
        
    Returns:
        Optimal num_ctx value
    """
    available_for_context = max(0, vram_gb - 2)
    num_ctx = int(available_for_context * 1024)
    num_ctx = max(2048, min(num_ctx, 16384))
    num_ctx = (num_ctx // 256) * 256
    
    logger.debug(f"Calculated num_ctx={num_ctx} for {vram_gb}GB VRAM")
    return num_ctx
