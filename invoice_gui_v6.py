#!/usr/bin/env python3
"""
Invoice Processor GUI - Multi-Agent Workflow with RapidOCR
Thin GUI layer over core business logic engine.

This is ONLY the presentation layer. All business logic is in core/engine.py

Core Engine Features:
- OCRExtractor: Hybrid OCR (RapidOCR + direct PDF text extraction)
- MultiAgentProcessor: 3 specialized AI agents + Consensus Engine
- FileDiscovery, InvoiceFilterAgent, InvoiceOrganizer

Usage:
    python invoice_gui_v6.py

CLI/API Alternative:
    from core import OCRExtractor, MultiAgentProcessor, process_invoice
    
    extractor = OCRExtractor()
    processor = MultiAgentProcessor()
    invoice = process_invoice(Path("invoice.pdf"), extractor, processor)

Dependencies:
    pip install rapidocr_onnxruntime opencv-python numpy fitz customtkinter
"""

# Potlačit varování Pydantic V1 s Pythonem 3.14+
import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*Python 3.14.*")

import os
import sys
import json
import logging
import threading
import gc
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# GUI knihovny
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk

# ─────────────────────────────────────────────
# Import business logic from core engine
# ─────────────────────────────────────────────
try:
    from core import (
        Invoice,
        OCRExtractor,
        MultiAgentProcessor,
        FileDiscovery,
        InvoiceFilterAgent,
        InvoiceOrganizer,
        SUPPORTED_EXTENSIONS,
        get_memory_usage,
        log_memory_state,
        save_raw_text,
        check_ollama_running,
        try_start_ollama,
        check_ollama_model,
        calculate_num_ctx,
        _set_ollama_privacy_mode,  # 🔒 Privacy settings
    )
    from core.engine import (
        AGENTS_AVAILABLE,
        MAX_MEMORY_PERCENT,
        DEBUG_MEMORY,
        MEMORY_CHECK_EVERY,
        GC_EVERY,
        SAVE_RAW_TEXT,
        RAW_TEXT_DIR,
        TEXT_MODEL,
        OCR_ZOOM,
    )
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    AGENTS_AVAILABLE = False
    print(f"⚠️ Warning: Core engine not available: {e}")
    print("💡 Ensure core/ directory exists with engine.py")

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "gui_output.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GUI nastavení
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ─────────────────────────────────────────────
# GUI Aplikace
# ─────────────────────────────────────────────
class InvoiceProcessorGUIV6(ctk.CTk):
    """
    GUI for Multi-Agent Workflow systém.
    
    This is a THIN LAYER - all business logic is delegated to core engine:
    - OCRExtractor: OCR text extraction
    - MultiAgentProcessor: Document analysis
    - FileDiscovery: File finding
    - InvoiceFilterAgent: Filtering
    - InvoiceOrganizer: Sorting and moving
    """

    def __init__(self):
        super().__init__()

        self.title("🧾 Invoice Processor")
        self.geometry("1200x900")
        self.minsize(1100, 800)

        # Proměnné
        self.source_dir = ctk.StringVar()
        self.target_dir = ctk.StringVar()
        self.sort_key = ctk.StringVar(value="issue_date")
        self.model_name = ctk.StringVar(value=TEXT_MODEL)

        # Filtr
        self.filter_key = ctk.StringVar(value="")
        self.filter_value = ctk.StringVar()

        # Memory settings - MUST BE BEFORE OCRExtractor!
        self.memory_settings = {
            'ocr_memory_mb': 2048,
            'ollama_vram_gb': 4,
            'memory_threshold': 60,
            'profile': 'balanced'
        }
        self.settings_file = Path(__file__).parent / ".memory_settings.json"
        self._load_memory_settings()

        # 🔒 PRIVACY: Activate Ollama privacy mode IMMEDIATELY on app start
        # This ensures Ollama runs in complete isolation from the very beginning
        try:
            _set_ollama_privacy_mode()
            logger.info("🔒 Privacy mode activated: Ollama poběží VÝHRADNĚ na localhostu")
        except Exception as e:
            logger.warning(f"Privacy mode setup failed: {e}")

        # Initialize core engine components
        ocr_memory = self.memory_settings.get('ocr_memory_mb', 2048)
        self.ocr_extractor = OCRExtractor(ocr_memory_mb=ocr_memory)
        self.agent_processor = None
        self.filter_agent = None

        self.is_processing = False
        self.stop_flag = False
        self.resume_index = 0
        self.processing_start_index = 0
        self.found_files: list[Path] = []
        self.invoices: list[Invoice] = []
        self.filtered_invoices: list[Invoice] = []
        self.review_queue: list[Invoice] = []
        self.selected_invoices: list[Invoice] = []
        self._last_selected_idx = None

        # Mapování
        self.sort_key_map = {
            "Datum vystavení": "issue_date",
            "Datum splatnosti": "due_date",
            "Odesílatel": "sender_name",
            "Příjemce": "recipient_name",
            "Částka": "total_amount",
        }

        self.filter_key_map = {
            "Vypnuto": "",
            "Odesílatel": "sender_name",
            "Příjemce": "recipient_name",
            "Datum vystavení": "issue_date",
            "Číslo faktury": "invoice_number",
        }

        self.decision_type_map = {
            "Všechny": "all",
            "Auto-Accept": "auto_accept",
            "Human Review": "human_review",
            "Auto-Reject": "auto_reject",
        }

        # Check components
        self.ocr_ready = self._check_ocr_status()
        self.agents_ready = AGENTS_AVAILABLE
        self.ollama_ready = self._check_and_start_ollama()
        self.agents_ready = self.agents_ready and self.ollama_ready

        self._setup_ui()

    def _check_ocr_status(self) -> bool:
        """Zkontroluje zda je RapidOCR připraven."""
        if self.ocr_extractor.rapidocr is None:
            logger.warning("RapidOCR není inicializován")
            return False

        logger.info("✓ RapidOCR připraven")
        return True

    def _check_and_start_ollama(self) -> bool:
        """Zkontroluje zda Ollama běží, pokud ne pokusí se ji spustit."""
        logger.info("🔍 Kontrola Ollama...")

        if check_ollama_running():
            logger.info("✓ Ollama již běží")
            if check_ollama_model(self.model_name.get()):
                logger.info(f"✓ Model {self.model_name.get()} je dostupný")
                return True
            else:
                logger.warning(f"⚠️ Model {self.model_name.get()} není stažen")
                return self._pull_ollama_model(self.model_name.get())

        logger.info("⚠️ Ollama neběží - pokus o spuštění...")
        if try_start_ollama():
            if check_ollama_model(self.model_name.get()):
                logger.info(f"✓ Model {self.model_name.get()} je dostupný")
                return True
            else:
                logger.warning(f"⚠️ Model {self.model_name.get()} není stažen")
                return self._pull_ollama_model(self.model_name.get())
        else:
            logger.error("✗ Nepodařilo se spustit Ollama")
            return False

    def _pull_ollama_model(self, model_name: str) -> bool:
        """Pokusí se stáhnout model z Ollama library."""
        logger.info(f"📥 Stahuji model {model_name}...")

        ollama_paths = [
            r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe",
            r"C:\Program Files\Ollama\ollama.exe",
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
            logger.error("✗ Ollama executable nenalezen")
            return False

        try:
            process = subprocess.Popen(
                [ollama_exe, "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            try:
                stdout, stderr = process.communicate(timeout=300)
                if process.returncode == 0:
                    logger.info(f"✓ Model {model_name} úspěšně stažen")
                    return True
                else:
                    logger.error(f"✗ Chyba při stahování modelu: {stderr}")
                    return False
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"✗ Timeout při stahování modelu {model_name}")
                return False

        except Exception as e:
            logger.error(f"✗ Chyba při stahování modelu: {e}")
            return False

    def _setup_ui(self):
        """Setup GUI components."""
        self.geometry("1400x950")
        self.minsize(1200, 850)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main scrollable container
        self.main_scroll_frame = ctk.CTkScrollableFrame(self, orientation="vertical")
        self.main_scroll_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.main_scroll_frame.grid_columnconfigure(0, weight=1)

        content_parent = self.main_scroll_frame

        # Header
        header_frame = ctk.CTkFrame(content_parent, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header_frame.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="🧾 Invoice Processor",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w")

        # Status
        status_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        status_frame.grid(row=1, column=0, sticky="w", pady=(5, 0))

        if self.agents_ready:
            agent_status = "✓ Multi-Agent připraven"
            agent_color = "#28a745"
        else:
            agent_status = "✗ Agent workflow chybí"
            agent_color = "#dc3545"

        if self.ocr_ready:
            ocr_status = "✓ RapidOCR připraven"
            ocr_color = "#28a745"
        else:
            ocr_status = "✗ RapidOCR nenalezen"
            ocr_color = "#dc3545"

        if self.ollama_ready:
            ollama_status = "✓ Ollama běží"
            ollama_color = "#28a745"
        else:
            ollama_status = "✗ Ollama neběží"
            ollama_color = "#dc3545"

        ctk.CTkLabel(
            status_frame,
            text="🤖 3 specializovaní AI agenti • Vážené hlasování • Detekce anomálií",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            status_frame,
            text=f"  {agent_status}  |  {ocr_status}  |  {ollama_status}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=agent_color if self.agents_ready else ollama_color
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))

        # Directory settings
        settings_frame = ctk.CTkFrame(content_parent)
        settings_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        settings_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings_frame, text="📁 Zdrojová složka:").grid(
            row=0, column=0, sticky="w", padx=(15, 10), pady=10
        )
        ctk.CTkEntry(settings_frame, textvariable=self.source_dir, height=35).grid(
            row=0, column=1, sticky="ew", padx=10, pady=10
        )
        ctk.CTkButton(settings_frame, text="Procházet", command=self._browse_source, width=100).grid(
            row=0, column=2, padx=15, pady=10
        )

        ctk.CTkLabel(settings_frame, text="📂 Cílová složka:").grid(
            row=1, column=0, sticky="w", padx=(15, 10), pady=10
        )
        ctk.CTkEntry(settings_frame, textvariable=self.target_dir, height=35).grid(
            row=1, column=1, sticky="ew", padx=10, pady=10
        )
        ctk.CTkButton(settings_frame, text="Procházet", command=self._browse_target, width=100).grid(
            row=1, column=2, padx=15, pady=10
        )

        # Memory/VRAM Settings Panel
        memory_frame = ctk.CTkFrame(content_parent)
        memory_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        memory_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            memory_frame,
            text="⚙️ Nastavení Paměti a Výkonu",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, columnspan=5, pady=(0, 10))

        # Profile
        ctk.CTkLabel(memory_frame, text="Profil:", width=100).grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.profile_var = ctk.StringVar(value=self.memory_settings['profile'])
        profile_combo = ctk.CTkOptionMenu(
            memory_frame, variable=self.profile_var,
            values=['conservative', 'balanced', 'aggressive', 'maximum'],
            command=self._on_profile_change, width=180
        )
        profile_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # OCR RAM
        ctk.CTkLabel(memory_frame, text="OCR RAM (MB):", width=100).grid(row=1, column=2, sticky="w", padx=10, pady=5)
        self.ocr_memory_var = ctk.IntVar(value=self.memory_settings['ocr_memory_mb'])
        ocr_slider = ctk.CTkSlider(
            memory_frame, from_=512, to=8192, number_of_steps=15,
            variable=self.ocr_memory_var, command=self._update_memory_labels, width=200
        )
        ocr_slider.grid(row=1, column=3, sticky="ew", padx=5, pady=5)
        self.ocr_memory_label = ctk.CTkLabel(
            memory_frame, text=f"{self.memory_settings['ocr_memory_mb']} MB", width=70
        )
        self.ocr_memory_label.grid(row=1, column=4, padx=5, pady=5)

        # VRAM
        ctk.CTkLabel(memory_frame, text="Ollama VRAM (GB):", width=100).grid(
            row=2, column=0, sticky="w", padx=10, pady=5
        )
        self.vram_var = ctk.IntVar(value=self.memory_settings['ollama_vram_gb'])
        vram_slider = ctk.CTkSlider(
            memory_frame, from_=2, to=24, number_of_steps=11,
            variable=self.vram_var, command=self._update_memory_labels, width=200
        )
        vram_slider.grid(row=2, column=1, sticky="ew", padx=5, pady=5, columnspan=2)
        self.vram_label = ctk.CTkLabel(
            memory_frame, text=f"{self.memory_settings['ollama_vram_gb']} GB", width=70
        )
        self.vram_label.grid(row=2, column=3, padx=5, pady=5, columnspan=2)

        # GC Threshold
        ctk.CTkLabel(memory_frame, text="GC Threshold (%):", width=100).grid(
            row=3, column=0, sticky="w", padx=10, pady=5
        )
        self.threshold_var = ctk.IntVar(value=self.memory_settings['memory_threshold'])
        threshold_slider = ctk.CTkSlider(
            memory_frame, from_=30, to=90, number_of_steps=12,
            variable=self.threshold_var, command=self._update_memory_labels, width=200
        )
        threshold_slider.grid(row=3, column=1, sticky="ew", padx=5, pady=5, columnspan=2)
        self.threshold_label = ctk.CTkLabel(
            memory_frame, text=f"{self.memory_settings['memory_threshold']}%", width=70
        )
        self.threshold_label.grid(row=3, column=3, padx=5, pady=5, columnspan=2)

        # Save & Display
        save_btn = ctk.CTkButton(
            memory_frame, text="💾 Uložit", command=self._save_memory_settings, width=120
        )
        save_btn.grid(row=4, column=0, padx=10, pady=10)
        self.memory_display_label = ctk.CTkLabel(
            memory_frame, text="💾 Paměť: -- MB (--%)", text_color="gray", width=200
        )
        self.memory_display_label.grid(row=4, column=1, padx=10, pady=10, columnspan=2)

        # Help
        help_label = ctk.CTkLabel(
            memory_frame,
            text="✅ VRAM ovlivňuje num_ctx | OCR RAM ovlivňuje počet vláken",
            text_color="gray", font=ctk.CTkFont(size=11)
        )
        help_label.grid(row=7, column=0, columnspan=5, padx=10, pady=(0, 5))

        self._update_memory_labels()
        self._monitor_memory()

        # Filters and sorting
        filter_frame = ctk.CTkFrame(content_parent)
        filter_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
        filter_frame.grid_columnconfigure(1, weight=1)
        filter_frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(filter_frame, text="🔍 Filtr:").grid(
            row=0, column=0, sticky="w", padx=(15, 5), pady=10
        )

        self.filter_key_combo = ctk.CTkComboBox(
            filter_frame,
            values=list(self.filter_key_map.keys()),
            variable=self.filter_key,
            width=150,
            command=self._on_filter_key_change
        )
        self.filter_key_combo.grid(row=0, column=1, sticky="w", padx=5, pady=10)
        self.filter_key_combo.set("Vypnuto")

        self.filter_value_entry = ctk.CTkEntry(
            filter_frame,
            textvariable=self.filter_value,
            width=200,
            placeholder_text="Hledaný výraz..."
        )
        self.filter_value_entry.grid(row=0, column=2, sticky="w", padx=10, pady=10)
        self.filter_value_entry.bind("<KeyRelease>", self._on_filter_change)

        # Decision type filter
        ctk.CTkLabel(filter_frame, text="📋 Rozhodnutí:").grid(
            row=0, column=3, sticky="w", padx=(20, 5), pady=10
        )

        self.decision_type_var = ctk.StringVar(value="Všechny")
        self.decision_type_combo = ctk.CTkComboBox(
            filter_frame,
            values=list(self.decision_type_map.keys()),
            variable=self.decision_type_var,
            width=150,
            command=self._on_decision_type_change
        )
        self.decision_type_combo.grid(row=0, column=4, sticky="w", padx=5, pady=10)

        # Action buttons
        action_frame = ctk.CTkFrame(content_parent)
        action_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=10)

        self.process_btn = ctk.CTkButton(
            action_frame,
            text="▶️ Spustit analýzu",
            command=self._start_processing,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#28a745",
            hover_color="#218838"
        )
        self.process_btn.pack(side="left", padx=10, pady=10)

        self.stop_btn = ctk.CTkButton(
            action_frame,
            text="⏹️ Zastavit",
            command=self._stop_processing,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#dc3545",
            hover_color="#c82333",
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=10, pady=10)

        self.resume_btn = ctk.CTkButton(
            action_frame,
            text="▶️ Pokračovat",
            command=self._resume_processing,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#007bff",
            hover_color="#0056b3",
            state="disabled"
        )
        self.resume_btn.pack(side="left", padx=10, pady=10)

        self.move_btn = ctk.CTkButton(
            action_frame,
            text="📂 Přesunout označené",
            command=self._move_selected_invoices,
            height=40,
            fg_color="#007bff",
            hover_color="#0056b3",
            state="disabled"
        )
        self.move_btn.pack(side="left", padx=10, pady=10)

        self.select_all_btn = ctk.CTkButton(
            action_frame,
            text="✅ Označit vše",
            command=self._select_all_invoices,
            height=40,
            fg_color="#6c757d",
            hover_color="#5a6268",
            state="disabled"
        )
        self.select_all_btn.pack(side="left", padx=10, pady=10)

        self.deselect_all_btn = ctk.CTkButton(
            action_frame,
            text="❌ Zrušit výběr",
            command=self._deselect_all_invoices,
            height=40,
            fg_color="#6c757d",
            hover_color="#5a6268",
            state="disabled"
        )
        self.deselect_all_btn.pack(side="left", padx=10, pady=10)

        # Progress
        self.progress_frame = ctk.CTkFrame(content_parent)
        self.progress_frame.grid(row=5, column=0, sticky="ew", padx=20, pady=10)
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="Připraven k analýze",
            font=ctk.CTkFont(size=14)
        )
        self.progress_label.grid(row=0, column=0, sticky="w", padx=15, pady=5)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=20, corner_radius=10)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 10))
        self.progress_bar.set(0)

        # Statistics
        self.stats_frame = ctk.CTkFrame(content_parent)
        self.stats_frame.grid(row=6, column=0, sticky="ew", padx=20, pady=10)
        self.stats_frame.grid_columnconfigure(0, weight=1)
        self.stats_frame.grid_columnconfigure(1, weight=1)
        self.stats_frame.grid_columnconfigure(2, weight=1)
        self.stats_frame.grid_columnconfigure(3, weight=1)
        self.stats_frame.grid_columnconfigure(4, weight=1)

        self.stat_labels = {}
        stats_config = [
            ("total", "📄 Celkem", 0),
            ("invoices", "✅ Faktury", 1),
            ("non_invoices", "❌ Odmítnuté", 2),
            ("review", "⚠️ Review", 3),
            ("errors", "⛔ Chyby", 4),
        ]

        for key, label, col in stats_config:
            frame = ctk.CTkFrame(self.stats_frame, fg_color="#2b2b2b")
            frame.grid(row=0, column=col, sticky="nsew", padx=5, pady=5)
            frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                frame,
                text=label,
                font=ctk.CTkFont(size=12),
                text_color="gray"
            ).grid(row=0, column=0, padx=10, pady=(10, 0))

            self.stat_labels[key] = ctk.CTkLabel(
                frame,
                text="0",
                font=ctk.CTkFont(size=24, weight="bold")
            )
            self.stat_labels[key].grid(row=1, column=0, padx=10, pady=(0, 10))

        # Results table
        result_frame = ctk.CTkFrame(content_parent)
        result_frame.grid(row=7, column=0, sticky="nsew", padx=20, pady=(0, 20))
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_rowconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")

        columns = ("file", "sender", "recipient", "date", "amount", "decision", "confidence")
        self.tree = ttk.Treeview(
            result_frame,
            columns=columns,
            show="headings",
            height=15,
            selectmode="extended"
        )

        self.tree.heading("file", text="Soubor")
        self.tree.heading("sender", text="Odesílatel")
        self.tree.heading("recipient", text="Příjemce")
        self.tree.heading("date", text="Datum")
        self.tree.heading("amount", text="Částka")
        self.tree.heading("decision", text="Rozhodnutí")
        self.tree.heading("confidence", text="Jistota")

        self.tree.column("file", width=200)
        self.tree.column("sender", width=150)
        self.tree.column("recipient", width=150)
        self.tree.column("date", width=100)
        self.tree.column("amount", width=100)
        self.tree.column("decision", width=100)
        self.tree.column("confidence", width=80)

        scrollbar = ctk.CTkScrollbar(result_frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.tree.bind("<Double-1>", self._show_document_detail)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    # ─────────────────────────────────────────────
    # Memory/VRAM Settings Methods
    # ─────────────────────────────────────────────
    def _on_profile_change(self, profile):
        """Apply performance profile."""
        profiles = {
            'conservative': {'ocr_memory_mb': 1024, 'ollama_vram_gb': 2, 'memory_threshold': 70},
            'balanced': {'ocr_memory_mb': 2048, 'ollama_vram_gb': 4, 'memory_threshold': 60},
            'aggressive': {'ocr_memory_mb': 4096, 'ollama_vram_gb': 8, 'memory_threshold': 50},
            'maximum': {'ocr_memory_mb': 8192, 'ollama_vram_gb': 16, 'memory_threshold': 40}
        }
        if profile in profiles:
            settings = profiles[profile]
            self.ocr_memory_var.set(settings['ocr_memory_mb'])
            self.vram_var.set(settings['ollama_vram_gb'])
            self.threshold_var.set(settings['memory_threshold'])
            self._update_memory_labels()

    def _update_memory_labels(self, value=None):
        """Update slider value labels."""
        self.ocr_memory_label.configure(text=f"{self.ocr_memory_var.get()} MB")
        self.vram_label.configure(text=f"{self.vram_var.get()} GB")
        self.threshold_label.configure(text=f"{self.threshold_var.get()}%")

    def _monitor_memory(self):
        """Update memory usage display periodically."""
        try:
            mem = get_memory_usage()
            if self.memory_display_label:
                self.memory_display_label.configure(
                    text=f"💾 Paměť: {mem['rss_mb']:.0f} MB ({mem['percent']:.1f}%)"
                )
        except:
            pass
        if hasattr(self, 'after_id'):
            self.after_cancel(self.after_id)
        self.after_id = self.after(2000, self._monitor_memory)

    def _save_memory_settings(self):
        """Save memory settings to file."""
        self.memory_settings = {
            'ocr_memory_mb': self.ocr_memory_var.get(),
            'ollama_vram_gb': self.vram_var.get(),
            'memory_threshold': self.threshold_var.get(),
            'profile': self.profile_var.get()
        }
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory_settings, f, indent=2)
        logger.info(f"💾 Memory settings saved: {self.memory_settings}")

        vram_gb = self.memory_settings['ollama_vram_gb']
        num_ctx = calculate_num_ctx(vram_gb)
        ocr_threads = max(1, min(8, self.memory_settings['ocr_memory_mb'] // 512))

        messagebox.showinfo(
            "Nastavení Uloženo",
            f"✅ Nastavení bylo uloženo.\n\n"
            f"📊 Ollama VRAM: {self.memory_settings['ollama_vram_gb']} GB\n"
            f"   → num_ctx: {num_ctx} tokenů\n"
            f"   → Ovlivňuje velikost kontextu pro LLM\n\n"
            f"📊 OCR RAM: {self.memory_settings['ocr_memory_mb']} MB\n"
            f"   → Počet vláken: {ocr_threads}\n"
            f"   → Ovlivňuje rychlost OCR zpracování\n\n"
            f"📊 GC Threshold: {self.memory_settings['memory_threshold']}%\n"
            f"   → Spustí GC když paměť překročí tuto hodnotu\n\n"
            f"⚠️ Změny se projeví po restartu aplikace!"
        )

    def _load_memory_settings(self):
        """Load memory settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.memory_settings = json.load(f)
                logger.info(f"📂 Memory settings loaded: {self.memory_settings}")
            except Exception as e:
                logger.warning(f"Error loading settings: {e}")
                self.memory_settings = {
                    'ocr_memory_mb': 2048,
                    'ollama_vram_gb': 4,
                    'memory_threshold': 60,
                    'profile': 'balanced'
                }

    # ─────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────
    def _browse_source(self):
        directory = filedialog.askdirectory(title="Vyberte zdrojovou složku")
        if directory:
            self.source_dir.set(directory)

    def _browse_target(self):
        directory = filedialog.askdirectory(title="Vyberte cílovou složku")
        if directory:
            self.target_dir.set(directory)

    def _on_filter_key_change(self, event=None):
        selected = self.filter_key_combo.get()
        self.filter_key.set(self.filter_key_map.get(selected, ""))
        self._apply_filters()

    def _on_filter_change(self, event=None):
        self._apply_filters()

    def _on_tree_select(self, event=None):
        """Obsluha změny výběru v tabulce."""
        selection = self.tree.selection()
        self.selected_invoices.clear()

        for item_id in selection:
            item = self.tree.item(item_id)
            filename = item['values'][0]

            for inv in self.filtered_invoices:
                if inv.source_path.name == filename:
                    self.selected_invoices.append(inv)
                    break

        count = len(self.selected_invoices)
        if count > 0:
            self.move_btn.configure(text=f"📂 Přesunout označené ({count})")
        else:
            self.move_btn.configure(text="📂 Přesunout označené")

    def _select_all_invoices(self):
        """Označit všechny faktury v aktuálním filtru."""
        all_items = self.tree.get_children()

        for inv in self.filtered_invoices:
            if inv.is_invoice:
                for item_id in all_items:
                    item = self.tree.item(item_id)
                    if item['values'][0] == inv.source_path.name:
                        self.tree.selection_add(item_id)
                        break

        self._on_tree_select()

    def _deselect_all_invoices(self):
        """Zrušit výběr všech faktur."""
        self.tree.selection_remove(self.tree.selection())
        self.selected_invoices.clear()
        self.move_btn.configure(text="📂 Přesunout označené")

    def _on_decision_type_change(self, event=None):
        self._apply_filters()

    def _apply_filters(self):
        """Aplikuje filtry na výsledky."""
        if not self.invoices:
            return

        decision_type = self.decision_type_map.get(self.decision_type_var.get(), "all")

        if decision_type == "all":
            filtered = self.invoices.copy()
        else:
            filtered = [inv for inv in self.invoices if inv.decision_type == decision_type]

        filter_key = self.filter_key.get()
        filter_value = self.filter_value.get()

        if filter_key and filter_value:
            final_filtered = []
            for inv in filtered:
                if inv.matches_filter(filter_key, filter_value):
                    final_filtered.append(inv)
            filtered = final_filtered

        self.filtered_invoices = filtered
        self._update_tree()

    def _update_tree(self):
        """Aktualizuje tabulku výsledků."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        for inv in self.filtered_invoices:
            if inv.decision_type == "auto_accept":
                tags = ("accept",)
            elif inv.decision_type == "human_review":
                tags = ("review",)
            elif inv.decision_type == "auto_reject":
                tags = ("reject",)
            else:
                tags = ()

            amount_str = f"{inv.total_amount} {inv.currency}" if inv.total_amount else "-"

            decision_display = {
                "auto_accept": "✅ Accept",
                "human_review": "⚠️ Review",
                "auto_reject": "❌ Reject",
                "error": "⛔ Error",
            }.get(inv.decision_type, inv.decision_type)

            self.tree.insert(
                "",
                "end",
                values=(
                    inv.source_path.name,
                    inv.sender_name[:30] + "..." if len(inv.sender_name) > 30 else inv.sender_name,
                    inv.recipient_name[:30] + "..." if len(inv.recipient_name) > 30 else inv.recipient_name,
                    inv.issue_date if inv.issue_date != "0000-00-00" else "-",
                    amount_str,
                    decision_display,
                    f"{inv.confidence:.0%}" if inv.confidence > 0 else "-"
                ),
                tags=tags
            )

        self.tree.tag_configure("accept", background="#28a745", foreground="white")
        self.tree.tag_configure("review", background="#ffc107", foreground="black")
        self.tree.tag_configure("reject", background="#dc3545", foreground="white")
        self.tree.tag_configure("selected", background="#007bff", foreground="white")

        has_invoices = any(inv.is_invoice for inv in self.filtered_invoices)
        self.select_all_btn.configure(state="normal" if has_invoices else "disabled")
        self.deselect_all_btn.configure(state="normal" if has_invoices else "disabled")
        self.move_btn.configure(state="normal" if has_invoices else "disabled")

    def _update_stats(self, stats: dict = None):
        """Aktualizuje statistiky."""
        if stats:
            self.stat_labels["total"].configure(text=str(stats.get('total', 0)))
            self.stat_labels["invoices"].configure(text=str(stats.get('invoices', 0)))
            self.stat_labels["non_invoices"].configure(text=str(stats.get('non_invoices', 0)))
            self.stat_labels["review"].configure(text=str(stats.get('human_review', 0)))
            self.stat_labels["errors"].configure(text=str(stats.get('errors', 0)))

    def _stop_processing(self):
        """Zastaví zpracování po dokončení aktuálního souboru."""
        if self.is_processing:
            self.stop_flag = True
            logger.info("⏹️ Zastavení zpracování požadováno...")
            self.progress_label.configure(text="⏳ Zastavuji po dokončení aktuálního souboru...")

    def _resume_processing(self):
        """Pokračuje ve zpracování od posledního zastaveného místa."""
        if not self.found_files or self.resume_index >= len(self.found_files):
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.processing_start_index = self.resume_index
        self.process_btn.configure(state="disabled", text="⏳ Zpracovávám...")
        self.stop_btn.configure(state="normal")
        self.resume_btn.configure(state="disabled")
        self.progress_label.configure(
            text=f"▶️ Pokračování od souboru {self.resume_index + 1}/{len(self.found_files)}..."
        )

        thread = threading.Thread(target=self._process_files, daemon=True)
        thread.start()

    def _start_processing(self):
        """Spustí zpracování na vlákně."""
        if self.is_processing:
            return

        source = self.source_dir.get()
        if not source or not Path(source).exists():
            messagebox.showerror("Chyba", "Zadejte platnou zdrojovou složku")
            return

        if not self.ollama_ready:
            messagebox.showerror(
                "Chyba",
                "Ollama neběží a nepodařilo se ji spustit.\n\n"
                "Možná řešení:\n"
                "1. Otevřete Ollama aplikaci a nechte ji běžet na pozadí\n"
                "2. Nainstalujte Ollama z: https://ollama.ai\n"
                "3. Spusťte příkaz: ollama serve"
            )
            return

        if not self.agents_ready:
            messagebox.showerror(
                "Chyba",
                "Agent workflow není dostupný.\n\n"
                "Nainstalujte: pip install ollama\n"
                "A spusťte: ollama pull llama3.2"
            )
            return

        self.stop_flag = False
        self.resume_index = 0
        self.processing_start_index = 0
        self.invoices = []
        self.review_queue = []

        # 🔒 THREAD-SAFETY: Read ALL Tkinter variables in the main thread
        # and cache them as plain Python attributes for the worker thread.
        # Tkinter/Tcl is NOT thread-safe — calling .get() from a background
        # thread can freeze or crash the entire application on Windows.
        self._cached_source_dir = source
        self._cached_vram_limit = self.vram_var.get() if hasattr(self, 'vram_var') else 4

        self.is_processing = True
        self.process_btn.configure(state="disabled", text="⏳ Zpracovávám...")
        self.stop_btn.configure(state="normal")
        self.resume_btn.configure(state="disabled")
        self.progress_bar.set(0)

        thread = threading.Thread(target=self._process_files, daemon=True)
        thread.start()

    def _process_files(self):
        """
        Hlavní zpracovací vlákno.
        
        Uses core engine for all processing:
        - FileDiscovery: Find files
        - OCRExtractor: Extract text
        - MultiAgentProcessor: Analyze document
        """
        try:
            log_memory_state("start")

            self.after(0, lambda: self.progress_label.configure(text="Inicializace agentů..."))

            if self.processing_start_index == 0:
                # 🔒 THREAD-SAFETY: Use cached values read in the main thread
                # instead of calling self.vram_var.get() here (Tkinter is NOT thread-safe)
                vram_limit = self._cached_vram_limit
                num_ctx = calculate_num_ctx(vram_limit)

                logger.info(f"Inicializace agentů: VRAM={vram_limit}GB, num_ctx={num_ctx}")
                self.agent_processor = MultiAgentProcessor(
                    vram_limit_gb=vram_limit,
                    num_ctx=num_ctx
                )
                self.filter_agent = InvoiceFilterAgent([])

            self.after(0, lambda: self.progress_label.configure(text="Vyhledávání souborů..."))

            if not self.found_files or self.processing_start_index == 0:
                # 🔒 THREAD-SAFETY: Use cached source_dir instead of self.source_dir.get()
                discovery = FileDiscovery(Path(self._cached_source_dir))
                self.found_files = discovery.find_files()

            total = len(self.found_files)
            if total == 0:
                self.after(0, lambda: messagebox.showinfo("Info", "Žádné soubory nenalezeny"))
                return

            if self.processing_start_index == 0:
                self.invoices = []
                self.review_queue = []

            start_index = self.processing_start_index
            if start_index >= total:
                self.after(0, lambda: messagebox.showinfo("Info", "Všechny soubory již byly zpracovány"))
                return

            # Zpracování každého souboru
            for i in range(start_index, total):
                if self.stop_flag:
                    logger.info(f"⏹️ Zastaveno po zpracování {i} souborů z {total}")
                    self.resume_index = i
                    self.after(
                        0,
                        lambda: (
                            self.progress_label.configure(text=f"⏹️ Zastaveno po {i}/{total} souborech"),
                            self._processing_stopped()
                        )
                    )
                    return

                file_path = self.found_files[i]
                try:
                    progress = (i + 1) / total
                    self.after(
                        0,
                        lambda p=progress, f=file_path: (
                            self.progress_bar.set(p),
                            self.progress_label.configure(text=f"[{i + 1}/{total}] {f.name}")
                        )
                    )

                    # Use core engine process_invoice function
                    from core import process_invoice
                    
                    invoice = process_invoice(
                        file_path=file_path,
                        ocr_extractor=self.ocr_extractor,
                        agent_processor=self.agent_processor,
                        save_raw=SAVE_RAW_TEXT
                    )

                    if invoice:
                        self.invoices.append(invoice)

                        if invoice.requires_review:
                            self.review_queue.append(invoice)
                    else:
                        logger.warning(f"  ⚠️ process_invoice vrátil None pro: {file_path.name}")

                    # Memory management
                    if (i + 1) % MEMORY_CHECK_EVERY == 0:
                        mem = get_memory_usage()
                        logger.debug(f"💾 Paměť [{i + 1}/{total}]: RSS={mem['rss_mb']:.1f}MB ({mem['percent']:.1f}%)")

                        if mem['percent'] > MAX_MEMORY_PERCENT:
                            self.after(0, lambda: self.progress_label.configure(text="⚠️ Uvolňování paměti..."))
                            gc.collect()
                            log_memory_state("after_gc")

                    if (i + 1) % GC_EVERY == 0:
                        gc.collect()
                        mem = get_memory_usage()
                        logger.info(f"💾 GC po {i + 1} souborech: RSS={mem['rss_mb']:.1f}MB ({mem['percent']:.1f}%)")

                except Exception as file_error:
                    logger.error(f"  ✗ Chyba při zpracování {file_path.name}: {file_error}")
                    import traceback
                    logger.error(f"  Traceback: {traceback.format_exc()}")
                    invoice = Invoice(
                        source_path=file_path,
                        is_invoice=False,
                        confidence=0.0,
                        decision_type='error'
                    )
                    invoice.raw_json = {'error': str(file_error)}
                    self.invoices.append(invoice)
                    continue

            gc.collect()
            log_memory_state("before_finalization")

            stats = self.agent_processor.get_statistics()
            self.after(0, lambda: self._update_stats(stats))

            self.filtered_invoices = self.invoices.copy()
            self.after(0, self._update_tree)

            self.after(
                0,
                lambda: (
                    self.progress_bar.set(1),
                    self.progress_label.configure(text="✅ Analýza dokončena"),
                    self._show_summary(stats)
                )
            )

        except Exception as e:
            logger.error(f"Chyba zpracování: {e}")
            import traceback
            error_msg = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("Chyba", f"Chyba zpracování:\n{error_msg}"))

        finally:
            if not self.stop_flag:
                self.is_processing = False
                self.processing_start_index = 0
                self.resume_index = 0
                self.after(0, lambda: (
                    self.process_btn.configure(state="normal", text="▶️ Spustit analýzu"),
                    self.stop_btn.configure(state="disabled"),
                    self.resume_btn.configure(state="disabled"),
                    log_memory_state("end")
                ))

    def _processing_stopped(self):
        """Called when processing is stopped by user."""
        self.is_processing = False
        self.stop_flag = False

        if self.agent_processor:
            stats = self.agent_processor.get_statistics()
            self.after(0, lambda: self._update_stats(stats))

        self.filtered_invoices = self.invoices.copy()
        self.after(0, self._update_tree)

        self.process_btn.configure(state="normal", text="▶️ Spustit analýzu")
        self.stop_btn.configure(state="disabled")
        if self.resume_index < len(self.found_files):
            self.resume_btn.configure(state="normal")

        processed = len(self.invoices)
        remaining = len(self.found_files) - self.resume_index
        self.after(
            0,
            lambda: messagebox.showinfo(
                "Zastaveno",
                f"Zpracování zastaveno.\n\n"
                f"✅ Zpracováno: {processed} souborů\n"
                f"⏳ Zbývá: {remaining} souborů\n\n"
                f"Můžete pokračovat kliknutím na '▶️ Pokračovat'"
            )
        )

        logger.info(f"✅ Zpracování zastaveno. Lze pokračovat od souboru {self.resume_index + 1}.")

    def _show_summary(self, stats: dict):
        """Zobrazí shrnutí výsledků."""
        total = stats.get('total', 0)
        if total == 0:
            return

        invoices = stats.get('invoices', 0)
        non_invoices = stats.get('non_invoices', 0)
        review = stats.get('human_review', 0)

        summary = (
            f"📊 Výsledky analýzy\n\n"
            f"Celkem dokumentů: {total}\n"
            f"✅ Faktury: {invoices} ({invoices/total*100:.1f}%)\n"
            f"❌ Odmítnuté: {non_invoices} ({non_invoices/total*100:.1f}%)\n"
            f"⚠️ K revizi: {review} ({review/total*100:.1f}%)\n\n"
            f"Multi-Agent workflow dokončil analýzu."
        )

        messagebox.showinfo("Shrnutí", summary)

    def _move_selected_invoices(self):
        """Přesune pouze označené faktury."""
        if not self.selected_invoices:
            messagebox.showinfo(
                "Info",
                "Nejdříve označte faktury které chcete přesunout.\n\n"
                "Výběr provedete:\n"
                "• Kliknutím na jednotlivé řádky\n"
                "• Shift+Klik pro výběr rozsahu\n"
                "• Ctrl+Klik pro výběr více řádků\n"
                "• Tlačítkem '✅ Označit vše'"
            )
            return

        target = self.target_dir.get()
        if not target:
            messagebox.showerror("Chyba", "Zadejte cílovou složku")
            return

        to_move = [inv for inv in self.selected_invoices if inv.is_invoice]

        if not to_move:
            messagebox.showinfo("Info", "Žádné označené faktury k přesunu")
            return

        if messagebox.askyesno(
            "Potvrzení",
            f"Přesunout {len(to_move)} označených faktur do:\n{target}?"
        ):
            organizer = InvoiceOrganizer(to_move, Path(target))
            success, errors = organizer.process("issue_date")

            for inv in to_move:
                if inv in self.selected_invoices:
                    self.selected_invoices.remove(inv)

            self._apply_filters()
            self._on_tree_select()

            messagebox.showinfo(
                "Hotovo",
                f"Přesunuto: {success}\nChyby: {errors}"
            )

    def _move_invoices(self):
        """Přesune identifikované faktury (legacy funkce)."""
        self._move_selected_invoices()

    def _show_document_detail(self, event):
        """Zobrazí detail dokumentu po double-click."""
        selection = self.tree.selection()
        if not selection:
            return

        item = self.tree.item(selection[0])
        filename = item['values'][0]

        invoice = None
        for inv in self.filtered_invoices:
            if inv.source_path.name == filename:
                invoice = inv
                break

        if not invoice:
            return

        detail_window = ctk.CTkToplevel(self)
        detail_window.title(f"📄 Detail: {filename}")
        detail_window.geometry("800x600")

        scroll_frame = ctk.CTkScrollableFrame(detail_window)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            scroll_frame,
            text="📋 Základní informace",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", pady=(0, 10))

        info_text = (
            f"Soubor: {invoice.source_path.name}\n"
            f"Odesílatel: {invoice.sender_name or '-'}\n"
            f"Adresa odesílatele: {getattr(invoice, 'sender_address', '') or '-'}\n"
            f"Příjemce: {invoice.recipient_name or '-'}\n"
            f"Adresa příjemce: {getattr(invoice, 'recipient_address', '') or '-'}\n"
            f"Datum vystavení: {invoice.issue_date or '-'}\n"
            f"Datum splatnosti: {invoice.due_date or '-'}\n"
            f"Částka: {invoice.total_amount or '-'} {invoice.currency or ''}\n"
            f"Číslo faktury: {invoice.invoice_number or '-'}\n"
        )
        ctk.CTkLabel(scroll_frame, text=info_text, justify="left").pack(anchor="w", pady=5)

        ctk.CTkLabel(
            scroll_frame,
            text="🗳️ Rozhodnutí",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", pady=(20, 10))

        decision_text = (
            f"Typ: {invoice.decision_type}\n"
            f"Jistota: {invoice.confidence:.0%}\n"
            f"Je faktura: {'Ano' if invoice.is_invoice else 'Ne'}"
        )
        ctk.CTkLabel(scroll_frame, text=decision_text, justify="left").pack(anchor="w", pady=5)

        if invoice.agent_results:
            ctk.CTkLabel(
                scroll_frame,
                text="🤖 Výsledky agentů",
                font=ctk.CTkFont(size=18, weight="bold")
            ).pack(anchor="w", pady=(20, 10))

            for agent_name, result in invoice.agent_results.items():
                ctk.CTkLabel(
                    scroll_frame,
                    text=f"\n{agent_name.upper()}:",
                    font=ctk.CTkFont(weight="bold")
                ).pack(anchor="w")

                result_json = json.dumps(result, indent=2, ensure_ascii=False)
                text_widget = ctk.CTkTextbox(scroll_frame, height=100, wrap="word")
                text_widget.insert("0.0", result_json)
                text_widget.configure(state="disabled")
                text_widget.pack(fill="x", pady=5)

            if invoice.consensus_result.get('reasoning'):
                ctk.CTkLabel(
                    scroll_frame,
                    text="\n💡 Konsenzus:",
                    font=ctk.CTkFont(weight="bold")
                ).pack(anchor="w", pady=(10, 5))
                ctk.CTkLabel(
                    scroll_frame,
                    text=invoice.consensus_result['reasoning'],
                    justify="left",
                    wraplength=750
                ).pack(anchor="w", pady=5)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    if not CORE_AVAILABLE:
        print("⚠️ Core engine module not available!")
        print("\nInstalace:")
        print("  pip install ollama rapidocr_onnxruntime")
        print("  ollama pull llama3.2")
        sys.exit(1)

    if not AGENTS_AVAILABLE:
        print("⚠️ Agent workflow module not available!")
        print("\nInstalace:")
        print("  pip install ollama")
        print("  ollama pull llama3.2")
        sys.exit(1)

    try:
        app = InvoiceProcessorGUIV6()
        app.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Ukončeno uživatelem")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
