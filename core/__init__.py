"""
Core Invoice Processing Engine

This module contains the business logic for invoice processing,
separated from the GUI layer. It can be used by GUI, CLI, or API.
"""

from .engine import (
    # Classes
    Invoice,
    OCRExtractor,
    MultiAgentProcessor,
    FileDiscovery,
    InvoiceFilterAgent,
    InvoiceOrganizer,

    # Functions
    process_invoice,
    get_memory_usage,
    log_memory_state,
    save_raw_text,
    check_ollama_running,
    try_start_ollama,
    check_ollama_model,
    calculate_num_ctx,
    _set_ollama_privacy_mode,  # 🔒 Privacy settings

    # Constants
    SUPPORTED_EXTENSIONS,
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

__all__ = [
    # Classes
    "Invoice",
    "OCRExtractor",
    "MultiAgentProcessor",
    "FileDiscovery",
    "InvoiceFilterAgent",
    "InvoiceOrganizer",

    # Functions
    "process_invoice",
    "get_memory_usage",
    "log_memory_state",
    "save_raw_text",
    "check_ollama_running",
    "try_start_ollama",
    "check_ollama_model",
    "calculate_num_ctx",
    "_set_ollama_privacy_mode",  # 🔒 Privacy settings

    # Constants
    "SUPPORTED_EXTENSIONS",
    "AGENTS_AVAILABLE",
    "MAX_MEMORY_PERCENT",
    "DEBUG_MEMORY",
    "MEMORY_CHECK_EVERY",
    "GC_EVERY",
    "SAVE_RAW_TEXT",
    "RAW_TEXT_DIR",
    "TEXT_MODEL",
    "OCR_ZOOM",
]
