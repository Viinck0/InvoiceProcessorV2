"""
Agent Workflow for Invoice Classification
Multi-agent system for maximum accuracy

Includes OCR text validation to detect and fix hallucinated words from Tesseract.
Includes LangGraph orchestration for structured workflow management.
"""

from .base_agent import BaseAgent
from .classifier_agent import ClassifierAgent
from .extractor_agent import ExtractorAgent
from .anomaly_agent import AnomalyDetectorAgent
from .consensus_engine import ConsensusEngine
from .ocr_validator import OCRTextValidator, validate_ocr_text
from .langgraph_workflow import (
    InvoiceWorkflowState,
    InvoiceWorkflowNodes,
    build_invoice_workflow,
    SimpleInvoicePipeline,
    create_invoice_processor
)

__all__ = [
    'BaseAgent',
    'ClassifierAgent',
    'ExtractorAgent',
    'AnomalyDetectorAgent',
    'ConsensusEngine',
    'OCRTextValidator',
    'validate_ocr_text',
    'InvoiceWorkflowState',
    'InvoiceWorkflowNodes',
    'build_invoice_workflow',
    'SimpleInvoicePipeline',
    'create_invoice_processor',
]
