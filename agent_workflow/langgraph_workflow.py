"""
LangGraph Workflow for Invoice Processing
Orchestrace všech komponent pomocí LangChain/LangGraph

Tento modul poskytuje:
1. State management pro celý workflow
2. Node-based processing pipeline
3. Error handling a retry logic
4. Validace mezi kroky

Workflow:
OCR → Validator → Pre-Filter → Classifier/Extractor/Anomaly → Consensus → Result
"""

from typing import TypedDict, List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import logging
import concurrent.futures

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangGraph not installed. Install: pip install langgraph langchain-core")

from .ocr_validator import OCRTextValidator, validate_ocr_text
from .classifier_agent import ClassifierAgent
from .extractor_agent import ExtractorAgent
from .anomaly_agent import AnomalyDetectorAgent
from .consensus_engine import ConsensusEngine
from .agent_memory import initialize_memory, finalize_processing, get_shared_memory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────
class InvoiceWorkflowState(TypedDict):
    """
    State pro celý invoice processing workflow.
    
    All fields are optional to allow partial state updates.
    """
    # Input
    file_path: str
    raw_ocr_text: str
    master_instruction: str
    
    # After validation
    validated_text: str
    validation_result: dict
    text_blocks: list
    enriched_text: str
    
    # After pre-filter
    pre_filter_result: tuple  # (classification, confidence, reason)
    skip_ai_processing: bool
    
    # Agent results
    classifier_result: dict
    extractor_result: dict
    anomaly_result: dict
    
    # Final result
    consensus_result: dict
    is_invoice: bool
    confidence: float
    decision_type: str  # auto_accept, human_review, auto_reject, error
    extracted_data: dict
    
    # Error handling
    errors: List[str]
    iterations: int


# ─────────────────────────────────────────────
# Workflow Nodes
# ─────────────────────────────────────────────
class InvoiceWorkflowNodes:
    """
    Node-based processing steps for the workflow.
    
    Each node:
    - Takes the current state
    - Performs a specific task
    - Returns state updates (dict)
    """
    
    def __init__(
        self,
        classifier: ClassifierAgent,
        extractor: ExtractorAgent,
        anomaly: AnomalyDetectorAgent,
        consensus: ConsensusEngine,
        ocr_validator: OCRTextValidator
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.anomaly = anomaly
        self.consensus = consensus
        self.ocr_validator = ocr_validator
    
    def ocr_validator_node(self, state: InvoiceWorkflowState) -> InvoiceWorkflowState:
        """
        Node: Validate OCR text and fix hallucinations.
        """
        raw_text = state.get("raw_ocr_text", "")
        
        if not raw_text:
            return {"errors": ["No OCR text to validate"], "validated_text": ""}
        
        try:
            validated_text, validation_result = validate_ocr_text(raw_text, language="auto")
            
            logger.debug(f"✓ OCR Validation: {validation_result['valid_words']}/{validation_result['original_words']} valid")
            
            if validation_result['hallucinated_words'] > 0:
                logger.info(f"  Removed {validation_result['hallucinated_words']} hallucinated words")
            
            return {
                "validated_text": validated_text,
                "validation_result": validation_result,
                "text_blocks": validation_result.get("valid_blocks", [])
            }
        
        except Exception as e:
            logger.error(f"OCR Validator error: {e}")
            return {
                "errors": [f"OCR validation failed: {str(e)}"],
                "validated_text": raw_text,  # Use original text as fallback
                "validation_result": {"confidence": 0.0, "error": str(e)}
            }
    
    def pre_filter_node(self, state: InvoiceWorkflowState) -> InvoiceWorkflowState:
        """
        Node: Quick rule-based pre-filter before AI processing.
        """
        # Import here to avoid circular dependency
        from invoice_gui_v6 import OCRExtractor
        
        text = state.get("validated_text", "") or state.get("raw_ocr_text", "")
        
        if not text:
            return {"skip_ai_processing": True, "pre_filter_result": ('reject', 1.0, 'No text')}
        
        # Use OCRExtractor's pre_filter method
        # Create a temporary instance if needed
        extractor = OCRExtractor()
        pre_class, pre_conf, pre_reason = extractor.pre_filter(text)
        
        # Decide whether to skip AI processing
        skip = (pre_class == 'reject' and pre_conf > 0.9)
        
        logger.debug(f"✓ Pre-filter: {pre_class} (confidence: {pre_conf:.0%}, reason: {pre_reason})")
        
        return {
            "pre_filter_result": (pre_class, pre_conf, pre_reason),
            "skip_ai_processing": skip
        }
    
    def classifier_node(self, state: InvoiceWorkflowState) -> InvoiceWorkflowState:
        """
        Node: Run classifier agent.

        ARCHITEKTURA: Classifier dostává POUZE Markdown data (žádný surový text!)
        """
        text_blocks = state.get("text_blocks", [])
        markdown_input = state.get("enriched_text", "")

        if not markdown_input or state.get("skip_ai_processing"):
            return {"classifier_result": {"is_invoice": False, "confidence": 0.0, "reason": "Skipped"}}

        try:
            result = self.classifier.analyze(
                markdown_input,
                metadata=None,
                text_blocks=text_blocks,
                master_instruction=state.get("master_instruction")
            )
            logger.debug(f"✓ Classifier: {'invoice' if result.get('is_invoice') else 'not invoice'} ({result.get('confidence', 0):.0%})")
            return {"classifier_result": result}

        except Exception as e:
            logger.error(f"Classifier error: {e}")
            return {"classifier_result": {"is_invoice": False, "confidence": 0.0, "error": str(e)}}

    def extractor_node(self, state: InvoiceWorkflowState) -> InvoiceWorkflowState:
        """
        Node: Run extractor agent.

        ARCHITEKTURA: Extractor dostává POUZE Markdown data (žádný surový text!)
        """
        markdown_input = state.get("enriched_text", "")

        if not markdown_input or state.get("skip_ai_processing"):
            return {"extractor_result": {"completeness_score": 0.0, "validation_errors": ["Skipped"]}}

        try:
            result = self.extractor.analyze(
                markdown_input,
                metadata=None,
                master_instruction=state.get("master_instruction")
            )
            logger.debug(f"✓ Extractor: completeness={result.get('completeness_score', 0):.0%}")
            return {"extractor_result": result}

        except Exception as e:
            logger.error(f"Extractor error: {e}")
            return {"extractor_result": {"completeness_score": 0.0, "validation_errors": [str(e)]}}

    def anomaly_node(self, state: InvoiceWorkflowState) -> InvoiceWorkflowState:
        """
        Node: Run anomaly detector agent.

        ARCHITEKTURA: Anomaly detector dostává POUZE Markdown data (žádný surový text!)
        """
        markdown_input = state.get("enriched_text", "")

        if not markdown_input or state.get("skip_ai_processing"):
            return {"anomaly_result": {"is_anomaly": False, "confidence": 0.0}}

        try:
            result = self.anomaly.analyze(
                markdown_input,
                metadata=None,
                master_instruction=state.get("master_instruction")
            )
            logger.debug(f"✓ Anomaly: {'anomaly detected' if result.get('is_anomaly') else 'normal'} ({result.get('confidence', 0):.0%})")
            return {"anomaly_result": result}

        except Exception as e:
            logger.error(f"Anomaly detector error: {e}")
            return {"anomaly_result": {"is_anomaly": False, "confidence": 0.0, "error": str(e)}}

    def enrichment_node(self, state: InvoiceWorkflowState) -> InvoiceWorkflowState:
        """
        Node: Enrich text with Markdown table and spatial layout.

        ARCHITEKTURA: Vytváří POUZE strukturovaná data pro agenty (žádný surový text!)
        """
        blocks = state.get("text_blocks", [])

        if not blocks:
            logger.debug("⏭️ Skipping enrichment (no blocks)")
            return {"enriched_text": "", "master_instruction": ""}

        try:
            # Lazy import to avoid circular dependency
            from core.engine import format_blocks_to_markdown_table, generate_master_instruction
            md_table = format_blocks_to_markdown_table(blocks)
            master_instruction = generate_master_instruction(blocks)

            # Reconstruct spatial layout (using ClassifierAgent's method from BaseAgent)
            mapped_blocks = []
            for b in blocks:
                bbox = b.get('bbox', {})
                mapped_blocks.append({
                    'text': b.get('text', ''),
                    'x': float(bbox.get('x0', 0)),
                    'y': float(bbox.get('y0', 0))
                })

            spatial_layout = ""
            try:
                spatial_layout = "\n\n### 📜 Vizualizace dokumentu (Reconstructed Layout)\n"
                spatial_layout += "Toto je simulovaný vzhled stránky. Použij ho pro pochopení struktury.\n"
                spatial_layout += "```\n"
                spatial_layout += self.classifier.reconstruct_spatial_layout(mapped_blocks)
                spatial_layout += "\n```\n"
            except Exception as layout_err:
                logger.warning(f"  ⚠️ Layout reconstruction failed: {layout_err}")

            # ⚠️ ARCHITEKTURA: Pouze Markdown data pro agenty (žádný surový text!)
            enriched = f"{md_table}\n\n{spatial_layout}".strip()
            logger.debug(f"✓ Text enriched and master instruction generated")

            # Initialize shared memory with master instruction
            file_path = state.get("file_path", "unknown")
            try:
                memory = initialize_memory(file_path, master_instruction)
                logger.debug(f"🧠 Shared memory initialized for {file_path}")
            except Exception as mem_err:
                logger.warning(f"Failed to initialize shared memory: {mem_err}")

            return {
                "enriched_text": enriched,
                "master_instruction": master_instruction
            }

        except Exception as e:
            logger.error(f"Enrichment error: {e}")
            return {"enriched_text": "", "master_instruction": ""}
    
    def consensus_node(self, state: InvoiceWorkflowState) -> InvoiceWorkflowState:
        """
        Node: Calculate consensus from all agent results.
        """
        classifier_result = state.get("classifier_result", {})
        extractor_result = state.get("extractor_result", {})
        anomaly_result = state.get("anomaly_result", {})

        try:
            # Compare agent reasoning with master instruction before consensus
            try:
                memory = get_shared_memory()
                comparison_results = memory.compare_all_agents()
                validation_result = memory.validate_consistency()
                logger.debug(f"🔍 Agent reasoning comparison completed")
                
                # Log any significant contradictions
                for agent_name, comparison in comparison_results.items():
                    if comparison.get('contradictions'):
                        logger.warning(f"  ⚠️ {agent_name} has contradictions: {comparison['contradictions']}")
                    if comparison.get('missing_elements'):
                        logger.debug(f"  ℹ️ {agent_name} missing elements: {comparison['missing_elements']}")
                
                if validation_result.get('conflicts'):
                    logger.warning(f"  ⚠️ Agent consistency conflicts: {validation_result['conflicts']}")
                    
            except Exception as mem_err:
                logger.warning(f"Failed to compare agent reasoning: {mem_err}")

            consensus = self.consensus.calculate_consensus(
                classifier_result,
                extractor_result,
                anomaly_result
            )

            logger.debug(f"✓ Consensus: {'invoice' if consensus.get('is_invoice') else 'not invoice'} "
                        f"({consensus.get('confidence', 0):.0%}, decision: {consensus.get('decision_type')})")

            # Finalize processing and clear memory
            try:
                memory_summary = finalize_processing()
                logger.debug(f"🧹 Memory finalized and cleared (processing time: {memory_summary.get('processing_duration', 0):.2f}s)")
            except Exception as mem_err:
                logger.warning(f"Failed to finalize shared memory: {mem_err}")

            return {
                "consensus_result": consensus,
                "is_invoice": consensus.get("is_invoice", False),
                "confidence": consensus.get("confidence", 0.0),
                "decision_type": consensus.get("decision_type", "unknown"),
                "extracted_data": consensus.get("extracted_data", {})
            }
        
        except Exception as e:
            logger.error(f"Consensus error: {e}")
            return {
                "consensus_result": {"error": str(e)},
                "is_invoice": False,
                "confidence": 0.0,
                "decision_type": "error",
                "extracted_data": {}
            }


# ─────────────────────────────────────────────
# Conditional Edge Functions
# ─────────────────────────────────────────────
def should_skip_ai_processing(state: InvoiceWorkflowState) -> str:
    """Conditional edge: skip AI processing if pre-filter says reject."""
    if state.get("skip_ai_processing"):
        return "skip_to_result"
    return "run_agents"


def should_retry(state: InvoiceWorkflowState) -> str:
    """Conditional edge: retry if errors and iterations < max."""
    max_iterations = 3
    errors = state.get("errors", [])
    iterations = state.get("iterations", 0)
    
    if errors and iterations < max_iterations:
        return "retry"
    return "finish"


# ─────────────────────────────────────────────
# Main Workflow Builder
# ─────────────────────────────────────────────
def build_invoice_workflow(
    classifier: ClassifierAgent,
    extractor: ExtractorAgent,
    anomaly: AnomalyDetectorAgent,
    consensus: ConsensusEngine,
    ocr_validator: OCRTextValidator
):
    """
    Build the LangGraph workflow for invoice processing.
    
    Returns:
        Compiled StateGraph ready to invoke
    """
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangGraph not available - using simple pipeline instead")
        return None
    
    # Create nodes
    nodes = InvoiceWorkflowNodes(
        classifier=classifier,
        extractor=extractor,
        anomaly=anomaly,
        consensus=consensus,
        ocr_validator=ocr_validator
    )
    
    # Build graph
    workflow = StateGraph(InvoiceWorkflowState)
    
    # Add nodes
    workflow.add_node("ocr_validator", nodes.ocr_validator_node)
    workflow.add_node("pre_filter", nodes.pre_filter_node)
    workflow.add_node("enrichment", nodes.enrichment_node)
    workflow.add_node("classifier", nodes.classifier_node)
    workflow.add_node("extractor", nodes.extractor_node)
    workflow.add_node("anomaly", nodes.anomaly_node)
    workflow.add_node("consensus", nodes.consensus_node)
    
    # Set entry point
    workflow.set_entry_point("ocr_validator")
    
    # Add edges
    workflow.add_edge("ocr_validator", "pre_filter")
    
    # Conditional edge after pre-filter
    workflow.add_conditional_edges(
        "pre_filter",
        should_skip_ai_processing,
        {
            "skip_to_result": "consensus",
            "run_agents": "enrichment"
        }
    )
    
    # After enrichment, run all agents in parallel
    workflow.add_edge("enrichment", ["classifier", "extractor", "anomaly"])
    
    # Send all agents to consensus afterwards
    workflow.add_edge("classifier", "consensus")
    workflow.add_edge("extractor", "consensus")
    workflow.add_edge("anomaly", "consensus")
    
    # Add retry logic
    workflow.add_conditional_edges(
        "consensus",
        should_retry,
        {
            "retry": "ocr_validator",
            "finish": END
        }
    )
    
    # Compile
    compiled_workflow = workflow.compile()
    
    logger.info("✓ LangGraph invoice workflow compiled successfully")
    
    return compiled_workflow


# ─────────────────────────────────────────────
# Simple Pipeline Fallback (no LangGraph)
# ─────────────────────────────────────────────
class SimpleInvoicePipeline:
    """
    Simple pipeline fallback when LangGraph is not available.
    
    Same workflow but implemented without LangGraph dependencies.
    """
    
    def __init__(
        self,
        classifier: ClassifierAgent,
        extractor: ExtractorAgent,
        anomaly: AnomalyDetectorAgent,
        consensus: ConsensusEngine,
        ocr_validator: OCRTextValidator
    ):
        self.nodes = InvoiceWorkflowNodes(
            classifier=classifier,
            extractor=extractor,
            anomaly=anomaly,
            consensus=consensus,
            ocr_validator=ocr_validator
        )
    
    def process(self, file_path: str, raw_ocr_text: str) -> Dict[str, Any]:
        """
        Process invoice through the pipeline.
        
        Args:
            file_path: Path to the invoice file
            raw_ocr_text: Raw text from OCR
            
        Returns:
            Final state dict with all results
        """
        state: InvoiceWorkflowState = {
            "file_path": file_path,
            "raw_ocr_text": raw_ocr_text,
            "validated_text": "",
            "validation_result": {},
            "pre_filter_result": (),
            "skip_ai_processing": False,
            "classifier_result": {},
            "extractor_result": {},
            "anomaly_result": {},
            "consensus_result": {},
            "is_invoice": False,
            "confidence": 0.0,
            "decision_type": "pending",
            "extracted_data": {},
            "errors": [],
            "iterations": 0
        }
        
        # Step 1: OCR Validation
        logger.debug(f"🔍 Validating OCR text for {Path(file_path).name}")
        state.update(self.nodes.ocr_validator_node(state))

        # Step 2: Pre-filter
        logger.debug(f"📋 Running pre-filter")
        state.update(self.nodes.pre_filter_node(state))

        # Step 3: Enrichment (if not skipped)
        if not state.get("skip_ai_processing"):
            logger.debug(f"📋 Enriching text basis")
            state.update(self.nodes.enrichment_node(state))
            
            # Initialize shared memory after enrichment (when master_instruction is available)
            try:
                master_instr = state.get("master_instruction", "")
                if master_instr:
                    memory = initialize_memory(file_path, master_instr)
                    logger.debug(f"🧠 Shared memory initialized for {file_path}")
            except Exception as mem_err:
                logger.warning(f"Failed to initialize shared memory: {mem_err}")

        # Step 4: AI Agents (Sequential with Feedback Loop)
        if not state.get("skip_ai_processing"):
            logger.debug(f"🤖 Running AI agents (Sequential with Feedback Loop)")
            
            # 1. Classifier
            logger.debug(f"  ⏳ Running Classifier...")
            state.update(self.nodes.classifier_node(state))
            clf_result = state.get("classifier_result", {})
            is_invoice = clf_result.get("is_invoice", False)
            
            # 2. Anomaly Agent (verifying classifier)
            logger.debug(f"  ⏳ Running Anomaly Detector...")
            state.update(self.nodes.anomaly_node(state))
            anom_result = state.get("anomaly_result", {})
            refutes = anom_result.get("refutes_classifier", False)
            
            # 3. Extractor (Conditional)
            if not is_invoice and not refutes:
                logger.info("  ⏭️ Extractor skipped (Classifier rejected, Anomaly confirmed)")
                state["extractor_result"] = {"completeness_score": 0.0, "validation_errors": ["Classifier rejected document"]}
            else:
                if not is_invoice and refutes:
                    logger.warning("  🔄 FEEDBACK LOOP: Anomaly agent refuted Classifier rejection. Running Extractor and neutralizing Classifier.")
                    state["classifier_result"]["is_invoice"] = True
                    state["classifier_result"]["confidence"] = 0.5
                    state["classifier_result"]["reasoning"] = state["classifier_result"].get("reasoning", "") + " | (Vyvráceno Anomaly Agentem - nalezeny prvky faktury)"

                logger.debug(f"  ⏳ Running Extractor...")
                state.update(self.nodes.extractor_node(state))
                
        else:
            logger.debug(f"⏭️ Skipping AI processing (pre-filter rejected)")
            state["classifier_result"] = {"is_invoice": False, "confidence": 0.0, "reason": "Pre-filter reject"}
            state["extractor_result"] = {"completeness_score": 0.0, "validation_errors": ["Skipped"]}
            state["anomaly_result"] = {"is_anomaly": False, "confidence": 0.0, "refutes_classifier": False}

        # Step 5: Consensus (with memory comparison and cleanup)
        logger.debug(f"🎯 Calculating consensus")
        
        # Compare agent reasoning with master instruction before consensus
        if not state.get("skip_ai_processing"):
            try:
                memory = get_shared_memory()
                comparison_results = memory.compare_all_agents()
                validation_result = memory.validate_consistency()
                logger.debug(f"🔍 Agent reasoning comparison completed")
                
                # Log any significant contradictions
                for agent_name, comparison in comparison_results.items():
                    if comparison.get('contradictions'):
                        logger.warning(f"  ⚠️ {agent_name} has contradictions: {comparison['contradictions']}")
                
                if validation_result.get('conflicts'):
                    logger.warning(f"  ⚠️ Agent consistency conflicts: {validation_result['conflicts']}")
                    
            except Exception as mem_err:
                logger.warning(f"Failed to compare agent reasoning: {mem_err}")
        
        state.update(self.nodes.consensus_node(state))
        
        # Finalize processing and clear memory (done in consensus_node, but ensure for simple pipeline)
        try:
            memory_summary = finalize_processing()
            logger.debug(f"🧹 Memory finalized and cleared")
        except Exception as mem_err:
            logger.warning(f"Failed to finalize shared memory: {mem_err}")

        return state


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────
def create_invoice_processor(
    classifier: ClassifierAgent,
    extractor: ExtractorAgent,
    anomaly: AnomalyDetectorAgent,
    consensus: ConsensusEngine,
    ocr_validator: OCRTextValidator,
    use_langgraph: bool = True
):
    """
    Factory function to create invoice processor.
    
    Uses LangGraph if available, otherwise falls back to simple pipeline.
    
    Args:
        classifier: Configured classifier agent
        extractor: Configured extractor agent
        anomaly: Configured anomaly detector agent
        consensus: Configured consensus engine
        ocr_validator: Configured OCR validator
        use_langgraph: Whether to try using LangGraph (falls back if not available)
        
    Returns:
        Processor object (either LangGraph workflow or SimplePipeline)
    """
    if use_langgraph and LANGCHAIN_AVAILABLE:
        workflow = build_invoice_workflow(
            classifier=classifier,
            extractor=extractor,
            anomaly=anomaly,
            consensus=consensus,
            ocr_validator=ocr_validator
        )
        
        if workflow:
            logger.info("✓ Using LangGraph workflow")
            return workflow
    
    logger.info("✓ Using simple pipeline (LangGraph not available or disabled)")
    return SimpleInvoicePipeline(
        classifier=classifier,
        extractor=extractor,
        anomaly=anomaly,
        consensus=consensus,
        ocr_validator=ocr_validator
    )
