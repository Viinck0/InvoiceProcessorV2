"""
Shared Agent Memory System
Provides shared reasoning storage for all agents with master instruction comparison

This module provides:
1. Centralized storage for agent reasoning during document processing
2. Comparison mechanisms between agent reasoning and master instructions
3. Automatic memory cleanup after processing completion
4. Reasoning validation and consistency checks
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentReasoning:
    """Stores reasoning from a single agent."""
    agent_name: str
    reasoning: str
    confidence: float
    is_invoice: Optional[bool]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentSharedMemory:
    """
    Shared memory system for agent reasoning during document processing.
    
    Features:
    - Stores reasoning from all agents (Classifier, Extractor, Anomaly Detector)
    - Compares agent reasoning against master instructions
    - Validates consistency between agents
    - Automatically clears after processing completion
    """
    
    def __init__(self):
        """Initialize empty shared memory."""
        self._reasoning_store: Dict[str, AgentReasoning] = {}
        self._master_instruction: str = ""
        self._current_file_path: str = ""
        self._processing_started: datetime = None
        self._processing_completed: datetime = None
        self._comparison_results: Dict[str, Any] = {}
        self._validation_errors: List[str] = []
    
    def start_processing(self, file_path: str, master_instruction: str) -> None:
        """
        Initialize memory for new document processing.
        
        Args:
            file_path: Path to the document being processed
            master_instruction: Master instruction text for comparison
        """
        self.clear()  # Ensure clean state
        self._current_file_path = file_path
        self._master_instruction = master_instruction
        self._processing_started = datetime.now()
        logger.debug(f"🧠 Agent memory initialized for {file_path}")
    
    def store_reasoning(
        self,
        agent_name: str,
        reasoning: str,
        confidence: float,
        is_invoice: Optional[bool],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store reasoning from an agent.
        
        Args:
            agent_name: Name of the agent (classifier, extractor, anomaly)
            reasoning: Agent's reasoning text
            confidence: Agent's confidence score
            is_invoice: Agent's invoice decision (True/False/None)
            metadata: Additional agent-specific metadata
        """
        self._reasoning_store[agent_name] = AgentReasoning(
            agent_name=agent_name,
            reasoning=reasoning,
            confidence=confidence,
            is_invoice=is_invoice,
            metadata=metadata or {}
        )
        logger.debug(f"🧠 Stored reasoning from {agent_name} (confidence: {confidence:.0%})")
    
    def compare_with_master_instruction(self, agent_name: str) -> Dict[str, Any]:
        """
        Compare specific agent's reasoning with master instruction.
        
        Args:
            agent_name: Name of the agent to compare
            
        Returns:
            Comparison result dict with alignment score and details
        """
        if agent_name not in self._reasoning_store:
            return {"error": f"No reasoning stored for {agent_name}"}
        
        agent_reasoning = self._reasoning_store[agent_name]
        
        # Check for keyword alignment
        alignment_score = self._calculate_alignment_score(agent_reasoning)
        
        # Check for contradictions
        contradictions = self._detect_contradictions(agent_reasoning)
        
        # Check for missing elements
        missing_elements = self._check_missing_elements(agent_reasoning)
        
        comparison_result = {
            "agent_name": agent_name,
            "alignment_score": alignment_score,
            "contradictions": contradictions,
            "missing_elements": missing_elements,
            "reasoning_quality": self._assess_reasoning_quality(agent_reasoning),
            "master_instruction_used": bool(self._master_instruction),
            "timestamp": datetime.now()
        }
        
        self._comparison_results[agent_name] = comparison_result
        
        logger.debug(f"🔍 Comparison for {agent_name}: alignment={alignment_score:.0%}, "
                    f"contradictions={len(contradictions)}, missing={len(missing_elements)}")
        
        return comparison_result
    
    def compare_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare all stored agent reasonings with master instruction.
        
        Returns:
            Dict of comparison results for each agent
        """
        results = {}
        for agent_name in self._reasoning_store:
            results[agent_name] = self.compare_with_master_instruction(agent_name)
        return results
    
    def validate_consistency(self) -> Dict[str, Any]:
        """
        Validate consistency between all agents' reasoning.
        
        Returns:
            Validation result with consistency score and conflicts
        """
        if len(self._reasoning_store) < 2:
            return {
                "consistent": True,
                "score": 1.0,
                "conflicts": [],
                "reason": "Insufficient agents for comparison"
            }
        
        conflicts = []
        agreements = []
        
        # Check for invoice decision conflicts
        decisions = {
            name: reasoning.is_invoice 
            for name, reasoning in self._reasoning_store.items()
            if reasoning.is_invoice is not None
        }
        
        if len(decisions) >= 2:
            decision_values = list(decisions.values())
            if not all(d == decision_values[0] for d in decision_values):
                conflicts.append({
                    "type": "invoice_decision_conflict",
                    "description": "Agents disagree on invoice classification",
                    "decisions": decisions
                })
            else:
                agreements.append({
                    "type": "invoice_decision_agreement",
                    "decision": decision_values[0]
                })
        
        # Check for confidence conflicts
        confidences = {
            name: reasoning.confidence
            for name, reasoning in self._reasoning_store.items()
        }
        
        if len(confidences) >= 2:
            conf_values = list(confidences.values())
            conf_range = max(conf_values) - min(conf_values)
            
            if conf_range > 0.5:  # High confidence divergence
                conflicts.append({
                    "type": "confidence_divergence",
                    "description": "Large confidence gap between agents",
                    "range": conf_range,
                    "confidences": confidences
                })
            elif conf_range < 0.2:  # Good confidence alignment
                agreements.append({
                    "type": "confidence_alignment",
                    "range": conf_range
                })
        
        # Check reasoning keyword overlap
        reasoning_keywords = self._extract_reasoning_keywords()
        keyword_overlap = self._calculate_keyword_overlap(reasoning_keywords)
        
        if keyword_overlap < 0.3:
            conflicts.append({
                "type": "low_keyword_overlap",
                "description": "Agents using very different terminology",
                "overlap_score": keyword_overlap
            })
        else:
            agreements.append({
                "type": "keyword_consistency",
                "overlap_score": keyword_overlap
            })
        
        consistency_score = len(agreements) / (len(agreements) + len(conflicts) + 0.001)
        
        validation_result = {
            "consistent": len(conflicts) == 0,
            "score": consistency_score,
            "conflicts": conflicts,
            "agreements": agreements,
            "agent_count": len(self._reasoning_store),
            "timestamp": datetime.now()
        }
        
        self._validation_errors = [c["description"] for c in conflicts]
        
        logger.debug(f"✓ Consistency validation: score={consistency_score:.0%}, "
                    f"conflicts={len(conflicts)}, agreements={len(agreements)}")
        
        return validation_result
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """
        Get summary of all stored reasoning.
        
        Returns:
            Summary dict with all agent reasonings and metadata
        """
        return {
            "file_path": self._current_file_path,
            "master_instruction_length": len(self._master_instruction),
            "processing_started": self._processing_started,
            "processing_completed": self._processing_completed,
            "agents": {
                name: {
                    "reasoning": reasoning.reasoning,
                    "confidence": reasoning.confidence,
                    "is_invoice": reasoning.is_invoice,
                    "timestamp": reasoning.timestamp.isoformat(),
                    "metadata": reasoning.metadata
                }
                for name, reasoning in self._reasoning_store.items()
            },
            "comparison_results": self._comparison_results,
            "validation_errors": self._validation_errors
        }
    
    def clear(self) -> None:
        """
        Clear all stored data and reset memory state.
        
        This method is called automatically after processing completion.
        """
        old_file = self._current_file_path
        self._reasoning_store.clear()
        self._master_instruction = ""
        self._current_file_path = ""
        self._processing_started = None
        self._processing_completed = None
        self._comparison_results.clear()
        self._validation_errors.clear()
        
        if old_file:
            logger.debug(f"🧹 Agent memory cleared for {old_file}")
    
    def complete_processing(self) -> Dict[str, Any]:
        """
        Mark processing as complete and return final summary.
        
        Returns:
            Final summary before memory cleanup
        """
        self._processing_completed = datetime.now()
        
        summary = self.get_reasoning_summary()
        summary["processing_duration"] = (
            self._processing_completed - self._processing_started
        ).total_seconds() if self._processing_started else 0
        
        logger.info(f"✓ Processing completed for {self._current_file_path} "
                   f"({summary['processing_duration']:.2f}s)")
        
        # Auto-clear after completion
        self.clear()
        
        return summary
    
    def _calculate_alignment_score(self, agent_reasoning: AgentReasoning) -> float:
        """
        Calculate how well agent reasoning aligns with master instruction.
        
        Args:
            agent_reasoning: Agent's reasoning to evaluate
            
        Returns:
            Alignment score from 0.0 (no alignment) to 1.0 (perfect alignment)
        """
        if not self._master_instruction:
            return 0.5  # Neutral if no master instruction
        
        master_lower = self._master_instruction.lower()
        reasoning_lower = agent_reasoning.reasoning.lower()
        
        # Extract key entities from master instruction
        master_entities = self._extract_entities(master_lower)
        
        # Check how many entities are referenced in reasoning
        referenced_entities = sum(
            1 for entity in master_entities
            if entity in reasoning_lower
        )
        
        entity_coverage = referenced_entities / (len(master_entities) + 0.001)
        
        # Check for positional references ([vlevo], [vpravo])
        position_refs_master = master_lower.count('[vlevo]') + master_lower.count('[vpravo]')
        position_refs_reasoning = reasoning_lower.count('[vlevo]') + reasoning_lower.count('[vpravo]')
        
        position_alignment = min(1.0, position_refs_reasoning / (position_refs_master + 0.001))
        
        # Weighted combination
        alignment = (entity_coverage * 0.6) + (position_alignment * 0.4)
        
        return min(1.0, max(0.0, alignment))
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text for comparison."""
        import re
        
        entities = []
        
        # Extract dates
        dates = re.findall(r'\d{1,2}\.\s*\d{1,2}\.\s*\d{4}', text)
        entities.extend(dates)
        
        # Extract amounts
        amounts = re.findall(r'\d+(?:[\s,.]\d+)*\s*(?:kč|eur|usd|czk)?', text, re.IGNORECASE)
        entities.extend(amounts)
        
        # Extract company-like patterns
        companies = re.findall(r'[A-Z][a-z]+\s+(?:s\.r\.o\.|a\.s\.|GmbH|Inc\.|Ltd\.)', text)
        entities.extend(companies)
        
        # Extract invoice keywords
        invoice_keywords = ['faktura', 'invoice', 'dodavatel', 'odběratel', 
                           'supplier', 'customer', 'částka', 'amount']
        entities.extend([kw for kw in invoice_keywords if kw in text])
        
        return list(set(entities))
    
    def _detect_contradictions(self, agent_reasoning: AgentReasoning) -> List[str]:
        """
        Detect contradictions between agent reasoning and master instruction.
        
        Returns:
            List of contradiction descriptions
        """
        contradictions = []
        
        if not self._master_instruction:
            return contradictions
        
        master_lower = self._master_instruction.lower()
        reasoning_lower = agent_reasoning.reasoning.lower()
        
        # Check for explicit denials that contradict master instruction content
        if 'není faktura' in reasoning_lower and 'faktura' in master_lower:
            contradictions.append("Agent denies invoice despite invoice keyword in master instruction")
        
        if 'neobsahuje' in reasoning_lower and any(
            kw in reasoning_lower for kw in ['dodavatel', 'odběratel', 'částka']
        ):
            # Check if master instruction contains these elements
            if any(
                kw in master_lower for kw in ['dodavatel', 'odběratel', 'částka', 'supplier', 'customer']
            ):
                contradictions.append("Agent claims missing elements that exist in master instruction")
        
        return contradictions
    
    def _check_missing_elements(self, agent_reasoning: AgentReasoning) -> List[str]:
        """
        Check if agent reasoning misses key elements from master instruction.
        
        Returns:
            List of missing element descriptions
        """
        missing = []
        
        if not self._master_instruction:
            return missing
        
        master_lower = self._master_instruction.lower()
        reasoning_lower = agent_reasoning.reasoning.lower()
        
        # Key elements to check
        key_elements = {
            'dodavatel': ['dodavatel', 'supplier', 'firma', 'společnost'],
            'odběratel': ['odběratel', 'customer', 'objednatel'],
            'částka': ['částka', 'amount', 'celkem', 'total', 'cena'],
            'datum': ['datum', 'date', 'vystavení', 'splatnost'],
            'faktura_type': ['faktura', 'invoice', 'daňový doklad']
        }
        
        for element_name, keywords in key_elements.items():
            # Check if element exists in master instruction
            in_master = any(kw in master_lower for kw in keywords)
            
            # Check if element is mentioned in reasoning
            in_reasoning = any(kw in reasoning_lower for kw in keywords)
            
            if in_master and not in_reasoning:
                missing.append(element_name)
        
        return missing
    
    def _assess_reasoning_quality(self, agent_reasoning: AgentReasoning) -> str:
        """
        Assess overall quality of agent reasoning.
        
        Returns:
            Quality rating: 'high', 'medium', 'low'
        """
        reasoning_length = len(agent_reasoning.reasoning)
        confidence = agent_reasoning.confidence
        
        # Length-based quality
        if reasoning_length < 50:
            length_quality = 'low'
        elif reasoning_length < 200:
            length_quality = 'medium'
        else:
            length_quality = 'high'
        
        # Confidence-based quality
        if confidence < 0.5:
            conf_quality = 'low'
        elif confidence < 0.75:
            conf_quality = 'medium'
        else:
            conf_quality = 'high'
        
        # Combined assessment
        quality_scores = {'high': 3, 'medium': 2, 'low': 1}
        avg_score = (quality_scores[length_quality] + quality_scores[conf_quality]) / 2
        
        if avg_score >= 2.5:
            return 'high'
        elif avg_score >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _extract_reasoning_keywords(self) -> Dict[str, List[str]]:
        """
        Extract keywords from all agents' reasoning.
        
        Returns:
            Dict mapping agent names to their keyword lists
        """
        keywords = {}
        
        for agent_name, reasoning in self._reasoning_store.items():
            # Simple keyword extraction
            words = reasoning.reasoning.lower().split()
            # Filter common words
            stop_words = {'je', 'to', 'se', 'v', 'na', 'a', 'i', 'že', 'jak', 'pro', 'bez'}
            filtered = [w for w in words if w not in stop_words and len(w) > 3]
            keywords[agent_name] = list(set(filtered))
        
        return keywords
    
    def _calculate_keyword_overlap(self, reasoning_keywords: Dict[str, List[str]]) -> float:
        """
        Calculate keyword overlap between agents.
        
        Returns:
            Overlap score from 0.0 (no overlap) to 1.0 (perfect overlap)
        """
        if len(reasoning_keywords) < 2:
            return 1.0  # Perfect overlap if only one agent
        
        keyword_sets = [set(kws) for kws in reasoning_keywords.values()]
        
        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(keyword_sets)):
            for j in range(i + 1, len(keyword_sets)):
                intersection = len(keyword_sets[i] & keyword_sets[j])
                union = len(keyword_sets[i] | keyword_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return sum(similarities) / len(similarities) if similarities else 0.0


# Singleton instance for global access
_shared_memory_instance: Optional[AgentSharedMemory] = None


def get_shared_memory() -> AgentSharedMemory:
    """
    Get the global shared memory instance.
    
    Returns:
        Singleton AgentSharedMemory instance
    """
    global _shared_memory_instance
    if _shared_memory_instance is None:
        _shared_memory_instance = AgentSharedMemory()
    return _shared_memory_instance


def initialize_memory(file_path: str, master_instruction: str) -> AgentSharedMemory:
    """
    Initialize shared memory for new document processing.
    
    Args:
        file_path: Path to the document
        master_instruction: Master instruction text
        
    Returns:
        Initialized shared memory instance
    """
    memory = get_shared_memory()
    memory.start_processing(file_path, master_instruction)
    return memory


def finalize_processing() -> Dict[str, Any]:
    """
    Finalize processing and clear memory.
    
    Returns:
        Final summary before cleanup
    """
    memory = get_shared_memory()
    return memory.complete_processing()
