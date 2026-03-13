"""
Markdown Utilities for Agent Communication

v7.0 (2026-02-28):
- Comprehensive Markdown parsing utilities for agent communication
- Replaces JSON-based communication with Markdown-first approach
- Provides robust parsing with fallback mechanisms

Usage:
    from core.markdown_utils import MarkdownParser
    
    parser = MarkdownParser()
    result = parser.parse_classifier_output(markdown_text)
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedResult:
    """Structured result from Markdown parsing."""
    success: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_markdown: str = ""


class MarkdownParser:
    """
    Comprehensive Markdown parser for agent communication.
    
    Handles extraction of structured data from Markdown-formatted agent outputs.
    """
    
    def __init__(self):
        """Initialize parser with common patterns."""
        # Boolean patterns
        self.true_patterns = ['✅', '✓', 'ANO', 'YES', 'True', 'pravda', 'present']
        self.false_patterns = ['❌', '✗', 'NE', 'NO', 'False', 'nepravda', 'absent', 'null', 'none']
        
        # Number patterns
        self.number_pattern = r'(\d+(?:[\s,.]\d+)*)'
        
    def parse_classifier_output(self, markdown_text: str) -> ParsedResult:
        """
        Parse Classifier Agent Markdown output.
        
        Args:
            markdown_text: Raw Markdown from classifier
            
        Returns:
            ParsedResult with classification data
        """
        result = ParsedResult(raw_markdown=markdown_text)
        
        if not markdown_text or not markdown_text.strip():
            result.errors.append("Empty markdown text")
            return result
        
        try:
            # Extract main classification
            classification_data = self._extract_classification_section(markdown_text)
            result.data.update(classification_data)
            
            # Extract elements
            elements = self._extract_elements_table(markdown_text)
            result.data['elements_present'] = elements
            
            # Extract values
            values = self._extract_values_table(markdown_text)
            result.data['extracted_values'] = values
            
            # Extract reasoning
            reasoning = self._extract_section(markdown_text, "🧠 Reasoning")
            if reasoning:
                result.data['reasoning'] = reasoning
            
            # Validate minimum required fields
            if 'is_invoice' not in result.data:
                result.errors.append("Missing is_invoice field")
            else:
                result.success = True
            
        except Exception as e:
            result.errors.append(f"Parsing error: {str(e)}")
        
        return result
    
    def parse_extractor_output(self, markdown_text: str) -> ParsedResult:
        """
        Parse Extractor Agent Markdown output.
        
        Args:
            markdown_text: Raw Markdown from extractor
            
        Returns:
            ParsedResult with extracted invoice data
        """
        result = ParsedResult(raw_markdown=markdown_text)
        
        if not markdown_text or not markdown_text.strip():
            result.errors.append("Empty markdown text")
            return result
        
        try:
            # Extract data table
            data = self._extract_key_value_table(markdown_text, "📋 Extrahovaná data")
            result.data.update(data)
            
            # Extract completeness score
            completeness = self._extract_completeness_score(markdown_text)
            if completeness is not None:
                result.data['completeness_score'] = completeness
            
            # Extract validation errors
            errors = self._extract_validation_errors(markdown_text)
            if errors:
                result.data['validation_errors'] = errors
            
            result.success = True
            
        except Exception as e:
            result.errors.append(f"Parsing error: {str(e)}")
        
        return result
    
    def parse_anomaly_output(self, markdown_text: str) -> ParsedResult:
        """
        Parse Anomaly Detector Agent Markdown output.
        
        Args:
            markdown_text: Raw Markdown from anomaly detector
            
        Returns:
            ParsedResult with anomaly detection data
        """
        result = ParsedResult(raw_markdown=markdown_text)
        
        if not markdown_text or not markdown_text.strip():
            result.errors.append("Empty markdown text")
            return result
        
        try:
            # Extract anomaly detection section
            anomaly_section = self._extract_section(markdown_text, "🚨 Detekce anomálií")
            
            if anomaly_section:
                # Extract is_anomaly boolean
                is_anomaly = self._extract_boolean(anomaly_section)
                result.data['is_anomaly'] = is_anomaly
                
                # Extract confidence
                confidence = self._extract_number(anomaly_section, "confidence")
                if confidence is not None:
                    result.data['confidence'] = confidence
                
                # Extract anomaly type
                anomaly_type = self._extract_value(anomaly_section, "Typ anomálie")
                if anomaly_type:
                    result.data['anomaly_type'] = anomaly_type
            
            # Extract detected keywords
            keywords = self._extract_keywords_table(markdown_text)
            if keywords:
                result.data['detected_keywords'] = keywords
            
            # Extract reasoning
            reasoning = self._extract_section(markdown_text, "🧠 Reasoning")
            if reasoning:
                result.data['reasoning'] = reasoning
            
            # Default values if not found
            if 'is_anomaly' not in result.data:
                result.data['is_anomaly'] = False
            if 'confidence' not in result.data:
                result.data['confidence'] = 0.95
            
            result.success = True
            
        except Exception as e:
            result.errors.append(f"Parsing error: {str(e)}")
        
        return result
    
    def _extract_classification_section(self, markdown_text: str) -> dict:
        """Extract main classification data."""
        section = self._extract_section(markdown_text, "📊 Klasifikace dokumentu")
        
        if not section:
            # Try alternative headers
            section = self._extract_section(markdown_text, "Klasifikace")
        
        result = {}
        
        if section:
            # Extract is_invoice
            is_invoice = self._extract_boolean(section)
            if is_invoice is not None:
                result['is_invoice'] = is_invoice
            
            # Extract confidence
            confidence = self._extract_number(section, "confidence")
            if confidence is not None:
                result['confidence'] = confidence
        
        return result
    
    def _extract_elements_table(self, markdown_text: str) -> dict:
        """Extract elements presence table."""
        section = self._extract_section(markdown_text, "🔍 Elementy")
        if not section:
            return {}
        
        return self._extract_boolean_table(section)
    
    def _extract_values_table(self, markdown_text: str) -> dict:
        """Extract values table."""
        section = self._extract_section(markdown_text, "📋 Extrahované hodnoty")
        if not section:
            return {}
        
        return self._extract_key_value_table(section)
    
    def _extract_section(self, markdown_text: str, section_name: str) -> str:
        """
        Extract content of a Markdown section.
        
        Args:
            markdown_text: Full Markdown text
            section_name: Section header to find
            
        Returns:
            Section content or empty string
        """
        # Pattern: ## Section Name or ### Section Name
        pattern = rf"##+\s*{re.escape(section_name)}\s*\n(.*?)(?=\n##+\s*|$)"
        match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_boolean(self, text: str) -> Optional[bool]:
        """Extract boolean value from text."""
        text_upper = text.upper()
        
        # Check for true indicators
        for pattern in self.true_patterns:
            if pattern.upper() in text_upper:
                # Make sure false patterns are not present
                is_false = any(fp.upper() in text_upper for fp in self.false_patterns)
                if not is_false:
                    return True
        
        # Check for false indicators
        for pattern in self.false_patterns:
            if pattern.upper() in text_upper:
                return False
        
        return None
    
    def _extract_number(self, text: str, key: str = None) -> Optional[float]:
        """Extract numeric value from text."""
        if key:
            # Look for Key: Number pattern
            pattern = rf"{re.escape(key)}:\s*([\d.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        # Fallback: find any number
        match = re.search(self.number_pattern, text)
        if match:
            try:
                num_str = match.group(1).replace(' ', '').replace(',', '.')
                return float(num_str)
            except ValueError:
                pass
        
        return None
    
    def _extract_value(self, text: str, key: str) -> Optional[str]:
        """Extract string value for a given key."""
        # Pattern: **Key**: Value or Key: Value
        patterns = [
            rf"\*\*{re.escape(key)}\*\*:\s*([^\n]+)",
            rf"{re.escape(key)}:\s*([^\n]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Filter out null/none values
                if value and value.lower() not in ['null', 'none', '—', '-', '']:
                    return value
        
        return None
    
    def _extract_key_value_table(self, markdown_text: str, section_name: str = None) -> dict:
        """
        Extract key-value pairs from Markdown table.
        
        Args:
            markdown_text: Markdown text
            section_name: Optional section to extract first
            
        Returns:
            Dictionary of key-value pairs
        """
        if section_name:
            markdown_text = self._extract_section(markdown_text, section_name)
        
        if not markdown_text:
            return {}
        
        result = {}
        
        # Find table pattern
        table_pattern = r"\|([^|]+)\|\s*\n\|[-\s|]+\|\s*\n((?:\|[^|]+\|\s*\n?)+)"
        table_match = re.search(table_pattern, markdown_text)
        
        if not table_match:
            # Try simple key: value format
            return self._extract_simple_key_values(markdown_text)
        
        # Parse headers
        headers = [h.strip() for h in table_match.group(1).split('|') if h.strip()]
        
        # Parse rows
        rows_text = table_match.group(2)
        for row_match in re.finditer(r"\|([^|]+)\|", rows_text):
            values = [v.strip() for v in row_match.group(1).split('|') if v.strip()]
            
            if len(values) >= 2 and len(headers) >= 2:
                key = values[0] if len(headers) == 2 else headers[0]
                value = values[1] if len(headers) == 2 else values[headers.index(key) + 1] if key in headers else values[1]
                
                if key and value and value.lower() not in ['null', 'none', '—', '-']:
                    result[key] = value
        
        return result
    
    def _extract_simple_key_values(self, text: str) -> dict:
        """Extract simple key: value pairs."""
        result = {}
        
        # Pattern: **Key**: Value or Key: Value
        patterns = [
            (r"\*\*([^*]+)\*\*:\s*([^\n]+)", True),
            (r"^([A-Za-z][\w\s]+):\s*([^\n]+)$", False)
        ]
        
        for pattern, is_bold in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                key = match.group(1).strip().rstrip(':')
                value = match.group(2).strip()
                
                if key and key not in result and value and value.lower() not in ['null', 'none', '—', '-']:
                    result[key] = value
        
        return result
    
    def _extract_boolean_table(self, markdown_text: str) -> dict:
        """Extract boolean values from table."""
        table_data = self._extract_key_value_table(markdown_text)
        
        result = {}
        for key, value in table_data.items():
            if isinstance(value, str):
                result[key] = self._extract_boolean(value)
        
        return result
    
    def _extract_completeness_score(self, markdown_text: str) -> Optional[float]:
        """Extract completeness score from text."""
        patterns = [
            r"completeness\s*score:\s*([\d.]+)",
            r"score:\s*([\d.]+)",
            r"\*\*Completeness Score:\*\*\s*([\d.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, markdown_text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def _extract_validation_errors(self, markdown_text: str) -> List[str]:
        """Extract validation errors from section."""
        section = self._extract_section(markdown_text, "⚠️ Validation Errors")
        if not section:
            section = self._extract_section(markdown_text, "Validation Errors")
        
        if not section:
            return []
        
        errors = []
        
        # Extract list items
        for match in re.finditer(r"[-•*]\s*([^\n]+)", section):
            errors.append(match.group(1).strip())
        
        # Extract table rows
        table_data = self._extract_key_value_table(section)
        errors.extend(table_data.values())
        
        return errors
    
    def _extract_keywords_table(self, markdown_text: str) -> List[str]:
        """Extract keywords from table."""
        section = self._extract_section(markdown_text, "🔍 Detekované klíčové body")
        if not section:
            return []
        
        keywords = []
        table_data = self._extract_key_value_table(section)
        
        for key, value in table_data.items():
            if value and value.lower() not in ['null', 'none']:
                keywords.append(value)
            elif key and key.lower() not in ['null', 'none', 'keyword', 'typ']:
                keywords.append(key)
        
        return keywords


# Convenience functions
def parse_classifier(markdown_text: str) -> dict:
    """Parse classifier output and return dict."""
    parser = MarkdownParser()
    result = parser.parse_classifier_output(markdown_text)
    return result.data if result.success else {}


def parse_extractor(markdown_text: str) -> dict:
    """Parse extractor output and return dict."""
    parser = MarkdownParser()
    result = parser.parse_extractor_output(markdown_text)
    return result.data if result.success else {}


def parse_anomaly(markdown_text: str) -> dict:
    """Parse anomaly detector output and return dict."""
    parser = MarkdownParser()
    result = parser.parse_anomaly_output(markdown_text)
    return result.data if result.success else {}
