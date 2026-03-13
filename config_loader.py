"""
Configuration Loader for Invoice Processing System.

This module provides centralized loading and access to all rules, keywords,
and patterns from external YAML configuration files.

Benefits:
- Tuning accuracy without modifying Python code
- Single source of truth for all patterns
- Easy to update keywords for different languages/regions
- Reduces risk of syntax errors in business logic

Usage:
    from config_loader import Config
    
    config = Config.load()
    keywords = config.get_classifier_keywords()
    patterns = config.get_extractor_patterns()
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import yaml, provide fallback if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning(
        "PyYAML not installed. Install with: pip install pyyaml. "
        "Using built-in configuration defaults."
    )


class Config:
    """
    Centralized configuration manager.
    
    Loads rules from YAML file and provides typed access to all patterns.
    Falls back to built-in defaults if file is missing or invalid.
    """
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "rules.yaml"
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_data: Pre-loaded configuration dictionary. If None, loads from file.
        """
        self._config = config_data or self._load_default_config()
    
    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, config_path: Optional[Path] = None) -> 'Config':
        """
        Load configuration from file (cached).
        
        Args:
            config_path: Optional custom path to config file.
            
        Returns:
            Config instance with loaded configuration.
        """
        path = config_path or cls.DEFAULT_CONFIG_PATH
        
        if not YAML_AVAILABLE:
            logger.info("Using built-in configuration (PyYAML not available)")
            return cls()
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}. Using built-in defaults.")
            return cls()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from {path}")

            # ⚠️ INTEGRITY CHECK DISABLED - Hash files (.sha256) were causing confusion
            # and providing no real benefit for local development.
            # If you need config integrity verification, you can re-enable this:
            # try:
            #     from core.security import verify_config_integrity
            #     is_valid, message = verify_config_integrity(path)
            #     if is_valid:
            #         logger.info(f"🔏 {message}")
            #     else:
            #         logger.warning(f"⚠️ {message}")
            # except ImportError:
            #     pass
            # except Exception as e:
            #     logger.warning(f"Config integrity check failed: {e}")

            return cls(config_data)
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using built-in defaults.")
            return cls()
    
    @classmethod
    def reload(cls, config_path: Optional[Path] = None) -> 'Config':
        """
        Force reload configuration (bypass cache).
        
        Use this when config file has been updated and you want to refresh.
        
        Args:
            config_path: Optional custom path to config file.
            
        Returns:
            Config instance with freshly loaded configuration.
        """
        # Clear cache
        cls.load.cache_clear()
        return cls.load(config_path)
    
    @classmethod
    def _load_default_config(cls) -> Dict[str, Any]:
        """Return built-in default configuration."""
        return {
            'classifier': {
                'strong_negative_indicators': [
                    'plná moc', 'plnomocenství', 'dluhopis', 'cenný papír', 'akcie',
                    'životopis', 'curriculum vitae', 'resume', 'vzdělání', 'skills',
                    'upomínka', 'reminder', 'výzva k úhradě'
                ],
                'fallback': {
                    'positive_keywords': ['faktura', 'invoice', 'daňový doklad'],
                    'negative_keywords': ['upomínka', 'smlouva', 'nabídka']
                }
            },
            'consensus_engine': {
                'invoice_keywords_cs': ['faktura', 'daňový doklad', 'IČO', 'DIČ'],
                'invoice_keywords_en': ['invoice', 'tax document', 'VAT'],
                'non_invoice_patterns': {}
            },
            'core_engine': {
                'invoice_keywords': ['faktura', 'invoice', 'daňový doklad'],
                'non_invoice_keywords': ['upomínka', 'reminder', 'smlouva']
            },
            'extractor_agent': {
                'patterns': {
                    'date': r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})',
                    'amount': r'(\d{1,3}(?:[\s\.,]\d{3})*(?:[\s\.,]\d{1,2})?)',
                    'ico': r'(?:ičo|ič|reg\.?\s*no\.?|company\s*no\.?|crn)\s*[:.]?\s*(\d{6,10})',
                    'dic': r'(?:dič|vat\s*id|tax\s*id|vat\s*no\.?)\s*[:.]?\s*([a-zA-Z]{2}\d{8,12})',
                    'account': r'(cz|sk)?\d{4,6}[- ]?\d{6,10}[- ]?\d{2,4}',
                    'iban': r'iban:\s*([a-z]{2}\d{2,24})'
                }
            },
            'anomaly_detection': {},
            'ocr_validator': {
                'hallucination_patterns': [],
                'valid_word_patterns': []
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Dot-separated key path (e.g., 'classifier.fallback.positive_keywords')
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    # =========================================================================
    # Classifier Agent Configuration
    # =========================================================================
    
    def get_classifier_strong_negatives(self) -> list:
        """Get strong negative indicators for immediate rejection."""
        return self.get('classifier.strong_negative_indicators', [])
    
    def get_classifier_context_negatives(self) -> dict:
        """Get context-dependent negative indicators."""
        return self.get('classifier.context_negative_indicators', {})
    
    def get_classifier_fallback_keywords(self) -> tuple:
        """Get fallback classification keywords (positive, negative)."""
        fallback = self.get('classifier.fallback', {})
        return (
            fallback.get('positive_keywords', []),
            fallback.get('negative_keywords', [])
        )
    
    # =========================================================================
    # Consensus Engine Configuration
    # =========================================================================
    
    def get_invoice_keywords_cs(self) -> list:
        """Get Czech invoice keywords."""
        return self.get('consensus_engine.invoice_keywords_cs', [])
    
    def get_invoice_keywords_en(self) -> list:
        """Get English invoice keywords."""
        return self.get('consensus_engine.invoice_keywords_en', [])
    
    def get_all_invoice_keywords(self) -> list:
        """Get all invoice keywords (Czech + English)."""
        return self.get_invoice_keywords_cs() + self.get_invoice_keywords_en()
    
    def get_non_invoice_patterns(self) -> dict:
        """Get non-invoice document patterns for anomaly detection."""
        return self.get('consensus_engine.non_invoice_patterns', {})
    
    # =========================================================================
    # Core Engine Configuration
    # =========================================================================
    
    def get_core_invoice_keywords(self) -> list:
        """Get basic invoice keywords for core filtering."""
        return self.get('core_engine.invoice_keywords', [])
    
    def get_core_non_invoice_keywords(self) -> list:
        """Get basic non-invoice keywords for core filtering."""
        return self.get('core_engine.non_invoice_keywords', [])
    
    # =========================================================================
    # Extractor Agent Configuration
    # =========================================================================
    
    def get_extractor_patterns(self) -> dict:
        """Get all extractor regex patterns."""
        return self.get('extractor_agent.patterns', {})
    
    def get_extractor_pattern(self, name: str) -> Optional[str]:
        """Get specific extractor pattern by name."""
        patterns = self.get_extractor_patterns()
        return patterns.get(name)
    
    def get_amount_patterns(self) -> list:
        """Get amount extraction patterns."""
        return self.get('extractor_agent.amount_patterns', [])
    
    def get_date_patterns(self) -> list:
        """Get date extraction patterns."""
        return self.get('extractor_agent.date_patterns', [])
    
    def get_currencies(self) -> dict:
        """Get currency symbols and codes."""
        return self.get('extractor_agent.currencies', {'symbols': [], 'codes': []})
    
    # =========================================================================
    # Anomaly Detection Configuration
    # =========================================================================
    
    def get_anomaly_rules(self) -> dict:
        """Get all anomaly detection rules."""
        return self.get('anomaly_detection', {})
    
    def get_anomaly_rule(self, rule_name: str) -> Optional[dict]:
        """Get specific anomaly rule by name."""
        return self.get(f'anomaly_detection.{rule_name}')
    
    def get_anomaly_keywords(self, rule_name: str) -> list:
        """Get keywords for specific anomaly type."""
        rule = self.get_anomaly_rule(rule_name)
        return rule.get('keywords', []) if rule else []
    
    def get_anomaly_threshold(self, rule_name: str) -> int:
        """Get detection threshold for specific anomaly type."""
        rule = self.get_anomaly_rule(rule_name)
        return rule.get('threshold', 1) if rule else 1
    
    def is_anomaly_veto(self, rule_name: str) -> bool:
        """Check if anomaly type has veto power."""
        rule = self.get_anomaly_rule(rule_name)
        return rule.get('veto', False) if rule else False
    
    # =========================================================================
    # OCR Validator Configuration
    # =========================================================================
    
    def get_hallucination_patterns(self) -> list:
        """Get OCR hallucination detection patterns."""
        return self.get('ocr_validator.hallucination_patterns', [])
    
    def get_valid_word_patterns(self) -> list:
        """Get valid word patterns (skip spell check)."""
        return self.get('ocr_validator.valid_word_patterns', [])
    
    def get_long_pattern_corrections(self) -> list:
        """Get OCR error corrections."""
        return self.get('ocr_validator.long_pattern_corrections', [])
    
    # =========================================================================
    # Element Detection Configuration
    # =========================================================================
    
    def get_element_keywords(self, element_type: str) -> list:
        """Get keywords for specific document element."""
        return self.get(f'element_detection.{element_type}', [])
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def validate(self) -> tuple:
        """
        Validate configuration structure.
        
        Returns:
            Tuple of (is_valid: bool, errors: list)
        """
        errors = []
        
        # Check required sections
        required_sections = [
            'classifier',
            'consensus_engine',
            'core_engine',
            'extractor_agent',
            'anomaly_detection'
        ]
        
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required section: {section}")
        
        # Check pattern validity
        patterns = self.get_extractor_patterns()
        import re
        for name, pattern in patterns.items():
            try:
                re.compile(pattern)
            except re.error as e:
                errors.append(f"Invalid regex pattern '{name}': {e}")
        
        return (len(errors) == 0, errors)
    
    def to_dict(self) -> dict:
        """Get full configuration as dictionary."""
        return self._config.copy()


# =============================================================================
# Global Configuration Instance
# =============================================================================

# Singleton instance - load once and reuse
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Optional custom path to config file.
        
    Returns:
        Config instance (cached).
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config.load(config_path)
    
    return _config_instance


def reload_config(config_path: Optional[Path] = None) -> Config:
    """
    Reload global configuration.
    
    Use this when config file has been updated.
    
    Args:
        config_path: Optional custom path to config file.
        
    Returns:
        Config instance (freshly loaded).
    """
    global _config_instance
    _config_instance = Config.reload(config_path)
    return _config_instance


# Convenience imports for common patterns
def get_invoice_keywords() -> list:
    """Get all invoice keywords."""
    return get_config().get_all_invoice_keywords()


def get_non_invoice_keywords() -> list:
    """Get all non-invoice keywords."""
    config = get_config()
    return config.get_core_non_invoice_keywords()


def get_extractor_pattern(name: str) -> Optional[str]:
    """Get extractor pattern by name."""
    return get_config().get_extractor_pattern(name)
