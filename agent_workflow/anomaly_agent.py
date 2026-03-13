"""
Anomaly Detector Agent
Detect non-invoice documents (CVs, certificates, contracts, etc.)

v7.1 (2026-03-04):
- ARCHITECTURAL CHANGE: Agents receive ONLY Markdown data, NO raw text!
- MARKDOWN-FIRST: Přepsáno z JSON formátu na Markdown (kompatibilita s BaseAgent v7)
- Agents work exclusively with:
  1. Markdown table (structured data from text_blocks)
  2. Master Instruction (fixed reference framework for validation)
- Odstraněny legacy JSON parsery
"""

from typing import Optional
import json  # Ponecháno pouze pro načítání lokálních config souborů (není pro LLM)
import logging
import re
from pathlib import Path
from .base_agent import BaseAgent
from .agent_memory import get_shared_memory

# Import security sanitizer
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.security import sanitize_llm_output
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False

logger = logging.getLogger(__name__)


class AnomalyDetectorAgent(BaseAgent):
    """
    Specialized agent for detecting non-invoice documents.

    Detects:
    - CVs and resumes (CZ + EN)
    - Certificates and licenses (CZ + EN)
    - Specific Contracts and agreements
    - Reminders and offers
    - Inquiries and requests
    - Internal documents
    """

    # Base prompt template - keywords will be injected dynamically from rules.yaml
    DETECTION_PROMPT_TEMPLATE = """
=== ANOMALY DETECTION SYSTEM ===
You are specialist for anomaly detection in documents. Your task is to identify documents that are NOT invoices.
The document might be in Czech or English.

=== PRIMARY DATA SOURCE ===
ALWAYS PREFER section "🎯 PLNÝ TEXT PODLE SOUŘADNIC (Master Instruction)".
Contains text sorted into logical blocks with position labels [vlevo] (left) and [vpravo] (right).

=== CRITICAL RULE - DETECTED KEYWORDS ===
⚠️ You MUST ONLY use keywords from the "VALID ANOMALY KEYWORDS" list below!
⚠️ DO NOT invent your own phrases like "Microsoft Word pokročilý"!
⚠️ Only report keywords that ACTUALLY EXIST in the document text!

=== VALID ANOMALY KEYWORDS (USE ONLY THESE) ===
{{valid_keywords_section}}

=== ANOMALY TYPES ===
1. cv_resume: curriculum vitae, resume, education, experience, skills (životopis, praxe, dovednosti, vzdělání).
2. certificate: certificate, certification, diplom (osvědčení, certifikát).
3. contract: contract (employment, lease) (smlouva, dohoda). WARNING: billing "per contract" is OK.
4. reminder: reminder, payment demand (upomínka, výzva k úhradě).
5. offer: price offer, quote, budget (nabídka, kalkulace, rozpočet).
6. inquiry: inquiry, request (poptávka).
7. internal: draft, internal document (koncept, interní).
8. research_report: research report, strategic document (výzkumná zpráva, mapování terénu).
9. logistics_report: pallet list, cargo manifest (without prices) (seznam palet, dodací list).

=== DECISION RULES ===
⚠️ IMPORTANT: If document contains invoice keywords → NOT an anomaly!

INVOICE KEYWORDS (if found → is_anomaly = FALSE):
- "faktura", "invoice", "daňový doklad"
- "total", "celkem", "amount", "částka", "cena", "price"
- "vat", "dpH", "tax", "daň"
- "net", "subtotal", "mezisoučet"
- "supplier", "dodavatel", "customer", "odběratel"
- "iban", "bic", "bank account", "účet"
- "invoice number", "číslo faktury"

RULES:
1. If you see invoice keywords → is_anomaly = FALSE, anomaly_type = null
2. If you see invoice keywords AND classifier rejected → refutes_classifier = TRUE
3. If document has amounts, supplier, customer → is_anomaly = FALSE
4. Use ONLY valid anomaly types from list above (no "life_story"!)
5. **ONLY report keywords from the VALID ANOMALY KEYWORDS list that ACTUALLY EXIST in the text!**

=== FORMAT RULES (STRICT MARKDOWN) ===
- If document looks like normal invoice → Not an anomaly.
- Respond ONLY IN MARKDOWN FORMAT.
- **NO JSON:** Absolute ban on curly braces {{}} anywhere.
- **IMPORTANT:** If you need to write pipe `|`, you MUST write it as `\\|`.
- **NO CONTRADICTIONS:** is_anomaly and anomaly_type must be consistent!
- **KEYWORDS MUST BE FROM THE VALID LIST ABOVE!**

=== USE EXACTLY THIS STRUCTURE ===

## 🚨 Anomaly Detection
**Is Anomaly:** ✅ YES or ❌ NO
**Anomaly Type:** [insert one of types above or null]
**Confidence:** 0.85
**Refutes Classifier (Invoice Found):** ✅ YES or ❌ NO

## 🔑 Detected Keywords
- [word 1 from VALID list ONLY, or "none"]
- [word 2 from VALID list ONLY]

## ⚠️ Flags
- [signal 1, if none write "none"]

## 🧠 Reasoning
[Explanation why this is/isn't anomaly based on Master Instruction]
[And why you're possibly refuting Classifier decision]

=== INPUT STRUCTURE ===
0. 🤖 CLASSIFIER DECISION: Information about whether Classifier rejected the document.
   Your task is to verify if it was wrong!
   (Refutation = YES if you see invoice here)
1. 🎯 PLNÝ TEXT PODLE SOUŘADNIC (Master Instruction): Your MAIN source
2. 📋 Tabulka textových bloků / Text blocks table: Summary of all parts
3. 📜 VIZUALIZACE DOKUMENTU / DOCUMENT VISUALIZATION: Graphic preview
4. SUROVÝ TEXT / RAW TEXT: Raw data

=== DOCUMENT TEXT ===
{{input_data}}"""

    def __init__(self, model: str = "llama3.2", timeout: int = 30, patterns_file: Optional[str] = None, vram_limit_gb: int = None, num_ctx: int = None):
        super().__init__(model, timeout, vram_limit_gb, num_ctx)

        # Load patterns EXCLUSIVELY from config/rules.yaml
        # NO hardcoded fallbacks - rules.yaml is the single source of truth
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config_loader import get_config
            config = get_config()
            self.patterns = config.get_anomaly_rules()
            logger.info("✓ Anomaly patterns loaded from config/rules.yaml")

            # Generate prompt with keywords from rules.yaml
            self.DETECTION_PROMPT = self._build_prompt_with_keywords()
        except Exception as e:
            logger.error(f"⚠️ Config loader failed: {e}. Using EMPTY patterns - anomaly detection will be disabled!")
            logger.error("⚠️ To enable anomaly detection, ensure config/rules.yaml exists and is valid.")
            self.patterns = {}
            self.DETECTION_PROMPT = self._build_prompt_with_keywords()  # Vrátí prompt s chybou

    def _build_prompt_with_keywords(self) -> str:
        """
        🔧 NOVÉ: Dynamicky generuje prompt s keywords z rules.yaml.

        Žádné hardcoded hodnoty - všechna keywords jsou načtena z configu!

        Returns:
            str: Complete prompt with valid keywords section
        """
        if not self.patterns:
            return self.DETECTION_PROMPT_TEMPLATE.replace(
                "{{valid_keywords_section}}", "ERROR: No keywords loaded from rules.yaml!"
            )

        # Build the valid keywords section from rules
        keywords_lines = []
        for anomaly_type, rule_config in self.patterns.items():
            keywords = rule_config.get('keywords', [])
            if keywords:
                keywords_lines.append(f"{anomaly_type}:")
                # Format keywords as comma-separated list
                kw_str = ", ".join(keywords[:15])  # Limit to 15 per category for brevity
                if len(keywords) > 15:
                    kw_str += f", ... ({len(keywords) - 15} more)"
                keywords_lines.append(f"  - {kw_str}")

        valid_keywords_section = "\n".join(keywords_lines)

        logger.debug(f"📋 Generated prompt with {len(self.patterns)} anomaly categories")

        return self.DETECTION_PROMPT_TEMPLATE.replace(
            "{{valid_keywords_section}}", valid_keywords_section
        )

    def analyze(self, markdown_input: str, metadata: Optional[dict] = None, master_instruction: Optional[str] = None) -> dict:
        """
        Analyzuje dokument pro detekci anomálií.

        ARCHITEKTURA: Agent dostává POUZE strukturovaná data:
        - markdown_input: Markdown tabulka + layout (žádný surový text!)
        - master_instruction: Pevný referenční rámec pro validaci

        Args:
            markdown_input: Strukturovaná Markdown data (tabulka + layout)
            metadata: Dodatečná metadata (obsahuje text_blocks pro prostorovou analýzu)
            master_instruction: Pevná instrukce pro validaci
        """
        # First, do fast rule-based detection (na Markdown datech)
        rule_result = self._rule_based_detection(markdown_input, metadata)

        # If rule-based found clear anomaly with high confidence, return immediately
        if rule_result.get('confidence', 0) > 0.9 and rule_result.get('is_anomaly'):
            logger.debug(f"✓ Anomalie detekována pravidly: {rule_result['anomaly_type']}")
            return rule_result

        # Inteligentní krácení textu - zachovat důležité Markdown sekce
        truncated = self._truncate_text_smart(markdown_input, max_chars=8000)

        try:
            master_instruction_section = f"=== MASTER INSTRUCTION (IMPORTANT) ===\n{master_instruction}\n\n" if master_instruction else ""

            classifier_info = ""
            if metadata and "classifier_is_invoice" in metadata:
                is_inv = "YES" if metadata["classifier_is_invoice"] else "NO"
                reasoning = metadata.get("classifier_reasoning", "Unknown")
                classifier_info = f"0. 🤖 CLASSIFIER DECISION:\nDecision (Is invoice?): {is_inv}\nClassifier Reasoning: {reasoning}\n\nCheck this decision. If Classifier rejected the invoice (NO), but you clearly see invoice elements in the table, you must refute it (Refutes Classifier: YES).\n\n"

            prompt = self.DETECTION_PROMPT.replace(
                "=== INPUT STRUCTURE ===",
                master_instruction_section + classifier_info + "=== INPUT STRUCTURE ==="
            ).replace("{{input_data}}", truncated)
        except (KeyError, IndexError) as e:
            logger.warning(f"Prompt format error: {e}")
            return rule_result
        
        try:
            client = self._get_client()
            response = client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "num_predict": 512,
                    "num_ctx": self.num_ctx,
                    "top_p": 0.1,
                    "repeat_penalty": 1.1,
                },
                keep_alive="0s"
            )
            
            raw_output = response.get("response", "")
            
            # v7.0: Volání vlastního parseru místo starého JSONu
            parsed = self._parse_markdown_output(raw_output)

            if parsed is None:
                logger.warning(f"Anomaly detector nevrátil platný Markdown: {raw_output[:150]}...")
                return rule_result

            # 🔧 FIX: Validate AI-detected keywords against actual text before merging
            if parsed.get('detected_keywords'):
                parsed['detected_keywords'] = self._validate_detected_keywords(
                    parsed['detected_keywords'], truncated
                )
            
            # If AI claimed anomaly but all keywords were fake, ignore AI result
            if parsed.get('is_anomaly') and not parsed.get('detected_keywords'):
                logger.warning(f"  🚫 AI tvrdí anomaly ale nemá žádná validní keywords - ignoruji AI výsledek")
                parsed['is_anomaly'] = False
                parsed['anomaly_type'] = None
                parsed['confidence'] = 0.0

            # Merge with rule-based results (pass text for validation)
            result = self._merge_results(rule_result, parsed, truncated)

            # 🔧 FIX: Auto-set refutes_classifier when invoice elements are detected
            # but classifier rejected the document. This prevents false rejections when
            # the classifier is confused but extracted values exist.
            if metadata and "classifier_is_invoice" in metadata:
                classifier_rejected = not metadata["classifier_is_invoice"]

                # Check if anomaly agent detected invoice-related keywords
                invoice_keywords = ['faktura', 'invoice', 'dodavatel', 'odběratel', 'supplier',
                                   'customer', 'částka', 'amount', 'total', 'iban', 'účet']
                detected_kw = result.get('detected_keywords', [])
                reasoning = result.get('reasoning', '').lower()

                has_invoice_keyword = any(kw in str(detected_kw).lower() for kw in invoice_keywords)
                mentions_invoice_elements = any(kw in reasoning for kw in invoice_keywords)

                # If classifier rejected but we found invoice elements, MUST refute
                if classifier_rejected and (has_invoice_keyword or mentions_invoice_elements):
                    if not result.get('refutes_classifier'):
                        logger.warning(f"  🔧 FIX: Auto-setting refutes_classifier=true (invoice elements detected)")
                        result['refutes_classifier'] = True
                        result['reasoning'] += " | Automaticky vyvráceno: detekovány prvky faktury (částky, dodavatel, odběratel)."

            logger.debug(f"✓ Anomalie: {'Detekována' if result['is_anomaly'] else 'Nedetektována'} ({result.get('anomaly_type', 'N/A')})")

            # Store reasoning in shared memory
            try:
                memory = get_shared_memory()
                memory.store_reasoning(
                    agent_name="anomaly",
                    reasoning=result.get('reasoning', ''),
                    confidence=result.get('confidence', 0.0),
                    is_invoice=not result.get('is_anomaly', False),  # Inverse of anomaly
                    metadata={
                        'anomaly_type': result.get('anomaly_type'),
                        'detected_keywords': result.get('detected_keywords', []),
                        'flags': result.get('flags', []),
                        'refutes_classifier': result.get('refutes_classifier', False)
                    }
                )
            except Exception as mem_err:
                logger.warning(f"Failed to store anomaly reasoning in shared memory: {mem_err}")

            return result
            
        except Exception as e:
            logger.error(f"Anomaly detector chyba: {e}")
            return rule_result

    def _parse_markdown_output(self, markdown_text: str) -> dict:
        """
        Rozebere Markdown výstup z AI modelu a převede ho na standardizovaný slovník.
        """
        if not markdown_text or not markdown_text.strip():
            return None

        result = {
            'is_anomaly': False,
            'anomaly_type': None,
            'confidence': 0.0,
            'detected_keywords': [],
            'flags': [],
            'reasoning': '',
            'refutes_classifier': False
        }

        # 1. Je anomálie
        is_anomaly_match = re.search(r'\*\*Is Anomaly:\*\*\s*(✅\s*YES|❌\s*NO|YES|NO|✅\s*ANO|❌\s*NE|ANO|NE|TRUE|FALSE)', markdown_text, re.IGNORECASE)
        if not is_anomaly_match:
            is_anomaly_match = re.search(r'\*\*Je anomálie:\*\*\s*(✅\s*ANO|❌\s*NE|ANO|NE|TRUE|FALSE)', markdown_text, re.IGNORECASE)
        
        if is_anomaly_match:
            val = is_anomaly_match.group(1).upper()
            result['is_anomaly'] = 'ANO' in val or 'TRUE' in val or 'YES' in val

        # 1.5 Vyvrácení Classifieru
        refutes_match = re.search(r'\*\*Refutes Classifier.*:\*\*\s*(✅\s*YES|❌\s*NO|YES|NO)', markdown_text, re.IGNORECASE)
        if not refutes_match:
             refutes_match = re.search(r'\*\*Vyvrácení Classifieru.*:\*\*\s*(✅ ANO|❌ NE)', markdown_text, re.IGNORECASE)
             
        if refutes_match:
            val = refutes_match.group(1).upper()
            result['refutes_classifier'] = 'ANO' in val or 'TRUE' in val or 'YES' in val

        # 2. Typ anomálie
        type_match = re.search(r'\*\*Anomaly Type:\*\*\s*([a-zA-Z_]+)', markdown_text)
        if not type_match:
            type_match = re.search(r'\*\*Typ anomálie:\*\*\s*([a-zA-Z_]+)', markdown_text)
        if type_match:
            val = type_match.group(1).lower()
            if val not in ['null', 'none', 'n/a']:
                result['anomaly_type'] = val

        # 3. Confidence
        conf_match = re.search(r'\*\*Confidence:\*\*\s*([\d.]+)', markdown_text)
        if conf_match:
            try:
                result['confidence'] = float(conf_match.group(1))
            except ValueError:
                pass

        # 4. Klíčová slova
        if "## 🔑 Detected Keywords" in markdown_text or "## 🔑 Nalezená klíčová slova" in markdown_text:
            kw_part = markdown_text.split("## 🔑 Detected Keywords")[1] if "## 🔑 Detected Keywords" in markdown_text else markdown_text.split("## 🔑 Nalezená klíčová slova")[1]
            kw_section = kw_part.split("## ")[0]
            words = re.findall(r'[-•]\s*(.+)', kw_section)
            result['detected_keywords'] = [w.strip() for w in words if w.strip() and "none" not in w.lower() and "žádné" not in w.lower()]

        # 5. Flags
        if "## ⚠️ Flags" in markdown_text or "## ⚠️ Varovné signály" in markdown_text:
            flags_part = markdown_text.split("## ⚠️ Flags")[1] if "## ⚠️ Flags" in markdown_text else markdown_text.split("## ⚠️ Varovné signály")[1]
            flags_section = flags_part.split("## ")[0]
            flags = re.findall(r'[-•]\s*(.+)', flags_section)
            result['flags'] = [f.strip() for f in flags if f.strip() and "none" not in f.lower() and "žádné" not in f.lower()]

        # 6. Reasoning
        if "## 🧠 Reasoning" in markdown_text:
            raw_reasoning = markdown_text.split("## 🧠 Reasoning", 1)[1].strip()
            # 🔒 SECURITY: Sanitize reasoning to prevent prompt injection
            if HAS_SECURITY:
                raw_reasoning = sanitize_llm_output(raw_reasoning)
            result['reasoning'] = raw_reasoning

        # 🔒 SANITY CHECK: Blokování kontradikcí a neplatných hodnot
        # 1. is_anomaly a anomaly_type musí být konzistentní
        if result['is_anomaly'] and not result['anomaly_type']:
            logger.warning(f"Sanity Check: is_anomaly=true ale anomaly_type=null - nastavuji na 'unknown'")
            result['anomaly_type'] = 'unknown'

        if not result['is_anomaly'] and result['anomaly_type']:
            logger.warning(f"Sanity Check: is_anomaly=false ale anomaly_type={result['anomaly_type']} - nulování typu")
            result['anomaly_type'] = None

        # 2. Blokování neplatných typů anomálií
        valid_anomaly_types = ['cv_resume', 'certificate', 'contract', 'reminder', 'offer',
                               'inquiry', 'internal', 'research_report', 'logistics_report', 'unknown']
        if result['anomaly_type'] and result['anomaly_type'] not in valid_anomaly_types:
            logger.warning(f"Sanity Check: Neplatný typ anomálie '{result['anomaly_type']}' - blokováno")
            result['anomaly_type'] = None

        # 3. Pokud jsou detekovány invoice keywords → is_anomaly musí být FALSE
        invoice_keywords = ['faktura', 'invoice', 'total', 'amount', 'vat', 'net', 'subtotal',
                           'supplier', 'customer', 'iban', 'bic']
        detected_kw_lower = [kw.lower() for kw in result.get('detected_keywords', [])]
        has_invoice_kw = any(kw in detected_kw_lower for kw in invoice_keywords)

        if has_invoice_kw and result['is_anomaly']:
            logger.warning(f"Sanity Check: Detekovány invoice keywords ale is_anomaly=true - přepis na FALSE")
            result['is_anomaly'] = False
            result['anomaly_type'] = None
            result['reasoning'] += " | Opraveno: invoice keywords detekovány → není anomálie."

        # 🔧 FIX 4: Sanity check - verify detected_keywords match the claimed anomaly_type
        # This catches cases where AI claims "certificate" but keywords are from different category
        if result['is_anomaly'] and result.get('anomaly_type') and result.get('detected_keywords'):
            anomaly_type = result['anomaly_type']
            detected_kw = result['detected_keywords']
            
            # Get expected keywords for this anomaly type from patterns
            expected_keywords = set()
            if anomaly_type in self.patterns:
                expected_keywords = set(kw.lower() for kw in self.patterns[anomaly_type].get('keywords', []))
            
            if expected_keywords:
                # Check if detected keywords actually belong to the claimed anomaly type
                matching_keywords = [kw for kw in detected_kw if kw.lower() in expected_keywords]
                
                if not matching_keywords:
                    logger.warning(f"  🚫 SANITY CHECK: detected_keywords {detected_kw} nepatří k anomaly_type '{anomaly_type}'!")
                    logger.warning(f"     Očekávaná keywords pro '{anomaly_type}': {list(expected_keywords)[:10]}...")
                    # Reset the result - AI made up a connection that doesn't exist
                    result['is_anomaly'] = False
                    result['anomaly_type'] = None
                    result['detected_keywords'] = []
                    result['confidence'] = 0.0
                    result['reasoning'] += " | SANITY CHECK: keywords neodpovídají typu anomálie."

        # 4. Fallback kontrola, zda se nepovedlo najít aspoň bool
        if 'is_anomaly' not in result and result['confidence'] == 0.0:
            return None

        return result

    def _validate_detected_keywords(self, detected_keywords: list, text: str) -> list:
        """
        Ověří, že detekovaná keywords skutečně existují v textu.

        🔧 FIX: Zabraňuje průchodu halucinovaných keywords z AI modelu.
        🔧 FIX 2: Tolerantnější matching - ignoruje extra slova ve frázích.
        🔧 FIX 3: Načítá keywords EXKLUZIVNĚ z rules.yaml (žádné hardcoded hodnoty!)

        Args:
            detected_keywords: Seznam keywords tvrzených AI modelem
            text: Původní text dokumentu (Markdown)

        Returns:
            Seznam validních keywords, která skutečně existují v textu
        """
        if not detected_keywords:
            return []

        text_lower = text.lower()
        valid_keywords = []

        # 🔧 NOVÉ: Načtení valid anomaly keywords EXKLUZIVNĚ z rules.yaml
        # Žádné hardcoded hodnoty - všechna pravidla jsou v config/rules.yaml
        valid_anomaly_keywords = set()
        try:
            from config_loader import get_config
            config = get_config()
            anomaly_rules = config.get_anomaly_rules()

            for anomaly_type, rule_config in anomaly_rules.items():
                keywords = rule_config.get('keywords', [])
                for kw in keywords:
                    valid_anomaly_keywords.add(kw.lower())

            logger.debug(f"  📋 Načteno {len(valid_anomaly_keywords)} valid anomaly keywords z rules.yaml")
        except Exception as e:
            logger.warning(f"  ⚠️ Nepodařilo se načíst anomaly keywords z rules.yaml: {e}")
            logger.warning(f"  ⚠️ Použiji prázdný seznam valid keywords - všechna keywords budou zamítnuta!")
            # Pokud se nepodaří načíst config, necháme valid_anomaly_keywords prázdné
            # To způsobí že všechna keywords budou zamítnuta (bezpečný fallback)

        for kw in detected_keywords:
            kw_lower = kw.lower().strip()

            # 🔧 NOVÉ: Odstranění "ne" z začátku pokud AI přidalo negaci
            if kw_lower.startswith('ne '):
                kw_lower = kw_lower[3:]

            # Skip "none" or empty
            if kw_lower in ['none', 'žádné', 'nic', '']:
                continue

            # 1. Přesná shoda s valid keywords z rules.yaml
            if kw_lower in valid_anomaly_keywords:
                if re.search(r'\b' + re.escape(kw_lower) + r'\b', text_lower, re.IGNORECASE):
                    valid_keywords.append(kw)
                    continue

            # 2. Zkusit najít část keywordu (pro případy kdy AI přidalo extra slova)
            # Např. "Microsoft Word pokročilý" → zkusit najít "word" nebo "pokročilý"
            found_partial = False
            for valid_kw in valid_anomaly_keywords:
                if valid_kw in kw_lower:
                    # AI přidalo extra slova, ale core keyword je validní
                    if re.search(r'\b' + re.escape(valid_kw) + r'\b', text_lower, re.IGNORECASE):
                        valid_keywords.append(valid_kw)
                        found_partial = True
                        logger.debug(f"  ✓ Partial match: '{kw}' → '{valid_kw}'")
                        break

            if not found_partial:
                # 🔒 SECURITY: AI model si vymyslel keyword, které není v textu
                # NEBO není v rules.yaml
                if valid_anomaly_keywords:
                    logger.warning(f"  🚫 FAKE KEYWORD DETECTED: AI tvrdila '{kw}' ale není v textu!")
                    logger.warning(f"     Text (prvních 500 znaků): {text_lower[:500]}...")
                else:
                    logger.warning(f"  🚫 KEYWORD REJECTED: '{kw}' - rules.yaml not loaded (config error)")

        if len(valid_keywords) < len(detected_keywords):
            logger.warning(f"  ⚠️ ODEBRÁNO {len(detected_keywords) - len(valid_keywords)} falešných keywords!")

        return valid_keywords

    def _check_invoice_keywords(self, text: str) -> dict:
        """
        🔧 NOVÉ: Kontrola zda text obsahuje invoice keywords.

        Pokud dokument obsahuje silné invoice keywords, NEMŮŽE být anomálie.

        Args:
            text: Markdown text dokumentu

        Returns:
            Dict s informacemi o nalezených invoice keywords
        """
        text_lower = text.lower()

        # Silná invoice keywords (když se najdou → určitě je to faktura)
        strong_invoice_keywords = [
            'faktura', 'invoice', 'daňový doklad', 'tax document',
            'celkem k úhradě', 'total amount', 'celkem', 'total',
            'k úhradě', 'amount due', 'částka', 'price',
            'dodavatel', 'odběratel', 'supplier', 'customer',
            'variabilní symbol', 'payment reference',
            'iban', 'bic', 'účet', 'bank account'
        ]

        found_strong = []
        for kw in strong_invoice_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower, re.IGNORECASE):
                found_strong.append(kw)

        # Slabší invoice keywords (podpůrné)
        weak_invoice_keywords = [
            's.r.o.', 'a.s.', 'gmbh', 'ltd', 'inc', 'company',
            'ičo', 'dič', 'vat id', 'tax id',
            'datum vystavení', 'issue date', 'datum splatnosti', 'due date',
            'kč', 'czk', 'eur', '€', '$', '£',
            'bez dph', 'dpH', 'sazba', 'základ daně'
        ]

        found_weak = []
        for kw in weak_invoice_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower, re.IGNORECASE):
                found_weak.append(kw)

        has_strong = len(found_strong) >= 1
        has_weak = len(found_weak) >= 2  # Slabších potřebujeme alespoň 2

        is_definitely_invoice = has_strong or (has_weak and len(found_weak) >= 3)

        result = {
            'has_strong_invoice_keywords': has_strong,
            'has_weak_invoice_keywords': has_weak,
            'found_strong_keywords': found_strong,
            'found_weak_keywords': found_weak,
            'is_definitely_invoice': is_definitely_invoice,
            'invoice_keyword_count': len(found_strong) + len(found_weak)
        }

        if is_definitely_invoice:
            logger.debug(f"  ✓ Invoice keywords detekovány: {found_strong + found_weak[:5]}")
        else:
            logger.debug(f"  ⚠️ Slabé invoice keywords: strong={len(found_strong)}, weak={len(found_weak)}")

        return result

    def _rule_based_detection(self, text: str, metadata: Optional[dict] = None) -> dict:
        """
        Fast rule-based anomaly detection.

        🔧 FIX: Added detailed logging for keyword debugging.
        🔧 FIX 2: Check invoice keywords BEFORE rejecting as anomaly!
        """
        text_lower = text.lower()

        # 🔧 NOVÉ: Nejprve zkontroluj invoice keywords
        invoice_check = self._check_invoice_keywords(text)

        detected_anomalies = []
        all_keywords = []

        logger.debug(f"  🔍 Rule-based detection started, checking {len(self.patterns)} anomaly types")

        for anomaly_type, config in self.patterns.items():
            keywords = config.get('keywords', [])
            threshold = config.get('threshold', 2)
            veto = config.get('veto', False)

            # 🔧 FIX: Use word boundary matching instead of substring matching
            found_keywords = []
            for kw in keywords:
                # 🚫 DŮLEŽITÉ: Ignoruj anomaly keywords pokud jsou v invoice kontextu
                # Např. "objednávka" v "faktura z objednávky č. X" není anomálie!
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower, re.IGNORECASE):
                    found_keywords.append(kw)

            if found_keywords:
                logger.debug(f"    {anomaly_type}: nalezeno {len(found_keywords)} keywords (threshold={threshold}, veto={veto}): {found_keywords}")

            if len(found_keywords) >= threshold:
                detected_anomalies.append({
                    'type': anomaly_type,
                    'keywords': found_keywords,
                    'count': len(found_keywords),
                    'threshold': threshold,
                    'veto': config.get('veto', False)
                })
                all_keywords.extend(found_keywords)

        # 🔧 NOVÉ: Pokud jsou invoice keywords → ignoruj anomálie!
        if invoice_check['is_definitely_invoice']:
            logger.debug(f"  ✓ Invoice keywords přebíjejí anomálie! Strong: {invoice_check['found_strong_keywords']}")
            return {
                'is_anomaly': False,
                'anomaly_type': None,
                'confidence': 0.95,
                'detected_keywords': [],
                'flags': [],
                'reasoning': f'Dokument obsahuje invoice keywords: {", ".join(invoice_check["found_strong_keywords"][:3])}'
            }

        if not detected_anomalies:
            logger.debug(f"  ✓ Rule-based: žádné anomálie detekovány")
            return {
                'is_anomaly': False,
                'anomaly_type': None,
                'confidence': 0.95,
                'detected_keywords': [],
                'flags': [],
                'reasoning': 'Žádné anomálie detekovány pravidly'
            }

        # Sort by severity (veto types first)
        detected_anomalies.sort(key=lambda x: (x['veto'], x['count']), reverse=True)
        primary = detected_anomalies[0]

        logger.debug(f"  ⚠️ Rule-based detekovala anomálii: {primary['type']} ({primary['count']} keywords, veto={primary['veto']})")

        # Calculate confidence
        base_confidence = 0.7
        keyword_bonus = min(0.25, (primary['count'] - primary['threshold'] + 1) * 0.05)
        confidence = base_confidence + keyword_bonus

        return {
            'is_anomaly': True,
            'anomaly_type': primary['type'],
            'confidence': confidence,
            'detected_keywords': list(set(all_keywords)),
            'flags': ['veto'] if primary.get('veto') else [],
            'reasoning': f"Nalezeno {primary['count']} keyword pro {primary['type']}"
        }
    
    def _merge_results(self, rule_result: dict, ai_result: dict, text: str = None) -> dict:
        """
        Merge rule-based and AI results.
        
        🔧 FIX: Added keyword validation and improved conflict resolution.
        """
        # Validate AI keywords if text is available
        if text and ai_result.get('detected_keywords'):
            ai_result['detected_keywords'] = self._validate_detected_keywords(
                ai_result['detected_keywords'], text
            )
        
        # If AI claimed anomaly but has no valid keywords, downgrade confidence
        if ai_result.get('is_anomaly') and not ai_result.get('detected_keywords'):
            logger.warning(f"  ⚠️ AI tvrdí anomaly bez keywords - snižuji confidence")
            ai_result['confidence'] = ai_result.get('confidence', 0.5) * 0.3
            ai_result['is_anomaly'] = False
            ai_result['anomaly_type'] = None
        
        if rule_result['is_anomaly'] and ai_result.get('is_anomaly'):
            confidence = max(rule_result['confidence'], ai_result.get('confidence', 0))
            merged_keywords = list(set(
                rule_result.get('detected_keywords', []) +
                ai_result.get('detected_keywords', [])
            ))
            logger.debug(f"  ✓ Merge: oba agenty detekovaly anomálii, confidence={confidence}, keywords={merged_keywords}")
            return {
                **ai_result,
                'confidence': confidence,
                'detected_keywords': merged_keywords
            }

        if rule_result['is_anomaly'] and not ai_result.get('is_anomaly'):
            if rule_result['confidence'] > 0.85:
                logger.debug(f"  ✓ Merge: rule-based vysoká confidence ({rule_result['confidence']}), přebíjím AI")
                return rule_result
            ai_result['confidence'] = ai_result.get('confidence', 0.5) * 0.8
            logger.debug(f"  ✓ Merge: rule-based nízká confidence, AI zamítá - kompromis")
            return ai_result

        if ai_result.get('is_anomaly') and not rule_result['is_anomaly']:
            # AI alone detected anomaly - reduce confidence due to lack of rule confirmation
            ai_result['confidence'] = ai_result.get('confidence', 0.5) * 0.7
            logger.debug(f"  ⚠️ Merge: pouze AI detekovalo anomálii (bez rule confirmation) - snižuji confidence na {ai_result['confidence']}")
            return ai_result

        logger.debug(f"  ✓ Merge: shoda - žádná anomálie detekována")
        return {
            'is_anomaly': False,
            'anomaly_type': None,
            'confidence': 0.95,
            'detected_keywords': [],
            'flags': [],
            'reasoning': 'Žádné anomálie detekovány (shoda)'
        }