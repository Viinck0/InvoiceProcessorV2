"""
Extractor Agent
Extract and validate invoice data fields

v7.5 (2026-03-04):
- ARCHITECTURAL CHANGE: Agents receive ONLY Markdown data, NO raw text!
- MARKDOWN-FIRST: Changed from JSON to Markdown output format
- Agents work exclusively with:
  1. Markdown table (structured data from text_blocks)
  2. Master Instruction (fixed reference framework for validation)
- Neprůstřelný regex parser pro Markdown tabulky nezávislý na přesných nadpisech.

Uses external configuration from config/rules.yaml for regex patterns.
Edit the YAML file to tune extraction rules without modifying this code.
"""

from typing import Optional
import logging
import re
from .base_agent import BaseAgent
from .agent_memory import get_shared_memory

# Import configuration loader
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config_loader import get_config
    CONFIG = get_config()
except Exception as e:
    logging.warning(f"Config loader not available: {e}. Using built-in defaults.")
    CONFIG = None

# Import security sanitizer
try:
    from core.security import sanitize_llm_output
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False

logger = logging.getLogger(__name__)


class ExtractorAgent(BaseAgent):
    """
    Specialized agent for invoice data extraction.
    Optimized for speed and Markdown output compliance.
    """

    # English-only prompt for invoice data extraction
    EXTRACTION_PROMPT = """
=== INVOICE DATA EXTRACTION ===
You are an AI for invoice data extraction. RETURN ONLY MARKDOWN.
The document might be in Czech or English.

=== PRIMARY DATA SOURCE ===
Your MOST IMPORTANT source is "🎯 PLNÝ TEXT PODLE SOUŘADNIC (Master Instruction)".
Contains text sorted into logical blocks with position labels:
* **[vlevo]**: Information in left column (often supplier).
* **[vpravo]**: Information in right column (often customer, date, amount).

=== CRITICAL ANTI-HALLUCINATION RULE ===
You MUST NOT INVENT values! Extract ONLY what is explicitly in the text. DO NOT TRANSLATE extract values into English, extract them EXACTLY as they appear, keeping Czech words if they are Czech.
If you don't see a value in text, return null. NEVER use template values!
NEVER extract just city name (e.g. "Prague", "Brno", "Praha") as supplier or customer.
Must be company name or person name.

=== EXTRACTION RULES ===
1. **Consistency:** Extract data exactly as they appear in the document. Don't change diacritics.

2. **Supplier vs Customer:**
   - Search in upper part of document.
   - Supplier is who issued the invoice.
   - Customer is who is being billed.

3. **Addresses - CRITICAL:**
   - Address consists of STREET + HOUSE NUMBER + CITY + ZIP CODE.
   - Search for lines BELOW supplier/customer name in SAME column.
   - Address may be split across multiple lines.
   - Example: "Ulice 111" + "111 Mesto" = full address.
   - DON'T extract ICO/DIC as address!

4. **Amount:** Extract TOTAL amount due.
   If multiple, search for words like: "Celkem", "K úhradě", "Grand Total", "Total", "Amount Due".
   Extract only the number.

5. **Currency:** Extract currency code (CZK, EUR, USD, etc.).

6. **LOGISTICS REPORTS - BAN:**
   If document looks like pallet loading report, truck cargo list, or other logistics report:
   (contains piece counts, weights, but MISSING clear supplier, customer and item prices)
   → RETURN EMPTY FIELDS (null).
   Don't try to map pieces or weights to invoice amounts!

7. **NO PROMPT COPYING:**
   Validation Errors section is only for real problems.
   NEVER copy text from instructions and hints (e.g. "to find:").

=== OUTPUT FORMAT (STRICT MARKDOWN) ===
- Respond ONLY IN EXACTLY ONE MARKDOWN TABLE for Extracted Data.
- DO NOT use bullet points or lists for the extracted data!
- **NO JSON:** Absolute ban on curly braces {{}} or JSON format anywhere.
- No accompanying text outside markdown.
- **IMPORTANT:** If you need to write pipe character `|`, you MUST write it as `\\|`.

=== EXAMPLE OF EXPECTED OUTPUT ===
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
| Total Amount Raw | 1500.50 CZK |
| Vendor IČO | 12345678 |
| Vendor DIČ | CZ12345678 |
| Customer IČO | null |

**Completeness Score:** 0.95

## ⚠️ Validation Errors
- [Missing customer IČO]

## 🧠 Reasoning & Location
Vendor found in top block [vlevo]...
Customer address: lines below name in column [vpravo]...

=== USE EXACTLY THIS STRUCTURE ===

## 📋 Extracted Data
| Field | Value |
|------|---------|
| Invoice Number | number or null |
| Vendor Name | supplier name or null |
| Vendor Address | supplier address or null |
| Customer Name | customer name or null |
| Customer Address | customer address or null |
| Issue Date | YYYY-MM-DD or null |
| Due Date | YYYY-MM-DD or null |
| Total Amount | number or null |
| Currency | CZK/EUR/USD or null |
| VAT Amount | number or null |
| Base Amount | number or null |
| Bank Account | account or null |
| Variable Symbol | VS or null |
| Total Amount Raw | amount with currency or null |
| Vendor IČO | IČO or null |
| Vendor DIČ | DIČ or null |
| Customer IČO | IČO or null |

**Completeness Score:** 0.8

## ⚠️ Validation Errors
- [Brief list of missing fields or problems. No hints!]

## 🧠 Reasoning & Location
[Brief description where you found data in Master Instruction]
e.g.: "Supplier found in top block [vlevo]..."
"Customer address: lines below name in column [vpravo]: Street 777, 99955 City"

=== INPUT STRUCTURE ===
1. 🎯 PLNÝ TEXT PODLE SOUŘADNIC (Master Instruction): Your MAIN source
2. 📋 Tabulka textových bloků / Text blocks table: Summary of all parts
3. 📜 VIZUALIZACE DOKUMENTU / DOCUMENT VISUALIZATION: Attempt at graphic reconstruction
4. SUROVÝ TEXT / RAW TEXT: Raw data

{classifier_info}=== DOCUMENT TEXT ===
{input_data}"""

    def __init__(self, model: str = "llama3.2", timeout: int = 60, vram_limit_gb: int = None, num_ctx: int = None):
        super().__init__(model, timeout, vram_limit_gb, num_ctx)
        
        # Load regex patterns from external config
        if CONFIG:
            patterns = CONFIG.get_extractor_patterns()
            self.date_pattern = patterns.get('date', r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})')
            self.amount_pattern = patterns.get('amount', r'(\d{1,3}(?:[\s\.,]\d{3})*(?:[\s\.,]\d{1,2})?)')
            self.ico_pattern = patterns.get('ico', r'(?:ičo|ič|reg\.?\s*no\.?|company\s*no\.?|crn)\s*[:.]?\s*(\d{6,10})')
            self.dic_pattern = patterns.get('dic', r'(?:dič|vat\s*id|tax\s*id|vat\s*no\.?)\s*[:.]?\s*([a-zA-Z]{2}\d{8,12})')
            self.account_pattern = patterns.get('account', r'(cz|sk)?\d{4,6}[- ]?\d{6,10}[- ]?\d{2,4}')
            self.iban_pattern = patterns.get('iban', r'iban:\s*([a-z]{2}\d{2,24})')
        else:
            self.date_pattern = r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})'
            self.amount_pattern = r'(\d{1,3}(?:[\s\.,]\d{3})*(?:[\s\.,]\d{1,2})?)'
            self.ico_pattern = r'(?:ičo|ič|reg\.?\s*no\.?|company\s*no\.?|crn)\s*[:.]?\s*(\d{6,10})'
            self.dic_pattern = r'(?:dič|vat\s*id|tax\s*id|vat\s*no\.?)\s*[:.]?\s*([a-zA-Z]{2}\d{8,12})'
            self.account_pattern = r'(cz|sk)?\d{4,6}[- ]?\d{6,10}[- ]?\d{2,4}'
            self.iban_pattern = r'iban:\s*([a-z]{2}\d{2,24})'

    def analyze(self, markdown_input: str, metadata: Optional[dict] = None, master_instruction: Optional[str] = None) -> dict:
        """
        Extrahuje data z faktury.

        ARCHITEKTURA: Agent dostává POUZE strukturovaná data:
        - markdown_input: Markdown tabulka + layout (žádný surový text!)
        - master_instruction: Pevný referenční rámec pro validaci
        - metadata: Obsahuje text_blocks pro prostorovou extrakci

        Args:
            markdown_input: Strukturovaná Markdown data (tabulka + layout)
            metadata: Dodatečná metadata (obsahuje text_blocks, classifier_elements, extracted_values)
            master_instruction: Pevná instrukce pro validaci
        """
        if not markdown_input or len(markdown_input.strip()) < 50:
            return self._empty_result()

        classifier_hints = {}
        classifier_found_elements = {}
        text_blocks = None
        classifier_is_invoice = False
        classifier_confidence = 0.0

        if metadata:
            text_blocks = metadata.get('text_blocks')
            elements = metadata.get('classifier_elements', {})
            classifier_found_elements['supplier'] = elements.get('supplier_and_buyer_present', False)
            classifier_found_elements['customer'] = elements.get('supplier_and_buyer_present', False)
            classifier_found_elements['amount'] = elements.get('total_amount_present', False)
            classifier_found_elements['date'] = elements.get('date_present', False)

            # === KLÍČOVÉ: Získat informaci zda classifier dokument označil jako fakturu ===
            classifier_is_invoice = metadata.get('classifier_is_invoice', False)
            classifier_confidence = metadata.get('classifier_confidence', 0.0)

            extracted_values = metadata.get('extracted_values', {})
            if extracted_values.get('supplier'): classifier_hints['vendor_name'] = extracted_values['supplier']
            if extracted_values.get('customer'): classifier_hints['customer_name'] = extracted_values['customer']
            if extracted_values.get('amount'): classifier_hints['total_amount_raw'] = extracted_values['amount']
            if extracted_values.get('date'): classifier_hints['issue_date'] = extracted_values['date']
            if extracted_values.get('document_type'): classifier_hints['document_type'] = extracted_values['document_type']

            if not classifier_hints and metadata.get('classifier_reasoning'):
                reasoning = metadata.get('classifier_reasoning', '')
                if elements.get('supplier_and_buyer_present'):
                    supplier_match = re.search(r'Dodavatel\s*[-:]\s*([^,\n]+)', reasoning)
                    buyer_match = re.search(r'Odběratel\s*[-:]\s*([^,\n]+)', reasoning)
                    if supplier_match: classifier_hints['vendor_name'] = supplier_match.group(1).strip()
                    if buyer_match: classifier_hints['customer_name'] = buyer_match.group(1).strip()
                if elements.get('total_amount_present'):
                    amount_match = re.search(r'(\d+(?:[\s,.]\d+)*)\s*(Kč|EUR|USD|CZK)', reasoning, re.IGNORECASE)
                    if amount_match: classifier_hints['total_amount_raw'] = amount_match.group(0).strip()
                if elements.get('date_present'):
                    date_match = re.search(r'(\d{1,2}[-./]\d{1,2}[-./]\d{2,4})', reasoning)
                    if date_match: classifier_hints['issue_date'] = date_match.group(0).strip()

        # === KRITICKÉ: Pokud classifier zamítl dokument s vysokou jistotou, extrakci přeskoč ===
        if not classifier_is_invoice and classifier_confidence >= 0.85:
            logger.info(f"  ⏭️ Extractor přeskočen: Classifier zamítl dokument (confidence: {classifier_confidence:.0%})")
            return self._empty_result("Classifier zamítl dokument - extrakce přeskočena")

        # Inteligentní krácení textu - zachovat důležité Markdown sekce
        truncated = self._truncate_text_smart(markdown_input, max_chars=8000)

        try:
            classifier_info = ""
            if classifier_hints:
                info_lines = []
                if 'vendor_name' in classifier_hints: info_lines.append(f"- Dodavatel k nalezení: {classifier_hints['vendor_name']}")
                if 'customer_name' in classifier_hints: info_lines.append(f"- Odběratel k nalezení: {classifier_hints['customer_name']}")
                if 'total_amount_raw' in classifier_hints: info_lines.append(f"- Částka k nalezení: {classifier_hints['total_amount_raw']}")
                if 'issue_date' in classifier_hints: info_lines.append(f"- Datum k nalezení: {classifier_hints['issue_date']}")
                if info_lines:
                    classifier_info = "\n".join(info_lines) + "\n\n"

            master_instruction_section = f"=== MASTER INSTRUCTION (IMPORTANT) ===\n{master_instruction}\n\n" if master_instruction else ""
            
            base_prompt = self.EXTRACTION_PROMPT.replace(
                "=== INPUT STRUCTURE ===",
                master_instruction_section + "=== INPUT STRUCTURE ==="
            )

            if classifier_info:
                prompt = base_prompt.replace(
                    "{classifier_info}", 
                    f"=== ADDITIONAL INFO FROM CLASSIFIER ===\n{classifier_info}\nUse this information for better extraction.\n\n"
                ).replace(
                    "{input_data}", truncated
                )
            else:
                prompt = base_prompt.replace("{classifier_info}", "").replace("{input_data}", truncated)
                
        except (KeyError, IndexError) as e:
            logger.warning(f"Prompt format error: {e}")
            return self._fallback_extraction(markdown_input)

        try:
            client = self._get_client()
            response = client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "num_predict": 1024,
                    "num_ctx": self.num_ctx,
                    "top_p": 0.1,
                    "repeat_penalty": 1.1,
                },
                keep_alive="0s"
            )

            raw_output = response.get("response", "")
            logger.debug(f"Extractor raw output ({len(raw_output)} chars): {raw_output[:300]}...")

            parsed = self._parse_markdown_output(raw_output)

            if parsed is None:
                logger.warning(f"Extractor nevrátil platný Markdown. Spouštím textový fallback.")
                parsed = self._fallback_markdown_parse(raw_output)
                if parsed is None:
                    return self._fallback_extraction(markdown_input)

            result = self._validate_extraction(parsed, markdown_input)
            
            # === NOVÉ: Záchranná logika z textu reasoning/errors ===
            result = self._post_process_recovery(result, raw_output)

            if text_blocks:
                spatial_result = self._extract_from_text_blocks(text_blocks)
                if spatial_result:
                    for key, value in spatial_result.items():
                        if value and not result.get(key):
                            result[key] = value
            else:
                spatial_result = {}

            filled_fields = set()
            if classifier_hints:
                result, filled_fields = self._fill_missing_with_hints(result, markdown_input, classifier_hints)
            elif classifier_found_elements:
                result = self._extract_with_classifier_guidance(result, markdown_input, classifier_found_elements)
                for elem in classifier_found_elements:
                    if classifier_found_elements[elem]:
                        filled_fields.add(elem.replace('supplier', 'vendor_name').replace('customer', 'customer_name'))

            # Exclude spatial and filled fields from verification (they may not match raw text exactly)
            exclude_from_verify = filled_fields | set(classifier_hints.keys()) | set(spatial_result.keys()) if classifier_hints else set(spatial_result.keys())
            result = self._verify_extraction(result, markdown_input, exclude_fields=exclude_from_verify)

            result['completeness_score'] = self._calculate_completeness(result)
            
            # Store reasoning in shared memory
            try:
                memory = get_shared_memory()
                memory.store_reasoning(
                    agent_name="extractor",
                    reasoning=result.get('reasoning', ''),
                    confidence=result.get('completeness_score', 0.0),
                    is_invoice=None,  # Extractor doesn't make invoice decision
                    metadata={
                        'extracted_data': {k: v for k, v in result.items() if k not in ['reasoning', 'validation_errors']},
                        'validation_errors': result.get('validation_errors', [])
                    }
                )
            except Exception as mem_err:
                logger.warning(f"Failed to store extractor reasoning in shared memory: {mem_err}")
            
            return result

        except Exception as e:
            logger.error(f"Extractor chyba: {e}")
            return self._fallback_extraction(markdown_input)

    def _parse_markdown_output(self, markdown_text: str) -> dict:
        """Robustní parser Markdownu, nezávislý na přesných nadpisech nebo emoji."""
        if not markdown_text or not markdown_text.strip():
            return None

        result = self._empty_result()
        has_data = False

        # Preferuj parsování pouze tabulky "Extracted Data"
        # Tím eliminujeme riziko, že parser omylem sebere jinou tabulku.
        # Fallback: pokud sekci nenajdeme, parsujeme celý text (legacy).
        def _extract_extracted_data_section(md: str) -> str:
            m = re.search(r'(?is)##\s*[^\n]*(?:extracted\s+data|extrahovan[áa]\s+data)\s*\n(.*?)(?:\n##\s+|\Z)', md)
            return (m.group(1).strip() if m and m.group(1) else "")

        md_for_table = _extract_extracted_data_section(markdown_text)
        if not md_for_table:
            md_for_table = markdown_text
        
        # 1. Extrakce dat z jakékoliv Markdown tabulky v textu (nejspolehlivější metoda)
        # Hledáme řádky ve formátu: | Klíč | Hodnota |
        table_rows = re.findall(r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|', md_for_table)
        
        key_mapping = {
            'invoice number': 'invoice_number', 'číslo faktury': 'invoice_number',
            'vendor name': 'vendor_name', 'dodavatel': 'vendor_name',
            'vendor address': 'vendor_address', 'adresa dodavatele': 'vendor_address',
            'customer name': 'customer_name', 'odběratel': 'customer_name',
            'customer address': 'customer_address', 'adresa odběratele': 'customer_address',
            'issue date': 'issue_date', 'datum vystavení': 'issue_date',
            'due date': 'due_date', 'datum splatnosti': 'due_date',
            'total amount': 'total_amount', 'celková částka': 'total_amount',
            'currency': 'currency', 'měna': 'currency',
            'vat amount': 'vat_amount', 'dph': 'vat_amount',
            'base amount': 'base_amount', 'základ bez dph': 'base_amount',
            'bank account': 'bank_account', 'číslo účtu': 'bank_account',
            'variable symbol': 'variable_symbol', 'variabilní symbol': 'variable_symbol',
            'total amount raw': 'total_amount_raw', 'částka s měnou': 'total_amount_raw',
            'vendor ičo': 'vendor_ico', 'ičo dodavatele': 'vendor_ico', 'vendor ico': 'vendor_ico',
            'vendor dič': 'vendor_dic', 'dič dodavatele': 'vendor_dic', 'vendor dic': 'vendor_dic',
            'customer ičo': 'customer_ico', 'ičo odběratele': 'customer_ico', 'customer ico': 'customer_ico',
        }

        for col1, col2 in table_rows:
            pole = col1.strip().lower()
            value = col2.strip()
            
            # Přeskočit formátovací řádky tabulky (---) a hlavičky
            if '---' in pole or pole in ['pole', 'field', 'klíč']:
                continue
                
            # Najít správný klíč (s podporou částečné shody)
            result_key = key_mapping.get(pole)
            if not result_key:
                for k, v in key_mapping.items():
                    if k in pole:
                        result_key = v
                        break

            # Pokud klíč existuje a hodnota není prázdná
            if result_key and value.lower() not in ['null', 'none', '—', '-', '', 'n/a']:
                has_data = True
                if result_key in ['total_amount', 'vat_amount', 'base_amount']:
                    try:
                        # Extrakce čistého čísla z řetězce, i když je tam nepořádek
                        num_str = re.sub(r'[^\d,.]', '', value).replace(',', '.')
                        # Ošetření tisícových oddělovačů (více teček)
                        if num_str.count('.') > 1:
                            parts = num_str.rsplit('.', 1)
                            num_str = parts[0].replace('.', '') + '.' + parts[1]
                        if num_str:
                            result[result_key] = float(num_str)
                    except ValueError:
                        pass
                else:
                    result[result_key] = value

        # 2. Extrakce Completeness Score (Nezávisle na formátování jako ** nebo :)
        score_match = re.search(r'(?i)completeness score[^\d]*([\d.]+)', markdown_text)
        if score_match:
            try:
                result['completeness_score'] = float(score_match.group(1))
            except ValueError:
                pass

        # 3. Extrakce chyb (Validation Errors) - S FILTREM PROTI PAPOUŠKOVÁNÍ
        errors_section = re.split(r'(?i)validation errors', markdown_text)
        if len(errors_section) > 1:
            errors = re.findall(r'[-•*]\s*(.+)', errors_section[1])
            cleaned_errors = []
            for e in errors:
                e_lower = e.strip().lower()
                # Zahoď řádky, které obsahují jen otrocky zkopírované instrukce z promptu
                if not e_lower or "seznam" in e_lower or "k nalezení:" in e_lower or "nápověd" in e_lower or "zde vypiš" in e_lower:
                    continue
                cleaned_errors.append(e.strip())
            result['validation_errors'] = cleaned_errors
            # 🔒 SECURITY: Sanitize validation errors to prevent injection
            if HAS_SECURITY:
                result['validation_errors'] = [sanitize_llm_output(e) for e in cleaned_errors]

        # Pokud jsme z tabulky nic nedostali a skóre je 0, LLM vygenerovalo nesmysl
        if not has_data and result['completeness_score'] == 0.0:
            return None

        return result

    def _fallback_markdown_parse(self, raw_output: str) -> Optional[dict]:
        """Záchrana v případě, že LLM nedodrží formát tabulky a vypíše seznam."""
        result = {}
        patterns_map = {
            'invoice_number': [r'[-•]?\s*Invoice number[:\s]+([^\n]+)', r'[-•]?\s*Číslo faktury[:\s]+([^\n]+)'],
            'vendor_name': [r'[-•]?\s*Vendor name[:\s]+([^\n]+)', r'[-•]?\s*Dodavatel[:\s]+([^\n]+)'],
            'customer_name': [r'[-•]?\s*Customer name[:\s]+([^\n]+)', r'[-•]?\s*Odběratel[:\s]+([^\n]+)'],
            'issue_date': [r'[-•]?\s*Issue date[:\s]+([^\n]+)', r'[-•]?\s*Datum vystavení[:\s]+([^\n]+)'],
            'due_date': [r'[-•]?\s*Due date[:\s]+([^\n]+)', r'[-•]?\s*Datum splatnosti[:\s]+([^\n]+)'],
            'total_amount': [r'[-•]?\s*Total amount[:\s]+([^\n]+)', r'[-•]?\s*Celkem[:\s]+([^\n]+)'],
            'currency': [r'[-•]?\s*Currency[:\s]+([^\n]+)'],
            'bank_account': [r'[-•]?\s*Bank account[:\s]+([^\n]+)', r'[-•]?\s*Účet[:\s]+([^\n]+)'],
            'vendor_ico': [r'[-•]?\s*Vendor ICO[:\s]+([^\n]+)', r'[-•]?\s*IČO[:\s]+([^\n]+)'],
            'vendor_dic': [r'[-•]?\s*Vendor DIC[:\s]+([^\n]+)', r'[-•]?\s*DIČ[:\s]+([^\n]+)'],
        }
        
        for field, field_patterns in patterns_map.items():
            for pattern in field_patterns:
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('-•: ')
                    if value and value.lower() not in ['null', 'none', 'n/a']:
                        if field == 'total_amount':
                            num_match = re.search(r'([\d\s,.]+)', value)
                            if num_match:
                                try:
                                    result[field] = float(num_match.group(1).replace(' ', '').replace(',', '.'))
                                except:
                                    pass
                        else:
                            result[field] = value
                    break

        if result:
            result['validation_errors'] = []
            return result

        return None

    def _extract_from_text_blocks(self, text_blocks: list) -> dict:
        if not text_blocks: return {}
        result = {}

        def _bbox_num(v, default=0.0):
            try:
                if isinstance(v, list):
                    return float(v[0]) if v else float(default)
                return float(v)
            except Exception:
                return float(default)

        def _get_bbox(block):
            bbox = block.get('bbox', {}) or {}
            x0 = _bbox_num(bbox.get('x0', 0))
            y0 = _bbox_num(bbox.get('y0', 0))
            x1 = _bbox_num(bbox.get('x1', 0))
            y1 = _bbox_num(bbox.get('y1', 0))
            return x0, y0, x1, y1

        def _clean_line(t: str) -> str:
            t = (t or "").strip()
            t = re.sub(r"\s+", " ", t)
            return t

        blocks = []
        for b in text_blocks:
            text = _clean_line(b.get('text', ''))
            if not text or len(text) < 2:
                continue
            x0, y0, x1, y1 = _get_bbox(b)
            blocks.append({"text": text, "text_lower": text.lower(), "x0": x0, "y0": y0, "x1": x1, "y1": y1})

        if not blocks:
            return {}

        by_y = sorted(blocks, key=lambda b: b.get('y0', 0))
        max_x = max((b.get('x1', 0) for b in blocks), default=600.0)
        max_y = max((b.get('y1', 0) for b in blocks), default=800.0)
        right_side_x = max_x * 0.6
        left_side_x = max_x * 0.4
        top_y = max_y * 0.3
        
        amount_keywords = ['celkem', 'úhradě', 'total', 'amount', 'částka', 'kč', 'eur', 'czk']
        for block in blocks:
            text = block.get('text_lower', '')
            x0 = block.get('x0', 0)
            if x0 > right_side_x:
                amount_match = re.search(r'(\d+(?:[\s,.]\d+)*)\s*(kč|eur|usd|czk|€|\$|£)?', text, re.IGNORECASE)
                if amount_match:
                    amount_str = amount_match.group(1).strip()
                    currency = amount_match.group(2) or ''
                    y0 = block.get('y0', 0)
                    for other in blocks:
                        other_y = other.get('y0', 0)
                        other_text = other.get('text_lower', '')
                        if abs(other_y - y0) < 20:
                            if any(kw in other_text for kw in amount_keywords):
                                result['total_amount_raw'] = f"{amount_str} {currency}".strip()
                                try:
                                    result['total_amount'] = float(amount_str.replace(' ', '').replace(',', '.'))
                                except: pass
                                break
        
        date_pattern = r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})'
        for block in by_y[:20]:
            y0 = block.get('y0', 0)
            if y0 > top_y: break
            text = block.get('text', '')
            date_match = re.search(date_pattern, text)
            if date_match:
                date_str = date_match.group(0)
                for other in blocks:
                    other_y = other.get('y0', 0)
                    other_text = other.get('text_lower', '')
                    if abs(other_y - y0) < 20:
                        if 'vystaven' in other_text or 'issue' in other_text:
                            result['issue_date'] = self._format_date(date_str)
                        elif 'splatnost' in other_text or 'due' in other_text:
                            result['due_date'] = self._format_date(date_str)
                        else:
                            if not result.get('issue_date'):
                                result['issue_date'] = self._format_date(date_str)
        
        # Vendor/Customer + Address (multi-line) z levé i pravé části stránky.
        supplier_keywords = ['dodavatel', 'supplier', 'vendor', 'odesílatel', 'seller', 'from']
        customer_keywords = ['odběratel', 'customer', 'příjemce', 'objednatel', 'buyer', 'to', 'for']

        def _looks_like_company_or_person(line: str) -> bool:
            l = (line or "").strip()
            if len(l) < 3:
                return False
            # Filtr proti samotnému městu
            if re.fullmatch(r"[A-Za-zÁÉĚÍÓÚÝČĎŇŘŠŤŽáéěíóúýčďňřšťž\s\-]{3,}", l) and len(l.split()) <= 2:
                # pokud je to jen 1-2 slova bez právní formy, je to často město
                if not re.search(r"\b(s\.r\.o\.|a\.s\.|spol\.|ltd\b|inc\b|llc\b|gmbh\b|z\.s\.)\b", l, re.IGNORECASE):
                    # může to být i jméno osoby, ale u faktur typicky bývá víc kontextu
                    return len(l.split()) >= 2
            return True

        def _is_noise(line_lower: str) -> bool:
            if not line_lower:
                return True
            if any(k in line_lower for k in ['faktura', 'invoice', 'daňový doklad', 'tax document']):
                return True
            return False

        # Najdi labely a vezmi následující 1-4 řádky ve stejném sloupci
        def _collect_entity(start_idx: int, left_bound: float, right_bound: float) -> list:
            lines = []
            base_y = by_y[start_idx]['y0']
            base_x = by_y[start_idx]['x0']
            # sbírej další řádky s rostoucím Y, dokud se výrazně neodskočí nebo nepřijde další label
            for j in range(start_idx + 1, min(start_idx + 10, len(by_y))):
                b = by_y[j]
                if b['x0'] < left_bound or b['x0'] > right_bound:
                    continue
                dy = b['y0'] - base_y
                if dy < -10:  # tolerance pro řádky se stejným Y (mohou být v jiném sloupci)
                    continue
                if dy > 160:
                    break
                t = b['text']
                tl = b['text_lower']
                if _is_noise(tl):
                    continue
                if any(kw in tl for kw in supplier_keywords) or any(kw in tl for kw in customer_keywords):
                    break
                if len(t.strip()) < 3:
                    continue
                lines.append(t.strip())
                if len(lines) >= 4:
                    break
            return lines

        def _find_next_line_in_column(start_idx: int, left_bound: float, right_bound: float, max_dy: float = 50) -> Optional[str]:
            """Najde první další řádek ve stejném sloupci s tolerancí Y."""
            base_y = by_y[start_idx]['y0']
            for j in range(start_idx + 1, min(start_idx + 8, len(by_y))):
                b = by_y[j]
                if b['x0'] < left_bound or b['x0'] > right_bound:
                    continue
                dy = b['y0'] - base_y
                if dy < -10 or dy > max_dy:
                    continue
                t = b['text'].strip()
                if len(t) >= 2 and not _is_noise(b['text_lower']):
                    return t
            return None

        def _extract_entity_from_column(blocks_list, start_idx, left_bound, right_bound, result_key_name, result_key_address):
            """Univerzální funkce pro extrakci entity (dodavatel/odběratel) z daného sloupce."""
            b = blocks_list[start_idx]
            tl = b['text_lower']

            if any(kw in tl for kw in supplier_keywords) or any(kw in tl for kw in customer_keywords):
                # Nejprve zkus najít jméno hned pod labelem
                next_line = _find_next_line_in_column(start_idx, left_bound, right_bound, max_dy=50)
                if next_line and not result.get(result_key_name) and _looks_like_company_or_person(next_line):
                    result[result_key_name] = next_line

                # Pak zkus sbírat více řádků pro adresu
                lines = _collect_entity(start_idx, left_bound, right_bound)
                if lines:
                    if not result.get(result_key_name) and _looks_like_company_or_person(lines[0]):
                        result[result_key_name] = lines[0]
                    if len(lines) > 1 and not result.get(result_key_address):
                        # adresa = zbytek řádků (bez IČO/DIČ apod.)
                        addr_lines = []
                        for ln in lines[1:]:
                            lnl = ln.lower()
                            if re.search(r"\b(ič\s*o|ičo|dič|vat|iban|bic|swift|účet|account|tel\.|e-?mail)\b", lnl):
                                continue
                            addr_lines.append(ln)
                        if addr_lines:
                            result[result_key_address] = ", ".join(addr_lines)

        # Hledání v levém sloupci (často dodavatel)
        left_bound = 0.0
        right_bound = left_side_x

        # Speciální detekce: první řádek vlevo nahoře může být vendor_name
        for i, b in enumerate(by_y[:10]):
            if b['y0'] > max_y * 0.2:  # jen horních 20%
                break
            if b['x0'] > right_bound:
                continue
            text = b['text'].strip()
            tl = b['text_lower']
            # První řádek vlevo nahoře, který není hlavička faktury
            if not _is_noise(tl) and 'isdoc' not in tl and 'faktura' not in tl:
                if _looks_like_company_or_person(text) and not result.get('vendor_name'):
                    result['vendor_name'] = text
                    logger.debug(f"Vendor detekován z prvního řádku vlevo: {text}")
                break

        for i, b in enumerate(by_y[:80]):
            if b['y0'] > max_y * 0.6:
                break
            if b['x0'] > right_bound:
                continue
            _extract_entity_from_column(by_y, i, left_bound, right_bound, 'vendor_name', 'vendor_address')
            _extract_entity_from_column(by_y, i, left_bound, right_bound, 'customer_name', 'customer_address')

        # Hledání v pravém sloupci (často odběratel)
        right_column_left_bound = right_side_x
        right_column_right_bound = max_x

        for i, b in enumerate(by_y[:80]):
            if b['y0'] > max_y * 0.6:
                break
            if b['x0'] < right_column_left_bound or b['x0'] > right_column_right_bound:
                continue
            _extract_entity_from_column(by_y, i, right_column_left_bound, right_column_right_bound, 'customer_name', 'customer_address')
            _extract_entity_from_column(by_y, i, right_column_left_bound, right_column_right_bound, 'vendor_name', 'vendor_address')
        
        invoice_num_pattern = r'faktura\s*č\.?\s*[:.]?\s*([^\n]+)'
        for block in by_y[:30]:
            match = re.search(invoice_num_pattern, block.get('text', ''), re.IGNORECASE)
            if match:
                result['invoice_number'] = match.group(1).strip()
                break
        return result

    def _format_date(self, date_str: str) -> Optional[str]:
        try:
            match = re.search(r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})', date_str)
            if match:
                day, month, year = match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except: pass
        return None

    def _validate_extraction(self, result: dict, markdown_input: str) -> dict:
        """Validuje extrahovaná data proti Markdown vstupu."""
        markdown_lower = markdown_input.lower()
        expected_fields = [
            'invoice_number', 'vendor_name', 'vendor_ico', 'vendor_dic',
            'customer_name', 'customer_ico', 'issue_date', 'due_date',
            'total_amount', 'total_amount_raw', 'currency',
            'vat_amount', 'base_amount', 'bank_account', 'variable_symbol'
        ]

        for field in expected_fields:
            if field not in result:
                result[field] = None

        if not result.get('vendor_ico'):
            ico_match = re.search(self.ico_pattern, markdown_lower)
            if ico_match: result['vendor_ico'] = ico_match.group(1)

        if not result.get('vendor_dic'):
            dic_match = re.search(self.dic_pattern, markdown_lower)
            if dic_match: result['vendor_dic'] = dic_match.group(1).upper()

        if not result.get('bank_account'):
            iban_match = re.search(self.iban_pattern, markdown_lower)
            if iban_match:
                result['bank_account'] = iban_match.group(1).upper()
            else:
                account_match = re.search(self.account_pattern, markdown_lower)
                if account_match: result['bank_account'] = account_match.group(0)

        if not result.get('currency'):
            if '€' in markdown_input or 'eur' in markdown_lower: result['currency'] = 'EUR'
            elif '$' in markdown_input or 'usd' in markdown_lower: result['currency'] = 'USD'
            else: result['currency'] = 'CZK'

        if result.get('issue_date'): result['issue_date'] = self._validate_date(result.get('issue_date'))
        if result.get('due_date'): result['due_date'] = self._validate_date(result.get('due_date'))
        if result.get('total_amount') is not None:
            validated_amount = self._validate_amount(result.get('total_amount'))
            if validated_amount is not None: result['total_amount'] = validated_amount

        if 'validation_errors' not in result:
            result['validation_errors'] = []

        return result
    
    def _validate_date(self, date_value) -> Optional[str]:
        """Převede datum do validního formátu (YYYY-MM-DD), i když obsahuje mezery."""
        if not date_value: return None
        
        # Odstraníme veškeré mezery, abychom bez problému našli "26.2.2026" i když původně bylo "26. 2. 2026"
        clean_date = re.sub(r'\s+', '', str(date_value))
        
        if re.match(r'\d{4}-\d{2}-\d{2}', clean_date):
            try:
                match = re.match(r'(\d{4})-(\d{2})-(\d{2})', clean_date)
                if match:
                    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    if month < 1 or month > 12: return None
                    if day < 1 or day > 31: return None
                    return clean_date
            except: pass
            return clean_date
            
        match = re.search(r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})', clean_date)
        if match: return f"{match.group(3)}-{match.group(2).zfill(2)}-{match.group(1).zfill(2)}"
        return None
    
    def _validate_amount(self, amount_value) -> Optional[float]:
        if amount_value is None: return None
        if isinstance(amount_value, (int, float)): return float(amount_value)
        try:
            return float(str(amount_value).replace(' ', '').replace(',', '.'))
        except (ValueError, TypeError): return None
    
    def _verify_extraction(self, result: dict, markdown_input: str, exclude_fields: set = None) -> dict:
        """Verifikuje že extrahovaná data jsou nalezena v Markdown vstupu."""
        markdown_lower = markdown_input.lower()
        exclude_fields = exclude_fields or set()

        # 🔒 SANITY CHECK: Blokování extrakce štítků místo hodnot
        # Zakázané patterny pro vendor_name/customer_name - to jsou ŠTÍTKY ne hodnoty!
        forbidden_label_patterns = [
            r'^datum ', r'^datum$',  # "Datum účinnosti", "Datum transakce"
            r'^číslo ', r'^číslo$',  # "Číslo faktury", "Číslo zákazníka"
            r'^id ', r'^id$',        # "ID transakce", "ID zákazníka"
            r'^daň', r'^dic', r'^ičo', r'^vat',  # "Daňové číslo", "DIČ", "IČO"
            r'^variabilní', r'^konstantní', r'^symbol',  # "Variabilní symbol"
            r'^e-mail', r'^email',   # "E-mail kupujícího"
            r'^platební', r'^frekvence', r'^metoda',  # "Platební metoda"
            r'^účinnosti', r'^transakce', r'^zákazníka', r'^kupujícího',  # Různé genitivy
            r'^přidejte', r'^podrobnosti', r'^správ',  # "Přidejte podrobnosti"
        ]

        for field in ['vendor_name', 'customer_name', 'invoice_number']:
            value = result.get(field)
            if value:
                value_lower = str(value).lower().strip()
                # Check if value matches any forbidden pattern
                for pattern in forbidden_label_patterns:
                    if re.search(pattern, value_lower):
                        logger.warning(f"Sanity Check: Blokována extrakce štítku '{value}' jako {field}")
                        result[field] = None
                        result['validation_errors'].append(f"{field} je štítek, ne hodnota")
                        break
                # Also check for very short values
                if result.get(field) and len(str(result[field])) < 3:
                    result[field] = None

        fields_to_check = [
            ('vendor_name', 'vendor_name'),
            ('customer_name', 'customer_name'),
            ('invoice_number', 'invoice_number'),
            ('bank_account', 'bank_account'),
        ]

        for field_key, _ in fields_to_check:
            if field_key in exclude_fields: continue
            value = result.get(field_key)
            if value:
                value_str = str(value).strip()
                if not value_str or value_str.lower() in ['null', 'none', 'n/a']:
                    result[field_key] = None
                    continue

                value_lower = value_str.lower()
                exact_match = value_lower in markdown_lower

                if not exact_match:
                    core_value = value_str.replace(' s.r.o.', '').replace(' a.s.', '').replace(' spol. s r.o.', '').strip()
                    if len(core_value) > 3 and core_value.lower() in markdown_lower: exact_match = True

                if not exact_match:
                    words = [w for w in value_str.split() if len(w) > 2 and w.lower() not in {'s.r.o.', 'a.s.', 'spol.', 'the', 'and', 'company'}]
                    if words:
                        significant_words = [w for w in words if len(w) >= 4]
                        if significant_words:
                            if sum(1 for w in significant_words if w.lower() in markdown_lower) > 0: exact_match = True
                        else:
                            if sum(1 for w in words if w.lower() in markdown_lower) >= len(words) * 0.5: exact_match = True

                if not exact_match and any(c.isdigit() for c in value_str):
                    digits = "".join(filter(str.isdigit, value_str))
                    if len(digits) >= 4 and digits in markdown_input.replace(" ", "").replace("-", ""): exact_match = True

                if not exact_match:
                    result[field_key] = None
                    result['validation_errors'].append(f"{field_key} nebyl nalezen v Markdown")

        # === Důkladná validace vendor_name a customer_name proti halucinacím ===
        for name_field in ['vendor_name', 'customer_name']:
            if name_field in exclude_fields:
                continue
            value = result.get(name_field)
            if value:  # Only validate if not already blocked by sanity check
                value_str = str(value).strip()
                validation_error = self._validate_name_field(value_str, markdown_input)
                if validation_error:
                    result[name_field] = None
                    result['validation_errors'].append(validation_error)

        # === Validace adres - musí vypadat jako adresa ===
        for addr_field in ['vendor_address', 'customer_address']:
            if addr_field in exclude_fields:
                continue
            value = result.get(addr_field)
            if value:
                value_str = str(value).strip()
                validation_error = self._validate_address_field(value_str, markdown_input)
                if validation_error:
                    result[addr_field] = None
                    result['validation_errors'].append(validation_error)

        # OPRAVA 1: Kontrola data pouze pokud není chráněno v exclude_fields
        if 'issue_date' not in exclude_fields:
            issue_date = result.get('issue_date')
            if issue_date and issue_date != '0000-00-00':
                date_found = False
                try:
                    y, m, d = issue_date.split('-')
                    if (f"{int(d)}" in markdown_lower or f"{d.zfill(2)}" in markdown_lower) and \
                       (f"{int(m)}" in markdown_lower or f"{m.zfill(2)}" in markdown_lower):
                        date_found = True
                except:
                    pass
                if not date_found:
                    date_found = any(re.search(p, markdown_input) for p in [r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', r'\d{4}[./-]\d{1,2}[./-]\d{1,2}'])
                if not date_found:
                    result['issue_date'] = None
                    result['validation_errors'].append("issue_date nebyl nalezen v Markdown")

        # OPRAVA 2: Kontrola částky pouze pokud není chráněna v exclude_fields
        if 'total_amount' not in exclude_fields:
            total_amount = result.get('total_amount')
            if total_amount is not None and total_amount > 0:
                amount_str = str(total_amount).replace('.', ',')
                markdown_normalized = markdown_input.replace(" ", "").replace(",", ".").lower()
                amount_val_str = f"{total_amount:.2f}".rstrip('0').rstrip('.')
                amount_found = (str(total_amount) in markdown_input or amount_str in markdown_input or amount_val_str in markdown_normalized or str(int(total_amount)) in markdown_input)
                if not amount_found:
                    result['total_amount'] = None
                    result['vat_amount'] = None
                    result['base_amount'] = None
                    result['validation_errors'].append("total_amount nebyl nalezen v Markdown")

        TEMPLATE_VALUES = [
            'vzorovy_dodavatel', 'vzorovy_odberatel', 'vzor', 'example',
            '2099-12-31', '2099-01-01', '99999', '111111111',
        ]
        for field in ['vendor_name', 'customer_name', 'invoice_number', 'bank_account']:
            value = result.get(field)
            if value and any(tv in value.lower() for tv in TEMPLATE_VALUES):
                result[field] = None
                result['validation_errors'].append(f"{field} je šablonová hodnota")

        return result

    def _validate_name_field(self, value: str, original_text: str) -> Optional[str]:
        """
        Validuje, že extracted name vypadá jako skutečné jméno firmy/osoby.
        Vrací chybovou hlášku nebo None pokud je vše v pořádku.
        """
        if not value or len(value.strip()) < 2:
            return f"{value} je prázdný"

        value = value.strip()

        # 1. Délka - jméno firmy by nemělo být extrémně dlouhé
        if len(value) > 150:
            return f"{value[:50]}... je příliš dlouhé (max 150 znaků)"

        # 2. Nesmí obsahovat větné zlomky (koncové čárky, tečky uprostřed, spojky na konci)
        if value.endswith(',') or value.endswith('.'):
            return f"{value} končí interpunkcí (větný zlomek)"

        # 3. Nesmí obsahovat spojky na začátku nebo konci (typické pro vytržený text)
        first_word = value.split()[0].lower() if value.split() else ''
        last_word = value.split()[-1].lower() if value.split() else ''
        forbidden_starts = ['a', 'i', 'že', 'který', 'která', 'které', 'zde', 'tento', 'tato', 'toto', 'ten', 'ta', 'to']
        forbidden_ends = ['a', 'i', 'že', 'se', 'si', 'je', 'by', 'v', 'na', 'o', 'u', 'k', 's', 'z']
        if first_word in forbidden_starts:
            return f"{value} začíná spojkou/zájmenem"
        if last_word in forbidden_ends and len(last_word) <= 2:
            return f"{value} končí krátkou spojkou"

        # 4. Detekce zda to není ve skutečnosti adresa místo jména
        # Pokud obsahuje PSČ (3-5 číslic) nebo ulici s číslem popisným, je to podezřelé
        has_postal_code = bool(re.search(r'\b\d{3}\s?\d{2}\b', value))
        has_street_address = bool(re.search(r'\b\d{1,4}\b', value)) and any(kw in value.lower() for kw in ['ulice', 'ul.', 'náměstí', 'nám.', 'třída', 'tř.', 'hlavní', 'masarykova'])
        if has_postal_code or (has_street_address and len(value.split()) >= 4):
            return f"{value[:50]}... vypadá jako adresa, ne jméno firmy/osoby"

        # 5. Nesmí obsahovat více než 5 čísel (jména firem je obvykle nemají, kromě IČO atd.)
        # Ale povolíme více pokud jsou součástí názvu (např. "Firma 2025")
        digit_count = sum(c.isdigit() for c in value)
        digit_sequences = re.findall(r'\d+', value)
        if digit_count > 8 or len(digit_sequences) > 3:
            return f"{value} obsahuje příliš čísel (možná adresa nebo jiný údaj)"

        # 6. Musí obsahovat alespoň jedno písmeno
        if not any(c.isalpha() for c in value):
            return f"{value} neobsahuje písmena"

        # 7. Kontrola proti "město pouze" - pokud je to jen 1-2 slova bez právní formy
        words = value.split()
        if 1 <= len(words) <= 2:
            # Pokud to vypadá jako město (velké písmeno, žádné s.r.o., a.s., atd.)
            if not re.search(r'\b(s\.r\.o\.|a\.s\.|spol\.|ltd|inc|llc|gmbh|z\.s\.|v\.o\.s\.|k\.s\.)\b', value, re.IGNORECASE):
                # Zkontroluj jestli to není jen obecné podstatné jméno
                common_nouns = ['ústav', 'skupina', 'pracoviště', 'firma', 'společnost', 'instituce', 'organizace', 'kategorie', 'mapa', 'pitch', 'průzkum', 'ekosystém']
                value_lower = value.lower()
                if any(noun in value_lower for noun in common_nouns):
                    return f"{value} je obecný pojem, ne jméno firmy"

        # 8. Nesmí obsahovat slova typická pro popisný text
        descriptive_words = ['často', 'ale', 'však', 'zde', 'tam', 'který', 'jenž', 'že', 'protože', 'jakmile', 'když', 'pokud', 'třeba', 'konkrétně', 'silně', 'silněji']
        value_lower = value.lower()
        if any(word in value_lower for word in descriptive_words):
            return f"{value[:50]}... obsahuje popisná slova"

        # 9. Musí být nalezen v textu jako celek nebo s malými úpravami
        text_lower = original_text.lower()
        value_lower = value.lower()
        if value_lower not in text_lower:
            # Zkusit najít alespoň významnou část
            core_value = re.sub(r'\s*(s\.r\.o\.|a\.s\.|spol\.|ltd|inc|llc|gmbh)\s*', '', value_lower, flags=re.IGNORECASE).strip()
            if len(core_value) > 5 and core_value not in text_lower:
                return f"{value} nebyl nalezen v textu"

        return None

    def _validate_address_field(self, value: str, original_text: str) -> Optional[str]:
        """
        Validuje, že extracted address vypadá jako skutečná adresa.
        Vrací chybovou hlášku nebo None pokud je vše v pořádku.
        """
        if not value or len(value.strip()) < 3:
            return f"{value} je příliš krátké"

        value = value.strip()

        # 1. Délka - adresa by neměla být extrémně dlouhá
        if len(value) > 250:
            return f"{value[:50]}... je příliš dlouhé (max 250 znaků)"

        # 2. Adresa typicky obsahuje číslo popisné nebo PSČ
        # Czech PSČ pattern: 3-5 číslic na začátku nebo v textu
        has_czech_postal_code = bool(re.search(r'\b\d{3}\s?\d{2}\b', value))
        has_house_number = bool(re.search(r'\b\d{1,4}\b', value))  # číslo popisné
        has_street_keywords = any(kw in value.lower() for kw in ['ulice', 'ul.', 'náměstí', 'nám.', 'třída', 'tř.', 'street', 'avenue', 'road'])

        # Adresa by měla mít alespoň nějaký strukturní prvek
        is_likely_address = has_czech_postal_code or has_house_number or has_street_keywords

        # 3. Nesmí končit čárkou nebo tečkou (větný zlomek)
        if value.endswith(',') or value.endswith('.'):
            return f"{value} končí interpunkcí (větný zlomek)"

        # 4. Kontrola proti popisnému textu - věty bez adresních prvků jsou podezřelé
        word_count = len(value.split())
        if word_count > 15 and not is_likely_address:
            return f"{value[:50]}... je příliš dlouhá věta bez adresních prvků"

        # 5. Nesmí obsahovat spojky na začátku
        first_word = value.split()[0].lower() if value.split() else ''
        forbidden_starts = ['a', 'i', 'že', 'který', 'která', 'které', 'zde', 'tento', 'tato', 'toto', 'ten', 'ta', 'to', 'diverzifikuje', 'propojuje', 'razí', 'nesnaží']
        if first_word in forbidden_starts:
            return f"{value} začíná spojkou/slovesem"

        # 6. Detekce "příliš mnoho čísel" - ale ignoruj PSČ a čísla popisná
        # Správná adresa může mít 2-3 čísla (PSČ, číslo popisné, číslo orientační)
        digit_sequences = re.findall(r'\d+', value)
        # Pokud má více než 4 samostatných číselných sekvencí, je to podezřelé
        # ALE: pokud vypadají jako PSČ+číslo domu, je to OK
        if len(digit_sequences) > 5 and not has_czech_postal_code:
            return f"{value} obsahuje příliš mnoho číselných sekvencí"

        # 7. Kontrola zda to není jen seznam IČO/DIČ bez skutečné adresy
        ico_dic_pattern = re.search(r'\b(ičo|ič|dič|vat|dic)\s*[:.]?\s*\d', value.lower())
        if ico_dic_pattern and not has_house_number and not has_street_keywords:
            # Pokud obsahuje jen IČO/DIČ bez ulice nebo čísla, není to adresa
            # Ale pokud je tam i něco jiného, může to být součást adresy
            non_ico_text = re.sub(r'\b(ičo|ič|dič|vat|dic)\s*[:.]?\s*\d+\b', '', value, flags=re.IGNORECASE).strip()
            if len(non_ico_text) < 5:
                return f"{value} obsahuje pouze IČO/DIČ bez skutečné adresy"

        # 8. Musí být nalezen v textu (alespoň část)
        text_lower = original_text.lower()
        value_lower = value.lower()
        if value_lower not in text_lower:
            # Zkusit najít alespoň významnou část adresy
            # Odstraníme IČO/DIČ pro kontrolu
            core_address = re.sub(r'\b(ičo|ič|dič|vat|dic)\s*[:.]?\s*\d+\b', '', value_lower, flags=re.IGNORECASE).strip()
            core_address = re.sub(r'\s+', ' ', core_address)
            if len(core_address) > 8 and core_address not in text_lower:
                # Zkusit najít alespoň část adresy (ulice nebo město)
                words = core_address.split()
                significant_words = [w for w in words if len(w) > 3]
                if significant_words:
                    found_count = sum(1 for w in significant_words if w in text_lower)
                    if found_count < len(significant_words) * 0.5:
                        return f"{value[:50]}... nebyla nalezena v textu"

        return None

    def _calculate_completeness(self, result: dict) -> float:
        required_fields = ['vendor_name', 'customer_name', 'issue_date', 'total_amount']
        optional_fields = ['invoice_number', 'due_date', 'currency', 'vendor_ico', 'bank_account']
        score = 0.0
        for field in required_fields:
            if result.get(field): score += 0.15
        for field in optional_fields:
            if result.get(field): score += 0.08
        return min(1.0, score)
    
    def _fill_missing_with_hints(self, result: dict, text: str, hints: dict) -> tuple:
        text_lower = text.lower()
        filled_fields = set()

        if not result.get('vendor_name') and hints.get('vendor_name'):
            hinted_value = hints['vendor_name']
            if hinted_value.lower() in text_lower:
                match = re.search(re.escape(hinted_value), text, re.IGNORECASE)
                if match:
                    result['vendor_name'] = match.group(0)
                    filled_fields.add('vendor_name')
            else:
                words = hinted_value.split()
                stop_words = {'firma', 's.r.o.', 'a.s.', 'spol.', 's', 'r.o.', 'o', 'z.s.', 'ič'}
                key_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
                for key_word in key_words:
                    match = re.search(r'\b' + re.escape(key_word) + r'\b', text, re.IGNORECASE)
                    if match:
                        result['vendor_name'] = match.group(0)
                        filled_fields.add('vendor_name')
                        break

        if not result.get('customer_name') and hints.get('customer_name'):
            hinted_value = hints['customer_name']
            if hinted_value.lower() in text_lower:
                match = re.search(re.escape(hinted_value), text, re.IGNORECASE)
                if match:
                    result['customer_name'] = match.group(0)
                    filled_fields.add('customer_name')
            else:
                words = hinted_value.split()
                stop_words = {'firma', 's.r.o.', 'a.s.', 'spol.', 's', 'r.o.', 'o', 'z.s.', 'ič'}
                key_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
                for key_word in key_words:
                    match = re.search(r'\b' + re.escape(key_word) + r'\b', text, re.IGNORECASE)
                    if match:
                        result['customer_name'] = match.group(0)
                        filled_fields.add('customer_name')
                        break

        if not result.get('issue_date') and hints.get('issue_date'):
            hinted_date = hints['issue_date']
            date_patterns = [re.escape(hinted_date), hinted_date.replace('.', r'[./-]'), hinted_date.replace('-', r'[./-]')]
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    normalized = self._validate_date(match.group(0))
                    if normalized:
                        result['issue_date'] = normalized
                        filled_fields.add('issue_date')
                    break

        if not result.get('total_amount') and hints.get('total_amount_raw'):
            hinted_amount = hints['total_amount_raw']
            amount_match = re.search(r'(\d+(?:[\s,.]\d+)*)', hinted_amount)
            if amount_match:
                amount_str = amount_match.group(1).replace(' ', '').replace(',', '.')
                try:
                    result['total_amount'] = float(amount_str)
                    result['total_amount_raw'] = hinted_amount
                    filled_fields.add('total_amount')
                except ValueError: pass

        return result, filled_fields

    def _extract_with_classifier_guidance(self, result: dict, text: str, found_elements: dict) -> dict:
        if found_elements.get('supplier') and not result.get('vendor_name'):
            for pattern in [r'dodavatel[:\s]+([^,\n]+)', r'vendor[:\s]+([^,\n]+)', r'from[:\s]+([^,\n]+)']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result['vendor_name'] = match.group(1).strip()
                    break
        
        if found_elements.get('customer') and not result.get('customer_name'):
            for pattern in [r'odběratel[:\s]+([^,\n]+)', r'customer[:\s]+([^,\n]+)', r'for[:\s]+([^,\n]+)', r'faktura pro[:\s]+([^,\n]+)']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result['customer_name'] = match.group(1).strip()
                    break
        
        if found_elements.get('amount') and not result.get('total_amount'):
            for pattern in [r'celkem[:\s]+(\d+(?:[\s,.]\d+)*)\s*(Kč|EUR|USD|CZK)', r'total[:\s]+(\d+(?:[\s,.]\d+)*)\s*(Kč|EUR|USD|CZK)', r'k úhradě[:\s]+(\d+(?:[\s,.]\d+)*)\s*(Kč|EUR|USD|CZK)', r'(\d+(?:[\s,.]\d+)*)\s*(Kč|EUR|USD|CZK)']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        result['total_amount'] = float(match.group(1).replace(' ', '').replace('.', '').replace(',', '.'))
                        result['total_amount_raw'] = f"{match.group(1)} {match.group(2)}"
                        break
                    except ValueError: pass
        
        if found_elements.get('date') and not result.get('issue_date'):
            for pattern in [r'datum vystavení[:\s]+(\d{1,2}\.\d{1,2}\.\d{4})', r'issue date[:\s]+(\d{1,2}\.\d{1,2}\.\d{4})', r'(\d{1,2}\.\d{1,2}\.\d{4})', r'(\d{4}-\d{2}-\d{2})']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    normalized = self._validate_date(match.group(0))
                    if normalized:
                        result['issue_date'] = normalized
                        break
        return result

    def _post_process_recovery(self, result: dict, raw_output: str) -> dict:
        """
        Záchranná logika: Pokud některé klíčové pole chybí v tabulce, ale LLM ho vypsalo
        v sekci 'Reasoning' nebo 'Validation Errors', pokusíme se ho odtud vytáhnout.
        """
        # 1. Hledání DATUMU (např. "Datum k nalezení: 26. 2. 2026")
        if not result.get('issue_date'):
            # Hledání v celém textu: "datum" -> libovolné znaky (ne dvojtečka) -> dvojtečka/mezera -> datum
            # Odstraněna nebezpečná nested kvantifikace (?:...)+
            date_match = re.search(r'(?i)datum[^:]*[:\s]+(\d{1,2}[\s.]+\d{1,2}[\s.]+\d{4})', raw_output)
            if not date_match:
                 # Fallback na jakýkoliv formát data v textu u kterého je "datum"
                 date_match = re.search(r'(?i)datum[^:]*[:\s]+([0-9.\-/]{6,10})', raw_output)
            
            if date_match:
                normalized = self._validate_date(date_match.group(1))
                if normalized:
                    result['issue_date'] = normalized
                    logger.debug(f"  ✨ Extractor Recovery: Datum nalezeno v textu -> {normalized}")

        # 2. Hledání ČÁSTKY (např. "Částka k nalezení: 177,00 Kč")
        if result.get('total_amount') is None:
            # Odstraněna nebezpečná nested kvantifikace (?:...)+
            amount_match = re.search(r'(?i)částka[^:]*[:\s]+([\d\s,.]+)\s*(?:Kč|CZK|EUR|USD|€)', raw_output)
            if amount_match:
                try:
                    num_str = amount_match.group(1).replace(' ', '').replace(',', '.')
                    # Ošetření teček jako tisíců
                    if num_str.count('.') > 1:
                        parts = num_str.rsplit('.', 1)
                        num_str = parts[0].replace('.', '') + '.' + parts[1]
                    
                    result['total_amount'] = float(num_str)
                    logger.debug(f"  ✨ Extractor Recovery: Částka nalezena v textu -> {result['total_amount']}")
                except ValueError:
                    pass

        # 3. Hledání VARIABILNÍHO SYMBOLU (častý problém)
        if not result.get('variable_symbol'):
            # Odstraněna nebezpečná nested kvantifikace (?:...)+
            vs_match = re.search(r'(?i)(?:vs|variabilní symbol|symbol)[^:]*[:\s]+(\d{1,10})', raw_output)
            if vs_match:
                result['variable_symbol'] = vs_match.group(1)
                logger.debug(f"  ✨ Extractor Recovery: VS nalezen v textu -> {result['variable_symbol']}")

        return result

    def _empty_result(self) -> dict:
        return {
            'invoice_number': None, 'vendor_name': None, 'vendor_ico': None, 'vendor_dic': None,
            'customer_name': None, 'customer_ico': None, 'issue_date': None, 'due_date': None,
            'total_amount': None, 'total_amount_raw': None, 'currency': None,
            'vat_amount': None, 'base_amount': None, 'bank_account': None, 'variable_symbol': None,
            'vendor_address': None, 'customer_address': None,
            'completeness_score': 0.0, 'validation_errors': ['Žádná data k extrakci']
        }
    
    def _fallback_extraction(self, markdown_input: str) -> dict:
        """Záchranná extrakce z Markdown vstupu."""
        markdown_lower = markdown_input.lower()
        result = self._empty_result()

        date_match = re.search(self.date_pattern, markdown_input)
        if date_match: result['issue_date'] = f"{date_match.group(3)}-{date_match.group(2).zfill(2)}-{date_match.group(1).zfill(2)}"

        amount_match = re.search(self.amount_pattern, markdown_lower)
        if amount_match:
            try: result['total_amount'] = float(amount_match.group(1).replace(' ', '').replace('.', '').replace(',', '.'))
            except ValueError: pass

        if '€' in markdown_input or 'EUR' in markdown_lower or 'eur' in markdown_lower: result['currency'] = 'EUR'
        elif '$' in markdown_input or 'USD' in markdown_lower or 'usd' in markdown_lower: result['currency'] = 'USD'
        elif 'Kč' in markdown_input or 'CZK' in markdown_lower or 'czk' in markdown_lower or 'KC' in markdown_input: result['currency'] = 'CZK'

        company_match = re.search(r'\b([A-ZČŠŽŘĎŤŇĚÁÉÍÓÚÝ][a-zčšžřďťňěáéíóúýA-Za-z\s]+(?:s\.r\.o\.|a\.s\.|spol\.\s+r\.o\.|Ltd\.?|Inc\.?|LLC|GmbH))', markdown_input)
        if company_match: result['vendor_name'] = company_match.group(1).strip()

        result['completeness_score'] = self._calculate_completeness(result)
        return result