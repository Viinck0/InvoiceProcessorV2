"""
OCR Text Validator
Validuje text po OCR z Tesseractu/RapidOCR:
- Kontroluje pravopis (čeština/angličtina)
- Detekuje halucinovaná slova (nesmyslné znaky)
- Opravuje běžné OCR chyby
- Vrací pouze smysluplná slova a zachovává souřadnice bloků pro prostorové formátování
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("⚠️ pyspellchecker not installed. Install: pip install pyspellchecker")

logger = logging.getLogger(__name__)


class OCRTextValidator:
    """
    Validuje a čistí text po OCR s podporou pro prostorové souřadnice (x, y, bbox).
    
    Workflow:
    1. Rozdělení textu/bloků na slova
    2. Detekce jazyka (čeština/angličtina)
    3. Kontrola pravopisu
    4. Oprava běžných OCR chyb
    5. Filtrace halucinovaných slov
    6. Rekonstrukce smysluplného textu a zachování validních bloků se souřadnicemi
    """
    
    # Běžné OCR chyby (Tesseract confusion matrix)
    OCR_CONFUSIONS = {
        # Číslo-písmeno-číslo opravy (pro adresy typu "2I6" → "216")
        '2I6': '216', '2I5': '215', '2I4': '214', '2I3': '213', '2I2': '212', '2I1': '211',
        '3I6': '316', '3I5': '315', '3I4': '314', '3I3': '313', '3I2': '312', '3I1': '311',
        '4I6': '416', '4I5': '415', '4I4': '414', '4I3': '413', '4I2': '412',
        '5I6': '516', '5I5': '515', '5I4': '514', '5I3': '513', '5I2': '512',
        'I6': '16', 'I5': '15', 'I4': '14', 'I3': '13', 'I2': '12', 'I1': '11',
        'l6': '16', 'l5': '15', 'l4': '14', 'l3': '13', 'l2': '12', 'l1': '11',
        
        # Speciální znaky
        'č': 'č', 'ć': 'č', 'ċ': 'č',
        'š': 'š', 'ś': 'š',
        'ž': 'ž', 'ź': 'ž',
        'ě': 'ě', 'e': 'e',
        'ř': 'ř', 'r': 'r',
        'ň': 'ň', 'n': 'n',
        'ť': 'ť', 't': 't',
        'ď': 'ď', 'd': 'd',
        'ů': 'ů', 'u': 'u',
        
        # Časté Tesseract chyby
        'rn': 'm', 'nv': 'm', 'cl': 'd', 'ct': 'd',
        'vv': 'w', 'Vv': 'W',
        '´': "'", '`': "'",
    }
    
    # Whitelist pro faktury - tato slova vždy považovat za platná
    INVOICE_WHITELIST = {
        'faktura', 'faktúry', 'daňový', 'doklad', 'dodavatel', 'odběratel',
        'ičo', 'dič', 'splatnosti', 'vystavení', 'úhradě', 'celkem', 'bez',
        'dpH', 'DPH', 'sazba', 'množství', 'jednotka', 'cena', 'celkem',
        'variabilní', 'symbol', 'konstantní', 'specifický', 'banka', 'účet',
        'IBAN', 'BIC', 'SWIFT', 'platba', 'převodem', 'hotově', 'dobírka',
        'datum', 'měsíc', 'rok', 'číslo', 'objednávky', 'smlouvy',
        'adresou', 'sídlem', 'zastoupený', 'společnost', 'firma',
        's.r.o.', 'a.s.', 'v.o.s.', 'spol.', 'r.o.', 'z.s.',
        'Kč', 'EUR', 'USD', 'CZK', 'Sk', 'zl',
        'ks', 'kg', 'km', 'hod', 'm2', 'm3', 'cm', 'mm', 'l', 'ml', 'g', 'mg',
        'invoice', 'tax', 'document', 'supplier', 'customer', 'vendor',
        'vat', 'amount', 'total', 'quantity', 'unit', 'price',
        'payment', 'bank', 'account', 'due', 'date', 'issue', 'number',
        'order', 'contract', 'company', 'address', 'registered', 'office',
        'represented', 'Ltd', 'Inc', 'GmbH', 'AG', 'SA', 'NV', 'GBP',
        'pcs', 'hours', 'hr', 'hrs',
        # Města a kraje - pouze pro validaci OCR (nejsou to fakturační klíčová slova!)
        'Ústecký', 'ústecký', 'Ústí', 'ústí', 'Praha', 'pražský',
        'Brno', 'brněnský', 'Ostrava', 'Plzeň', 'Liberec', 'Zlín',
        # Příjmení - pouze pro validaci OCR (nejsou to fakturační klíčová slova!)
        'Novák', 'Svoboda', 'Dvořák', 'Černý', 'Procházka',
        'Krejčí', 'Němec', 'Navrátil', 'Havlíček', 'Horák',
        'Pokorný', 'Jelínek', 'Kovář', 'Adam', 'Tichý',
        'Beneš', 'Čech', 'Moravec', 'Liška', 'Růžička',
    }
    
    HALLUCINATION_PATTERNS = [
        r'(.)\1{5,}',  
        r'^[aeiouyáéíóúýěů]{6,}$',  
        r'[\u0400-\u04FF]', 
        r'[^a-zA-Z0-9\s.,;:!?\-_\/\\()""\'\áéíóúýčďěňřšťžůÁÉÍÓÚÝČĎĚŇŘŠŤŽŮ€$£@&%]', 
    ]
    
    VALID_WORD_PATTERNS = [
        r'^\d{1,2}\.\d{1,2}\.\d{4}$',  
        r'^\d{4}-\d{2}-\d{2}$',  
        r'^CZ\d{2,3}\d{4,6}[- ]?\d{2,4}[- ]?\d{2,4}$',  
        r'^\d{6,10}$',  
        r'^[A-Z]{2}\d{8,12}$',  
        r'^[A-Z0-9-]{3,15}$',  
        r'^\d+([.,]\d{1,2})?$',  
        r'^[A-Z]\.$',  
    ]
    
    SHORT_WORDS_CS = {
        'a', 'i', 'k', 'o', 's', 'u', 'v', 'z', 'na', 've', 'se', 'ke',
        'ze', 'do', 'po', 'pro', 'bez', 'při', 'už', 'jen', 'tak',
        'jak', 'kdy', 'kde', 'kam', 'kým', 'mu', 'mi', 'ti', 'to', 'ta',
        'ten', 'tam', 'tu', 'té', 'tí', 'tě', 'mí', 'ní', 'jí',
        'by', 'li', 'zdali', 'což', 'jež', 'je', 'či', 'ci', 'zí', 'rí', 'ne'
    }

    SHORT_WORDS_EN = {
        'a', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'hi', 'if',
        'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so',
        'to', 'up', 'us', 'we', 'am', 'id', 'en', 'ex', 'ad', 're', 
        'th', 'er', 'po'
    }

    SHORT_WORDS_COMMON = {
        'www', 'http', 'https', 'ftp', 'pdf', 'jpg', 'png', 'doc', 'xls',
        'id', 'ID', 'no', 'No', 'nr', 'Nr', 'č', 'Č',
        'IBAN', 'BIC', 'SWIFT', 'VAT', 'TAX', 'GmbH', 'AG', 'SA', 'NV', 'LLC'
    }
    
    def __init__(self, language: str = "auto"):
        self.language = language
        self.spell_cs = None
        self.spell_en = None
        
        if SPELLCHECKER_AVAILABLE:
            try:
                self.spell_cs = SpellChecker(language="cs", local_dictionary=Path(__file__).parent / "cs.json")
            except (ValueError, FileNotFoundError) as e:
                logger.debug(f"Czech spellchecker not available: {e}")
                self.spell_cs = SpellChecker()
            
            try:
                self.spell_en = SpellChecker(language="en")
            except ValueError:
                self.spell_en = SpellChecker()
        else:
            logger.warning("SpellChecker not available - using pattern-based validation only")
    
    def validate(self, data: Union[str, List[Dict]]) -> Dict:
        """Hlavní validace OCR textu nebo strukturovaných OCR bloků."""
        
        # 1. Normalizace vstupu na formát bloků
        if isinstance(data, str):
            if not data or len(data.strip()) < 10:
                return self._empty_result(data)
            blocks = [{"text": data}]
        elif isinstance(data, list):
            blocks = data
            if not blocks:
                return self._empty_result("")
        else:
            raise ValueError("Input data must be a string or a list of dictionaries.")

        # 2. Extrahování všech slov pro detekci jazyka
        all_words = []
        for block in blocks:
            text = block.get('text', '')
            if isinstance(text, str):
                all_words.extend(self._tokenize(text))

        # Detekce jazyka
        detected_lang = self._detect_language(all_words)
        
        # 3. Validace po blocích
        corrections = []
        hallucinations = []
        valid_words_flat = []
        valid_blocks = [] # Ukládání opravených bloků i se souřadnicemi
        
        for block in blocks:
            original_text = block.get("text", "")
            if not isinstance(original_text, str) or not original_text.strip():
                continue

            words_in_block = self._tokenize(original_text)
            valid_block_words = []
            
            for word in words_in_block:
                result = self._validate_word(word, detected_lang)
                
                if result["status"] == "valid":
                    valid_block_words.append(word)
                    valid_words_flat.append(word)
                elif result["status"] == "corrected":
                    valid_block_words.append(result["corrected"])
                    valid_words_flat.append(result["corrected"])
                    corrections.append({
                        "original": word,
                        "corrected": result["corrected"],
                        "reason": result["reason"]
                    })
                elif result["status"] == "hallucination":
                    hallucinations.append(word)
                    corrections.append({
                        "original": word,
                        "corrected": None,
                        "reason": result["reason"]
                    })
            
            # Pokud v bloku zůstalo nějaké smysluplné slovo, zrekonstruujeme text a uložíme blok
            if valid_block_words:
                new_block = block.copy() # Zachová 'bbox', 'x', 'y' atd.
                new_block["text"] = " ".join(valid_block_words)
                valid_blocks.append(new_block)
        
        # Rekonstrukce plochého textu
        valid_text = " ".join(valid_words_flat)
        
        total_words = len(all_words)
        valid_count = len(valid_words_flat)
        hallucinated_count = len(hallucinations)
        
        confidence = (valid_count / total_words) if total_words > 0 else 0.0
        
        if hallucinated_count > 0:
            hallucination_penalty = min(0.5, hallucinated_count / total_words * 0.5)
            confidence -= hallucination_penalty
        
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "valid_text": valid_text,
            "valid_blocks": valid_blocks, # Předání bloků pro další zpracování
            "original_words": total_words,
            "valid_words": valid_count,
            "corrected_words": len(corrections),
            "hallucinated_words": hallucinated_count,
            "corrections": corrections,
            "hallucinations": hallucinations,
            "confidence": round(confidence, 3),
            "detected_language": detected_lang
        }

    def _empty_result(self, raw_text: str) -> Dict:
        """Pomocná metoda pro vrácení prázdného výsledku."""
        return {
            "valid_text": raw_text if isinstance(raw_text, str) else "",
            "valid_blocks": [],
            "original_words": 0,
            "valid_words": 0,
            "corrected_words": 0,
            "hallucinated_words": 0,
            "corrections": [],
            "hallucinations": [],
            "confidence": 0.0,
            "detected_language": "unknown"
        }
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.replace('\n', ' ').replace('\t', ' ')
        raw_words = text.split()
        
        cleaned_words = []
        for word in raw_words:
            word = word.strip()
            if not word:
                continue
            
            if word in ('€', '$', '£', '%'):
                cleaned_words.append(word)
                continue
                
            cleaned = word.strip('.,;:!?()[]{}"\'')
            if cleaned:
                cleaned_words.append(cleaned)
                
        return cleaned_words
    
    def _detect_language(self, words: List[str]) -> str:
        cs_count = 0
        en_count = 0
        cs_chars = {'á', 'é', 'í', 'ó', 'ú', 'ý', 'č', 'ď', 'ě', 'ň', 'ř', 'š', 'ť', 'ž', 'ů'}
        
        for word in words[:100]:
            word_lower = word.lower()
            if any(c in word_lower for c in cs_chars):
                cs_count += 2
                continue
            if word_lower in {'faktura', 'daňový', 'doklad', 'dodavatel', 'odběratel', 'částka', 'splatnosti'}:
                cs_count += 3
                continue
            if word_lower in {'invoice', 'supplier', 'customer', 'amount', 'vat', 'tax'}:
                en_count += 3
                continue
            if word_lower in self.SHORT_WORDS_CS:
                cs_count += 1
            if word_lower in self.SHORT_WORDS_EN:
                en_count += 1
        
        if cs_count > en_count * 1.5:
            return "cs"
        elif en_count > cs_count * 1.5:
            return "en"
        else:
            return "mixed"
    
    def _validate_word(self, word: str, language: str) -> Dict:
        word_lower = word.lower()
        
        if any(char.isdigit() for char in word):
             return {"status": "valid", "reason": "contains_number"}

        if word_lower in self.INVOICE_WHITELIST:
            return {"status": "valid", "reason": "whitelist"}
        
        for pattern in self.VALID_WORD_PATTERNS:
            if re.match(pattern, word, re.IGNORECASE):
                return {"status": "valid", "reason": f"pattern:{pattern[:30]}"}
        
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, word):
                if len(word) <= 3 and word_lower in (self.SHORT_WORDS_CS | self.SHORT_WORDS_EN):
                    return {"status": "valid", "reason": "short_word_exception"}
                if word in self.SHORT_WORDS_COMMON or word_lower in self.SHORT_WORDS_COMMON:
                    return {"status": "valid", "reason": "common_abbreviation"}
                return {"status": "hallucination", "reason": f"pattern:{pattern[:30]}"}
        
        if language in ("cs", "mixed") and self.spell_cs:
            corrected = self.spell_cs.correction(word_lower)
            if corrected and corrected != word_lower:
                if corrected in self.INVOICE_WHITELIST:
                    return {"status": "corrected", "corrected": corrected, "reason": "spellcheck_cs"}
                if self.spell_cs.unknown([word_lower]) and not self.spell_cs.unknown([corrected]):
                    return {"status": "corrected", "corrected": corrected, "reason": "spellcheck_cs"}
        
        if language in ("en", "mixed") and self.spell_en:
            corrected = self.spell_en.correction(word_lower)
            if corrected and corrected != word_lower:
                if corrected in self.INVOICE_WHITELIST:
                    return {"status": "corrected", "corrected": corrected, "reason": "spellcheck_en"}
                if self.spell_en.unknown([word_lower]) and not self.spell_en.unknown([corrected]):
                    return {"status": "corrected", "corrected": corrected, "reason": "spellcheck_en"}
        
        corrected = self._apply_ocr_corrections(word)
        if corrected != word:
            return {"status": "corrected", "corrected": corrected, "reason": "ocr_confusion"}
        
        return {"status": "valid", "reason": "unknown_but_allowed"}
    
    def _apply_ocr_corrections(self, word: str) -> str:
        corrected = word

        long_patterns = [
            ('2I6', '216'), ('2I5', '215'), ('2I4', '214'), ('2I3', '213'), ('2I2', '212'), ('2I1', '211'),
            ('3I6', '316'), ('3I5', '315'), ('3I4', '314'), ('3I3', '313'), ('3I2', '312'), ('3I1', '311'),
            ('4I6', '416'), ('4I5', '415'), ('4I4', '414'), ('4I3', '413'), ('4I2', '412'),
            ('5I6', '516'), ('5I5', '515'), ('5I4', '514'), ('5I3', '513'), ('5I2', '512'),
            ('I6', '16'), ('I5', '15'), ('I4', '14'), ('I3', '13'), ('I2', '12'), ('I1', '11'),
            ('l6', '16'), ('l5', '15'), ('l4', '14'), ('l3', '13'), ('l2', '12'), ('l1', '11'),
            ('rn', 'm'), ('nv', 'm'), ('cl', 'd'), ('ct', 'd'),
            ('vv', 'w'), ('Vv', 'W'),
        ]
        for wrong, right in long_patterns:
            if wrong in corrected:
                corrected = corrected.replace(wrong, right)

        if not any(c.isdigit() for c in word):
            char_subs = {
                '´': "'", '`': "'",
            }
            for wrong, right in char_subs.items():
                corrected = corrected.replace(wrong, right)

        diacritics = {
            'č': 'č', 'ć': 'č', 'ċ': 'č',
            'š': 'š', 'ś': 'š',
            'ž': 'ž', 'ź': 'ž',
        }
        for wrong, right in diacritics.items():
            corrected = corrected.replace(wrong, right)

        return corrected
    
    def get_summary(self, validation_result: Dict) -> str:
        lines = [
            f"📝 OCR Text Validation Summary",
            f"  Original words: {validation_result['original_words']}",
            f"  Valid words: {validation_result['valid_words']}",
            f"  Corrected: {validation_result['corrected_words']}",
            f"  Hallucinations: {validation_result['hallucinated_words']}",
            f"  Confidence: {validation_result['confidence']:.0%}",
            f"  Language: {validation_result['detected_language']}",
        ]
        
        if validation_result['hallucinations']:
            lines.append(f"  ⚠️ Hallucinated words: {', '.join(validation_result['hallucinations'][:10])}")
        
        if validation_result['corrections']:
            lines.append(f"  ✓ Corrections:")
            for corr in validation_result['corrections'][:5]:
                lines.append(f"    '{corr['original']}' → '{corr['corrected']}' ({corr['reason']})")
        
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Integration helper
# ─────────────────────────────────────────────
def validate_ocr_text(data: Union[str, List[Dict]], language: str = "auto") -> Tuple[str, Dict]:
    """
    Rychlá pomocná funkce pro validaci OCR.
    Vrací dvojici: (očištěný text jako string, výsledek analýzy jako dict obsahující i valid_blocks)
    """
    validator = OCRTextValidator(language=language)
    result = validator.validate(data)
    
    logger.debug(f"OCR Validation: {result['valid_words']}/{result['original_words']} words valid ({result['confidence']:.0%})")
    
    if result['hallucinations']:
        logger.warning(f"  Detected {result['hallucinated_words']} hallucinated words: {result['hallucinations'][:5]}")
    
    # Vracíme pouze dvě hodnoty očekávané enginem. Bloky s pozicemi jsou zabalené ve slovníku result.
    return result["valid_text"], result


if __name__ == "__main__":
    # Testovací data (simulující to, co pošle tvůj OCRExtractor)
    test_blocks = [
        {"text": "FAKTURA", "bbox": {"x0": 10, "y0": 10, "x1": 50, "y1": 20}},
        {"text": "č. 2024001", "bbox": {"x0": 100, "y0": 10, "x1": 150, "y1": 20}},
        {"text": "Dodavatel:", "bbox": {"x0": 10, "y0": 30, "x1": 60, "y1": 40}},
        {"text": "ABC s.r.o., IČO: 12345678", "bbox": {"x0": 100, "y0": 30, "x1": 250, "y1": 40}},
        {"text": "XyZasdfg", "bbox": {"x0": 10, "y0": 50, "x1": 40, "y1": 60}}, # Halucinace
    ]
    
    validator = OCRTextValidator()
    result = validator.validate(test_blocks)
    print(validator.get_summary(result))
    print("\nValid blocks:")
    for b in result["valid_blocks"]:
        print(f" - {b}")