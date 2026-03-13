"""
Base Agent Class
Abstract base class for all agents in the workflow

v6.8:
- Podpora pro nastavení VRAM a num_ctx z GUI
- PŘIDÁNO: Nástroje pro formátování OCR dat (Markdown a prostorové uspořádání)
- NOVĚ: Komunikace agentů VÝHRADNĚ přes Markdown tabulky (bez JSONu)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List, Dict
import logging
import os
import re

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Základní třída pro všechny agenty analyzující dokumenty.
    Poskytuje sdílené metody pro parsování výstupů (Markdown) a formátování vstupů (prostorové OCR).
    """

    # Class-level VRAM setting (sdílené napříč agenty)
    _vram_limit_gb: int = 4  # Výchozí 4GB
    _num_ctx: int = 4096     # Výchozí velikost kontextového okna
    
    def __init__(self, model: str = "llama3.2", timeout: int = 30, vram_limit_gb: int = None, num_ctx: int = None):
        """
        Inicializace základního agenta.

        Args:
            model: Jméno modelu v Ollama
            timeout: Časový limit požadavku v sekundách
            vram_limit_gb: Limit VRAM v GB (volitelný, přepíše nastavení třídy)
            num_ctx: Velikost kontextu (volitelný, přepíše nastavení třídy)
        """
        self.model = model
        self.timeout = timeout
        
        # Nastavení specifické pro instanci nebo třídu
        self.vram_limit_gb = vram_limit_gb if vram_limit_gb is not None else BaseAgent._vram_limit_gb
        self.num_ctx = num_ctx if num_ctx is not None else BaseAgent._num_ctx

    def _get_client(self) -> Any:
        """
        Získá a vrátí instanci klienta pro komunikaci s modelem (např. Ollama).
        """
        import ollama
        return ollama

    def _truncate_text(self, text: str, max_chars: int = 15000) -> str:
        """
        Zkrátí text na maximální povolenou délku, aby se vešel do kontextového okna.
        """
        if not text:
            return ""
        if len(text) > max_chars:
            logger.warning(f"Text byl zkrácen z {len(text)} na {max_chars} znaků.")
            return text[:max_chars]
        return text

    def _truncate_text_smart(self, text: str, max_chars: int = 8000, preserve_markdown_table: bool = True) -> str:
        """
        Inteligentní krácení textu s prioritou zachování důležitých částí.

        ⚠️ DŮLEŽITÉ: Markdown tabulky se souřadnicemi (📍 Prostorové rozložení)
        se NIKDY neořezávají - mají povolenu libovolnou délku!

        Priorita:
        1. Začátek textu (úvodní informace) - může být ořezán
        2. Markdown tabulka se souřadnicemi (📍 Prostorové rozložení) - NIKDY neořezovat
        3. Vizualizace dokumentu (📜 VIZUALIZACE) - NIKDY neořezovat

        Pokud je text příliš dlouhý, ořeže se POUZE střední část mezi úvodem a tabulkou.
        Výsledná délka může překročit max_chars kvůli zachování tabulky.
        """
        if not text:
            return ""

        if len(text) <= max_chars:
            return text

        # Najít důležité sekce - tabulka souřadnic
        spatial_table_start = text.find("## 📍 Textové bloky s pozicemi")
        if spatial_table_start == -1:
            spatial_table_start = text.find("📍 Prostorové rozložení")
        if spatial_table_start == -1:
            spatial_table_start = text.find("| Text | X0 | Y0")

        # Najít vizualizaci
        visualization_start = text.find("## 📜 VIZUALIZACE")
        if visualization_start == -1:
            visualization_start = text.find("📜 VIZUALIZACE")

        # Pokud nemáme tabulku ani vizualizaci, použít jednoduché krácení
        if spatial_table_start == -1 and visualization_start == -1:
            logger.warning(f"Text byl zkrácen z {len(text)} na {max_chars} znaků.")
            return text[:max_chars]

        # Tabulka/vizualizace nalezena - NIKDY ji neořezávat!
        # Ořezat pouze úvodní část před tabulkou

        if spatial_table_start != -1:
            # Najít konec úvodní části (před tabulkou)
            intro_end = spatial_table_start
            
            # Výpočet kolik můžeme nechat ze začátku
            max_intro_length = max_chars // 2  # Polovina pro úvod
            
            if intro_end > max_intro_length:
                # Najít poslední konec věty před oříznutím
                last_period = text.rfind('.', 0, max_intro_length)
                last_newline = text.rfind('\n', 0, max_intro_length)
                cut_point = max(last_period, last_newline) if last_period > 10 or last_newline > 10 else max_intro_length
                
                # Zachovat začátek + celou tabulku (bez ořezávání!)
                truncated_text = text[:cut_point] + "\n\n[... střed textu byl zkrácen ...]\n\n" + text[spatial_table_start:]
                logger.info(f"Text byl inteligentně zkrácen: úvod z {intro_end} na {cut_point} znaků, tabulka zachována celá ({len(text) - spatial_table_start} znaků). Celková délka: {len(truncated_text)} znaků.")
                return truncated_text
            else:
                # Úvod se vejde, vrátit celý text i s tabulkou
                logger.info(f"Text {len(text)} znaků se vejde do limitu (zachována celá tabulka)")
                return text

        elif visualization_start != -1:
            # Pouze vizualizace, bez tabulky souřadnic
            max_intro_length = max_chars // 2
            
            if visualization_start > max_intro_length:
                last_period = text.rfind('.', 0, max_intro_length)
                last_newline = text.rfind('\n', 0, max_intro_length)
                cut_point = max(last_period, last_newline) if last_period > 10 or last_newline > 10 else max_intro_length
                
                truncated_text = text[:cut_point] + "\n\n[... střed textu byl zkrácen ...]\n\n" + text[visualization_start:]
                logger.info(f"Text byl inteligentně zkrácen: úvod z {visualization_start} na {cut_point} znaků, vizualizace zachována celá. Celková délka: {len(truncated_text)} znaků.")
                return truncated_text
            else:
                logger.info(f"Text {len(text)} znaků se vejde do limitu (zachována celá vizualizace)")
                return text

        # Fallback - pokud něco selže
        logger.warning(f"Text byl zkrácen z {len(text)} na {max_chars} znaků.")
        return text[:max_chars]

    def reconstruct_spatial_layout(self, ocr_blocks: List[Dict[str, Any]], y_tolerance: int = 10) -> str:
        """
        Rekonstruuje text se zachováním vizuálního formátování (sloupce, řádky)
        pomocí prostorových souřadnic x a y z OCR.
        """
        if not ocr_blocks or not isinstance(ocr_blocks, list):
            return ""

        if not all(isinstance(b, dict) and 'text' in b and 'x' in b and 'y' in b for b in ocr_blocks):
            logger.warning("Data neobsahují 'x' a 'y' souřadnice. Formátuji jako běžný text.")
            return " ".join([b.get('text', str(b)) for b in ocr_blocks if isinstance(b, dict)])

        ocr_blocks.sort(key=lambda b: b['y'])
        
        lines = []
        current_line = []
        current_y = ocr_blocks[0]['y']

        for block in ocr_blocks:
            if abs(block['y'] - current_y) <= y_tolerance:
                current_line.append(block)
            else:
                current_line.sort(key=lambda b: b['x'])
                lines.append("    |    ".join([b['text'] for b in current_line]))
                current_line = [block]
                current_y = block['y']
                
        if current_line:
            current_line.sort(key=lambda b: b['x'])
            lines.append("    |    ".join([b['text'] for b in current_line]))
            
        return "\n".join(lines)

    def _extract_markdown_table(self, text: str) -> Optional[dict]:
        """
        Robustní parser Markdown tabulky. Odolný vůči změnám názvů v hlavičce.
        """
        if not text or not text.strip():
            return None
            
        result = {}
        lines = text.strip().split('\n')
        is_table = False
        
        for line in lines:
            line = line.strip()
            
            # 1. Detekce formátovacího oddělovače (např. |---|---|)
            # Tímto s jistotou poznáme, že datové řádky začínají HNED pod tímto řádkem
            if re.match(r'^\|[\-\s\|]+\|$', line):
                is_table = True
                continue
                
            # 2. Ignorování hlavičky (řádek s tabulkou před oddělovačem)
            if not is_table and line.startswith('|') and line.endswith('|'):
                continue
                
            # 3. Zpracování datových řádků
            if is_table and line.startswith('|') and line.endswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 2:
                    key = parts[0].replace('**', '').strip()
                    value = parts[1].replace('**', '').strip()
                    
                    if value.lower() in ('null', 'none', '', '-', 'n/a'):
                        result[key] = None
                    else:
                        result[key] = value
            
            # 4. Ukončení čtení, pokud tabulka skončila a začal běžný text
            elif is_table and not line.startswith('|'):
                break
                        
        if not result:
            logger.warning("V odpovědi agenta nebyla nalezena platná Markdown tabulka.")
            return None
            
        return result

    @abstractmethod
    def analyze(self, text: str, metadata: Optional[dict] = None) -> dict:
        """
        Hlavní metoda, kterou musí implementovat každý konkrétní agent.
        """
        pass