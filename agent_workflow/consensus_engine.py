"""
Consensus Engine
Combines results from multiple agents using weighted voting

v7.3 (2026-02-28):
- ODSTRANĚNA HLOUPÁ TEXTOVÁ ANALÝZA REASONINGU (Zabraňuje sabotáži Markdown výstupu)
- Plná podpora pro Markdown boolean výstupy z ClassifierAgenta
- Oprava AttributeError u NON_INVOICE_ANOMALY_TYPES
- Sjednocení a rozšíření NON_INVOICE typů
- Čištění duplicitního kódu

Uses external configuration from config/rules.yaml for keywords and patterns.
Edit the YAML file to tune accuracy without modifying this code.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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

logger = logging.getLogger(__name__)

# Konfigurační konstanty
MIN_REALISTIC_AMOUNT = 10
CHECK_AMOUNT_REALISTIC = False


@dataclass
class AgentResult:
    """Result from a single agent."""
    agent_name: str
    result: dict
    weight: float
    veto_power: bool = False


class ConsensusEngine:
    """
    Combines results from multiple agents using weighted voting.
    """

    def __init__(
        self,
        threshold_accept: float = 0.7,
        threshold_review: float = 0.5,
        anomaly_veto_threshold: float = 0.85
    ):
        self.threshold_accept = threshold_accept
        self.threshold_review = threshold_review
        self.anomaly_veto_threshold = anomaly_veto_threshold

        # Agent weights
        self.weights = {
            'classifier': 0.4,
            'extractor': 0.3,
            'anomaly': 0.3
        }
        
        # Load keywords and patterns from external config
        if CONFIG:
            self.invoice_keywords_cs = CONFIG.get_invoice_keywords_cs()
            self.invoice_keywords_en = CONFIG.get_invoice_keywords_en()
            self.non_invoice_patterns = CONFIG.get_non_invoice_patterns()
        else:
            # Fallback to built-in defaults
            self.invoice_keywords_cs = [
                'faktura', 'faktúry', 'daňový doklad', 'zálohová faktura', 'konečná faktura', 'proforma',
                'dodavatel', 'odběratel', 'objednatel', 'zhotovitel', 'poskytovatel', 'příjemce',
                'ičo', 'dič', 'společnost', 'firma', 's.r.o.', 'a.s.', 'v.o.s.',
                'datum vystavení', 'datum splatnosti', 'vystaveno', 'splatnost', 'du', 'dv',
                'celkem', 'k úhradě', 'částka', 'cena', 'součet', 'úhrada', 'platba',
                'bez dph', 'dpH', 'sazba', 'základ daně',
                'variabilní symbol', 'konstantní symbol', 'specifický symbol', 'banka', 'účet', 'iban', 'bic', 'swift',
                'kč', 'czk', 'eur', '€', 'usd', '$', 'gbp', '£',
            ]
            self.invoice_keywords_en = [
                'invoice', 'tax document', 'proforma', 'bill', 'receipt',
                'supplier', 'vendor', 'customer', 'buyer', 'seller', 'contractor',
                'company', 'limited', 'inc.', 'corp.', 'gmbh',
                'vat', 'tax id', 'registration no',
                'issue date', 'due date', 'date of issue', 'dated',
                'total', 'amount', 'price', 'sum', 'payment', 'balance',
                'excl. vat', 'incl. vat', 'vat rate', 'tax base', 'subtotal',
                'payment reference', 'bank account', 'account no', 'iban', 'bic', 'swift',
                'eur', 'usd', 'gbp', 'czk', 'pln', 'huf', '€', '$', '£',
            ]
            self.non_invoice_patterns = {}
        
        # Combined invoice keywords for easy access
        self.all_invoice_keywords = self.invoice_keywords_cs + self.invoice_keywords_en
        
        # Non-invoice patterns as flat list for quick lookup
        self.non_invoice_anomaly_types = []
        for category, keywords in self.non_invoice_patterns.items():
            if isinstance(keywords, list):
                self.non_invoice_anomaly_types.extend(keywords)
            elif isinstance(keywords, dict):
                self.non_invoice_anomaly_types.extend(keywords.get('keywords', []))
    
    @property
    def NON_INVOICE_ANOMALY_TYPES(self) -> list:
        """Property pro zajištění zpětné kompatibility velkých/malých písmen."""
        return self.non_invoice_anomaly_types

    def calculate_consensus(
        self,
        classifier_result: dict,
        extractor_result: dict,
        anomaly_result: dict,
        raw_text: str = None 
    ) -> dict:
        
        classifier_conf = classifier_result.get('confidence', 0)
        classifier_is_invoice = classifier_result.get('is_invoice', False)

        veto_result = self._check_anomaly_veto(anomaly_result, classifier_result)
        if veto_result:
            return veto_result

        # === ODSTRANĚNA HLOUPÁ TEXTOVÁ ANALÝZA REASONINGU ===
        # Dříve se zde hledala slova jako "není faktura" v reasoningu,
        # což způsobovalo false-positives zamítnutí. Nyní věříme strukturovaným datům.

        # === PŘÍPRAVA DAT PRO KONTROLY ===
        field_check = {
            'číslo faktury': bool(extractor_result.get('invoice_number')),
            'dodavatel': bool(extractor_result.get('vendor_name')),
            'odběratel': bool(extractor_result.get('customer_name')),
            'datum vystavení': bool(extractor_result.get('issue_date') and extractor_result.get('issue_date') != '0000-00-00'),
            'datum splatnosti': bool(extractor_result.get('due_date') and extractor_result.get('due_date') != '0000-00-00'),
            'částka': bool(extractor_result.get('total_amount') is not None or bool(extractor_result.get('total_amount_raw'))),
            'účet/IBAN': bool(extractor_result.get('bank_account')),
        }
        extractor_fields_found = sum(field_check.values())
        missing_fields = [k for k, v in field_check.items() if not v]

        extractor_completeness = extractor_result.get('completeness_score', 0)

        # === KONTROLA ŽIVOTOPISU ===
        # Kontrolujeme vždy když je text k dispozici - i když classifier zamítl!
        if raw_text:
            cv_check = self._check_cv_resume(raw_text)
            if cv_check['is_cv']:
                logger.warning(f"  ⚠️ CV DETEKOVÁN: Text obsahuje znaky životopisu ({cv_check['indicators']})")
                return {
                    'is_invoice': False,
                    'confidence': max(cv_check['confidence'], 0.95),
                    'decision_type': 'auto_reject',
                    'weighted_score': 0.05,
                    'agent_scores': {
                        'classifier': classifier_conf if classifier_is_invoice else 0.0,
                        'extractor': 0.0,
                        'anomaly': 1.0
                    },
                    'agent_agreement': {
                        'full_agreement': True,
                        'majority_agreement': True,
                        'agreement_count': 3,
                        'total_agents': 3
                    },
                    'reasoning': f"Dokument je životopis/CV - detekovány znaky: {', '.join(cv_check['indicators'])}",
                    'cv_detected': True,
                    'cv_indicators': cv_check['indicators']
                }

        # === KONTROLA VÝZKUMNÉ ZPRÁVY / PRŮZKUMU ===
        if classifier_is_invoice and classifier_conf >= 0.5 and raw_text:
            research_check = self._check_research_report(raw_text)
            if research_check['is_research']:
                logger.warning(f"  ⚠️ RESEARCH REPORT DETEKOVÁN: Text obsahuje znaky výzkumné zprávy ({research_check['indicators']})")
                return {
                    'is_invoice': False,
                    'confidence': max(research_check['confidence'], 0.9),
                    'decision_type': 'auto_reject',
                    'weighted_score': 0.1,
                    'agent_scores': {
                        'classifier': 0.0,
                        'extractor': 0.0,
                        'anomaly': 1.0
                    },
                    'agent_agreement': {
                        'full_agreement': True,
                        'majority_agreement': True,
                        'agreement_count': 3,
                        'total_agents': 3
                    },
                    'reasoning': f"Dokument je výzkumná zpráva/průzkum - detekovány znaky: {', '.join(research_check['indicators'])}",
                    'research_report_detected': True,
                    'research_indicators': research_check['indicators']
                }

        # === RESPEKTOVAT VYSOKOU JISTOTU ZAMÍTNUTÍ OD CLASSIFIERU ===
        # Pokud classifier zamítl dokument s vysokou jistotou (≥90%) a extractor
        # nenašel kompletní data faktury, respektovat rozhodnutí classifieru.
        # Tím se předejde situacím kdy extractor "halucinuie" data z CV/resumé.
        if not classifier_is_invoice and classifier_conf >= 0.90:
            # Počítat kolik polí extractor skutečně našel
            if extractor_fields_found < 5:  # Nenašel všech 5 klíčových elementů
                logger.info(f"  ✗ Classifier zamítl s vysokou jistotou ({classifier_conf:.0%}): {classifier_result.get('reasoning', 'N/A')}")
                return {
                    'is_invoice': False,
                    'confidence': classifier_conf,
                    'decision_type': 'auto_reject',
                    'weighted_score': 1.0 - classifier_conf,  # Nízké skóre pro zamítnutí
                    'agent_scores': {
                        'classifier': classifier_conf,
                        'extractor': extractor_completeness,
                        'anomaly': 1 - anomaly_result.get('confidence', 0)
                    },
                    'agent_agreement': {
                        'full_agreement': False,
                        'majority_agreement': True,
                        'agreement_count': 2,  # Classifier + Anomaly (pokud není anomálie)
                        'total_agents': 3
                    },
                    'reasoning': f"Classifier zamítl s vysokou jistotou ({classifier_conf:.0%}): {classifier_result.get('reasoning', 'Neznámý důvod')} | Extractor našel pouze {extractor_fields_found} elementů (potřebuje ≥5)",
                    'classifier_veto': True,
                    'classifier_veto_reason': classifier_result.get('reasoning', 'Silný negativní indikátor')
                }

            # === KONTROLA HALUCINACE I PŘI 5+ POLECH ===
            # Pokud chybí invoice_number NEBO bank_account, pravděpodobně jde o halucinaci
            # (skutečné faktury téměř vždy mají číslo faktury nebo bankovní účet)
            has_invoice_number = bool(extractor_result.get('invoice_number'))
            has_bank_account = bool(extractor_result.get('bank_account'))
            
            if not has_invoice_number and not has_bank_account:
                logger.info(f"  ✗ Classifier zamítl ({classifier_conf:.0%}) + chybí invoice_number i bank_account → HALUCINACE")
                return {
                    'is_invoice': False,
                    'confidence': classifier_conf,
                    'decision_type': 'auto_reject',
                    'weighted_score': 1.0 - classifier_conf,
                    'agent_scores': {
                        'classifier': classifier_conf,
                        'extractor': extractor_completeness,
                        'anomaly': 1 - anomaly_result.get('confidence', 0)
                    },
                    'agent_agreement': {
                        'full_agreement': False,
                        'majority_agreement': True,
                        'agreement_count': 2,
                        'total_agents': 3
                    },
                    'reasoning': f"Classifier zamítl ({classifier_conf:.0%}): {classifier_result.get('reasoning', 'N/A')} | Extractor sice našel {extractor_fields_found} polí ALE chybí invoice_number i bank_account → HALUCINACE",
                    'classifier_veto': True,
                    'classifier_veto_reason': classifier_result.get('reasoning', 'Silný negativní indikátor')
                }

        keyword_check = None
        if classifier_is_invoice and classifier_conf >= 0.5 and raw_text:
            keyword_check = self._check_invoice_keywords(raw_text)
            if not keyword_check['found']:
                logger.warning(f"  ⚠️ RETRY: Classifier řekl JE faktura ({classifier_conf:.0%}) ale text neobsahuje invoice keywords!")
                return {
                    'is_invoice': None,
                    'confidence': 0.0,
                    'decision_type': 'retry',
                    'retry_reason': 'missing_invoice_keywords',
                    'keyword_analysis': keyword_check,
                    'reasoning': f"Classifier potvrdil fakturu ({classifier_conf:.0%}) ale text neobsahuje klíčová invoice slova. Nutná kontrola."
                }
        elements = classifier_result.get('elements_present', {})
        has_identification = elements.get('identification', False)
        has_financial = elements.get('financial', False)
        has_subjects = elements.get('subjects', False)
        has_dates = elements.get('dates', False)
        has_payment_info = elements.get('payment_info', False)

        if classifier_is_invoice and classifier_conf >= 0.6 and extractor_fields_found <= 1:
            logger.debug(f"  ⚠️ DETEKOVÁNA HALUCINACE: Classifier řekl JE faktura ({classifier_conf:.0%}) ale Extractor našel pouze {extractor_fields_found} elementů → REJECT")
            return {
                'is_invoice': False,
                'confidence': max(classifier_conf, 0.9),
                'decision_type': 'auto_reject',
                'weighted_score': 0.1,
                'agent_scores': {
                    'classifier': 0.0,
                    'extractor': extractor_completeness,
                    'anomaly': 1 - anomaly_result.get('confidence', 0)
                },
                'agent_agreement': {
                    'full_agreement': False,
                    'majority_agreement': True,
                    'agreement_count': 2,
                    'total_agents': 3
                },
                'reasoning': f"Klasifikátor tvrdil že všechny elementy jsou přítomny ({classifier_conf:.0%}) ale Extraktor našel pouze {extractor_fields_found} elementů → Klasifikátor PRAVDĚPODOBNĚ HALUCINUJE → NENÍ faktura",
                'extracted_data': {},
                'classifier_hallucination_detected': True
            }

        if classifier_is_invoice and classifier_conf >= 0.5 and 2 <= extractor_fields_found <= 3:
            logger.debug(f"  ⚠️ ROZPOR: Classifier řekl JE faktura ({classifier_conf:.0%}) ale Extractor našel pouze {extractor_fields_found} elementy → HUMAN REVIEW")
            return {
                'is_invoice': None,
                'confidence': 0.5,
                'decision_type': 'human_review',
                'weighted_score': 0.5,
                'agent_scores': {
                    'classifier': classifier_conf,
                    'extractor': extractor_completeness,
                    'anomaly': 0.5
                },
                'agent_agreement': {
                    'full_agreement': False,
                    'majority_agreement': False,
                    'agreement_count': 1,
                    'total_agents': 3
                },
                'reasoning': f"Klasifikátor tvrdí že je to faktura ({classifier_conf:.0%}) ale Extraktor našel pouze {extractor_fields_found} elementy. Chybí: {', '.join(missing_fields)} → Nutná lidská kontrola",
                'extracted_data': self._extract_final_data(classifier_result, extractor_result),
                'classifier_extractor_conflict': True
            }

        if not classifier_is_invoice and extractor_fields_found < 3:
            logger.debug(f"  ✗ Classifier: NENÍ faktura + Extractor našel jen {extractor_fields_found} elementů → AUTO-REJECT")
            anomaly_confirms = anomaly_result.get('is_anomaly', False)
            return {
                'is_invoice': False,
                'confidence': max(classifier_conf, 0.85) if anomaly_confirms else max(classifier_conf, 0.8),
                'decision_type': 'auto_reject',
                'weighted_score': 0.2,
                'agent_scores': {
                    'classifier': 1 - classifier_conf,
                    'extractor': extractor_completeness,
                    'anomaly': 1 - anomaly_result.get('confidence', 0)
                },
                'agent_agreement': {
                    'full_agreement': anomaly_confirms,
                    'majority_agreement': True,
                    'agreement_count': 3 if anomaly_confirms else 2,
                    'total_agents': 3
                },
                'reasoning': f"Classifier řekl není faktura + Extractor našel jen {extractor_fields_found} elementů (potřebuje ≥3) → NENÍ faktura" + (" + Anomalie potvrzuje" if anomaly_confirms else ""),
                'extracted_data': {}
            }

        if not classifier_is_invoice and extractor_fields_found >= 4:
            anomaly_type = anomaly_result.get('anomaly_type', '')
            
            is_definite_non_invoice = any(
                keyword in (anomaly_type or '').lower() 
                for keyword in self.NON_INVOICE_ANOMALY_TYPES
            )
            
            if is_definite_non_invoice and anomaly_result.get('is_anomaly', False):
                logger.debug(f"  ✗ Classifier: NENÍ faktura + Extractor našel {extractor_fields_found} elementů ALE Anomaly detekovala '{anomaly_type}' → REJECT (Extractor halucinace)")
                return {
                    'is_invoice': False,
                    'confidence': max(classifier_conf, 0.9),
                    'decision_type': 'auto_reject',
                    'weighted_score': 0.15,
                    'agent_scores': {
                        'classifier': 1 - classifier_conf,
                        'extractor': 0.0,
                        'anomaly': anomaly_result.get('confidence', 0)
                    },
                    'agent_agreement': {
                        'full_agreement': True,
                        'majority_agreement': True,
                        'agreement_count': 2,
                        'total_agents': 3
                    },
                    'reasoning': f"Classifier řekl není faktura + Anomaly detekovala '{anomaly_type}' → NENÍ faktura (Extractor halucinoval {extractor_fields_found} elementů)",
                    'extracted_data': {},
                    'extractor_hallucination_detected': True
                }
            
            extracted_amount = extractor_result.get('total_amount')
            extracted_amount_raw = extractor_result.get('total_amount_raw')
            has_real_amount = (extracted_amount is not None and extracted_amount > 0) or bool(extracted_amount_raw)

            if not has_real_amount:
                logger.debug(f"  ✗ Classifier: NENÍ faktura + Extractor našel {extractor_fields_found} elementů ALE žádná skutečná částka → REJECT (halucinace)")
                return {
                    'is_invoice': False,
                    'confidence': max(classifier_conf, 0.9),
                    'decision_type': 'auto_reject',
                    'weighted_score': 0.15,
                    'agent_scores': {
                        'classifier': 1 - classifier_conf,
                        'extractor': 0.0,
                        'anomaly': 1 - anomaly_result.get('confidence', 0)
                    },
                    'agent_agreement': {
                        'full_agreement': True,
                        'majority_agreement': True,
                        'agreement_count': 2,
                        'total_agents': 3
                    },
                    'reasoning': f"Classifier řekl není faktura + Extractor nemůže najít skutečnou částku → NENÍ faktura (halucinace)",
                    'extracted_data': {},
                    'extractor_hallucination_detected': True
                }

            # === ZMĚNA ARCHITEKTURY (v7.5): Odstraněna Extractor Priorita ===
            # Původně Extractor mohl přebít Klasifikátora pokud našel 4+ pole.
            # To způsobovalo halucinace u paletových listů. Nyní to jde na HUMAN REVIEW.
            logger.debug(f"  ⚠️ ROZPOR: Classifier: NENÍ faktura ALE Extractor našel {extractor_fields_found} elementů → HUMAN REVIEW")
            return {
                'is_invoice': None,
                'confidence': 0.5,
                'decision_type': 'human_review',
                'weighted_score': 0.5,
                'agent_scores': {
                    'classifier': 0.0, # Započítáno jako nesouhlas
                    'extractor': extractor_completeness,
                    'anomaly': 1 - anomaly_result.get('confidence', 0)
                },
                'agent_agreement': {
                    'full_agreement': False,
                    'majority_agreement': False,
                    'agreement_count': 1,
                    'total_agents': 3
                },
                'reasoning': f"Rozpor: Klasifikátor zamítl dokument ALE Extraktor našel {extractor_fields_found} elementů (vč. částky) → Možná halucinace, nutná kontrola.",
                'extracted_data': self._extract_final_data(classifier_result, extractor_result),
                'classifier_extractor_conflict': True
            }

        if classifier_is_invoice and classifier_conf >= 0.7 and extractor_fields_found == 0:
            logger.debug(f"  ✗ Classifier: JE faktura ({classifier_conf:.0%}) ALE Extractor nenašel ŽÁDNÉ elementy → NENÍ faktura (Extractor priorita)")
            return {
                'is_invoice': False,
                'confidence': max(classifier_conf, 0.75),
                'decision_type': 'auto_reject',
                'weighted_score': 0.2,
                'agent_scores': {
                    'classifier': 0.0,
                    'extractor': 0.0,
                    'anomaly': 1 - anomaly_result.get('confidence', 0)
                },
                'agent_agreement': {
                    'full_agreement': False,
                    'majority_agreement': True,
                    'agreement_count': 2,
                    'total_agents': 3
                },
                'reasoning': f"Classifier řekl faktura ({classifier_conf:.0%}) ale Extractor nenašel žádné elementy → NENÍ faktura (halucinace)",
                'extracted_data': {}
            }

        # Tento blok byl v původním kódu prázdný (pass), nechávám jej pro úplnost
        if classifier_is_invoice and extractor_fields_found >= 1:
            pass

        all_elements_present = all([
            has_identification, has_subjects, has_dates, has_financial, has_payment_info
        ])

        if classifier_conf >= 0.85 and all_elements_present and classifier_is_invoice:
            logger.debug(f"  ✓ Classifier velmi jistý ({classifier_conf:.0%}) + všech 5 elementů → AUTO-ACCEPT")
            return {
                'is_invoice': True,
                'confidence': classifier_conf,
                'decision_type': 'auto_accept',
                'weighted_score': classifier_conf,
                'agent_scores': {
                    'classifier': classifier_conf,
                    'extractor': extractor_completeness,
                    'anomaly': 1 - anomaly_result.get('confidence', 0)
                },
                'agent_agreement': {
                    'full_agreement': False,
                    'majority_agreement': True,
                    'agreement_count': 2,
                    'total_agents': 3
                },
                'reasoning': f"Classifier velmi jistý ({classifier_conf:.0%}) + všech 5 elementů přítomno",
                'extracted_data': self._extract_final_data(classifier_result, extractor_result)
            }

        weighted_score = self._calculate_weighted_score(classifier_result, extractor_result, anomaly_result)
        final_score = self._apply_penalties(weighted_score, extractor_result, classifier_result)
        decision, decision_type = self._make_decision(final_score)

        return {
            'is_invoice': decision,
            'confidence': round(final_score, 3),
            'decision_type': decision_type,
            'weighted_score': round(weighted_score, 3),
            'agent_scores': {
                'classifier': classifier_result.get('confidence', 0),
                'extractor': extractor_result.get('completeness_score', 0),
                'anomaly': 1 - anomaly_result.get('confidence', 0)
            },
            'agent_agreement': self._check_agreement(classifier_result, extractor_result, anomaly_result),
            'reasoning': self._build_reasoning(classifier_result, extractor_result, anomaly_result, final_score, extractor_fields_found),
            'extracted_data': self._extract_final_data(classifier_result, extractor_result)
        }
    
    def _check_anomaly_veto(self, anomaly_result: dict, classifier_result: dict = None) -> Optional[dict]:
        if not anomaly_result.get('is_anomaly'):
            return None

        confidence = anomaly_result.get('confidence', 0)
        anomaly_type = anomaly_result.get('anomaly_type', '')
        flags = anomaly_result.get('flags', [])

        classifier_agrees = False
        if classifier_result:
            clf_is_invoice = classifier_result.get('is_invoice', False)
            clf_conf = classifier_result.get('confidence', 0)
            if not clf_is_invoice and clf_conf >= 0.5:
                classifier_agrees = True

        is_definite_non_invoice = any(
            keyword in (anomaly_type or '').lower() 
            for keyword in self.NON_INVOICE_ANOMALY_TYPES
        )

        effective_threshold = self.anomaly_veto_threshold
        if classifier_agrees:
            effective_threshold = 0.5
        
        if is_definite_non_invoice and classifier_agrees:
            if confidence >= 0.4:
                logger.debug(f"🚫 ANOMALY VETO: {anomaly_type} (conf: {confidence:.2f}) + Classifier agrees = DEFINITE NON-INVOICE")
                return {
                    'is_invoice': False,
                    'confidence': max(confidence, 0.9),
                    'decision_type': 'anomaly_veto',
                    'veto_reason': f"Detekován konkrétní typ dokumentu: {anomaly_type} (Classifier souhlasí)",
                    'anomaly_type': anomaly_type,
                    'agent_scores': {
                        'classifier': 1 - classifier_result.get('confidence', 0),
                        'extractor': 0,
                        'anomaly': confidence
                    },
                    'is_definite_non_invoice': True
                }

        if confidence >= effective_threshold or 'veto' in flags:
            veto_confidence = confidence
            if classifier_agrees:
                veto_confidence = max(confidence, 0.85)
                logger.debug(f"🚫 Anomaly veto + Classifier agrees: {anomaly_type} (confidence: {veto_confidence:.2f})")
            else:
                logger.debug(f"🚫 Anomaly veto: {anomaly_type} (confidence: {confidence:.2f})")

            return {
                'is_invoice': False,
                'confidence': veto_confidence,
                'decision_type': 'anomaly_veto',
                'veto_reason': f"Detekována anomálie: {anomaly_type}" + (" (Classifier souhlasí)" if classifier_agrees else ""),
                'anomaly_type': anomaly_type,
                'agent_scores': {
                    'classifier': 1 - classifier_result.get('confidence', 0) if classifier_agrees else 0,
                    'extractor': 0,
                    'anomaly': confidence
                }
            }

        return None
    
    def _calculate_weighted_score(self, classifier: dict, extractor: dict, anomaly: dict) -> float:
        classifier_score = classifier.get('confidence', 0)
        if not classifier.get('is_invoice'):
            classifier_score = 1 - classifier_score
        
        extractor_score = extractor.get('completeness_score', 0)
        anomaly_score = 1 - anomaly.get('confidence', 0) if anomaly.get('is_anomaly') else 0.95
        
        weighted = (
            classifier_score * self.weights['classifier'] +
            extractor_score * self.weights['extractor'] +
            anomaly_score * self.weights['anomaly']
        )
        return min(1.0, max(0.0, weighted))
    
    def _apply_penalties(self, score: float, extractor_result: dict, classifier_result: dict = None) -> float:
        penalty = 0.0

        validation_errors = extractor_result.get('validation_errors', [])
        completeness = extractor_result.get('completeness_score', 0)

        has_vendor = bool(extractor_result.get('vendor_name'))
        has_customer = bool(extractor_result.get('customer_name'))
        has_amount = extractor_result.get('total_amount') is not None or bool(extractor_result.get('total_amount_raw'))
        has_issue_date = bool(extractor_result.get('issue_date') and extractor_result.get('issue_date') != '0000-00-00')

        classifier_confirms_invoice = False
        extractor_technical_failure = False
        extractor_found_data_no_amount = False
        
        if classifier_result:
            clf_conf = classifier_result.get('confidence', 0)
            clf_is_invoice = classifier_result.get('is_invoice', False)
            elements = classifier_result.get('elements_present', {})
            all_elements_present = all([
                elements.get('identification', False),
                elements.get('subjects', False),
                elements.get('dates', False),
                elements.get('financial', False),
                elements.get('payment_info', False)
            ])
            if clf_conf >= 0.85 and clf_is_invoice and all_elements_present:
                classifier_confirms_invoice = True
                extractor_completeness = extractor_result.get('completeness_score', 0)
                if extractor_completeness < 0.3:
                    extractor_technical_failure = True
                elif extractor_completeness > 0.3 and not has_amount:
                    extractor_found_data_no_amount = True

        if not has_amount:
            if classifier_confirms_invoice and extractor_found_data_no_amount:
                logger.debug("  ✗ Chybí částka - extractor běžel ale nenašel → AUTO-REJECT")
                penalty += 0.9 
            elif classifier_confirms_invoice and extractor_technical_failure:
                logger.debug("  ⚠️ Chybí částka, ale classifier potvrdil fakturu a extractor selhal technicky → ignoruji penalizaci")
            elif classifier_confirms_invoice:
                logger.debug("  ⚠️ Chybí částka - classifier řekl faktura ale extractor nenašel → penalty")
                penalty += 0.6
            else:
                logger.debug("  ⚠️ Chybí částka - PRAVDĚPODOBNĚ NENÍ faktura")
                penalty += 0.6
        else:
            if CHECK_AMOUNT_REALISTIC:
                amount = extractor_result.get('total_amount')
                if amount is not None:
                    try:
                        amount_value = float(amount) if isinstance(amount, str) else amount
                        if amount_value < MIN_REALISTIC_AMOUNT:
                            logger.debug(f"  ⚠️ Nízká částka: {amount_value} - mírná penalizace")
                            penalty += 0.10
                    except (ValueError, TypeError):
                        pass
        
        if has_vendor and not has_customer:
            logger.debug("  ⚠️ Chybí odběratel - některé faktury nemusí mít")
            penalty += 0.05
        
        if not extractor_result.get('invoice_number'):
            logger.debug("  ⚠️ Chybí číslo faktury")
            penalty += 0.10
        
        if not extractor_result.get('bank_account') and not extractor_result.get('variable_symbol'):
            logger.debug("  ⚠️ Chybí platební údaje - nemusí být vždy vyžadováno")
            penalty += 0.05
        
        if completeness < 0.5:
            penalty += 0.15
        elif completeness < 0.6:
            penalty += 0.10
        elif completeness < 0.7:
            penalty += 0.05
        
        if len(validation_errors) >= 3:
            penalty += 0.15
        elif len(validation_errors) >= 2:
            penalty += 0.08
        elif len(validation_errors) >= 1:
            penalty += 0.03
        
        final_score = score * (1 - penalty)
        logger.debug(f"  Penalizace: {penalty:.2f}, finální skóre: {final_score:.2f}")
        return min(1.0, max(0.0, final_score))
    
    def _make_decision(self, score: float) -> Tuple[bool, str]:
        if score >= self.threshold_accept:
            return True, 'auto_accept'
        elif score >= self.threshold_review:
            return None, 'human_review'
        else:
            return False, 'auto_reject'
    
    def _check_agreement(self, classifier: dict, extractor: dict, anomaly: dict) -> dict:
        classifier_says_invoice = classifier.get('is_invoice', False)
        extractor_says_invoice = extractor.get('completeness_score', 0) > 0.5
        anomaly_says_clean = not anomaly.get('is_anomaly')
        
        agreements = [classifier_says_invoice, extractor_says_invoice, anomaly_says_clean]
        agreement_count = sum(agreements)
        
        return {
            'full_agreement': agreement_count == 3,
            'majority_agreement': agreement_count >= 2,
            'agreement_count': agreement_count,
            'total_agents': 3
        }
    
    def _build_reasoning(self, classifier: dict, extractor: dict, anomaly: dict, final_score: float, extractor_fields_found: int = None) -> str:
        reasons = []

        if classifier.get('is_invoice'):
            reasons.append(f"Klasifikátor: faktura ({classifier.get('confidence', 0):.0%})")
        else:
            reasons.append(f"Klasifikátor: není faktura ({classifier.get('confidence', 0):.0%})")

        completeness = extractor.get('completeness_score', 0)
        errors = extractor.get('validation_errors', [])
        
        if extractor_fields_found is not None:
            reasons.append(f"Extraktor: {extractor_fields_found} elementů")
        elif completeness > 0.7:
            reasons.append(f"Extraktor: kompletní data ({completeness:.0%})")
        elif errors:
            reasons.append(f"Extraktor: chybí {len(errors)} polí")
        else:
            reasons.append(f"Extraktor: data ({completeness:.0%})")

        if anomaly.get('is_anomaly'):
            reasons.append(f"Anomalie: {anomaly.get('anomaly_type')} ({anomaly.get('confidence', 0):.0%})")
        else:
            reasons.append("Anomalie: žádné detekovány")

        if final_score >= self.threshold_accept:
            reasons.append(f"→ Auto-accept (score: {final_score:.2f})")
        elif final_score >= self.threshold_review:
            reasons.append(f"→ Human review (score: {final_score:.2f})")
        else:
            reasons.append(f"→ Auto-reject (score: {final_score:.2f})")

        return " | ".join(reasons)
    
    def _extract_final_data(self, classifier: dict, extractor: dict) -> dict:
        return {
            'invoice_number': extractor.get('invoice_number'),
            'vendor_name': extractor.get('vendor_name'),
            'customer_name': extractor.get('customer_name'),
            'issue_date': extractor.get('issue_date'),
            'due_date': extractor.get('due_date'),
            'total_amount': extractor.get('total_amount'),
            'currency': extractor.get('currency'),
            'vendor_ico': extractor.get('vendor_ico'),
            'vendor_dic': extractor.get('vendor_dic'),
            'bank_account': extractor.get('bank_account'),
            'variable_symbol': extractor.get('variable_symbol'),
        }

    def _check_invoice_keywords(self, text: str) -> dict:
        """Check for invoice keywords and return analysis."""
        text_lower = text.lower()

        # Use keywords from config, or fallback to defaults
        categories = {
            'identification': {
                'cs': [kw for kw in self.invoice_keywords_cs if 'faktura' in kw or 'doklad' in kw or 'proforma' in kw],
                'en': [kw for kw in self.invoice_keywords_en if 'invoice' in kw or 'tax' in kw or 'bill' in kw]
            },
            'subjects': {
                'cs': [kw for kw in self.invoice_keywords_cs if kw in ['dodavatel', 'odběratel', 'objednatel', 'zhotovitel', 'ičo', 'dič', 's.r.o.', 'a.s.']],
                'en': [kw for kw in self.invoice_keywords_en if kw in ['supplier', 'vendor', 'customer', 'contractor', 'vat', 'tax id', 'limited', 'inc.', 'gmbh']]
            },
            'dates': {
                'cs': [kw for kw in self.invoice_keywords_cs if 'datum' in kw or 'splatnost' in kw or 'vystaveno' in kw],
                'en': [kw for kw in self.invoice_keywords_en if 'date' in kw.lower() or 'dated' in kw]
            },
            'financial': {
                'cs': [kw for kw in self.invoice_keywords_cs if kw in ['celkem', 'k úhradě', 'částka', 'cena', 'úhrada', 'bez dph', 'dpH', 'sazba']],
                'en': [kw for kw in self.invoice_keywords_en if kw in ['total', 'amount', 'price', 'payment', 'balance', 'excl. vat', 'incl. vat', 'subtotal']]
            },
            'payment_info': {
                'cs': [kw for kw in self.invoice_keywords_cs if kw in ['variabilní symbol', 'banka', 'účet', 'iban', 'bic', 'swift']],
                'en': [kw for kw in self.invoice_keywords_en if 'payment' in kw or 'bank' in kw or 'account' in kw or kw in ['iban', 'bic', 'swift']]
            },
            'currency': {
                'cs': [kw for kw in self.invoice_keywords_cs if kw in ['kč', 'czk', 'eur', '€', 'usd', '$', '£', 'gbp']],
                'en': [kw for kw in self.invoice_keywords_en if kw in ['eur', 'usd', 'gbp', 'czk', '€', '$', '£']]
            }
        }

        found_keywords = []
        categories_found = []
        categories_missing = []

        for category, keywords in categories.items():
            category_keywords = keywords['cs'] + keywords['en']
            # Use regex with word boundaries for whole word matching only
            found_in_category = [
                kw for kw in category_keywords
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower, re.IGNORECASE)
            ]

            if found_in_category:
                found_keywords.extend(found_in_category)
                categories_found.append(category)
            else:
                categories_missing.append(category)

        has_identification = 'identification' in categories_found
        has_financial = 'financial' in categories_found

        category_score = len(categories_found) / len(categories)

        critical_bonus = 0.0
        if has_identification:
            critical_bonus += 0.2
        if has_financial:
            critical_bonus += 0.2

        final_score = min(1.0, category_score * 0.6 + critical_bonus)

        is_found = (
            (has_identification and has_financial and len(categories_found) >= 3) or
            len(categories_found) >= 4
        )

        # Use regex with word boundaries for whole word matching only
        if re.search(r'\b' + re.escape('faktura') + r'\b', text_lower, re.IGNORECASE) or \
           re.search(r'\b' + re.escape('invoice') + r'\b', text_lower, re.IGNORECASE):
            is_found = True

        return {
            'found': is_found,
            'keywords_found': list(set(found_keywords)),
            'keywords_missing': categories_missing,
            'score': round(final_score, 3),
            'categories_found': categories_found,
            'categories_missing': categories_missing
        }

    def _check_cv_resume(self, text: str) -> dict:
        """Check if text contains CV/resume indicators."""
        text_lower = text.lower()

        # CV/Resume indicators from rules.yaml
        cv_indicators = [
            'professional profile',
            'professional summary',
            'career objective',
            'core competencies',
            'technical competencies',
            'technical projects',
            'professional working proficiency',
            'curriculum vitae',
            'curriculum',
            'resume',
            'work experience',
            'education',
            'skills',
            'životopis',
            'životopisy',
            'pracovní zkušenosti',
            'vzdělání',
            'dovednosti',
            'jazykové znalosti',
        ]

        # Additional CV patterns
        cv_patterns = [
            r'professional\s+profile',
            r'core\s+technical\s+competenc(ies|y)',
            r'technical\s+projects',
            r'professional\s+working\s+proficiency',
            r'curriculum\s+vitae?',
            r'work\s+experience',
            r'career\s+objective',
        ]

        found_indicators = []

        # Check direct keywords - using word boundaries for whole word matching
        for indicator in cv_indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower, re.IGNORECASE):
                found_indicators.append(indicator)

        # Check regex patterns
        for pattern in cv_patterns:
            if re.search(pattern, text_lower) and pattern not in found_indicators:
                found_indicators.append(f'regex:{pattern}')

        # Calculate confidence based on number of indicators
        confidence = 0.0
        if found_indicators:
            # More indicators = higher confidence
            confidence = min(0.95, 0.5 + (len(found_indicators) * 0.15))

        return {
            'is_cv': len(found_indicators) >= 2,  # Need at least 2 indicators
            'confidence': confidence,
            'indicators': found_indicators,
            'indicator_count': len(found_indicators)
        }

    def _check_research_report(self, text: str) -> dict:
        """Check if text contains research report / survey indicators."""
        text_lower = text.lower()

        # Research report indicators
        research_indicators = [
            'mapování terénu',
            'strategický pitch',
            'průzkum českého',
            'výzkumných skupin',
            'pracovišť',
            'český ai ekosystém',
            'mapa české ai',
            'tuzemského ekosystému',
            'konzervativní umělé inteligence',
            'explainable ai',
            'medicínské ai',
            'výzkumná zpráva',
            'research report',
            'survey',
            'ekosystém',
            'výzkumný tým',
        ]

        # Additional research report patterns
        research_patterns = [
            r'mapov[áa]n[íi]\s+ter[ée]nu',
            r'pr[ůu]zkum\s+\w+',
            r'ekosyst[ée]m',
            r'v[ýy]zkumn[ýy]ch\s+skupin',
            r'pracovi[šs][tť]',
            r'strategick[ýy]\s+pitch',
        ]

        found_indicators = []

        # Check direct keywords - using word boundaries for whole word matching
        for indicator in research_indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', text_lower, re.IGNORECASE):
                found_indicators.append(indicator)

        # Check regex patterns
        for pattern in research_patterns:
            if re.search(pattern, text_lower) and pattern not in found_indicators:
                found_indicators.append(f'regex:{pattern}')

        # Calculate confidence based on number of indicators
        confidence = 0.0
        if found_indicators:
            # More indicators = higher confidence
            confidence = min(0.95, 0.5 + (len(found_indicators) * 0.15))

        return {
            'is_research': len(found_indicators) >= 2,  # Need at least 2 indicators
            'confidence': confidence,
            'indicators': found_indicators,
            'indicator_count': len(found_indicators)
        }

    def get_statistics(self, results: List[dict]) -> dict:
        if not results:
            return {}
        
        total = len(results)
        invoices = sum(1 for r in results if r.get('is_invoice'))
        non_invoices = sum(1 for r in results if r.get('is_invoice') is False)
        reviews = sum(1 for r in results if r.get('is_invoice') is None)
        
        confidences = [r.get('confidence', 0) for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        full_agreements = sum(
            1 for r in results 
            if r.get('agent_agreement', {}).get('full_agreement')
        )
        
        return {
            'total': total,
            'invoices': invoices,
            'non_invoices': non_invoices,
            'human_review': reviews,
            'invoice_percentage': round(invoices / total * 100, 1) if total > 0 else 0,
            'average_confidence': round(avg_confidence, 3),
            'full_agreement_rate': round(full_agreements / total * 100, 1) if total > 0 else 0,
        }