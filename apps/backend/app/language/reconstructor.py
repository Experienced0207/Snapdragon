"""
Gloss-to-Sentence Reconstruction Engine for SignBridge.
Converts sequences of ISL sign glosses (e.g., ["I", "WATER", "NEED"]) into fluent English sentences (e.g., "I need water.").

Supports dual execution modes:
  1. DEV_MODE: Local Mac execution utilizing HuggingFace Qwen2.5-0.5B-Instruct LLM or robust ISL grammar heuristic fallback.
  2. SNAPDRAGON_MODE: Prepared runtime interface for Qualcomm GenieX / QAIRT execution on Snapdragon ARM64.
"""

import os
import re
from typing import List, Optional, Dict


class ISLRuleBasedReconstructor:
    """
    Fast rule-based & grammar heuristic engine for Indian Sign Language (ISL) gloss conversion.
    ISL typically follows Subject-Object-Verb (SOV) order, whereas English follows Subject-Verb-Object (SVO).
    """
    def __init__(self):
        # Dictionary of common ISL gloss sequence mappings
        self.exact_mappings = {
            ("i", "water", "need"): "I need water.",
            ("me", "water", "need"): "I need water.",
            ("you", "name", "what"): "What is your name?",
            ("me", "home", "go"): "I am going home.",
            ("i", "home", "go"): "I am going home.",
            ("help", "me"): "Please help me.",
            ("thank", "you"): "Thank you.",
            ("nice", "meet", "you"): "Nice to meet you.",
            ("where", "hospital"): "Where is the hospital?",
            ("doctor", "need"): "I need a doctor.",
            ("food", "want"): "I want food.",
            ("good", "morning"): "Good morning.",
            ("good", "night"): "Good night."
        }

        # Sub-phrase verb mappings (SOV -> SVO reordering)
        self.verb_corrections = {
            "need": "need",
            "want": "want",
            "go": "am going to",
            "come": "am coming to",
            "eat": "am eating",
            "drink": "am drinking",
            "help": "help",
            "like": "like"
        }

    def reconstruct(self, gloss_list: List[str]) -> str:
        if not gloss_list:
            return ""

        # Normalize gloss list
        clean_glosses = [str(g).strip().lower() for g in gloss_list if str(g).strip()]
        if not clean_glosses:
            return ""

        key = tuple(clean_glosses)
        if key in self.exact_mappings:
            return self.exact_mappings[key]

        # Simple SOV -> SVO reordering heuristic
        # If pattern is [Subject, Object, Verb] e.g. ["i", "food", "want"]
        if len(clean_glosses) == 3:
            subj, obj, verb = clean_glosses[0], clean_glosses[1], clean_glosses[2]
            if verb in self.verb_corrections:
                subj_str = "I" if subj in ["i", "me"] else subj.capitalize()
                verb_str = self.verb_corrections[verb]
                return f"{subj_str} {verb_str} {obj}."

        # Default fallback: Capitalize first word, join with spaces, add period
        words = [g.upper() if len(g) <= 2 else g for g in clean_glosses]
        sentence = " ".join(words).capitalize()
        if not sentence.endswith((".", "?", "!")):
            sentence += "."

        return sentence


class GlossToSentenceEngine:
    """
    Gloss-to-Sentence Reconstruction Architecture supporting DEV_MODE and SNAPDRAGON_MODE.
    """
    def __init__(self, mode: str = "DEV_MODE", model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.mode = mode.upper()
        self.model_name = model_name
        self.rule_engine = ISLRuleBasedReconstructor()
        self.hf_pipeline = None

        if self.mode == "DEV_MODE":
            self._init_dev_mode()
        elif self.mode == "SNAPDRAGON_MODE":
            self._init_snapdragon_mode()
        else:
            print(f"[GlossToSentenceEngine] Unknown mode '{mode}'. Defaulting to DEV_MODE.")
            self.mode = "DEV_MODE"
            self._init_dev_mode()

    def set_mode(self, mode: str):
        """Switches operational mode."""
        self.mode = mode.upper()
        print(f"[GlossToSentenceEngine] Operational mode set to: {self.mode}")
        if self.mode == "DEV_MODE" and self.hf_pipeline is None:
            self._init_dev_mode()
        elif self.mode == "SNAPDRAGON_MODE":
            self._init_snapdragon_mode()

    def _init_dev_mode(self):
        """Attempts to load local HuggingFace LLM pipeline; falls back gracefully to rule engine."""
        print(f"[GlossToSentenceEngine] Initializing DEV_MODE (Model: {self.model_name})...")
        try:
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            print(f"[GlossToSentenceEngine] Loading HuggingFace model '{self.model_name}'...")
            self.hf_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                max_new_tokens=64,
                temperature=0.3
            )
            print("[GlossToSentenceEngine] HuggingFace LLM loaded successfully.")
        except Exception as e:
            print(f"[GlossToSentenceEngine] Note: Local HF model unavailable ({e}). Using fast ISL rule engine.")
            self.hf_pipeline = None

    def _init_snapdragon_mode(self):
        """Qualcomm GenieX / QAIRT NPU Runtime initialization wrapper."""
        print("[GlossToSentenceEngine] Initializing SNAPDRAGON_MODE (Qualcomm GenieX / QAIRT NPU Engine)...")
        # Ready interface for Qualcomm GenieX C++/Python bindings on Snapdragon ARM64
        pass

    def _llm_prompt_reconstruct(self, gloss_list: List[str]) -> str:
        """Formulates LLM system prompt for ISL gloss translation."""
        gloss_str = ", ".join([f'"{g.upper()}"' for g in gloss_list])
        prompt = (
            f"<|im_start|>system\n"
            f"You are an expert translator converting Indian Sign Language (ISL) glosses into fluent, grammatically correct English sentences. Output only the translation.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Convert these ISL sign glosses into a natural English sentence: [{gloss_str}]<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            output = self.hf_pipeline(prompt, max_new_tokens=32, return_full_text=False)
            text = output[0]["generated_text"].strip()
            # Clean up response string
            text = re.sub(r'<\|im_end\|>.*', '', text).strip()
            return text
        except Exception as e:
            print(f"[GlossToSentenceEngine] LLM generation error: {e}. Falling back to rule engine.")
            return self.rule_engine.reconstruct(gloss_list)

    def reconstruct(self, gloss_list: List[str]) -> str:
        """
        Reconstructs fluent English sentence from input list of sign glosses.

        Args:
            gloss_list: List of string glosses e.g. ["I", "WATER", "NEED"]

        Returns:
            Reconstructed English sentence e.g. "I need water."
        """
        if not gloss_list:
            return ""

        if self.mode == "DEV_MODE":
            if self.hf_pipeline is not None:
                return self._llm_prompt_reconstruct(gloss_list)
            else:
                return self.rule_engine.reconstruct(gloss_list)

        elif self.mode == "SNAPDRAGON_MODE":
            # Qualcomm GenieX NPU execution fallback
            return self.rule_engine.reconstruct(gloss_list)

        return self.rule_engine.reconstruct(gloss_list)


if __name__ == "__main__":
    engine = GlossToSentenceEngine(mode="DEV_MODE")
    test_glosses = ["I", "WATER", "NEED"]
    sentence = engine.reconstruct(test_glosses)
    print(f"Glosses: {test_glosses} -> Reconstructed Sentence: '{sentence}'")
