# bert_skill_extractor.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

class SkillExtractor:
    def __init__(
        self,
        skill_list_path: str = 'data/skills.csv',
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        min_similarity: float = 0.7
    ):
        print("[DEBUG] Before loading model")
        self.model = SentenceTransformer(model_name)
        print("[DEBUG] After loading model")
        self.kb = KeyBERT(model=self.model)  # backend uses SBERT :contentReference[oaicite:2]{index=2}

        skills_df = pd.read_csv(skill_list_path)
        print("[DEBUG] Encoding skill embeddings...")
        self.skill_list = (
            skills_df['skill']
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
        self.skill_embeds = self.model.encode(self.skill_list, convert_to_tensor=True)
        print("[DEBUG] Model and embeddings ready")

        self.min_similarity = min_similarity

    def extract_skills(self, text: str, top_n: int = 20) -> list[str]:
        if not text or len(text.strip()) < 20:
            return []

        candidates = [
            kw[0] for kw in self.kb.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=top_n
            )
        ]
        if not candidates:
            return []

        cand_embeds = self.model.encode(candidates, convert_to_tensor=True)
        sim_matrix = util.cos_sim(cand_embeds, self.skill_embeds)

        extracted = [
            candidates[i]
            for i in range(len(candidates))
            if float(sim_matrix[i].max()) >= self.min_similarity
        ]

        return sorted(set(extracted))
