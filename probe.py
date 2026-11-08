"""
probe.py — Spatial Dialect Realizer and Telemetry Observer.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Sequence
import numpy as np

from phonetics import (
    Phoneme,
    repair_phonemes,
    trajectory_distance,
    relax_gesture_gradient,
    apply_syncope_and_clustering,
    render_and_extract_tones,
)
from word import Word, Morpheme, LexicalCategory
from language import Language, LanguageProfile
from grammar import GrammarEngine
from translator import SemanticTranslator
from semantics import SemanticVector, DerivationRecord, DerivationType, LoanProvenance
from world_grid import WorldGrid


CONCEPT_DOMAINS = {
    "nature": [
        ((1.0, 0.0, 0.7, 0.4, 0.4, 0.1, 0.3), "water"),
        ((0.9, 0.1, 0.4, 0.7, 0.9, 0.1, 0.3), "fire"),
        ((1.0, 0.0, 0.6, 0.7, 0.1, 0.1, 0.7), "earth"),
        ((1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.1), "stone"),
        ((1.0, 0.4, 0.7, 0.6, 0.2, 0.1, 0.4), "tree"),
        ((0.8, 0.0, 0.9, 1.0, 0.6, 0.1, 0.9), "sun"),
        ((0.8, 0.0, 0.6, 0.6, 0.3, 0.1, 0.7), "moon"),
        ((0.6, 0.0, 0.6, 0.5, 0.8, 0.1, 0.8), "wind"),
        ((0.8, 0.0, 0.6, 0.4, 0.9, 0.1, 0.6), "rain"),
        ((0.9, 0.0, 0.4, 0.6, 0.2, 0.1, 0.4), "ice"),
    ],
    "body": [
        ((0.9, 0.8, 0.6, 0.7, 0.4, 0.3, 0.2), "head"),
        ((1.0, 0.8, 0.6, 0.6, 0.6, 0.4, 0.2), "hand"),
        ((0.9, 0.8, 0.6, 0.4, 0.5, 0.3, 0.2), "eye"),
        ((0.7, 0.9, 0.7, 0.6, 0.4, 0.6, 0.2), "heart"),
        ((0.9, 0.6, 0.4, 0.7, 0.5, 0.3, 0.2), "blood"),
        ((1.0, 0.5, 0.5, 0.8, 0.1, 0.1, 0.2), "bone"),
        ((1.0, 0.9, 0.6, 0.6, 0.5, 0.4, 0.5), "body"),
    ],
    "action": [
        ((0.5, 0.8, 0.6, 0.6, 0.8, 0.4, 0.3), "make"),
        ((0.6, 0.7, 0.5, 0.4, 0.7, 0.2, 0.5), "walk"),
        ((0.6, 0.8, 0.3, 0.7, 0.8, 0.4, 0.4), "hunt"),
        ((0.4, 0.8, 0.6, 0.4, 0.5, 0.3, 0.4), "see"),
        ((0.4, 0.9, 0.6, 0.4, 0.7, 1.0, 0.3), "speak"),
        ((0.8, 0.8, 0.8, 0.5, 0.6, 0.3, 0.2), "eat"),
        ((0.7, 0.8, 0.2, 0.85, 0.9, 0.2, 0.3), "strike"),
        ((0.6, 0.8, 0.85, 0.6, 0.6, 0.8, 0.4), "heal"),
        ((0.5, 0.8, 0.1, 0.4, 0.4, 0.3, 0.5), "die"),
        ((0.5, 0.7, 0.5, 0.4, 0.4, 0.3, 0.5), "journey"),
    ],
    "quality": [
        ((0.2, 0.3, 0.9, 0.6, 0.3, 0.7, 0.5), "good"),
        ((0.2, 0.3, 0.1, 0.6, 0.4, 0.3, 0.5), "bad"),
        ((0.4, 0.1, 0.6, 0.9, 0.3, 0.1, 0.9), "big"),
        ((0.4, 0.1, 0.5, 0.1, 0.3, 0.1, 0.1), "small"),
        ((0.6, 0.0, 0.7, 0.8, 0.6, 0.1, 0.3), "hot"),
        ((0.7, 0.0, 0.4, 0.6, 0.2, 0.1, 0.3), "cold"),
        ((0.3, 0.0, 0.2, 0.6, 0.2, 0.1, 0.7), "dark"),
        ((0.3, 0.0, 0.9, 0.6, 0.4, 0.1, 0.8), "bright"),
    ],
    "social": [
        ((0.9, 1.0, 0.6, 0.5, 0.5, 0.7, 0.3), "person"),
        ((0.9, 1.0, 0.85, 0.4, 0.4, 0.95, 0.2), "mother"),
        ((0.9, 1.0, 0.75, 0.75, 0.6, 0.85, 0.3), "father"),
        ((0.8, 1.0, 0.8, 0.4, 0.4, 0.9, 0.4), "friend"),
        ((0.8, 1.0, 0.2, 0.8, 0.7, 0.1, 0.4), "enemy"),
        ((0.8, 1.0, 0.7, 0.5, 0.5, 0.9, 0.7), "tribe"),
        ((0.9, 0.9, 0.5, 0.6, 0.7, 0.2, 0.4), "animal"),
    ]
}


class DialectProbe:
    @classmethod
    def evoke_pixel_dialect(
        cls,
        grid: WorldGrid,
        x: int,
        y: int,
        proto_roots: Sequence[Word],
        epoch: int,
    ) -> dict:
        cell = grid.get_cell_state(x, y)
        alt = cell["elevation"]

        if alt <= 0.0:
            depth_m = int(round(abs(alt) * 3800))
            return {
                "is_water": True,
                "coordinates": (cell["x"], cell["y"]),
                "dialect_name": "Marine Expanse (Open Water)",
                "epoch": epoch,
                "geography": {
                    "elevation": cell["elevation"],
                    "depth_meters": depth_m,
                    "temperature": cell["temperature"],
                    "humidity": cell["humidity"],
                    "vegetation": 0.0,
                    "terrain_permeability": cell["permeability"],
                },
                "demographics": {
                    "population": 0,
                    "carrying_capacity": 0.0,
                    "infrastructure": 0.0,
                    "prestige": 0.0,
                },
                "cultural_attention": cell["attention"],
                "message": "Open ocean waters.",
            }

        bias = np.asarray(cell["phonetic_bias"], dtype=np.float32)
        h = cell["head_directionality"]
        s_idx = cell["synthesis_index"]
        s_comp = cell["syllable_complexity"]
        temp = cell["temperature"]
        hum = cell["humidity"]
        veg = cell["vegetation"]
        pop = cell["population"]
        infra = cell["infrastructure"]
        prestige = cell["prestige"]
        attn = np.asarray(cell["attention"], dtype=np.float32)

        biomes = [
            ("Alpine Highlands", math.exp(-((alt - 0.75) ** 2) / 0.08)),
            ("Coastal Littoral", math.exp(-((hum - 0.85) ** 2) / 0.06)),
            ("Forest Basin",     math.exp(-((veg - 0.80) ** 2) / 0.06)),
            ("Arid Steppe",      math.exp(-((temp - 0.75) ** 2 + (hum - 0.15) ** 2) / 0.08)),
            ("Cradle Valley",    math.exp(-((alt - 0.22) ** 2 + (veg - 0.45) ** 2) / 0.08)),
        ]
        reg_name = max(biomes, key=lambda b: b[1])[0]

        profile = LanguageProfile(
            name=f"{reg_name}_({x},{y})",
            head_directionality=h,
            synthesis_index=s_idx,
            syllable_complexity=s_comp,
            max_transition_cost=1.80 + 3.60 * s_comp,
        )

        dialect_lang = Language(name=f"{reg_name}_({x},{y})", profile=profile, population=pop)
        dialect_lang.cultural_attention = list(attn)

        base_n = min(12, len(proto_roots)) if epoch < 10 else len(proto_roots)
        time_growth = 260.0 * math.sqrt(epoch / 100.0) + (0.85 * epoch)
        civ_mult = 0.35 + 0.65 * math.tanh(pop * 0.30 + infra * 0.15)
        total_vocab_count = max(12, int(base_n + time_growth * civ_mult))
        render_vocab_limit = min(600, total_vocab_count)

        cluster_seed = (int(x // 3) * 73856093) ^ (int(y // 3) * 19349663) ^ (epoch // 5 * 83492791)
        rng = random.Random(cluster_seed)

        living_words: list[Word] = [w.copy() for w in proto_roots[:min(len(proto_roots), render_vocab_limit)]]

        # Horizontal loanwords
        loan_sources = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.W and 0 <= ny < grid.H:
                n_cell = grid.get_cell_state(nx, ny)
                if n_cell["elevation"] > 0.0 and n_cell["prestige"] > prestige + 0.40:
                    loan_sources.append(n_cell)

        num_loans = int(render_vocab_limit * 0.12) if (loan_sources and render_vocab_limit > 50) else 0
        loan_words = []
        for l_idx in range(num_loans):
            source = rng.choice(loan_sources)
            donor_root = rng.choice(proto_roots)
            p_nativized = [
                Phoneme(np.clip(p.g + bias * 0.5, 0.0, 1.0), duration=p.duration)
                for p in donor_root.phonemes
            ]
            repaired_loan = repair_phonemes(p_nativized, profile=profile)
            if repaired_loan:
                w_loan = Word(
                    word_id=5000 + l_idx,
                    vector=donor_root.vector,
                    phonemes=repaired_loan,
                    derivation=DerivationRecord(
                        root_family_id=donor_root.derivation.root_family_id,
                        derivation_type=DerivationType.LOANWORD,
                        provenance=LoanProvenance(
                            source_language=f"Culture_({source['x']},{source['y']})",
                            donor_word_id=donor_root.id,
                            donor_form_at_borrowing=donor_root.plain_form,
                            donor_vector_at_borrowing=donor_root.vector,
                            generation_borrowed=max(0, epoch - rng.randint(10, 500)),
                        )
                    ),
                    usage_frequency=rng.uniform(0.7, 1.5),
                    category=LexicalCategory.CONTENT_OPEN,
                )
                loan_words.append(w_loan)

        living_words.extend(loan_words)

        # Compounding
        domain_keys = list(CONCEPT_DOMAINS.keys())
        num_neologisms = max(0, render_vocab_limit - len(living_words))
        domain_pools: dict[str, list[Word]] = {k: [] for k in domain_keys}

        for w in living_words:
            if w.category != LexicalCategory.CONTENT_OPEN:
                continue
            pos = SemanticTranslator.get_pos(w.vector)
            if pos == "Verb": domain_pools["action"].append(w)
            elif pos == "Adjective": domain_pools["quality"].append(w)
            elif w.vector.animacy >= 0.70: domain_pools["social"].append(w)
            elif w.vector.concreteness >= 0.75: domain_pools["nature"].append(w)
            else: domain_pools["body"].append(w)

        content_all = [w for w in living_words if w.category == LexicalCategory.CONTENT_OPEN]
        for k in domain_keys:
            if not domain_pools[k]:
                domain_pools[k] = list(content_all)

        frame_rules = [
            ("nature", "action"), ("body", "nature"), ("social", "action"),
            ("quality", "nature"), ("nature", "quality"), ("social", "social"), ("action", "nature")
        ]
        t_drift = min(4.5, epoch / 500.0)

        for neo_idx in range(num_neologisms):
            d1_name, d2_name = rng.choice(frame_rules)
            w1 = rng.choice(domain_pools[d1_name] or content_all)
            w2 = rng.choice(domain_pools[d2_name] or content_all)
            if w1.id == w2.id:
                continue
            blended_vec = w1.vector.blend(w2.vector, weight=rng.uniform(0.42, 0.58))
            comp_word = dialect_lang.compound_concepts(
                w1=w1, w2=w2, blended_vec=blended_vec, new_id=2000 + neo_idx,
                epoch=epoch, alt=alt, hum=hum, temp=temp, t_drift=t_drift, rng=rng
            )
            if comp_word:
                living_words.append(comp_word)

        # Emergent Tonogenesis: condition of monosyllabicity and coda loss
        coda_drain_pressure = 1.0 / (1.0 + math.exp(6.0 * (s_comp - 0.40)))
        analytic_pressure = 1.0 / (1.0 + math.exp(6.0 * (s_idx - 0.40)))
        has_phonemic_tone = (coda_drain_pressure * 0.6 + analytic_pressure * 0.4) >= 0.55

        # Extract vowel repelling pool across dialect sample
        vowel_pool: list[np.ndarray] = []
        for w in living_words[:40]:
            for p in w.phonemes:
                if p.is_vowel:
                    vowel_pool.append(p.g)
                    if len(vowel_pool) >= 28:
                        break

        transformed_words: list[Word] = []
        dissonance_acc = 0.0

        for w0 in living_words:
            X_w0 = w0.tensor
            if len(X_w0) == 0:
                continue

            valence_intensity = abs(w0.vector.valence - 0.50) + (w0.vector.potency * 0.4)
            mutated_phonemes = []
            for i, p_seg in enumerate(w0.phonemes):
                adj = w0.phonemes[i - 1] if i > 0 else None
                m_seg = relax_gesture_gradient(
                    p=p_seg,
                    adjacent_p=adj,
                    neighbor_g_pool=vowel_pool,
                    valence_intensity=valence_intensity,
                    bias=bias,
                    alt=alt,
                    hum=hum,
                    temp=temp,
                    t_drift=t_drift,
                    rng=rng,
                )
                mutated_phonemes.append(m_seg)

            eroded_phonemes = apply_syncope_and_clustering(
                mutated_phonemes,
                usage_frequency=w0.usage_frequency,
                alt=alt,
                hum=hum,
                temp=temp,
                profile=profile,
            )

            X_trans = np.stack([p.vector for p in eroded_phonemes], axis=0)
            d_dist = trajectory_distance(X_trans[:, :8], X_w0[:, :8])
            dissonance_acc += d_dist

            m_stem = Morpheme(eroded_phonemes, root_family_id=w0.derivation.root_family_id)
            w_loc = Word(
                word_id=w0.id,
                vector=w0.vector,
                phonemes=eroded_phonemes,
                morphemes=[m_stem],
                derivation=w0.derivation,
                category=w0.category,
                usage_frequency=w0.usage_frequency,
                senses=w0.senses,
            )

            form_w, plain_w, _ = render_and_extract_tones(
                w_loc.tensor,
                directional_bias=h,
                is_tonal=has_phonemic_tone,
            )
            w_loc._form = form_w
            w_loc._plain_form = plain_w

            dialect_lang.add_word(w_loc)
            transformed_words.append(w_loc)

        mean_diss = dissonance_acc / max(1, len(living_words))
        intelligibility = float(np.clip(math.exp(-1.80 * mean_diss) * (0.25 + 0.75 * cell["permeability"]), 0.0, 1.0))

        contrastive_vowels, contrastive_consonants = dialect_lang.extract_phonemic_inventory()

        tone_inventory = []
        minimal_tone_pairs = []

        if has_phonemic_tone:
            tone_names = {
                "˥": "High-Level (55)", "˧": "Mid-Level (33)", "˩": "Low-Level (11)",
                "˥˩": "High-Falling (51)", "˩˥": "Low-Rising (15)", "˨˩˦": "Dipping (214)",
            }
            tone_counts = Counter()
            grouped_by_plain: dict[str, list[dict]] = {}

            for w_loc in transformed_words:
                plain = w_loc.plain_form
                tone_part = "".join([c for c in w_loc.form if c in "˩˨˧˦˥"])
                if tone_part:
                    tone_counts[tone_part] += 1
                    if plain not in grouped_by_plain:
                        grouped_by_plain[plain] = []
                    gloss_str = SemanticTranslator.translate(w_loc, dialect_lang)
                    if not any(item["tone"] == tone_part for item in grouped_by_plain[plain]):
                        grouped_by_plain[plain].append({"form": w_loc.form, "gloss": gloss_str, "tone": tone_part})

            for t_sym, cnt in tone_counts.most_common():
                tone_inventory.append({
                    "symbol": t_sym,
                    "name": tone_names.get(t_sym, f"Tone ({t_sym})"),
                    "frequency": round((cnt / max(1, len(transformed_words))) * 100, 1)
                })

            for plain, variants in grouped_by_plain.items():
                if len(variants) >= 2:
                    minimal_tone_pairs.append({"base": plain, "variants": variants[:3]})
                if len(minimal_tone_pairs) >= 6:
                    break

        paradigm = GrammarEngine.discover_grammar_system(dialect_lang)

        content_nouns = [w for w in transformed_words if w.category == LexicalCategory.CONTENT_OPEN and w.vector.concreteness >= 0.60]
        content_verbs = [w for w in transformed_words if w.category == LexicalCategory.CONTENT_OPEN and w.vector.dynamism >= 0.50]
        sample_noun = content_nouns[0] if content_nouns else transformed_words[0]
        sample_verb = content_verbs[0] if content_verbs else (content_nouns[1] if len(content_nouns) > 1 else transformed_words[0])

        noun_table = GrammarEngine.get_noun_declension_table(sample_noun, dialect_lang, paradigm=paradigm)
        verb_table = GrammarEngine.get_verb_conjugation_table(sample_verb, dialect_lang, paradigm=paradigm)

        lexicon_payload = []
        cohort_sample = transformed_words[:32]
        for w_loc, w0 in zip(transformed_words, living_words):
            gloss = SemanticTranslator.translate(w_loc, dialect_lang)
            if w_loc.derivation.derivation_type == DerivationType.LOANWORD and w_loc.derivation.provenance:
                gloss = f"{gloss} [Loan: {w_loc.derivation.provenance.source_language}]"

            # Energy now integrates ambiguity and modulation costs
            total_cost = w_loc.energetic_cost(profile=profile, cohort_sample=cohort_sample)

            lexicon_payload.append({
                "id": w_loc.id,
                "form": w_loc.form,
                "plain": w_loc.plain_form,
                "gloss": gloss,
                "proto_form": w0.plain_form,
                "pos": GrammarEngine.get_part_of_speech(w_loc),
                "energy_cost": total_cost,
            })

        pronouns_display = {k: f"/{v.form}/" for k, v in paradigm.pronouns.items()}
        numerals_display = {val: GrammarEngine.format_numeral(val, dialect_lang, paradigm) for val in range(1, 6)}
        human_pop = int(round(pop * 250000))
        tone_label = f"Tonal ({len(tone_inventory)} Tones)" if has_phonemic_tone else "Non-Tonal (Stress-Accent)"

        return {
            "is_water": False,
            "coordinates": (cell["x"], cell["y"]),
            "dialect_name": reg_name,
            "epoch": epoch,
            "geography": {
                "elevation": cell["elevation"],
                "temperature": cell["temperature"],
                "humidity": cell["humidity"],
                "vegetation": cell["vegetation"],
                "terrain_permeability": cell["permeability"],
            },
            "demographics": {
                "population": human_pop,
                "raw_pop": pop,
                "carrying_capacity": cell["carrying_capacity"],
                "infrastructure": cell["infrastructure"],
                "prestige": prestige,
            },
            "cultural_attention": cell["attention"],
            "typology": {
                "word_order": profile.dominant_word_order,
                "head_directionality": h,
                "synthesis_index": s_idx,
                "syllable_complexity": s_comp,
                "morphological_strategy": paradigm.verb_classes[0].morph_strategy.capitalize() if paradigm.verb_classes else "Analytic",
                "alignment": GrammarEngine.get_alignment_system(dialect_lang),
                "article_system": paradigm.article_system,
                "number_system": paradigm.number_system,
                "tone_system": tone_label,
            },
            "phonology": {
                "vowels": contrastive_vowels,
                "consonants": contrastive_consonants,
                "has_ejectives": any("ʼ" in c for c in contrastive_consonants),
                "has_implosives": any(c in {"ɓ", "ɗ", "ɠ", "ʄ"} for c in contrastive_consonants),
                "has_pharyngealized": any("ˁ" in c for c in contrastive_consonants),
                "has_tones": has_phonemic_tone,
                "tone_inventory": tone_inventory,
                "minimal_tone_pairs": minimal_tone_pairs,
                "total_phonemes": len(contrastive_vowels) + len(contrastive_consonants),
            },
            "noun_paradigm": {
                "noun": sample_noun.form,
                "noun_gloss": SemanticTranslator.translate(sample_noun, dialect_lang),
                "alignment": GrammarEngine.get_alignment_system(dialect_lang),
                "table": noun_table,
            },
            "verb_paradigm": {
                "verb": sample_verb.form,
                "verb_gloss": SemanticTranslator.translate(sample_verb, dialect_lang),
                "table": verb_table,
            },
            "pronouns": pronouns_display,
            "numerals": numerals_display,
            "vocabulary_stats": {
                "total_lexicon_size": total_vocab_count,
                "displayed_sample": len(lexicon_payload),
            },
            "lexicon": lexicon_payload,
            "cradle_intelligibility": round(intelligibility, 3),
        }