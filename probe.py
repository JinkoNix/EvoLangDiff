"""
probe.py — Spatial Dialect Realizer and Continuous Tensor Collapse Engine.
Optimized for instantaneous (<40ms) inspection via cohort capping and probe caching.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
from typing import Sequence
from collections import Counter, namedtuple
import numpy as np

from phonetics import (
    Phoneme,
    repair_phonemes,
    render_and_extract_tones,
    enforce_acoustic_contrast,
)
from word import Word, LexicalCategory
from language import Language, LanguageProfile
from grammar import GrammarEngine
from translator import SemanticTranslator
from semantics import SemanticVector, SemanticSpace
from world_grid import WorldGrid

EnvRecord = namedtuple("EnvRecord", ["altitude", "humidity", "temperature", "vegetation", "fauna", "roughness"])


class DialectProbe:
    @classmethod
    def evoke_pixel_dialect(
        cls,
        grid: WorldGrid,
        x: int,
        y: int,
        epoch: int | None = None,
        proto_roots: Sequence[Word] | None = None,
        **kwargs,
    ) -> dict:
        current_epoch = epoch if epoch is not None else kwargs.get("epoch", grid.epoch)
        cache_key = (x, y, current_epoch)
        if cache_key in grid._probe_cache:
            return grid._probe_cache[cache_key]

        cell = grid.get_cell_state(x, y)
        alt = cell["elevation"]
        pop = cell["population"]

        # 1. Marine Expanse
        if alt <= 0.0:
            res = {
                "is_water": True, "is_uninhabited": True, "coordinates": (cell["x"], cell["y"]),
                "dialect_name": "Marine Expanse (Open Water)", "epoch": current_epoch,
                "geography": {
                    "elevation": cell["elevation"], "depth_meters": int(round(abs(alt) * 3800)),
                    "temperature": cell["temperature"], "humidity": cell["humidity"],
                    "vegetation": 0.0, "fauna": 0.0, "terrain_permeability": cell["permeability"], "nearby_landmarks": [],
                },
                "demographics": {"population": 0, "carrying_capacity": 0.0, "infrastructure": 0.0, "settlement_tier": "Open Ocean", "net_calories": 0.0, "prestige": 0.0},
                "cultural_attention": cell["attention"], "message": "Open ocean waters.",
            }
            grid._probe_cache[cache_key] = res
            return res

        # 2. Uninhabited Wilderness
        nearby_landmarks = [
            {"name": lm.name_endonym or f"Site_{lm.id}", "gloss": lm.gloss or "Landmark", "feature_type": lm.feature_type, "distance_km": round(d * 25.0, 1)}
            for lm in grid.landmarks.values()
            if (d := math.hypot(lm.x - x, lm.y - y)) <= 8.0
        ]
        nearby_landmarks.sort(key=lambda item: item["distance_km"])

        if pop < 0.05:
            res = {
                "is_water": False, "is_uninhabited": True, "coordinates": (cell["x"], cell["y"]),
                "dialect_name": "Pristine Wilderness (Uninhabited)", "epoch": current_epoch,
                "geography": {
                    "elevation": cell["elevation"], "temperature": cell["temperature"],
                    "humidity": cell["humidity"], "vegetation": cell["vegetation"],
                    "fauna": cell["fauna"], "terrain_permeability": cell["permeability"],
                    "nearby_landmarks": nearby_landmarks,
                },
                "demographics": {"population": 0, "raw_pop": 0.0, "carrying_capacity": cell["carrying_capacity"], "infrastructure": cell["infrastructure"], "settlement_tier": cell["settlement_tier"], "net_calories": cell["net_calories"], "prestige": cell["prestige"]},
                "polity": {"dominant_id": -1, "name": "Unclaimed Territory", "literal_meaning": "No Sovereign", "control_percentage": 0.0, "is_capital": False, "color": [80, 80, 80]},
                "cultural_attention": cell["attention"], "message": "Unsettled wilderness.",
            }
            grid._probe_cache[cache_key] = res
            return res

        # 3. Speech Community Context
        h, s_idx, s_comp = cell["head_directionality"], cell["synthesis_index"], cell["syllable_complexity"]
        profile = LanguageProfile(name=f"Dialect_({x},{y})", head_directionality=h, synthesis_index=s_idx, syllable_complexity=s_comp, max_transition_cost=1.80 + 3.60 * s_comp)
        dialect_lang = Language(name=profile.name, profile=profile, population=pop)
        dialect_lang.cultural_attention = list(np.asarray(cell["attention"], dtype=np.float32))

        env = EnvRecord(
            altitude=alt,
            humidity=cell["humidity"],
            temperature=cell["temperature"],
            vegetation=cell["vegetation"],
            fauna=cell["fauna"],
            roughness=cell["permeability"]
        )
        has_tone = ((1.0 / (1.0 + math.exp(6.0 * (s_comp - 0.40)))) * 0.6 + (1.0 / (1.0 + math.exp(6.0 * (s_idx - 0.40)))) * 0.4) >= 0.55

        # 4. Collapse Word Presence Tensor U(x, y)
        n_manifold = len(grid.lexical_manifold)
        active_indices = [i for i in range(n_manifold) if grid.word_presence[i, y, x] > 0.035]

        # SPEED OPTIMIZATION: Cap cohort to top 35 words for fast UI response
        if len(active_indices) > 35:
            ranked_active = sorted(active_indices, key=lambda idx: grid.word_presence[idx, y, x], reverse=True)
            active_indices = ranked_active[:35]
        elif len(active_indices) < 12:
            active_indices = list(np.argsort([-grid.word_presence[i, y, x] for i in range(n_manifold)])[:28])

        local_presence = [grid.word_presence[i, y, x] for i in active_indices]
        local_base_words = [grid.lexical_manifold[i] for i in active_indices]

        # 5. Realize Local Dialect via 9D Phonetic Displacement & Imperial Pull
        phi_local = grid.phonetic_drift_field[y, x] if hasattr(grid, "phonetic_drift_field") else np.zeros(9, dtype=np.float32)
        dom_polity_id = cell["polity"]["dominant_id"]
        dom_polity = next((p for p in grid.polities if p.id == dom_polity_id), None)

        living_words: list[Word] = []
        is_suzerain_province = dom_polity is not None and not cell["polity"]["is_capital"] and dom_polity.is_standardized

        for w_base, presence_weight in zip(local_base_words, local_presence):
            is_superstrate_root = (dom_polity and dom_polity.is_standardized and w_base.id in dom_polity.canonical_roots)

            if is_superstrate_root:
                canon_phonemes = dom_polity.canonical_roots[w_base.id]
                h_weight = cell["polity"]["control_percentage"] / 100.0
                dist_factor = 1.0 / (1.0 + float(dom_polity.dist_cells[y, x]) * 0.08)
                pull_weight = float(np.clip(dom_polity.centralization * h_weight * dist_factor * cell["permeability"], 0.0, 0.90))

                warped_phonemes = []
                for p_idx, p in enumerate(w_base.phonemes):
                    g_displaced = np.clip(p.g + phi_local, 0.0, 1.0)
                    if p_idx < len(canon_phonemes):
                        g_std = canon_phonemes[p_idx].g
                        g_final = (1.0 - pull_weight) * g_displaced + pull_weight * g_std
                    else:
                        g_final = g_displaced
                    warped_phonemes.append(Phoneme(g_final, duration=p.duration))
            else:
                warped_phonemes = [Phoneme(np.clip(p.g + phi_local, 0.0, 1.0), duration=p.duration) for p in w_base.phonemes]

            repaired = repair_phonemes(warped_phonemes, profile=profile, max_passes=2)
            if not repaired:
                continue

            dynamic_usage = float(presence_weight * math.log1p(pop * 1.8) * 3.5)

            w_loc = Word(
                word_id=w_base.id,
                vector=w_base.vector,
                phonemes=repaired,
                derivation=w_base.derivation,
                category=w_base.category,
                usage_frequency=round(dynamic_usage, 2),
                generation_born=w_base.generation_born,
            )
            living_words.append(w_loc)

        ambient_noise = float(cell["infrastructure"] * 0.05 + 0.05)
        enforce_acoustic_contrast(living_words[:25], min_auditory_margin=0.28, ambient_noise=ambient_noise)

        for w_loc in living_words:
            form_w, plain_w, _ = render_and_extract_tones(w_loc.tensor, directional_bias=h, is_tonal=has_tone)
            w_loc._form, w_loc._plain_form = form_w, plain_w
            dialect_lang.add_word(w_loc)

        # 6. Paradigm Discovery
        paradigm = GrammarEngine.discover_grammar_system(dialect_lang)
        vowels, consonants = dialect_lang.extract_phonemic_inventory(paradigm=paradigm)

        counts = Counter()
        grouped: dict[str, list[dict]] = {}
        if has_tone:
            tone_names = {"˥": "High (55)", "˧": "Mid (33)", "˩": "Low (11)", "˥˩": "Falling (51)", "˩˥": "Rising (15)", "˨˩˦": "Dipping (214)"}
            for w in living_words:
                tone_str = "".join(c for c in w.form if c in "˩˨˧˦˥")
                if tone_str:
                    counts[tone_str] += 1
                    grouped.setdefault(w.plain_form, []).append({"form": w.form, "gloss": w.plain_form, "tone": tone_str})

            tones_inv = [{"symbol": sym, "name": tone_names.get(sym, f"Tone ({sym})"), "frequency": round((cnt / max(1, len(living_words))) * 100, 1)} for sym, cnt in counts.most_common()]
            tone_pairs = [{"base": base, "variants": vars[:3]} for base, vars in grouped.items() if len(vars) >= 2][:6]
        else:
            tones_inv, tone_pairs = [], []

        pol = cell["polity"]
        if dom_polity and not dom_polity.name:
            capital_anchor = min(living_words, key=lambda w: w.vector.distance_to(SemanticSpace.PERSON))
            gov_tag = dom_polity.government_type
            endonym_title = capital_anchor.form.capitalize()
            dom_polity.name = f"{gov_tag} of {endonym_title}"
            dom_polity.literal_meaning = f"Authority of {SemanticTranslator.translate(capital_anchor, dialect_lang)}"
            pol["name"] = dom_polity.name
            pol["literal_meaning"] = dom_polity.literal_meaning

        true_nouns = [w for w in living_words if w.category == LexicalCategory.CONTENT_OPEN and SemanticTranslator.get_pos(w.vector) == "Noun"]
        true_verbs = [w for w in living_words if w.category == LexicalCategory.CONTENT_OPEN and SemanticTranslator.get_pos(w.vector) == "Verb"]
        sample_noun = true_nouns[0] if true_nouns else living_words[0]
        sample_verb = true_verbs[0] if true_verbs else living_words[0]

        # Fast UI Lexicon Serialization (Avoiding duplicate syllabifications)
        lexicon_payload = []
        for w in living_words:
            is_superstrate = (dom_polity and dom_polity.is_standardized and w.id in dom_polity.canonical_roots)
            pos_label = GrammarEngine.get_part_of_speech(w)
            lexicon_payload.append({
                "id": w.id, "form": w.form, "root_form": w.form, "plain": w.plain_form,
                "gloss": SemanticTranslator.translate(w, dialect_lang), "pos": pos_label,
                "grammar_info": {"pos": pos_label},
                "stratification": "Superstrate (Imperial Lingua Franca)" if is_superstrate else "Substrate (Local Vernacular)",
                "energy_cost": round(float(w.energy_cost if hasattr(w, 'energy_cost') else 3.5), 2),
                "usage_frequency": w.usage_frequency,
            })

        dialect_label = pol["name"] if pol["dominant_id"] >= 0 else "Autonomous Dialect Continuum"
        if cell["is_danger_zone"]:
            status_label = "Contested Frontier [Active Conflict]"
        elif is_suzerain_province:
            status_label = f"Vassal Province [Superstrate: {dom_polity.name}]"
        elif pol["dominant_id"] >= 0:
            status_label = f"Sovereign Core [{dom_polity.government_type}]"
        else:
            status_label = "Independent Clan Territory"

        res = {
            "is_water": False, "is_uninhabited": False, "coordinates": (x, y), "epoch": current_epoch,
            "dialect_name": f"{dialect_label} — {status_label}",
            "geography": {
                "elevation": cell["elevation"], "temperature": cell["temperature"],
                "humidity": cell["humidity"], "vegetation": cell["vegetation"],
                "fauna": cell["fauna"], "terrain_permeability": cell["permeability"],
                "nearby_landmarks": nearby_landmarks,
            },
            "demographics": {
                "population": int(round(pop * 250000)), "raw_pop": pop,
                "carrying_capacity": cell["carrying_capacity"],
                "infrastructure": cell["infrastructure"],
                "settlement_tier": cell["settlement_tier"],
                "net_calories": cell["net_calories"],
                "prestige": cell["prestige"]
            },
            "polity": pol,
            "cultural_attention": cell["attention"],
            "typology": {
                "word_order": profile.dominant_word_order, "head_directionality": h,
                "synthesis_index": s_idx, "syllable_complexity": s_comp,
                "morphological_strategy": paradigm.verb_classes[0].morph_strategy.capitalize() if paradigm.verb_classes else "Analytic",
                "alignment": GrammarEngine.get_alignment_system(dialect_lang),
                "article_system": paradigm.article_system, "number_system": paradigm.number_system,
                "numeral_radix": paradigm.numeral_chain.name,
                "noun_classes": [c.label for c in paradigm.noun_classes],
                "tone_system": f"Tonal ({len(tones_inv)} Tones)" if has_tone else "Non-Tonal",
            },
            "phonology": {
                "vowels": vowels, "consonants": consonants, "total_phonemes": len(vowels) + len(consonants),
                "has_ejectives": any("ʼ" in c for c in consonants), "has_implosives": any(c in {"ɓ", "ɗ", "ɠ", "ʄ"} for c in consonants),
                "has_pharyngealized": any("ˁ" in c for c in consonants), "has_tones": has_tone,
                "tone_inventory": tones_inv, "minimal_tone_pairs": tone_pairs,
            },
            "noun_paradigm": {
                "noun": sample_noun.form, "noun_gloss": SemanticTranslator.translate(sample_noun, dialect_lang),
                "title_display": f"«{SemanticTranslator.translate(sample_noun, dialect_lang)}» /{sample_noun.form}/",
                "alignment": GrammarEngine.get_alignment_system(dialect_lang), "table": GrammarEngine.get_noun_declension_table(sample_noun, dialect_lang, paradigm),
            },
            "verb_paradigm": {
                "verb": sample_verb.form, "verb_gloss": SemanticTranslator.translate(sample_verb, dialect_lang),
                "title_display": f"«{SemanticTranslator.translate(sample_verb, dialect_lang)}» {GrammarEngine.get_infinitive_form(sample_verb, dialect_lang, paradigm)}",
                "table": GrammarEngine.get_verb_conjugation_table(sample_verb, dialect_lang, paradigm),
            },
            "pronouns": {k: f"/{v.form}/" for k, v in paradigm.pronouns.items()},
            "numerals": {val: GrammarEngine.format_numeral(val, dialect_lang, paradigm) for val in range(1, paradigm.numeral_chain.primary_base + 3)},
            "vocabulary_stats": {"total_lexicon_size": len(living_words), "displayed_sample": len(lexicon_payload)},
            "lexicon": lexicon_payload,
            "cradle_intelligibility": round(float(np.clip(1.0 - np.linalg.norm(phi_local) * 0.8, 0.05, 1.0)), 3),
        }
        grid._probe_cache[cache_key] = res
        return res