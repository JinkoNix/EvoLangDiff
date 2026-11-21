"""
probe.py — Read-only Dialect Realizer and Diachronic Lexicon Extractor.
"""

from __future__ import annotations

import math
from collections import Counter, namedtuple
from typing import Sequence
import numpy as np

from grammar import GrammarEngine
from language import Language, LanguageProfile, inventory_size
from phonetics import render_and_extract_tones, trajectory_distance, word_energy
from semantics import SemanticSpace
from translator import SemanticTranslator
from word import LexicalCategory, Phoneme, Word, repair_phonemes
from world_grid import WorldGrid

EnvRecord = namedtuple("EnvRecord", ["altitude", "humidity", "temperature", "vegetation", "fauna", "roughness", "population"])


def _uninhabited(cell: dict, current_epoch: int, nearby_landmarks: list, name: str, message: str) -> dict:
    return {
        "is_water": False, "is_uninhabited": True, "coordinates": (cell["x"], cell["y"]),
        "dialect_name": name, "epoch": current_epoch,
        "geography": {
            "elevation": cell["elevation"], "temperature": cell["temperature"],
            "humidity": cell["humidity"], "vegetation": cell["vegetation"],
            "fauna": cell["fauna"], "terrain_permeability": cell["permeability"],
            "nearby_landmarks": nearby_landmarks,
        },
        "demographics": {
            "population": 0, "raw_pop": 0.0,
            "carrying_capacity": cell["carrying_capacity"],
            "infrastructure": cell["infrastructure"],
            "settlement_tier": cell["settlement_tier"],
            "net_calories": cell["net_calories"],
            "prestige": cell["prestige"],
        },
        "polity": {
            "dominant_id": -1, "name": "Unclaimed Territory",
            "literal_meaning": "No Sovereign", "control_percentage": 0.0,
            "is_capital": False, "color": [80, 80, 80],
        },
        "cultural_attention": cell["attention"], "message": message,
    }


class DialectProbe:
    @classmethod
    def evoke_pixel_dialect(
        cls,
        grid: WorldGrid,
        x: int,
        y: int,
        epoch: int | None = None,
        **kwargs,
    ) -> dict:
        current_epoch = epoch if epoch is not None else grid.epoch
        cache_key = (x, y, current_epoch)
        if cache_key in grid._probe_cache:
            return grid._probe_cache[cache_key]

        cell = grid.get_cell_state(x, y)
        alt, pop = cell["elevation"], cell["population"]

        if alt <= 0.0:
            res = {
                "is_water": True, "is_uninhabited": True, "coordinates": (cell["x"], cell["y"]),
                "dialect_name": "Marine Expanse (Open Water)", "epoch": current_epoch,
                "geography": {
                    "elevation": cell["elevation"], "depth_meters": int(round(abs(alt) * 3800)),
                    "temperature": cell["temperature"], "humidity": cell["humidity"],
                    "vegetation": 0.0, "fauna": 0.0, "terrain_permeability": cell["permeability"], "nearby_landmarks": [],
                },
                "demographics": {
                    "population": 0, "carrying_capacity": 0.0, "infrastructure": 0.0,
                    "settlement_tier": "Open Ocean", "net_calories": 0.0, "prestige": 0.0,
                },
                "cultural_attention": cell["attention"], "message": "Open ocean waters.",
            }
            grid._probe_cache[cache_key] = res
            return res

        nearby_landmarks = [
            {"name": lm.name_endonym or f"Site_{lm.id}", "gloss": lm.gloss or "Landmark", "feature_type": lm.feature_type, "distance_km": round(d * 25.0, 1)}
            for lm in grid.landmarks.values()
            if (d := math.hypot(lm.x - x, lm.y - y)) <= 8.0
        ]
        nearby_landmarks.sort(key=lambda item: item["distance_km"])

        if pop < 0.008:
            res = _uninhabited(cell, current_epoch, nearby_landmarks, "Pristine Wilderness (Uninhabited)", "Unsettled wilderness.")
            grid._probe_cache[cache_key] = res
            return res

        h, s_idx, s_comp = cell["head_directionality"], cell["synthesis_index"], cell["syllable_complexity"]
        profile = LanguageProfile(
            name=f"Dialect_({x},{y})",
            head_directionality=h,
            synthesis_index=s_idx,
            syllable_complexity=s_comp,
            max_transition_cost=1.80 + 3.60 * s_comp,
        )
        dialect_lang = Language(name=profile.name, profile=profile, population=pop)
        dialect_lang.cultural_attention = list(np.asarray(cell["attention"], dtype=np.float32))
        dialect_lang.infrastructure = float(cell["infrastructure"])

        rough_field = getattr(grid, "roughness", None)
        env = EnvRecord(
            altitude=alt, humidity=cell["humidity"], temperature=cell["temperature"],
            vegetation=cell["vegetation"], fauna=cell["fauna"],
            roughness=float(rough_field[cell["y"], cell["x"]]) if rough_field is not None else 0.10,
            population=pop,
        )
        has_tone = ((1.0 / (1.0 + math.exp(6.0 * (s_comp - 0.40)))) * 0.6 + (1.0 / (1.0 + math.exp(6.0 * (s_idx - 0.40)))) * 0.4) >= 0.55
        dialect_lang.is_tonal = has_tone

        n_manifold = len(grid.lexical_manifold)
        pres = grid.get_presence_column(x, y) if hasattr(grid, "get_presence_column") else grid.word_presence[:n_manifold, y, x]
        order = np.argsort(-pres)
        active_indices = order[pres[order] > 0.003]
        if len(active_indices) < 20:
            active_indices = order[:min(n_manifold, max(24, int(n_manifold * 0.4)))]
        active_indices = [int(i) for i in active_indices]

        phi = np.zeros(9, dtype=np.float32)
        if hasattr(grid, "phonetic_bias"):
            phi += grid.phonetic_bias[y, x]
        if hasattr(grid, "phonetic_drift_field"):
            phi += grid.phonetic_drift_field[y, x]
        phi_v = phi.copy()
        phi_v[7], phi_v[8] = 0.0, 0.0
        phi_c = phi.copy()
        phi_c[4] *= 0.25

        dom_polity_id = cell["polity"]["dominant_id"]
        dom_polity = next((p for p in grid.polities if p.id == dom_polity_id), None)

        h_vec = grid.hegemony[y, x]
        h_total = float(np.sum(h_vec))
        pol_weights: list[tuple[object, float]] = []
        if h_total > 0.05:
            for p_obj in grid.polities:
                if p_obj.id < len(h_vec) and h_vec[p_obj.id] > 0.02:
                    pol_weights.append((p_obj, float(h_vec[p_obj.id]) / h_total))
            pol_weights.sort(key=lambda item: item[1], reverse=True)
            pol_weights = pol_weights[:2]

        macro_lingua = max(
            (p_obj for p_obj in grid.polities if getattr(p_obj, "standard_tier", 0) >= 2),
            key=lambda p_obj: p_obj.prestige / (1.0 + float(p_obj.dist_cells[y, x]) * 0.05),
            default=None,
        )

        living_words: list[Word] = []
        divergence_dists: list[float] = []
        is_suzerain_province = dom_polity is not None and not cell["polity"]["is_capital"] and dom_polity.is_standardized

        for idx_m in active_indices:
            w_base = grid.lexical_manifold[idx_m]
            presence_weight = float(pres[idx_m])

            if pol_weights:
                lead_p, lead_w = pol_weights[0]
                base_phonemes = lead_p.lexicon_phonemes.get(w_base.id, w_base.phonemes)
                if len(pol_weights) == 2 and pol_weights[1][1] > 0.10:
                    sec_p, sec_w = pol_weights[1]
                    sec_ph = sec_p.lexicon_phonemes.get(w_base.id, w_base.phonemes)
                    mix = sec_w / (lead_w + sec_w)
                    base_phonemes = [
                        Phoneme((1.0 - mix) * p0.g + mix * sec_ph[i_p].g, duration=(1.0 - mix) * p0.duration + mix * sec_ph[i_p].duration)
                        if i_p < len(sec_ph) else p0
                        for i_p, p0 in enumerate(base_phonemes)
                    ]
            else:
                base_phonemes = w_base.phonemes

            warped_phonemes = [
                Phoneme(np.clip(p.g + (phi_v if p.is_vowel else phi_c), 0.0, 1.0), duration=p.duration)
                for p in base_phonemes
            ]

            if dom_polity and dom_polity.is_standardized and w_base.id in dom_polity.canonical_roots:
                canon_ph = dom_polity.canonical_roots[w_base.id]
                h_weight = cell["polity"]["control_percentage"] / 100.0
                dist_factor = 1.0 / (1.0 + float(dom_polity.dist_cells[y, x]) * 0.08)
                pull = float(np.clip(dom_polity.centralization * h_weight * dist_factor * cell["permeability"], 0.0, 0.85))
                warped_phonemes = [
                    Phoneme((1.0 - pull) * wp.g + pull * canon_ph[i_p].g if i_p < len(canon_ph) else wp.g, duration=wp.duration)
                    for i_p, wp in enumerate(warped_phonemes)
                ]

            if macro_lingua and macro_lingua is not dom_polity and w_base.id in macro_lingua.canonical_roots:
                macro_ph = macro_lingua.canonical_roots[w_base.id]
                m_dist = 1.0 / (1.0 + float(macro_lingua.dist_cells[y, x]) * 0.10)
                m_pull = float(np.clip(0.25 * macro_lingua.centralization * m_dist * cell["permeability"], 0.0, 0.45))
                warped_phonemes = [
                    Phoneme((1.0 - m_pull) * wp.g + m_pull * macro_ph[i_p].g if i_p < len(macro_ph) else wp.g, duration=wp.duration)
                    for i_p, wp in enumerate(warped_phonemes)
                ]

            repaired = repair_phonemes(warped_phonemes, profile=profile, max_passes=2)
            if not repaired:
                continue

            if w_base.phonemes and w_base.generation_born == 0:
                divergence_dists.append(trajectory_distance(np.stack([p.g for p in repaired], axis=0), w_base.tensor[:, :9]))

            w_dial = w_base.copy()
            w_dial.phonemes = repaired
            w_dial.usage_frequency = round(float(presence_weight * w_base.usage_frequency * math.log1p(pop * 1.8) * 1.6 / (1.0 + 0.12 * len(repaired))), 2)
            w_dial.record_evolution(epoch=current_epoch, new_phonemes=repaired, event=f"Dialect Reflex ({cell['settlement_tier']})", min_divergence=0.08)
            living_words.append(w_dial)

        if not living_words:
            res = _uninhabited(cell, current_epoch, nearby_landmarks, "Silent Settlement", "No active lexicon at this cell.")
            grid._probe_cache[cache_key] = res
            return res

        grid._probe_words_cache[cache_key] = {w.id: w for w in living_words}

        for w_loc in living_words:
            form_w, plain_w, _ = render_and_extract_tones(w_loc.tensor, directional_bias=h, is_tonal=has_tone)
            w_loc._form, w_loc._plain_form = form_w, plain_w
            dialect_lang.add_word(w_loc)

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
                    grouped.setdefault(w.plain_form, []).append({"form": w.form, "gloss": SemanticTranslator.translate(w, dialect_lang), "tone": tone_str})
            tones_inv = [{"symbol": sym, "name": tone_names.get(sym, f"Tone ({sym})"), "frequency": round((cnt / max(1, len(living_words))) * 100, 1)} for sym, cnt in counts.most_common()]
            tone_pairs = [{"base": base, "variants": vars[:3]} for base, vars in grouped.items() if len(vars) >= 2][:6]
        else:
            tones_inv, tone_pairs = [], []

        pol = cell["polity"]
        if dom_polity and not dom_polity.name:
            if (x, y) != (dom_polity.x, dom_polity.y):
                cls.evoke_pixel_dialect(grid, dom_polity.x, dom_polity.y, epoch=current_epoch)
            if not dom_polity.name:
                capital_anchor = min(living_words, key=lambda w: w.vector.distance_to(SemanticSpace.PERSON))
                dom_polity.name = f"{dom_polity.government_type} of {capital_anchor.form.capitalize()}"
                dom_polity.literal_meaning = f"Authority of {SemanticTranslator.translate(capital_anchor, dialect_lang)}"
            pol["name"] = dom_polity.name
            pol["literal_meaning"] = dom_polity.literal_meaning

        true_nouns = [w for w in living_words if w.category == LexicalCategory.CONTENT_OPEN and SemanticTranslator.get_pos(w.vector) == "Noun"]
        true_verbs = [w for w in living_words if w.category == LexicalCategory.CONTENT_OPEN and SemanticTranslator.get_pos(w.vector) == "Verb"]
        sample_noun = true_nouns[0] if true_nouns else living_words[0]
        sample_verb = true_verbs[0] if true_verbs else living_words[0]

        lexicon_payload = []
        word_energies = []
        seen_pairs: set[tuple[str, str]] = set()
        gloss_counts: Counter = Counter()
        ancestor_root_map = {w.id: w for w in grid.lexical_manifold}

        for w in living_words:
            is_superstrate = bool(dom_polity and dom_polity.is_standardized and w.id in dom_polity.canonical_roots)
            gloss = SemanticTranslator.translate(w, dialect_lang)
            if (w.form, gloss) in seen_pairs:
                continue
            if gloss_counts[gloss] >= (2 if is_superstrate else 1):
                continue
            seen_pairs.add((w.form, gloss))
            gloss_counts[gloss] += 1

            pos_label = GrammarEngine.get_part_of_speech(w)
            e_w = float(word_energy(w, profile=profile, env=env))
            word_energies.append(e_w)

            lexicon_payload.append({
                "id": w.id,
                "form": w.form,
                "root_form": ancestor_root_map.get(w.id, w).form,
                "plain": w.plain_form,
                "gloss": gloss,
                "pos": pos_label,
                "grammar_info": {"pos": pos_label},
                "stratification": "Superstrate (Imperial)" if is_superstrate else "Vernacular Substrate",
                "energy_cost": round(e_w, 2),
                "usage_frequency": w.usage_frequency,
                "history": w.get_history_log(directional_bias=h),
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

        mean_div = float(np.mean(divergence_dists)) if divergence_dists else 0.0
        cradle_intell = round(float(np.clip(math.exp(-1.6 * mean_div), 0.05, 1.0)), 3)

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
                "prestige": cell["prestige"],
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
                "has_ejectives": any("ʼ" in c for c in consonants),
                "has_implosives": any(c in {"ɓ", "ɗ", "ɠ", "ʄ"} for c in consonants),
                "has_pharyngealized": any("ˁ" in c for c in consonants),
                "has_tones": has_tone,
                "tone_inventory": tones_inv,
                "minimal_tone_pairs": tone_pairs,
                "nonpulmonic_segments": inventory_size(living_words)["nonpulmonic"],
                "mean_word_energy": round(float(np.mean(word_energies)), 2),
                "has_evolved_phonology": any(bool(p_obj.lexicon_phonemes) for p_obj, _ in pol_weights),
            },
            "noun_paradigm": {
                "noun": sample_noun.form,
                "noun_gloss": SemanticTranslator.translate(sample_noun, dialect_lang),
                "title_display": f"«{SemanticTranslator.translate(sample_noun, dialect_lang)}» /{sample_noun.form}/",
                "alignment": GrammarEngine.get_alignment_system(dialect_lang),
                "table": GrammarEngine.get_noun_declension_table(sample_noun, dialect_lang, paradigm),
            },
            "verb_paradigm": {
                "verb": sample_verb.form,
                "verb_gloss": SemanticTranslator.translate(sample_verb, dialect_lang),
                "title_display": f"«{SemanticTranslator.translate(sample_verb, dialect_lang)}» {GrammarEngine.get_infinitive_form(sample_verb, dialect_lang, paradigm)}",
                "table": GrammarEngine.get_verb_conjugation_table(sample_verb, dialect_lang, paradigm),
            },
            "pronouns": {k: f"/{GrammarEngine.render_surface(v, dialect_lang)}/" for k, v in paradigm.pronouns.items()},
            "numerals": {val: GrammarEngine.format_numeral(val, dialect_lang, paradigm) for val in range(1, paradigm.numeral_chain.primary_base + 3)},
            "vocabulary_stats": {"total_lexicon_size": len(living_words), "displayed_sample": len(lexicon_payload)},
            "lexicon": lexicon_payload,
            "cradle_intelligibility": cradle_intell,
        }
        grid._probe_cache[cache_key] = res
        return res