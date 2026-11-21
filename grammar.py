"""
grammar.py — Morphosyntax, Sociolinguistic Population Law,
Alignment Rotation, and Numeral Radices (5, 10, 12, 20, 60).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from language import Language
from phonetics import render_and_extract_tones, transition_cost
from semantics import ClineStage, DerivationRecord, DerivationType, SemanticSpace, SemanticVector
from word import LexicalCategory, Morpheme, Phoneme, Word, repair_phonemes


@dataclass
class EmergentClass:
    class_id: int
    label: str
    centroid: SemanticVector
    marker: Word | None = None


@dataclass
class EmergentVerbClass:
    class_id: int
    label: str
    centroid: SemanticVector
    thematic_marker: Word | None = None
    aspect_label: str = "Aspectual"
    morph_strategy: str = "analytic"


@dataclass
class NumeralChain:
    primary_base: int = 10
    name: str = "Decimal (Base-10)"
    higher_units: dict[str, int] = field(default_factory=dict)


@dataclass
class EmergentParadigm:
    temporal_operators: list[Word] = field(default_factory=list)
    spatial_operators: list[Word] = field(default_factory=list)
    determiners: list[Word] = field(default_factory=list)
    modal_operators: list[Word] = field(default_factory=list)
    noun_classes: list[EmergentClass] = field(default_factory=list)
    verb_classes: list[EmergentVerbClass] = field(default_factory=list)
    relativizers: list[Word] = field(default_factory=list)
    complementizers: list[Word] = field(default_factory=list)
    pronouns: dict[str, Word] = field(default_factory=dict)
    number_system: str = "Singular-Plural"
    number_operators: dict[str, Word] = field(default_factory=dict)
    article_system: str = "Transdefinite (No Articles)"
    articles: dict[str, Word] = field(default_factory=dict)
    numeral_chain: NumeralChain = field(default_factory=NumeralChain)
    digits: dict[int, Word] = field(default_factory=dict)
    radix_words: dict[str, Word] = field(default_factory=dict)
    formatted_cache: dict[int, str] = field(default_factory=dict)
    alignment_angle: float = 0.0


def _render_surface(word: Word, lang: Language) -> str:
    if not word.phonemes:
        return ""
    is_tonal = getattr(lang, "is_tonal", False)
    bias = getattr(lang.profile, "metrical_bias", 0.50)
    form, _, _ = render_and_extract_tones(word.tensor, directional_bias=bias, is_tonal=is_tonal)
    return form


class ContinuousNumeralSystem:
    @classmethod
    def determine_base(cls, attn: Sequence[float], pop: float, infra: float = 0.0) -> tuple[int, str, dict[str, int]]:
        scores = [
            (attn[0] * 0.70 + max(0.0, 0.8 - pop * 0.6), 5, "Quinary (Base-5)", {"hundred": 25, "thousand": 125}),
            (attn[1] * 0.50 + attn[3] * 0.50 + max(0.0, 0.5 - infra * 0.2), 20, "Vigesimal (Base-20)", {"hundred": 400, "thousand": 8000}),
            (attn[5] * 0.50 + attn[6] * 0.40 + math.tanh(pop * 0.25), 10, "Decimal (Base-10)", {"hundred": 100, "thousand": 1000}),
            (attn[6] * 0.60 + math.tanh(infra * 0.20) * 0.50, 12, "Duodecimal (Base-12)", {"hundred": 144, "thousand": 1728}),
            (math.tanh(infra * 0.35) * 0.80 + math.tanh(pop * 0.20) * 0.40, 60, "Sexagesimal (Base-60)", {"hundred": 3600, "thousand": 216000}),
        ]
        best = max(scores, key=lambda x: x[0])
        return best[1], best[2], best[3]

    @classmethod
    def resolve_numeral_word(
        cls,
        val: int,
        lang: Language,
        semantic_anchor: SemanticVector,
        usage_boost: float = 12.0,
        exclude_ids: set[int] | None = None,
    ) -> Word:
        all_words = list(lang.words.values())
        if exclude_ids:
            filtered = [w for w in all_words if w.id not in exclude_ids]
            if filtered:
                all_words = filtered

        if all_words:
            best_lex = min(all_words, key=lambda w: w.vector.distance_to(semantic_anchor))
            if exclude_ids is not None:
                exclude_ids.add(best_lex.id)
            w = best_lex.copy()
            w.usage_frequency = max(w.usage_frequency, usage_boost)
            return w

        num_vec = SemanticVector(0.10, 0.0, 0.50, 0.20 + (val * 0.04) % 0.60, 0.0, 0.10, 0.20 + (val * 0.05) % 0.70)
        return Word(
            word_id=90000 + val,
            vector=num_vec,
            phonemes=[Phoneme([1.0, 0.0, 0.0, 0.25, 0.8, 0.5, 0.0, 0.6, 0.5], duration=1.0)],
            derivation=DerivationRecord(root_family_id=90000 + val),
            category=LexicalCategory.FUNCTIONAL_CLOSED,
            usage_frequency=usage_boost,
        )

    @classmethod
    def _decompose_subbase(cls, val: int, digits: dict[int, Word], sub_base: int, h: float, lang: Language) -> list[str]:
        if val <= 0:
            return []
        if val in digits:
            return [_render_surface(digits[val], lang)]
        if sub_base > 1 and sub_base in digits:
            q, r = divmod(val, sub_base)
            toks: list[str] = []
            if q > 1 and q in digits:
                sb_str, q_str = _render_surface(digits[sub_base], lang), _render_surface(digits[q], lang)
                toks.extend([sb_str, q_str] if h < 0.50 else [q_str, sb_str])
            else:
                toks.append(_render_surface(digits[sub_base], lang))
            if r > 0:
                toks.extend(cls._decompose_subbase(r, digits, sub_base, h, lang))
            return toks
        return [str(val)]

    @classmethod
    def format_combinatorial_numeral(cls, val: int, lang: Language, paradigm: EmergentParadigm) -> str:
        if val in paradigm.formatted_cache:
            return paradigm.formatted_cache[val]

        base = paradigm.numeral_chain.primary_base
        digits, radix_words = paradigm.digits, paradigm.radix_words
        h = lang.profile.head_directionality
        sub_base = 10 if 10 in digits else (5 if 5 in digits else max(digits.keys(), default=1))

        if val in digits:
            res = f"/{_render_surface(digits[val], lang)}/"
        elif val == base and "base" in radix_words:
            res = f"/{_render_surface(radix_words['base'], lang)}/"
        elif val == paradigm.numeral_chain.higher_units.get("hundred", base * base) and "hundred" in radix_words:
            res = f"/{_render_surface(radix_words['hundred'], lang)}/"
        else:
            q, r = divmod(val, base)
            tokens: list[str] = []
            if q > 0 and "base" in radix_words:
                b_str = _render_surface(radix_words["base"], lang)
                if q > 1:
                    q_toks = cls._decompose_subbase(q, digits, sub_base, h, lang)
                    tokens.extend(([b_str] + q_toks) if h < 0.50 else (q_toks + [b_str]))
                else:
                    tokens.append(b_str)
            if r > 0:
                tokens.extend(cls._decompose_subbase(r, digits, sub_base, h, lang))
            res = f"/{'-'.join(tokens)}/" if tokens else f"/{val}/"

        paradigm.formatted_cache[val] = res
        return res


@dataclass
class NounPhrase:
    head_noun: Word
    modifier: Word | None = None
    determiner: Word | None = None
    possessor: Word | None = None
    case_operator: Word | None = None
    number_operator: Word | None = None
    numeral_value: int | None = None
    is_definite: bool | None = None
    case_label: str = "Nom"
    number_label: str = "Sg"
    assigned_class: EmergentClass | None = None
    is_subject: bool = False

    def render(self, lang: Language) -> tuple[str, str]:
        h = lang.profile.head_directionality
        pop = getattr(lang, "population", 0.50)
        paradigm = GrammarEngine.discover_grammar_system(lang)

        agr_op = self.assigned_class.marker if self.assigned_class else None
        cls_tag = f"Cls{self.assigned_class.class_id}" if self.assigned_class else ""

        base_noun = self.head_noun
        is_class_bound = False
        if agr_op and len(paradigm.noun_classes) >= 2 and pop < 1.20:
            base_noun, is_class_bound = GrammarEngine.apply_spatial_relator(self.head_noun, agr_op, lang, category_type="class")

        num_op = self.number_operator
        num_lbl = self.number_label
        if self.numeral_value and self.numeral_value > 1 and num_lbl == "Sg" and "pl" in paradigm.number_operators:
            num_op = paradigm.number_operators["pl"]
            num_lbl = "Pl"

        is_num_bound = False
        if num_op and num_lbl != "Sg":
            base_noun, is_num_bound = GrammarEngine.apply_spatial_relator(base_noun, num_op, lang, category_type="number")

        num_tokens, num_gloss = [], []
        if self.numeral_value is not None:
            num_tokens.append(GrammarEngine.format_numeral(self.numeral_value, lang, paradigm).strip("/"))
            num_gloss.append(f"[Num:{self.numeral_value}]")

        art_op = paradigm.articles["def"] if (self.is_definite is True and "def" in paradigm.articles) else None
        art_label = "Def" if art_op else ""
        art_tokens, art_gloss = [], []
        is_art_bound = False
        if art_op:
            base_noun, is_art_bound = GrammarEngine.apply_spatial_relator(base_noun, art_op, lang, category_type="case")
            if not is_art_bound:
                art_tokens = [_render_surface(art_op, lang)]
                art_gloss = [f"[{art_label}:{art_op.id}]"]

        adp_tokens, adp_gloss = [], []
        if self.case_operator:
            inflected_noun, is_bound = GrammarEngine.apply_spatial_relator(base_noun, self.case_operator, lang, category_type="case")
            if is_bound:
                n_form = _render_surface(inflected_noun, lang)
                tags = [self.case_label]
                if is_art_bound: tags.append(art_label)
                if is_num_bound: tags.append(num_lbl)
                if is_class_bound and cls_tag: tags.append(cls_tag)
                n_gloss = f"{self.head_noun.id}[{'+'.join(tags)}]"
            else:
                n_form = _render_surface(base_noun, lang)
                tags = []
                if is_art_bound: tags.append(art_label)
                if is_num_bound: tags.append(num_lbl)
                if is_class_bound and cls_tag: tags.append(cls_tag)
                tag_str = f"[{'+'.join(tags)}]" if tags else ""
                n_gloss = f"{self.head_noun.id}{tag_str}"
                adp_tokens = [_render_surface(self.case_operator, lang)]
                adp_gloss = [f"[{self.case_label}:{self.case_operator.id}]"]
        else:
            n_form = _render_surface(base_noun, lang)
            tags = [self.case_label] if self.case_label not in ("Nom", "Abs") else []
            if is_art_bound: tags.append(art_label)
            if is_num_bound: tags.append(num_lbl)
            if is_class_bound and cls_tag: tags.append(cls_tag)
            tag_str = f"[{'+'.join(tags)}]" if tags else ""
            n_gloss = f"{self.head_noun.id}{tag_str}"

        mod_tokens = [_render_surface(self.modifier, lang)] if self.modifier else []
        mod_gloss = [f"{self.modifier.id}[Adj]"] if self.modifier else []
        det_tokens = [_render_surface(self.determiner, lang)] if self.determiner else []
        det_gloss = [f"[Det:{self.determiner.id}]"] if self.determiner else []
        poss_tokens = [_render_surface(self.possessor, lang)] if self.possessor else []
        poss_gloss = [f"[Poss:{self.possessor.id}]"] if self.possessor else []

        components = [
            (adp_tokens, adp_gloss, 0.5 * (1.0 - h) + 7.5 * h),
            (art_tokens, art_gloss, 1.0 * (1.0 - h) + 6.0 * h),
            (poss_tokens, poss_gloss, 2.0 * (1.0 - h) + 5.0 * h),
            (num_tokens, num_gloss, 3.0 * (1.0 - h) + 4.0 * h),
            (det_tokens, det_gloss, 3.5 * (1.0 - h) + 2.0 * h),
            ([n_form], [n_gloss], 5.0 * (1.0 - h) + 2.5 * h),
            (mod_tokens, mod_gloss, 6.0 * (1.0 - h) + 3.0 * h),
        ]
        components.sort(key=lambda item: item[2])

        tokens, glosses = [], []
        for t_list, g_list, _ in components:
            tokens.extend(t_list)
            glosses.extend(g_list)
        return " ".join(t for t in tokens if t), " ".join(g for g in glosses if g)


class GrammarEngine:
    @classmethod
    def render_surface(cls, word: Word, lang: Language) -> str:
        return _render_surface(word, lang)

    @classmethod
    def get_part_of_speech(cls, word: Word) -> str:
        if word.category == LexicalCategory.FUNCTIONAL_CLOSED:
            return "Particle"
        from translator import SemanticTranslator
        return SemanticTranslator.get_pos(word.vector)

    @classmethod
    def compress_bound_affix(cls, operator: Word) -> list[Phoneme]:
        p_list = list(operator.phonemes)
        if len(p_list) <= 2:
            return p_list
        vowels = [p for p in p_list if p.is_vowel]
        if vowels:
            v_idx = p_list.index(vowels[0])
            if v_idx > 0:
                return [p_list[v_idx - 1], p_list[v_idx]]
            if v_idx + 1 < len(p_list):
                return [p_list[v_idx], p_list[v_idx + 1]]
            return [vowels[0]]
        return [max(p_list, key=lambda p: p.energy.acoustic_salience)]

    @classmethod
    def should_fuse_morpheme(cls, host_word: Word, operator: Word, lang: Language, is_prefix: bool = False) -> bool:
        s_idx = getattr(lang.profile, "synthesis_index", 0.50)
        pop = getattr(lang, "population", 0.50)
        eff_s = s_idx * max(0.20, 1.0 - pop * 0.35)

        if host_word.phonemes and operator.phonemes:
            p_host = host_word.phonemes[0] if is_prefix else host_word.phonemes[-1]
            p_op = operator.phonemes[-1] if is_prefix else operator.phonemes[0]
            boundary_friction = transition_cost(p_host, p_op)
        else:
            boundary_friction = 0.0

        fusion_potential = (eff_s * 4.0) - (0.95 + boundary_friction * 0.25)
        return (1.0 / (1.0 + math.exp(-5.0 * fusion_potential))) >= 0.50

    @classmethod
    def apply_spatial_relator(
        cls,
        noun: Word,
        operator: Word,
        lang: Language,
        category_type: str = "case",
    ) -> tuple[Word, bool]:
        h = lang.profile.head_directionality
        is_prefix = (category_type in ("class", "infinitive")) or (h > 0.50)

        if not cls.should_fuse_morpheme(noun, operator, lang, is_prefix=is_prefix):
            return noun.copy(), False

        affix_phonemes = cls.compress_bound_affix(operator)
        m_stem = [m.copy() for m in noun.morphemes] if noun.morphemes else [Morpheme(noun.phonemes, root_family_id=noun.derivation.root_family_id)]
        m_case = Morpheme(affix_phonemes, root_family_id=operator.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)

        inflected_word = Word(
            vector=noun.vector,
            phonemes=repair_phonemes(affix_phonemes + noun.phonemes if is_prefix else noun.phonemes + affix_phonemes, profile=lang.profile),
            morphemes=[m_case] + m_stem if is_prefix else m_stem + [m_case],
            derivation=DerivationRecord(root_family_id=noun.derivation.root_family_id, derivation_type=DerivationType.AFFIXATION, local_parent_ids=(noun.id,)),
            category=LexicalCategory.CONTENT_OPEN,
            is_ephemeral=True,
        )
        return inflected_word, True

    @classmethod
    def assign_noun_class(cls, noun: Word, paradigm: EmergentParadigm) -> EmergentClass | None:
        if not paradigm.noun_classes:
            return None
        return min(paradigm.noun_classes, key=lambda c: noun.vector.distance_to(c.centroid))

    @classmethod
    def get_alignment_system(cls, lang: Language) -> str:
        para = getattr(lang, "_cached_grammar_paradigm", None)
        if para and hasattr(para, "alignment_angle"):
            return "ergative_absolutive" if para.alignment_angle >= (math.pi / 4.0) else "nominative_accusative"
        return getattr(lang.profile, "alignment", "nominative_accusative")

    @classmethod
    def get_infinitive_form(cls, verb: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> str:
        para = paradigm or cls.discover_grammar_system(lang)
        h = lang.profile.head_directionality
        theme_op = para.verb_classes[0].thematic_marker if para.verb_classes else None
        if not theme_op:
            return f"/{cls.render_surface(verb, lang)}/"

        inf_verb, is_b = cls.apply_spatial_relator(verb, theme_op, lang, category_type="infinitive")
        if is_b:
            return f"/{cls.render_surface(inf_verb, lang)}/"

        t_str, v_str = cls.render_surface(theme_op, lang), cls.render_surface(verb, lang)
        return f"/{t_str} {v_str}/" if h >= 0.5 else f"/{v_str} {t_str}/"

    @classmethod
    def inflect_predicate(
        cls,
        verb: Word,
        temporal_op: Word | None,
        modal_op: Word | None,
        agreement_op: Word | None,
        person_op: Word | None = None,
        person_label: str = "3sg",
        lang: Language | None = None,
        is_past_or_perfective: bool = False,
    ) -> tuple[Word, list[Word], bool, list[str]]:
        bound_labels, free_particles = [], []
        stem_word = verb
        s_idx = getattr(lang.profile, "synthesis_index", 0.50) if lang else 0.50
        h = getattr(lang.profile, "head_directionality", 0.50) if lang else 0.50

        active_person_op = person_op if s_idx >= 0.35 else None
        for op, label in ((temporal_op, "TAM"), (modal_op, "Mod"), (active_person_op, person_label)):
            if op is None:
                continue
            if lang and cls.should_fuse_morpheme(stem_word, op, lang, is_prefix=(h > 0.50)):
                affix_p = cls.compress_bound_affix(op)
                m_op = Morpheme(affix_p, root_family_id=op.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)
                m_stem = Morpheme(stem_word.phonemes, root_family_id=stem_word.derivation.root_family_id)
                stem_word = Word(
                    vector=stem_word.vector,
                    phonemes=affix_p + stem_word.phonemes if h > 0.50 else stem_word.phonemes + affix_p,
                    morphemes=[m_op, m_stem] if h > 0.50 else [m_stem, m_op],
                    derivation=stem_word.derivation,
                    category=LexicalCategory.CONTENT_OPEN,
                    is_ephemeral=True,
                )
                bound_labels.append(label)
            else:
                free_particles.append(op)

        if bound_labels and lang:
            stem_word.phonemes = repair_phonemes(stem_word.phonemes, profile=lang.profile)

        return stem_word, free_particles, False, bound_labels

    @classmethod
    def discover_grammar_system(cls, lang: Language) -> EmergentParadigm:
        cached = getattr(lang, "_cached_grammar_paradigm", None)
        if cached is not None:
            return cached

        attn = lang.cultural_attention
        s_idx = getattr(lang.profile, "synthesis_index", 0.50)
        pop = getattr(lang, "population", 0.50)
        infra = getattr(lang, "infrastructure", 0.0)

        radix_base, radix_name, higher_powers = ContinuousNumeralSystem.determine_base(attn, pop, infra)
        num_chain = NumeralChain(primary_base=radix_base, name=radix_name, higher_units=higher_powers)

        digit_anchors = [
            SemanticSpace.ONE, SemanticSpace.OTHER, SemanticSpace.EYE,
            SemanticSpace.STONE, SemanticSpace.HAND, SemanticSpace.HEAD,
            SemanticSpace.MOON, SemanticSpace.TREE, SemanticSpace.HEART,
            SemanticSpace.SUN, SemanticSpace.EARTH,
        ]
        used_digit_ids: set[int] = set()
        digits = {
            d: ContinuousNumeralSystem.resolve_numeral_word(d, lang, digit_anchors[d - 1], max(10.0, 20.0 - d), exclude_ids=used_digit_ids)
            for d in range(1, min(len(digit_anchors), radix_base - 1) + 1)
        }

        radix_words = {
            "base": ContinuousNumeralSystem.resolve_numeral_word(radix_base, lang, SemanticSpace.BODY, 20.0),
            "hundred": ContinuousNumeralSystem.resolve_numeral_word(higher_powers["hundred"], lang, SemanticSpace.TREE, 15.0),
            "thousand": ContinuousNumeralSystem.resolve_numeral_word(higher_powers["thousand"], lang, SemanticSpace.EARTH, 14.0),
        }

        spatials = [
            ContinuousNumeralSystem.resolve_numeral_word(101, lang, SemanticVector(0.1, 0.0, 0.5, 0.2, 0.0, 0.0, 0.9), 16.0),
            ContinuousNumeralSystem.resolve_numeral_word(102, lang, SemanticVector(0.2, 0.4, 0.6, 0.4, 0.5, 0.8, 0.3), 16.0),
            ContinuousNumeralSystem.resolve_numeral_word(103, lang, SemanticVector(0.2, 0.3, 0.5, 0.9, 0.8, 0.3, 0.4), 16.0),
            ContinuousNumeralSystem.resolve_numeral_word(104, lang, SemanticVector(0.3, 0.0, 0.4, 0.2, 0.2, 0.2, 0.5), 16.0),
        ]
        v_past = ContinuousNumeralSystem.resolve_numeral_word(105, lang, SemanticVector(0.1, 0.1, 0.5, 0.3, 0.9, 0.2, 0.7), 16.0)
        v_mod = ContinuousNumeralSystem.resolve_numeral_word(106, lang, SemanticVector(0.2, 0.4, 0.6, 0.8, 0.5, 0.4, 0.2), 14.0)

        pronouns = {
            "1sg": ContinuousNumeralSystem.resolve_numeral_word(201, lang, SemanticSpace.SELF, 22.0),
            "2sg": ContinuousNumeralSystem.resolve_numeral_word(202, lang, SemanticSpace.OTHER, 20.0),
            "3sg": ContinuousNumeralSystem.resolve_numeral_word(203, lang, SemanticSpace.PERSON, 22.0),
            "1pl": ContinuousNumeralSystem.resolve_numeral_word(204, lang, SemanticSpace.MANY, 18.0),
        }

        align_angle = float(min(math.pi / 2.0, max(0.0, (attn[3] / (attn[5] + 0.1)) * 0.45 * (math.pi / 2.0))))

        noun_classes: list[EmergentClass] = []
        if s_idx > 0.35 and pop < 1.20:
            noun_classes = [
                EmergentClass(1, "Animate", SemanticSpace.PERSON, ContinuousNumeralSystem.resolve_numeral_word(301, lang, SemanticSpace.PERSON, 18.0)),
                EmergentClass(2, "Inanimate", SemanticSpace.STONE, ContinuousNumeralSystem.resolve_numeral_word(302, lang, SemanticSpace.STONE, 18.0)),
            ]

        articles = {}
        if (1.0 - s_idx) * attn[6] > 0.35 or pop > 1.50:
            articles["def"] = ContinuousNumeralSystem.resolve_numeral_word(401, lang, SemanticSpace.THIS, 24.0)
            art_system = "Definite Article"
        else:
            art_system = "Transdefinite (No Articles)"

        number_operators = {"pl": ContinuousNumeralSystem.resolve_numeral_word(501, lang, SemanticSpace.MANY, 20.0)}
        strat_mode = "fusional" if (s_idx > 0.65 and pop < 1.0) else ("agglutinative" if s_idx > 0.35 else "analytic")
        theme_cand = ContinuousNumeralSystem.resolve_numeral_word(601, lang, SemanticVector(0.4, 0.6, 0.5, 0.6, 0.7, 0.4, 0.3), 16.0)

        paradigm = EmergentParadigm(
            temporal_operators=[v_past],
            spatial_operators=spatials,
            determiners=[ContinuousNumeralSystem.resolve_numeral_word(701, lang, SemanticSpace.THIS)],
            modal_operators=[v_mod],
            noun_classes=noun_classes,
            verb_classes=[EmergentVerbClass(1, f"Regular ({strat_mode.capitalize()})", theme_cand.vector, theme_cand, strat_mode.capitalize(), strat_mode)],
            relativizers=[ContinuousNumeralSystem.resolve_numeral_word(702, lang, SemanticSpace.THIS)],
            complementizers=[ContinuousNumeralSystem.resolve_numeral_word(703, lang, SemanticSpace.THAT)],
            pronouns=pronouns,
            number_system="Singular / Plural",
            number_operators=number_operators,
            article_system=art_system,
            articles=articles,
            numeral_chain=num_chain,
            digits=digits,
            radix_words=radix_words,
            formatted_cache={},
            alignment_angle=align_angle,
        )
        lang._cached_grammar_paradigm = paradigm
        return paradigm

    @classmethod
    def format_numeral(cls, value: int, lang: Language, paradigm: EmergentParadigm | None = None) -> str:
        return ContinuousNumeralSystem.format_combinatorial_numeral(value, lang, paradigm or cls.discover_grammar_system(lang))

    @classmethod
    def get_citation_presentation(
        cls, word: Word, lang: Language, paradigm: EmergentParadigm | None = None
    ) -> tuple[str, str, dict]:
        para = paradigm or cls.discover_grammar_system(lang)
        pos = cls.get_part_of_speech(word)
        alignment = cls.get_alignment_system(lang)

        if pos == "Noun":
            assigned_cls = cls.assign_noun_class(word, para)
            cls_label = assigned_cls.label if assigned_cls else None
            surf_word = cls.render_surface(word, lang)
            if assigned_cls and assigned_cls.marker and len(para.noun_classes) >= 2 and getattr(lang, "population", 0.5) < 1.20:
                aff_str = "".join(p.ipa for p in cls.compress_bound_affix(assigned_cls.marker))
                clean_display = f"{aff_str}-{surf_word}"
            else:
                clean_display = surf_word
            return clean_display, word.plain_form, {"pos": "Noun", "class": cls_label, "case": "Nom" if alignment == "nominative_accusative" else "Abs"}
        if pos == "Verb":
            inf_str = cls.get_infinitive_form(word, lang, para)
            return inf_str.strip("/"), word.plain_form, {"pos": "Verb", "infinitive": inf_str, "strategy": para.verb_classes[0].morph_strategy if para.verb_classes else "analytic"}
        return cls.render_surface(word, lang), word.plain_form, {"pos": pos}

    @classmethod
    def get_noun_declension_table(cls, noun: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> dict[str, str]:
        para = paradigm or cls.discover_grammar_system(lang)
        alignment = cls.get_alignment_system(lang)
        spatials = para.spatial_operators
        if not spatials:
            return {}

        loc_word = spatials[0]
        dat_word = spatials[1] if len(spatials) > 1 else spatials[0]
        erg_word = spatials[2] if len(spatials) > 2 else spatials[0]
        acc_word = spatials[3] if len(spatials) > 3 else spatials[0]
        assigned_cls = cls.assign_noun_class(noun, para)

        cases = [
            ("Abs" if alignment == "ergative_absolutive" else "Nom", None),
            ("Erg" if alignment == "ergative_absolutive" else "Acc", erg_word if alignment == "ergative_absolutive" else acc_word),
            ("Dat", dat_word),
            ("Loc", loc_word),
        ]

        grid: dict[str, str] = {}
        pl_op = para.number_operators.get("pl")
        for num_label in ("Sg", "Pl"):
            num_op = pl_op if num_label == "Pl" else None
            for case_label, case_op in cases:
                form_bare, _ = NounPhrase(
                    head_noun=noun, assigned_class=assigned_cls, case_operator=case_op,
                    case_label=case_label, number_operator=num_op, number_label=num_label,
                ).render(lang)
                grid[f"{case_label}.{num_label}"] = f"/{form_bare}/"
        return grid

    @classmethod
    def get_verb_conjugation_table(cls, verb: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> dict[str, str]:
        para = paradigm or cls.discover_grammar_system(lang)
        conj: dict[str, str] = {"Infinitive": cls.get_infinitive_form(verb, lang, para)}
        h = lang.profile.head_directionality
        t_op = para.temporal_operators[0] if para.temporal_operators else None

        for p_key in ("1sg", "2sg", "3sg", "1pl"):
            v_form, free_p, _, _ = cls.inflect_predicate(verb, None, None, None, para.pronouns.get(p_key), p_key, lang)
            v_str, p_strs = cls.render_surface(v_form, lang), [cls.render_surface(p, lang) for p in free_p]
            conj[f"Pres.{p_key}"] = f"/{' '.join(p_strs + [v_str] if h >= 0.5 else [v_str] + p_strs)}/"

        for p_key in ("1sg", "3sg"):
            v_form, free_p, _, _ = cls.inflect_predicate(verb, t_op, None, None, para.pronouns.get(p_key), p_key, lang, is_past_or_perfective=True)
            v_str, p_strs = cls.render_surface(v_form, lang), [cls.render_surface(p, lang) for p in free_p]
            conj[f"Past.{p_key}"] = f"/{' '.join(p_strs + [v_str] if h >= 0.5 else [v_str] + p_strs)}/"

        return conj