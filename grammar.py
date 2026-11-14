"""
grammar.py — Autonomous Morphosyntax, Sociolinguistic Population Law,
Continuous Alignment Rotation, and Diverse Numeral Radices (5, 10, 12, 20, 60).
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence
import numpy as np

from phonetics import (
    Phoneme,
    transition_cost,
    repair_phonemes,
    apply_prosody,
    sensory_distance,
)
from word import Word, Morpheme, LexicalCategory
from language import Language
from semantics import SemanticVector, SemanticSpace, DerivationRecord, DerivationType, ClineStage


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


# -----------------------------------------------------------------------------
# 1. NATURAL NUMERAL BASES & COMPOSITIONAL COUNTING
# -----------------------------------------------------------------------------

class ContinuousNumeralSystem:
    @classmethod
    def determine_base(cls, attn: Sequence[float], pop: float, infra: float = 0.0) -> tuple[int, str, dict[str, int]]:
        """
        Derives cultural radix continuously from ecological attention, population, and infrastructure:
        - Quinary (5): Foraging, isolated bands (1 hand)
        - Decimal (10): Moderate settlements (2 hands)
        - Duodecimal (12): Trade confluences counting 12 finger phalanxes with thumb
        - Vigesimal (20): Full body counting (hands + feet, "one person")
        - Sexagesimal (60): Highly developed institutional/astronomical empires (5 x 12)
        """
        p_quinary = attn[0] * 0.70 + max(0.0, 0.8 - pop * 0.6)
        p_vigesimal = attn[1] * 0.50 + attn[3] * 0.50 + max(0.0, 0.5 - infra * 0.2)
        p_decimal = attn[5] * 0.50 + attn[6] * 0.40 + math.tanh(pop * 0.25)
        p_duodecimal = attn[6] * 0.60 + math.tanh(infra * 0.20) * 0.50
        p_sexagesimal = math.tanh(infra * 0.35) * 0.80 + math.tanh(pop * 0.20) * 0.40

        scores = [
            (p_quinary, 5, "Quinary (Base-5)", {"hundred": 25, "thousand": 125}),
            (p_vigesimal, 20, "Vigesimal (Base-20)", {"hundred": 400, "thousand": 8000}),
            (p_decimal, 10, "Decimal (Base-10)", {"hundred": 100, "thousand": 1000}),
            (p_duodecimal, 12, "Duodecimal (Base-12)", {"hundred": 144, "thousand": 1728}),
            (p_sexagesimal, 60, "Sexagesimal (Base-60)", {"hundred": 3600, "thousand": 216000}),
        ]
        best = max(scores, key=lambda x: x[0])
        return best[1], best[2], best[3]

    @classmethod
    def resolve_numeral_word(
        cls,
        val: int,
        lang: Language,
        semantic_anchor: SemanticVector,
        usage_boost: float = 12.0
    ) -> Word:
        all_words = list(lang.words.values())
        if all_words:
            best_lex = min(all_words, key=lambda w: w.vector.distance_to(semantic_anchor))
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
    def format_combinatorial_numeral(cls, val: int, lang: Language, paradigm: EmergentParadigm) -> str:
        if val in paradigm.formatted_cache:
            return paradigm.formatted_cache[val]

        base = paradigm.numeral_chain.primary_base
        digits = paradigm.digits
        radix_words = paradigm.radix_words
        h = lang.profile.head_directionality

        if val in digits:
            res = f"/{digits[val].form}/"
            paradigm.formatted_cache[val] = res
            return res

        if val == base and "base" in radix_words:
            res = f"/{radix_words['base'].form}/"
            paradigm.formatted_cache[val] = res
            return res

        hundred_val = paradigm.numeral_chain.higher_units.get("hundred", base * base)
        if val == hundred_val and "hundred" in radix_words:
            res = f"/{radix_words['hundred'].form}/"
            paradigm.formatted_cache[val] = res
            return res

        q, r = divmod(val, base)
        tokens = []

        if q > 1:
            q_word = digits.get(q)
            b_word = radix_words.get("base")
            if q_word and b_word:
                tokens.extend([b_word.form, q_word.form] if h < 0.50 else [q_word.form, b_word.form])
            elif b_word:
                tokens.append(b_word.form)
        elif "base" in radix_words:
            tokens.append(radix_words["base"].form)

        if r > 0 and r in digits:
            tokens.append(digits[r].form)

        res = f"/{'-'.join(tokens)}/" if tokens else f"/{val}/"
        paradigm.formatted_cache[val] = res
        return res


# -----------------------------------------------------------------------------
# 2. NOUN PHRASE & MORPHOSYNTAX
# -----------------------------------------------------------------------------

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
        # Trudgill law: Noun class prefixing erodes and drops in larger populations
        if agr_op and len(paradigm.noun_classes) >= 2 and pop < 1.20:
            base_noun, is_class_bound = GrammarEngine.apply_spatial_relator(self.head_noun, agr_op, lang, category_type="class")

        if self.numeral_value and self.numeral_value > 1 and self.number_label == "Sg":
            if "pl" in paradigm.number_operators:
                self.number_operator = paradigm.number_operators["pl"]
                self.number_label = "Pl"

        is_num_bound = False
        if self.number_operator and self.number_label != "Sg":
            base_noun, is_num_bound = GrammarEngine.apply_spatial_relator(base_noun, self.number_operator, lang, category_type="number")

        num_tokens, num_gloss = [], []
        if self.numeral_value is not None:
            raw_num = GrammarEngine.format_numeral(self.numeral_value, lang, paradigm).strip("/")
            num_tokens.append(raw_num)
            num_gloss.append(f"[Num:{self.numeral_value}]")

        art_op = None
        art_label = ""
        if self.is_definite is True and "def" in paradigm.articles:
            art_op = paradigm.articles["def"]
            art_label = "Def"

        art_tokens, art_gloss = [], []
        is_art_bound = False
        if art_op:
            base_noun, is_art_bound = GrammarEngine.apply_spatial_relator(base_noun, art_op, lang, category_type="case")
            if not is_art_bound:
                art_tokens = [art_op.form]
                art_gloss = [f"[{art_label}:{art_op.id}]"]

        adp_tokens, adp_gloss = [], []

        if self.case_operator:
            inflected_noun, is_bound = GrammarEngine.apply_spatial_relator(base_noun, self.case_operator, lang, category_type="case")
            if is_bound:
                n_form = inflected_noun.form
                tags = [self.case_label]
                if is_art_bound: tags.append(art_label)
                if is_num_bound: tags.append(self.number_label)
                if is_class_bound and cls_tag: tags.append(cls_tag)
                n_gloss = f"{self.head_noun.id}[{'+'.join(tags)}]"
            else:
                n_form = base_noun.form
                tags = []
                if is_art_bound: tags.append(art_label)
                if is_num_bound: tags.append(self.number_label)
                if is_class_bound and cls_tag: tags.append(cls_tag)
                tag_str = f"[{'+'.join(tags)}]" if tags else ""
                n_gloss = f"{self.head_noun.id}{tag_str}"
                adp_tokens = [self.case_operator.form]
                adp_gloss = [f"[{self.case_label}:{self.case_operator.id}]"]
        else:
            n_form = base_noun.form
            tags = [self.case_label] if self.case_label not in ("Nom", "Abs") else []
            if is_art_bound: tags.append(art_label)
            if is_num_bound: tags.append(self.number_label)
            if is_class_bound and cls_tag: tags.append(cls_tag)
            tag_str = f"[{'+'.join(tags)}]" if tags else ""
            n_gloss = f"{self.head_noun.id}{tag_str}"

        mod_tokens, mod_gloss = [], []
        if self.modifier:
            mod_tokens.append(self.modifier.form)
            mod_gloss.append(f"{self.modifier.id}[Adj]")

        det_tokens, det_gloss = [], []
        if self.determiner:
            det_tokens.append(self.determiner.form)
            det_gloss.append(f"[Det:{self.determiner.id}]")

        poss_tokens, poss_gloss = [], []
        if self.possessor:
            poss_tokens.append(self.possessor.form)
            poss_gloss.append(f"[Poss:{self.possessor.id}]")

        r_adp = 0.5 * (1.0 - h) + 7.5 * h
        r_art = 1.0 * (1.0 - h) + 6.0 * h
        r_poss = 2.0 * (1.0 - h) + 5.0 * h
        r_num = 3.0 * (1.0 - h) + 4.0 * h
        r_det = 3.5 * (1.0 - h) + 2.0 * h
        r_noun = 5.0 * (1.0 - h) + 2.5 * h
        r_mod = 6.0 * (1.0 - h) + 3.0 * h

        components = [
            ("adp", adp_tokens, adp_gloss, r_adp),
            ("art", art_tokens, art_gloss, r_art),
            ("poss", poss_tokens, poss_gloss, r_poss),
            ("num", num_tokens, num_gloss, r_num),
            ("det", det_tokens, det_gloss, r_det),
            ("noun", [n_form], [n_gloss], r_noun),
            ("mod", mod_tokens, mod_gloss, r_mod),
        ]
        components.sort(key=lambda item: item[3])

        tokens: list[str] = []
        glosses: list[str] = []
        for _, t_list, g_list, _ in components:
            tokens.extend(t_list)
            glosses.extend(g_list)

        return " ".join(t for t in tokens if t), " ".join(g for g in glosses if g)


# -----------------------------------------------------------------------------
# 3. GRAMMAR ENGINE
# -----------------------------------------------------------------------------

class GrammarEngine:
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
            elif v_idx + 1 < len(p_list):
                return [p_list[v_idx], p_list[v_idx + 1]]
            return [vowels[0]]

        return [max(p_list, key=lambda p: p.energy.acoustic_salience)]

    @classmethod
    def should_fuse_morpheme(cls, host_word: Word, operator: Word, lang: Language, is_prefix: bool = False) -> bool:
        s_idx = getattr(lang.profile, "synthesis_index", 0.50)
        pop = getattr(lang, "population", 0.50)

        # High population penalizes synthesis (Lupyan & Dale sociolinguistic law)
        eff_s = s_idx * max(0.20, 1.0 - pop * 0.35)

        if host_word.phonemes and operator.phonemes:
            p_host = host_word.phonemes[0] if is_prefix else host_word.phonemes[-1]
            p_op = operator.phonemes[-1] if is_prefix else operator.phonemes[0]
            boundary_friction = transition_cost(p_host, p_op)
        else:
            boundary_friction = 0.0

        fusion_potential = (eff_s * 4.0) - (0.95 + boundary_friction * 0.25)
        p_fuse = 1.0 / (1.0 + math.exp(-5.0 * fusion_potential))
        return p_fuse >= 0.50

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
        m_stem_list = [m.copy() for m in noun.morphemes] if noun.morphemes else [Morpheme(noun.phonemes, root_family_id=noun.derivation.root_family_id)]
        m_case = Morpheme(affix_phonemes, root_family_id=operator.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)

        morphemes = [m_case] + m_stem_list if is_prefix else m_stem_list + [m_case]
        raw_phonemes = affix_phonemes + noun.phonemes if is_prefix else noun.phonemes + affix_phonemes

        inflected_word = Word(
            vector=noun.vector,
            phonemes=raw_phonemes,
            morphemes=morphemes,
            derivation=DerivationRecord(root_family_id=noun.derivation.root_family_id, derivation_type=DerivationType.AFFIXATION, local_parent_ids=(noun.id,)),
            category=LexicalCategory.CONTENT_OPEN,
            is_ephemeral=True,
        )

        prof = lang.profile if lang else None
        if prof:
            inflected_word.phonemes = repair_phonemes(inflected_word.phonemes, profile=prof)
            apply_prosody(
                inflected_word,
                directional_bias=getattr(prof, "metrical_bias", 0.50),
                syllable_complexity=getattr(prof, "syllable_complexity", 0.50),
            )
        return inflected_word, True

    @classmethod
    def assign_noun_class(cls, noun: Word, paradigm: EmergentParadigm) -> EmergentClass | None:
        if not paradigm.noun_classes:
            return None
        return min(paradigm.noun_classes, key=lambda c: noun.vector.distance_to(c.centroid))

    # --- RESTORED TYPOLOGICAL QUERY METHODS ---

    @classmethod
    def get_alignment_system(cls, lang: Language) -> str:
        """
        Derives continuous morphosyntactic alignment.
        Rotates between Ergative-Absolutive and Nominative-Accusative
        based on cultural attention tension between Potency and Sociality.
        """
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
            return f"/{verb.form}/"

        inf_verb, is_b = cls.apply_spatial_relator(verb, theme_op, lang, category_type="infinitive")
        if is_b:
            return f"/{inf_verb.form}/"

        return f"/{theme_op.form} {verb.form}/" if h >= 0.5 else f"/{verb.form} {theme_op.form}/"

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
        bound_labels = []
        free_particles = []
        stem_word = verb

        s_idx = getattr(lang.profile, "synthesis_index", 0.50) if lang else 0.50
        h = getattr(lang.profile, "head_directionality", 0.50) if lang else 0.50

        active_person_op = person_op if s_idx >= 0.35 else None
        ops = [(temporal_op, "TAM"), (modal_op, "Mod"), (active_person_op, person_label)]

        for op, label in ops:
            if op is None:
                continue
            if lang and cls.should_fuse_morpheme(stem_word, op, lang, is_prefix=(h > 0.50)):
                affix_p = cls.compress_bound_affix(op)
                m_op = Morpheme(affix_p, root_family_id=op.derivation.root_family_id, is_grammatical=True, stage=ClineStage.BOUND_AFFIX)
                m_stem = Morpheme(stem_word.phonemes, root_family_id=stem_word.derivation.root_family_id)
                new_morphemes = [m_op, m_stem] if h > 0.50 else [m_stem, m_op]
                raw_p = affix_p + stem_word.phonemes if h > 0.50 else stem_word.phonemes + affix_p

                stem_word = Word(
                    vector=stem_word.vector,
                    phonemes=raw_p,
                    morphemes=new_morphemes,
                    derivation=stem_word.derivation,
                    category=LexicalCategory.CONTENT_OPEN,
                    is_ephemeral=True
                )
                bound_labels.append(label)
            else:
                free_particles.append(op)

        if bound_labels and lang:
            prof = lang.profile
            stem_word.phonemes = repair_phonemes(stem_word.phonemes, profile=prof)
            apply_prosody(
                stem_word,
                directional_bias=getattr(prof, "metrical_bias", 0.50),
                syllable_complexity=getattr(prof, "syllable_complexity", 0.50),
            )

        return stem_word, free_particles, False, bound_labels

    # --- PARADIGM DISCOVERY ---

    @classmethod
    def discover_grammar_system(cls, lang: Language) -> EmergentParadigm:
        cached = getattr(lang, "_cached_grammar_paradigm", None)
        if cached is not None:
            return cached

        attn = lang.cultural_attention
        s_idx = getattr(lang.profile, "synthesis_index", 0.50)
        pop = getattr(lang, "population", 0.50)
        infra = getattr(lang, "infrastructure", 0.0)

        # Natural Numeral Radix & Powers from Cultural State
        radix_base, radix_name, higher_powers = ContinuousNumeralSystem.determine_base(attn, pop, infra)
        num_chain = NumeralChain(primary_base=radix_base, name=radix_name, higher_units=higher_powers)

        digits: dict[int, Word] = {
            1: ContinuousNumeralSystem.resolve_numeral_word(1, lang, SemanticSpace.ONE, 18.0),
            2: ContinuousNumeralSystem.resolve_numeral_word(2, lang, SemanticSpace.OTHER, 16.0),
        }
        for d in range(3, min(6, radix_base)):
            digits[d] = ContinuousNumeralSystem.resolve_numeral_word(d, lang, SemanticSpace.HAND, 12.0)

        radix_words: dict[str, Word] = {
            "base": ContinuousNumeralSystem.resolve_numeral_word(radix_base, lang, SemanticSpace.BODY, 20.0),
            "hundred": ContinuousNumeralSystem.resolve_numeral_word(higher_powers["hundred"], lang, SemanticSpace.TREE, 15.0),
            "thousand": ContinuousNumeralSystem.resolve_numeral_word(higher_powers["thousand"], lang, SemanticSpace.EARTH, 14.0),
        }

        # Case & Relational Operators
        v_loc = ContinuousNumeralSystem.resolve_numeral_word(101, lang, SemanticVector(0.1, 0.0, 0.5, 0.2, 0.0, 0.0, 0.9), 16.0)
        v_dat = ContinuousNumeralSystem.resolve_numeral_word(102, lang, SemanticVector(0.2, 0.4, 0.6, 0.4, 0.5, 0.8, 0.3), 16.0)
        v_erg = ContinuousNumeralSystem.resolve_numeral_word(103, lang, SemanticVector(0.2, 0.3, 0.5, 0.9, 0.8, 0.3, 0.4), 16.0)
        v_acc = ContinuousNumeralSystem.resolve_numeral_word(104, lang, SemanticVector(0.3, 0.0, 0.4, 0.2, 0.2, 0.2, 0.5), 16.0)
        spatials = [v_loc, v_dat, v_erg, v_acc]

        v_past = ContinuousNumeralSystem.resolve_numeral_word(105, lang, SemanticVector(0.1, 0.1, 0.5, 0.3, 0.9, 0.2, 0.7), 16.0)
        v_mod  = ContinuousNumeralSystem.resolve_numeral_word(106, lang, SemanticVector(0.2, 0.4, 0.6, 0.8, 0.5, 0.4, 0.2), 14.0)

        pronouns = {
            "1sg": ContinuousNumeralSystem.resolve_numeral_word(201, lang, SemanticSpace.SELF, 22.0),
            "2sg": ContinuousNumeralSystem.resolve_numeral_word(202, lang, SemanticSpace.OTHER, 20.0),
            "3sg": ContinuousNumeralSystem.resolve_numeral_word(203, lang, SemanticSpace.PERSON, 22.0),
            "1pl": ContinuousNumeralSystem.resolve_numeral_word(204, lang, SemanticSpace.MANY, 18.0),
        }

        # Continuous Alignment Rotation (Potency vs Sociality)
        align_angle = float(min(math.pi / 2.0, max(0.0, (attn[3] / (attn[5] + 0.1)) * 0.45 * (math.pi / 2.0))))

        # Sociolinguistic Law: Noun classes dissolve in large populations
        noun_classes: list[EmergentClass] = []
        if s_idx > 0.35 and pop < 1.20:
            m_anim = ContinuousNumeralSystem.resolve_numeral_word(301, lang, SemanticSpace.PERSON, 18.0)
            m_inan = ContinuousNumeralSystem.resolve_numeral_word(302, lang, SemanticSpace.STONE, 18.0)
            noun_classes = [
                EmergentClass(class_id=1, label="Animate", centroid=SemanticSpace.PERSON, marker=m_anim),
                EmergentClass(class_id=2, label="Inanimate", centroid=SemanticSpace.STONE, marker=m_inan),
            ]

        articles = {}
        if (1.0 - s_idx) * attn[6] > 0.35 or pop > 1.50:
            articles["def"] = ContinuousNumeralSystem.resolve_numeral_word(401, lang, SemanticSpace.THIS, 24.0)
            art_system = "Definite Article"
        else:
            art_system = "Transdefinite (No Articles)"

        number_operators = {
            "pl": ContinuousNumeralSystem.resolve_numeral_word(501, lang, SemanticSpace.MANY, 20.0)
        }

        strat_mode = "fusional" if (s_idx > 0.65 and pop < 1.0) else ("agglutinative" if s_idx > 0.35 else "analytic")
        theme_cand = ContinuousNumeralSystem.resolve_numeral_word(601, lang, SemanticVector(0.4, 0.6, 0.5, 0.6, 0.7, 0.4, 0.3), 16.0)

        emergent_verb_classes = [
            EmergentVerbClass(
                class_id=1,
                label=f"Regular ({strat_mode.capitalize()})",
                centroid=theme_cand.vector,
                thematic_marker=theme_cand,
                aspect_label=strat_mode.capitalize(),
                morph_strategy=strat_mode,
            )
        ]

        paradigm = EmergentParadigm(
            temporal_operators=[v_past], spatial_operators=spatials, determiners=[ContinuousNumeralSystem.resolve_numeral_word(701, lang, SemanticSpace.THIS)],
            modal_operators=[v_mod], noun_classes=noun_classes, verb_classes=emergent_verb_classes,
            relativizers=[ContinuousNumeralSystem.resolve_numeral_word(702, lang, SemanticSpace.THIS)],
            complementizers=[ContinuousNumeralSystem.resolve_numeral_word(703, lang, SemanticSpace.THAT)],
            pronouns=pronouns, number_system="Singular / Plural", number_operators=number_operators,
            article_system=art_system, articles=articles, numeral_chain=num_chain, digits=digits,
            radix_words=radix_words, formatted_cache={}, alignment_angle=align_angle
        )

        lang._cached_grammar_paradigm = paradigm
        return paradigm

    @classmethod
    def format_numeral(cls, value: int, lang: Language, paradigm: EmergentParadigm | None = None) -> str:
        para = paradigm or cls.discover_grammar_system(lang)
        return ContinuousNumeralSystem.format_combinatorial_numeral(value, lang, para)

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
            np = NounPhrase(head_noun=word, assigned_class=assigned_cls, case_label="Nom" if alignment == "nominative_accusative" else "Abs")
            surf_form, _ = np.render(lang)

            if assigned_cls and assigned_cls.marker and len(para.noun_classes) >= 2 and getattr(lang, "population", 0.5) < 1.20:
                p_aff = cls.compress_bound_affix(assigned_cls.marker)
                aff_str = "".join(p.ipa for p in p_aff)
                clean_display = f"{aff_str}-{word.form}"
            else:
                clean_display = word.form

            return clean_display, word.plain_form, {
                "pos": "Noun", "class": cls_label, "case": "Nom" if alignment == "nominative_accusative" else "Abs"
            }
        elif pos == "Verb":
            inf_str = cls.get_infinitive_form(word, lang, para)
            return inf_str.strip("/"), word.plain_form, {
                "pos": "Verb", "infinitive": inf_str,
                "strategy": para.verb_classes[0].morph_strategy if para.verb_classes else "analytic",
            }
        else:
            return word.form, word.plain_form, {"pos": pos}

    @classmethod
    def get_noun_declension_table(cls, noun: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> dict[str, str]:
        para = paradigm or cls.discover_grammar_system(lang)
        alignment = cls.get_alignment_system(lang)
        grid: dict[str, str] = {}
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
            ("Loc", loc_word)
        ]

        for num_label in ["Sg", "Pl"]:
            num_val = 2 if num_label == "Pl" else 1
            for case_label, case_op in cases:
                np_bare = NounPhrase(
                    head_noun=noun, assigned_class=assigned_cls, case_operator=case_op,
                    case_label=case_label, numeral_value=num_val, number_label=num_label
                )
                form_bare, _ = np_bare.render(lang)
                grid[f"{case_label}.{num_label}"] = f"/{form_bare}/"

        return grid

    @classmethod
    def get_verb_conjugation_table(cls, verb: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> dict[str, str]:
        para = paradigm or cls.discover_grammar_system(lang)
        conj: dict[str, str] = {}
        h = lang.profile.head_directionality

        conj["Infinitive"] = cls.get_infinitive_form(verb, lang, para)
        t_op = para.temporal_operators[0] if para.temporal_operators else None

        for p_key in ["1sg", "2sg", "3sg", "1pl"]:
            p_op = para.pronouns.get(p_key)
            v_form, free_p, _, _ = cls.inflect_predicate(verb, temporal_op=None, modal_op=None, agreement_op=None, person_op=p_op, person_label=p_key, lang=lang)
            tokens = [p.form for p in free_p] + [v_form.form] if h >= 0.5 else [v_form.form] + [p.form for p in free_p]
            conj[f"Pres.{p_key}"] = f"/{' '.join(tokens)}/"

        for p_key in ["1sg", "3sg"]:
            p_op = para.pronouns.get(p_key)
            v_form, free_p, _, _ = cls.inflect_predicate(verb, temporal_op=t_op, modal_op=None, agreement_op=None, person_op=p_op, person_label=p_key, lang=lang, is_past_or_perfective=True)
            tokens = [p.form for p in free_p] + [v_form.form] if h >= 0.5 else [v_form.form] + [p.form for p in free_p]
            conj[f"Past.{p_key}"] = f"/{' '.join(tokens)}/"

        return conj