"""
grammar.py — Autonomous Morphosyntax, Continuous Typological Paradigms,
and Emergent Noun/Verb Classification.
Part of the EvoLangDiff Engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from phonetics import (
    Phoneme,
    transition_cost,
    repair_phonemes,
    apply_prosody,
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
    sub_base: int | None = None
    primary_base: int = 10
    super_base: int | None = None
    name: str = "Decimal (Base-10)"


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


class ClauseValency(Enum):
    INTRANSITIVE = "Intransitive (1-Arg: Subject)"
    TRANSITIVE = "Transitive (2-Arg: Agent + Patient)"
    DITRANSITIVE = "Ditransitive (3-Arg: Agent + Recipient + Theme)"


@dataclass
class NounPhrase:
    head_noun: Word
    modifier: Word | None = None
    determiner: Word | None = None
    possessor: Word | None = None
    relative_clause: Clause | None = None
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
        paradigm = GrammarEngine.discover_grammar_system(lang)

        agr_op = self.assigned_class.marker if self.assigned_class else None
        cls_tag = f"Cls{self.assigned_class.class_id}" if self.assigned_class else ""

        base_noun = self.head_noun
        is_class_bound = False
        if agr_op and len(paradigm.noun_classes) >= 2:
            base_noun, is_class_bound = GrammarEngine.apply_spatial_relator(self.head_noun, agr_op, lang, category_type="class")

        if self.numeral_value and self.numeral_value > 1 and self.number_label == "Sg":
            if self.numeral_value == 2 and "du" in paradigm.number_operators:
                self.number_operator = paradigm.number_operators["du"]
                self.number_label = "Du"
            elif "pl" in paradigm.number_operators:
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
        elif self.is_definite is False and "indef" in paradigm.articles:
            art_op = paradigm.articles["indef"]
            art_label = "Indef"

        art_tokens, art_gloss = [], []
        is_art_bound = False
        if art_op:
            base_noun, is_art_bound = GrammarEngine.apply_spatial_relator(base_noun, art_op, lang, category_type="case")
            if not is_art_bound:
                art_tokens = [art_op.form]
                art_gloss = [f"[{art_label}:{art_op.id}]"]

        prep_tokens, prep_gloss = [], []
        post_tokens, post_gloss = [], []

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

                p_prep = 1.0 / (1.0 + math.exp(-8.0 * (h - 0.50)))
                if p_prep >= 0.50:
                    prep_tokens = [self.case_operator.form]
                    prep_gloss = [f"[{self.case_label}:{self.case_operator.id}]"]
                else:
                    post_tokens = [self.case_operator.form]
                    post_gloss = [f"[{self.case_label}:{self.case_operator.id}]"]
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

        components = [
            ("prep", prep_tokens, prep_gloss, 0.0 * (1.0 - h) + 0.0 * h),
            ("art", art_tokens, art_gloss, 1.0 * (1.0 - h) + 1.0 * h),
            ("poss", poss_tokens, poss_gloss, 2.0 * (1.0 - h) + 5.0 * h),
            ("num", num_tokens, num_gloss, 3.0 * (1.0 - h) + 4.0 * h),
            ("det", det_tokens, det_gloss, 4.0 * (1.0 - h) + 1.5 * h),
            ("noun", [n_form], [n_gloss], 6.0 * (1.0 - h) + 2.0 * h),
            ("mod", mod_tokens, mod_gloss, 5.0 * (1.0 - h) + 3.0 * h),
            ("post", post_tokens, post_gloss, 7.0 * (1.0 - h) + 7.0 * h),
        ]
        components.sort(key=lambda item: item[3])

        tokens: list[str] = []
        glosses: list[str] = []
        for _, t_list, g_list, _ in components:
            tokens.extend(t_list)
            glosses.extend(g_list)

        return " ".join(t for t in tokens if t), " ".join(g for g in glosses if g)


@dataclass
class Clause:
    predicate: Word
    valency: ClauseValency
    subject: NounPhrase
    direct_object: NounPhrase | None = None
    indirect_object: NounPhrase | None = None
    oblique_location: NounPhrase | None = None
    temporal_op: Word | None = None
    modal_op: Word | None = None
    agreement_op: Word | None = None
    person_marker: Word | None = None
    person_label: str = "3sg"
    complement_clause: Clause | None = None
    complementizer: Word | None = None
    relativizer: Word | None = None
    is_subordinate: bool = False
    is_past_or_perfective: bool = False

    def render(self, lang: Language) -> tuple[str, str]:
        h = lang.profile.head_directionality

        inflected_verb, free_particles, is_suppletive, bound_labels = GrammarEngine.inflect_predicate(
            verb=self.predicate,
            temporal_op=self.temporal_op,
            modal_op=self.modal_op,
            agreement_op=self.agreement_op,
            person_op=self.person_marker,
            person_label=self.person_label,
            lang=lang,
            is_past_or_perfective=self.is_past_or_perfective,
        )

        v_tokens = [p.form for p in free_particles] + [inflected_verb.form]
        v_tag = f"{self.predicate.id}[V" + (f"+{'+'.join(bound_labels)}" if bound_labels else "") + "]"
        if is_suppletive:
            v_tag += "[Supp]"
        v_gloss = [f"[Part:{p.id}]" for p in free_particles] + [v_tag]

        v_str = " ".join(v_tokens)
        vg_str = " ".join(v_gloss)

        subj_ipa, subj_gl = self.subject.render(lang)
        obj_ipa, obj_gl = self.direct_object.render(lang) if self.direct_object else ("", "")
        ind_ipa, ind_gl = self.indirect_object.render(lang) if self.indirect_object else ("", "")
        obl_ipa, obl_gl = self.oblique_location.render(lang) if self.oblique_location else ("", "")

        r_subj = 1.0
        r_verb = 5.0 * (1.0 - h) + 0.5 * h
        r_obj  = 3.0 * (1.0 - h) + 2.5 * h
        r_ind  = 2.0 * (1.0 - h) + 3.5 * h
        r_obl  = 4.0 * (1.0 - h) + 4.5 * h

        items = [
            (r_subj, subj_ipa, subj_gl),
            (r_verb, v_str, vg_str),
            (r_obj, obj_ipa, obj_gl),
            (r_ind, ind_ipa, ind_gl),
            (r_obl, obl_ipa, obl_gl),
        ]
        items.sort(key=lambda x: x[0])

        clause_ipa = [it[1] for it in items if it[1]]
        clause_gl  = [it[2] for it in items if it[2]]

        return " ".join(clause_ipa).strip(), " ".join(clause_gl).strip()


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
        if host_word.phonemes and operator.phonemes:
            p_host = host_word.phonemes[0] if is_prefix else host_word.phonemes[-1]
            p_op = operator.phonemes[-1] if is_prefix else operator.phonemes[0]
            boundary_friction = transition_cost(p_host, p_op)
        else:
            boundary_friction = 0.0

        fusion_potential = (s_idx * 4.0) - (0.95 + boundary_friction * 0.25)
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
    def _rank_bleached_candidates(cls, lang: Language, projection_vector: Sequence[float]) -> list[Word]:
        attn = lang.cultural_attention
        candidates = list(lang.words.values())
        if not candidates:
            return []

        scored = []
        for w in candidates:
            c = w.vector.concreteness
            a = w.vector.animacy
            bleach_score = (1.0 - c * 0.70 - a * 0.30)
            u_score = math.log(w.usage_frequency + 1.2)
            brevity = 1.0 / (len(w.phonemes) ** 1.2 + 0.1)

            alignment = sum(w.vector.coords[i] * projection_vector[i] * attn[i] for i in range(7)) / max(0.1, sum(attn))
            total_fitness = bleach_score * u_score * brevity * (1.0 + alignment)
            scored.append((total_fitness, w))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [w for _, w in scored]

    @classmethod
    def assign_noun_class(cls, noun: Word, paradigm: EmergentParadigm) -> EmergentClass | None:
        """Assigns noun to an emergent class based on cognitive animacy and sociality."""
        if not paradigm.noun_classes:
            return None
        # Class 1: Animate / Social; Class 2: Inanimate / Physical
        return paradigm.noun_classes[0] if noun.vector.animacy >= 0.45 else paradigm.noun_classes[1]

    @classmethod
    def discover_grammar_system(cls, lang: Language) -> EmergentParadigm:
        cached = getattr(lang, "_cached_grammar_paradigm", None)
        if cached is not None:
            return cached

        attn = lang.cultural_attention
        s_idx = getattr(lang.profile, "synthesis_index", 0.50)
        candidates = list(lang.words.values())
        if not candidates:
            return EmergentParadigm()

        assigned_word_ids: set[int] = set()

        def pick_emergent(projection: Sequence[float], max_count: int = 1, base_usage: float = 12.0) -> list[Word]:
            ranked = cls._rank_bleached_candidates(lang, projection)
            chosen = []
            for w in ranked:
                if w.id in assigned_word_ids:
                    continue
                w_gram = w.copy()
                w_gram.usage_frequency = max(w.usage_frequency, base_usage)
                chosen.append(w_gram)
                assigned_word_ids.add(w.id)
                if len(chosen) >= max_count:
                    break
            return chosen or ([candidates[0]] if candidates else [])

        temporals = pick_emergent([0.1, 0.1, 0.5, 0.3, 0.9, 0.2, 0.7], max_count=4, base_usage=14.0)
        spatials = pick_emergent([0.2, 0.0, 0.5, 0.2, 0.1, 0.1, 0.9], max_count=6, base_usage=16.0)
        modals = pick_emergent([0.2, 0.4, 0.6, 0.8, 0.5, 0.4, 0.2], max_count=4, base_usage=12.0)
        determiners = pick_emergent([0.2, 0.0, 0.5, 0.3, 0.1, 0.2, 0.4], max_count=4, base_usage=18.0)
        relativizers = pick_emergent([0.1, 0.0, 0.5, 0.2, 0.1, 0.1, 0.6], max_count=2, base_usage=15.0)
        complementizers = pick_emergent([0.2, 0.6, 0.5, 0.3, 0.6, 0.8, 0.2], max_count=2, base_usage=15.0)

        # Emergent Noun Classes (Animate vs Inanimate gender distinction)
        noun_classes: list[EmergentClass] = []
        if s_idx >= 0.35 or attn[1] >= 0.90:
            m_anim = pick_emergent([0.8, 1.0, 0.6, 0.5, 0.5, 0.8, 0.2], max_count=1, base_usage=18.0)
            m_inan = pick_emergent([0.8, 0.1, 0.5, 0.5, 0.2, 0.1, 0.6], max_count=1, base_usage=18.0)
            noun_classes = [
                EmergentClass(class_id=1, label="Animate", centroid=SemanticSpace.PERSON, marker=m_anim[0] if m_anim else None),
                EmergentClass(class_id=2, label="Inanimate", centroid=SemanticSpace.STONE, marker=m_inan[0] if m_inan else None),
            ]

        pronouns: dict[str, Word] = {}
        p_1sg = pick_emergent([0.5, 0.9, 0.6, 0.4, 0.4, 0.3, 0.1], max_count=1, base_usage=22.0)
        p_2sg = pick_emergent([0.5, 0.9, 0.6, 0.4, 0.4, 0.8, 0.1], max_count=1, base_usage=20.0)
        p_3sg = pick_emergent([0.6, 0.9, 0.5, 0.4, 0.4, 0.5, 0.3], max_count=1, base_usage=22.0)
        p_pl = pick_emergent([0.3, 0.7, 0.5, 0.5, 0.3, 0.7, 0.7], max_count=2, base_usage=18.0)

        if p_1sg: pronouns["1sg"] = p_1sg[0]
        if p_2sg: pronouns["2sg"] = p_2sg[0]
        if p_3sg: pronouns["3sg"] = p_3sg[0]
        if p_pl:
            pronouns["1pl"] = p_pl[0]
            pronouns["2pl"] = p_pl[1] if len(p_pl) > 1 else p_pl[0]
            pronouns["3pl"] = p_pl[1] if len(p_pl) > 1 else p_pl[0]

        articles: dict[str, Word] = {}
        p_article_emerge = 1.0 / (1.0 + math.exp(6.0 * (s_idx - 0.45))) * (attn[6] / 1.0)
        if p_article_emerge >= 0.45:
            art_cand = pick_emergent([0.2, 0.0, 0.5, 0.2, 0.0, 0.1, 0.3], max_count=1, base_usage=26.0)
            if art_cand:
                articles["def"] = art_cand[0]
                art_system = "Definite Article"
            else:
                art_system = "Transdefinite (No Articles)"
        else:
            art_system = "Transdefinite (No Articles)"

        number_operators: dict[str, Word] = {}
        p_number_emerge = 1.0 / (1.0 + math.exp(-8.0 * (s_idx - 0.35)))
        if p_number_emerge >= 0.50:
            num_system = "Singular / Plural"
            pl_cand = pick_emergent([0.2, 0.2, 0.5, 0.5, 0.1, 0.3, 0.8], max_count=1, base_usage=20.0)
            if pl_cand:
                number_operators["pl"] = pl_cand[0]
        else:
            num_system = "Transnumeral (No Grammatical Number)"

        digits: dict[int, Word] = {}
        d1 = pick_emergent([0.1, 0.0, 0.5, 0.2, 0.0, 0.0, 0.1], max_count=1, base_usage=24.0)
        if d1: digits[1] = d1[0]
        for d_idx in range(2, 6):
            cands = pick_emergent([0.1, 0.0, 0.5, 0.4, 0.0, 0.1, 0.5], max_count=1, base_usage=18.0)
            if cands: digits[d_idx] = cands[0]

        p_an = math.exp(-10.0 * (s_idx - 0.20) ** 2)
        p_fu = math.exp(-10.0 * (s_idx - 0.50) ** 2)
        p_ag = math.exp(-10.0 * (s_idx - 0.80) ** 2)
        strat_mode = max([("analytic", p_an), ("fusional", p_fu), ("concatenative", p_ag)], key=lambda x: x[1])[0]

        theme_cand = pick_emergent([0.4, 0.6, 0.5, 0.6, 0.7, 0.4, 0.3], max_count=1, base_usage=16.0)
        emergent_verb_classes = [
            EmergentVerbClass(
                class_id=1,
                label=f"Regular ({strat_mode.capitalize()})",
                centroid=theme_cand[0].vector if theme_cand else candidates[0].vector,
                thematic_marker=theme_cand[0] if theme_cand else None,
                aspect_label=strat_mode.capitalize(),
                morph_strategy=strat_mode,
            )
        ]

        paradigm = EmergentParadigm(
            temporal_operators=temporals, spatial_operators=spatials, determiners=determiners, modal_operators=modals,
            noun_classes=noun_classes, verb_classes=emergent_verb_classes, relativizers=relativizers, complementizers=complementizers,
            pronouns=pronouns, number_system=num_system, number_operators=number_operators, article_system=art_system,
            articles=articles, numeral_chain=NumeralChain(primary_base=10), digits=digits, radix_words={},
        )

        lang._cached_grammar_paradigm = paradigm
        return paradigm

    @classmethod
    def format_numeral(cls, value: int, lang: Language, paradigm: EmergentParadigm | None = None) -> str:
        para = paradigm or cls.discover_grammar_system(lang)
        if value in para.formatted_cache:
            return para.formatted_cache[value]
        if value in para.digits:
            res = f"/{para.digits[value].form}/"
            para.formatted_cache[value] = res
            return res
        return f"/{value}/"

    @classmethod
    def get_infinitive_form(cls, verb: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> str:
        """Derives standard grammatical infinitive (fused thematic marker or analytic particle)."""
        para = paradigm or cls.discover_grammar_system(lang)
        s_idx = lang.profile.synthesis_index
        h = lang.profile.head_directionality
        theme_op = para.verb_classes[0].thematic_marker if para.verb_classes else None

        if not theme_op:
            return f"/{verb.form}/"

        if s_idx >= 0.38:
            inf_verb, is_b = cls.apply_spatial_relator(verb, theme_op, lang, category_type="infinitive")
            if is_b:
                return f"/{inf_verb.form}/"

        p_head = 1.0 / (1.0 + math.exp(-6.0 * (h - 0.50)))
        return f"/{theme_op.form} {verb.form}/" if p_head >= 0.5 else f"/{verb.form} {theme_op.form}/"

    @classmethod
    def get_citation_presentation(
        cls, word: Word, lang: Language, paradigm: EmergentParadigm | None = None
    ) -> tuple[str, str, dict]:
        """Provides correctly declined/inflected surface citation form for vocabulary display."""
        para = paradigm or cls.discover_grammar_system(lang)
        pos = cls.get_part_of_speech(word)

        if pos == "Verb":
            inf_str = cls.get_infinitive_form(word, lang, para)
            return inf_str.strip("/"), word.plain_form, {
                "pos": "Verb",
                "infinitive": inf_str,
                "strategy": para.verb_classes[0].morph_strategy if para.verb_classes else "analytic",
            }
        elif pos == "Noun":
            assigned_cls = cls.assign_noun_class(word, para)
            cls_label = assigned_cls.label if assigned_cls else None
            has_def = ("def" in para.articles)
            np = NounPhrase(
                head_noun=word,
                assigned_class=assigned_cls,
                is_definite=True if has_def else None,
                case_label="Nom" if cls.get_alignment_system(lang) == "nominative_accusative" else "Abs",
            )
            surf_form, _ = np.render(lang)
            return surf_form, word.plain_form, {
                "pos": "Noun",
                "class": cls_label,
                "article": "Def" if has_def else None,
                "case": "Nom" if cls.get_alignment_system(lang) == "nominative_accusative" else "Abs",
            }
        else:
            return word.form, word.plain_form, {"pos": pos}

    @classmethod
    def get_valency_frame(cls, verb: Word) -> ClauseValency:
        d = verb.vector.dynamism
        p = verb.vector.potency
        return ClauseValency.TRANSITIVE if (d * 0.6 + p * 0.4) >= 0.55 else ClauseValency.INTRANSITIVE

    @classmethod
    def get_alignment_system(cls, lang: Language) -> str:
        return getattr(lang.profile, "alignment", "nominative_accusative")

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

    @classmethod
    def get_noun_declension_table(cls, noun: Word, lang: Language, paradigm: EmergentParadigm | None = None) -> dict[str, str]:
        para = paradigm or cls.discover_grammar_system(lang)
        alignment = cls.get_alignment_system(lang)
        grid: dict[str, str] = {}

        words = list(lang.words.values())
        if not words:
            return {}

        spatials = para.spatial_operators or words[:4]
        loc_word = spatials[0]
        dat_word = spatials[1] if len(spatials) > 1 else spatials[0]
        erg_word = spatials[2] if len(spatials) > 2 else spatials[0]
        acc_word = spatials[3] if len(spatials) > 3 else spatials[0]

        has_plural = (para.number_system != "Transnumeral (No Grammatical Number)" and "pl" in para.number_operators)
        assigned_cls = cls.assign_noun_class(noun, para)
        has_def = ("def" in para.articles)

        # Build table entries through NounPhrase to guarantee correct class and article incorporation
        cases = [
            ("Abs" if alignment == "ergative_absolutive" else "Nom", None),
            ("Erg" if alignment == "ergative_absolutive" else "Acc", erg_word if alignment == "ergative_absolutive" else acc_word),
            ("Dat", dat_word),
            ("Loc", loc_word),
        ]

        for num_label in (["Sg", "Pl"] if has_plural else ["Sg"]):
            num_val = 2 if num_label == "Pl" else 1
            for case_label, case_op in cases:
                # 1. Bare / Indefinite Declension
                np_bare = NounPhrase(
                    head_noun=noun,
                    assigned_class=assigned_cls,
                    case_operator=case_op,
                    case_label=case_label,
                    numeral_value=num_val,
                    number_label=num_label,
                    is_definite=False if has_def else None,
                )
                form_bare, _ = np_bare.render(lang)
                grid[f"{case_label}.{num_label}"] = f"/{form_bare}/"

                # 2. Definite Declension (if articles exist in dialect)
                if has_def:
                    np_def = NounPhrase(
                        head_noun=noun,
                        assigned_class=assigned_cls,
                        case_operator=case_op,
                        case_label=case_label,
                        numeral_value=num_val,
                        number_label=num_label,
                        is_definite=True,
                    )
                    form_def, _ = np_def.render(lang)
                    grid[f"{case_label}.{num_label} (Def)"] = f"/{form_def}/"

        return grid

    @classmethod
    def assemble_complex_clause(
        cls,
        lang: Language,
        subject: NounPhrase,
        predicate: Word,
        direct_object: NounPhrase | None = None,
        indirect_object: NounPhrase | None = None,
        oblique_location: NounPhrase | None = None,
        complement_clause: Clause | None = None,
        relativizer: Word | None = None,
        is_past_or_perfective: bool = False,
    ) -> Clause:
        paradigm = cls.discover_grammar_system(lang)
        valency = cls.get_valency_frame(predicate)
        alignment = cls.get_alignment_system(lang)

        spatials = paradigm.spatial_operators or list(lang.words.values())[:4]
        loc_word = spatials[0]
        dat_word = spatials[1] if len(spatials) > 1 else spatials[0]
        erg_word = spatials[2] if len(spatials) > 2 else spatials[0]
        acc_word = spatials[3] if len(spatials) > 3 else spatials[0]

        if alignment == "ergative_absolutive":
            if direct_object:
                subject.case_operator = erg_word
                subject.case_label = "Erg"
                direct_object.case_operator = None
                direct_object.case_label = "Abs"
            else:
                subject.case_operator = None
                subject.case_label = "Abs"
        else:
            subject.case_operator = None
            subject.case_label = "Nom"
            if direct_object:
                direct_object.case_operator = acc_word
                direct_object.case_label = "Acc"

        if indirect_object:
            indirect_object.case_operator = dat_word
            indirect_object.case_label = "Dat"

        if oblique_location:
            oblique_location.case_operator = loc_word
            oblique_location.case_label = "Loc"

        temporal_op = paradigm.temporal_operators[0] if (is_past_or_perfective and paradigm.temporal_operators) else None

        return Clause(
            predicate=predicate,
            valency=valency,
            subject=subject,
            direct_object=direct_object,
            indirect_object=indirect_object,
            oblique_location=oblique_location,
            temporal_op=temporal_op,
            modal_op=None,
            agreement_op=None,
            person_marker=paradigm.pronouns.get("3sg"),
            person_label="3sg",
            complement_clause=complement_clause,
            complementizer=None,
            relativizer=relativizer,
            is_subordinate=(relativizer is not None),
            is_past_or_perfective=is_past_or_perfective,
        )