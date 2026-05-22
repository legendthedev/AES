"""
Explainable AI (XAI) Feedback Generation Module
Translates model feature weights and scores into natural-language diagnostic feedback.
Bridges the "Black Box" gap — making AI decisions transparent to students.
"""

from typing import Dict, List, Any, Tuple


# Feedback templates keyed by trait and performance tier
FEEDBACK_TEMPLATES = {
    'Grammar & Mechanics': {
        'excellent': {
            'summary': "Your grammar and mechanics are excellent.",
            'details': [
                "Very few or no spelling errors were detected in your essay.",
                "Your sentences are consistently well-formed and properly punctuated.",
                "Capital letters are used correctly throughout the essay.",
            ],
            'suggestions': [
                "Proofread one final time to catch any overlooked minor errors.",
            ]
        },
        'good': {
            'summary': "Your grammar and mechanics are generally solid.",
            'details': [
                "A small number of spelling errors were found, which slightly affect readability.",
                "Most sentences are grammatically correct, though a few need revision.",
            ],
            'suggestions': [
                "Use a spell-checker or tool like Grammarly for a final review.",
                "Read your essay aloud — errors that are hard to spot visually often become obvious when spoken.",
            ]
        },
        'satisfactory': {
            'summary': "Your grammar and mechanics need some attention.",
            'details': [
                "Several spelling and/or punctuation errors were detected.",
                "Some sentences may be grammatically incomplete or incorrectly constructed.",
            ],
            'suggestions': [
                "Review the rules for comma usage, especially in compound and complex sentences.",
                "Double-check the spelling of academic terms and subject-specific vocabulary.",
                "Consider using a grammar-checking tool like LanguageTool or Grammarly.",
            ]
        },
        'poor': {
            'summary': "Grammar and mechanical errors significantly impact this essay.",
            'details': [
                "A high density of spelling errors was detected throughout the essay.",
                "Multiple sentences contain structural errors that obscure meaning.",
                "Inconsistent capitalization was observed.",
            ],
            'suggestions': [
                "Focus on correcting spelling errors first — they are the most distracting to readers.",
                "Re-read each sentence individually to check it makes grammatical sense.",
                "Practice writing shorter, clearer sentences before building toward complexity.",
                "Use NLTK or an online grammar checker to identify error patterns.",
            ]
        }
    },
    'Vocabulary Sophistication': {
        'excellent': {
            'summary': "Your vocabulary is impressively rich and sophisticated.",
            'details': [
                "You demonstrate a wide range of vocabulary with strong lexical diversity.",
                "Academic vocabulary is used effectively and appropriately throughout.",
                "Word choices precisely convey the intended meaning.",
            ],
            'suggestions': [
                "Maintain this level of vocabulary in your revisions.",
                "Continue reading academic texts to further expand your lexical range.",
            ]
        },
        'good': {
            'summary': "Your vocabulary is varied and shows good command of language.",
            'details': [
                "You use a reasonably broad range of words, though some repetition is present.",
                "Some academic vocabulary is evident, contributing positively to the essay's tone.",
            ],
            'suggestions': [
                "Replace repeated words with synonyms — use a thesaurus to find alternatives.",
                "Study the Academic Word List (AWL) to incorporate more formal vocabulary.",
            ]
        },
        'satisfactory': {
            'summary': "Your vocabulary is functional but somewhat limited in range.",
            'details': [
                "Many common words appear repeatedly, reducing the lexical richness of the essay.",
                "Academic and discipline-specific vocabulary is underused.",
            ],
            'suggestions': [
                "Before writing, brainstorm synonyms for the key terms of your topic.",
                "Use the Academic Word List (AWL) as a reference guide for formal writing.",
                "Aim to use at least 3–5 academic or topic-specific terms per paragraph.",
            ]
        },
        'poor': {
            'summary': "Vocabulary is repetitive and lacks sophistication for academic writing.",
            'details': [
                "The same words are used very frequently, suggesting limited lexical range.",
                "Academic register is largely absent — the writing reads as informal or simplistic.",
            ],
            'suggestions': [
                "Read one academic article in your subject area each week and note new vocabulary.",
                "Use resources like Quizlet or Anki to actively build your academic word bank.",
                "Replace words like 'good', 'bad', 'nice', 'thing' with more precise alternatives.",
                "Study the Academic Word List (AWL): https://www.eapfoundation.com/vocab/academic/awl/",
            ]
        }
    },
    'Sentence Fluency': {
        'excellent': {
            'summary': "Your sentences flow excellently with great variety.",
            'details': [
                "You effectively vary sentence length and structure, creating a natural and engaging rhythm.",
                "Complex, compound, and simple sentences are all used appropriately.",
                "Sentence structures enhance rather than impede readability.",
            ],
            'suggestions': [
                "Continue this varied approach in all your academic writing.",
            ]
        },
        'good': {
            'summary': "Your sentence structure is generally fluent and readable.",
            'details': [
                "You show some variety in sentence construction.",
                "Most sentences are clear, though a few could be restructured for better flow.",
            ],
            'suggestions': [
                "Try starting some sentences with adverbial clauses (e.g., 'Although...', 'While...').",
                "Vary your sentence openings — avoid starting too many sentences with 'I' or 'The'.",
            ]
        },
        'satisfactory': {
            'summary': "Sentence structure is somewhat monotonous and could be more varied.",
            'details': [
                "Many sentences follow the same structural pattern, reducing variety.",
                "Some sentences may be too long (run-ons) or too short (choppy).",
            ],
            'suggestions': [
                "Combine short, related sentences using conjunctions (and, but, or, however).",
                "Break very long sentences (30+ words) into two clearer sentences.",
                "Vary sentence starters: try adverbs, prepositional phrases, or dependent clauses.",
            ]
        },
        'poor': {
            'summary': "Sentence fluency needs significant improvement.",
            'details': [
                "Many sentences are structurally similar, creating a repetitive and monotonous flow.",
                "Run-on sentences or sentence fragments may be present.",
            ],
            'suggestions': [
                "Practice writing 3 types of sentences: simple, compound, and complex.",
                "Identify and fix all sentence fragments (incomplete sentences without a main verb).",
                "Read each sentence aloud — if you run out of breath, the sentence is probably too long.",
            ]
        }
    },
    'Organization & Structure': {
        'excellent': {
            'summary': "The essay is excellently organized and structured.",
            'details': [
                "A clear introduction and conclusion frame the essay effectively.",
                "Discourse markers and transition words guide the reader logically through the argument.",
                "Paragraphs are well-balanced and focused on distinct ideas.",
            ],
            'suggestions': [
                "Ensure each body paragraph has a clear topic sentence as a best practice.",
            ]
        },
        'good': {
            'summary': "The essay is well-organized with clear logical flow.",
            'details': [
                "The overall structure is evident, though some transitions could be smoother.",
                "Introduction and/or conclusion markers are present.",
            ],
            'suggestions': [
                "Use more explicit transition phrases between paragraphs (e.g., 'Furthermore...', 'In contrast...').",
                "Ensure every paragraph relates clearly to the essay's central thesis.",
            ]
        },
        'satisfactory': {
            'summary': "The essay's organization is present but needs strengthening.",
            'details': [
                "The structure is partially evident, but the logical flow between ideas could be clearer.",
                "Transition words and discourse markers are underused.",
            ],
            'suggestions': [
                "Follow the standard 5-paragraph essay structure: Introduction → 3 Body Paragraphs → Conclusion.",
                "Add transition sentences at the beginning or end of each paragraph to link ideas.",
                "Ensure your introduction ends with a clear thesis statement.",
            ]
        },
        'poor': {
            'summary': "The essay lacks clear organizational structure.",
            'details': [
                "Ideas appear randomly arranged without a clear logical progression.",
                "An introduction and/or conclusion may be missing or very underdeveloped.",
                "Paragraphing is either absent or inconsistent.",
            ],
            'suggestions': [
                "Create an outline BEFORE writing: list your main points and organize them logically.",
                "Every essay must have 3 parts: Introduction (what you'll say), Body (the argument), Conclusion (what you said).",
                "Use 'signposting language': 'Firstly...', 'In addition...', 'Finally...' to guide the reader.",
            ]
        }
    },
    'Content Development': {
        'excellent': {
            'summary': "The essay shows excellent content development and depth.",
            'details': [
                "Ideas are thoroughly developed with supporting evidence and examples.",
                "The essay maintains strong semantic coherence throughout.",
                "Content is relevant and directly addresses the prompt requirements.",
            ],
            'suggestions': [
                "Continue this level of depth and elaboration in your academic writing.",
            ]
        },
        'good': {
            'summary': "Content is well-developed with good depth of analysis.",
            'details': [
                "Main ideas are adequately supported, though some points could be elaborated further.",
                "The essay addresses the prompt, though some areas may need more specific examples.",
            ],
            'suggestions': [
                "For each claim you make, add at least one specific example or piece of evidence.",
                "Expand your analysis of complex points — explain the 'why' behind each argument.",
            ]
        },
        'satisfactory': {
            'summary': "Content development is adequate but lacks sufficient depth.",
            'details': [
                "Some ideas are introduced but not fully developed or supported.",
                "The essay may be too short to fully address all aspects of the prompt.",
            ],
            'suggestions': [
                "Each body paragraph should develop one main idea with: a topic sentence, 2–3 supporting points, and evidence.",
                "Ask yourself 'Why?' and 'How?' after each claim to push your analysis deeper.",
                "Aim for a minimum of 250–300 words for short-answer essays; 400–500 for argument essays.",
            ]
        },
        'poor': {
            'summary': "Content is underdeveloped and insufficient.",
            'details': [
                "The essay is significantly below the expected length for the given prompt.",
                "Ideas are stated but not supported, analyzed, or elaborated.",
            ],
            'suggestions': [
                "Significantly expand your essay — address the prompt from multiple angles.",
                "Use the PEEL paragraph method: Point → Evidence → Explanation → Link.",
                "Brainstorm at least 5 different points before writing and select the 3 strongest.",
            ]
        }
    },
    'Readability': {
        'excellent': {
            'summary': "The essay is highly readable and accessible.",
            'details': [
                "The writing flows naturally and is easy to follow.",
                "Sentence complexity is appropriate for an academic audience.",
            ],
            'suggestions': [
                "Maintain this balance between accessibility and academic complexity.",
            ]
        },
        'good': {
            'summary': "The essay is generally readable with minor issues.",
            'details': [
                "Most of the essay is easy to read and well-paced.",
            ],
            'suggestions': [
                "Check for any passages that feel particularly dense or hard to follow and revise them.",
            ]
        },
        'satisfactory': {
            'summary': "Readability is adequate but some passages are difficult to follow.",
            'details': [
                "Some sections may be overly complex, dense, or conversely too simplistic.",
            ],
            'suggestions': [
                "Aim for the Flesch Reading Ease score of 60–70 for academic essays.",
                "Break down overly complex sentences into two or more clearer statements.",
            ]
        },
        'poor': {
            'summary': "The essay is difficult to read and follow.",
            'details': [
                "Readability scores indicate the text is either too complex or too simplistic for the academic level.",
                "The reader may struggle to extract meaning from many passages.",
            ],
            'suggestions': [
                "Use shorter, clearer sentences as the foundation of your writing.",
                "Avoid unnecessary jargon — clarity is more important than complexity.",
                "Read your essay to a friend: if they cannot understand it easily, revise it.",
            ]
        }
    }
}

HOLISTIC_FEEDBACK = {
    'excellent': [
        "This is a strong essay that demonstrates excellent command of academic writing conventions.",
        "Your work shows a sophisticated blend of vocabulary, structured argumentation, and coherent development.",
        "Minor refinements will polish this into an outstanding academic submission.",
    ],
    'good': [
        "This is a solid essay that meets most of the expectations for academic writing.",
        "Your strengths outweigh your weaknesses — with targeted revision, this essay can reach a higher level.",
        "Focus on the areas marked below for the most impactful improvements.",
    ],
    'satisfactory': [
        "This essay addresses the task but needs meaningful revision in several key areas.",
        "There is clear potential here — with focused effort on organization and development, your score can improve significantly.",
        "Pay special attention to the feedback areas marked 'Needs Work' below.",
    ],
    'poor': [
        "This essay requires substantial revision before it meets academic writing standards.",
        "Focus on the foundational skills first: clear sentences, organized paragraphs, and adequate content development.",
        "Using the specific suggestions below as a revision checklist will help you make steady, measurable progress.",
    ]
}


def _get_tier(score: float) -> str:
    """Map a 0–1 trait score to a feedback tier."""
    if score >= 0.80:
        return 'excellent'
    elif score >= 0.65:
        return 'good'
    elif score >= 0.45:
        return 'satisfactory'
    else:
        return 'poor'


def _get_holistic_tier(pct: float) -> str:
    if pct >= 80:
        return 'excellent'
    elif pct >= 65:
        return 'good'
    elif pct >= 45:
        return 'satisfactory'
    else:
        return 'poor'


class FeedbackEngine:
    """
    XAI Feedback Engine: translates model outputs into actionable natural-language advice.
    Implements the Highly Informative Feedback (HIF) framework.
    """

    def generate(
        self,
        scoring_result: Dict[str, Any],
        linguistic_feats: Dict[str, Any],
        neural_feats: Dict[str, Any],
        highlight_sentences: List[str],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive XAI feedback from the scoring result.
        Returns structured feedback for each trait plus holistic advice.
        """
        traits = scoring_result['trait_scores']
        pct = scoring_result['percentage_score']

        # Per-trait feedback
        trait_feedback = {}
        for trait_name, trait_score in traits.items():
            tier = _get_tier(trait_score)
            template = FEEDBACK_TEMPLATES.get(trait_name, {}).get(tier, {})
            trait_feedback[trait_name] = {
                'score': trait_score,
                'score_pct': round(trait_score * 100, 1),
                'tier': tier,
                'summary': template.get('summary', ''),
                'details': template.get('details', []),
                'suggestions': template.get('suggestions', []),
                'weight': scoring_result['weights'].get(trait_name, 0.0),
            }

        # Holistic feedback
        holistic_tier = _get_holistic_tier(pct)
        holistic_msgs = HOLISTIC_FEEDBACK.get(holistic_tier, [])

        # Specific metric callouts
        metric_insights = self._generate_metric_insights(linguistic_feats, neural_feats)

        # Strengths and weaknesses
        sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
        strengths = [t for t, s in sorted_traits if s >= 0.70][:2]
        weaknesses = [t for t, s in sorted_traits if s < 0.55][:3]

        return {
            'holistic_tier': holistic_tier,
            'holistic_messages': holistic_msgs,
            'trait_feedback': trait_feedback,
            'metric_insights': metric_insights,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'highlighted_sentences': highlight_sentences,
            'priority_action': weaknesses[0] if weaknesses else None,
        }

    def _generate_metric_insights(
        self,
        linguistic_feats: Dict,
        neural_feats: Dict
    ) -> List[Dict[str, str]]:
        """Generate specific, data-driven insights from extracted metrics."""
        insights = []

        wc = linguistic_feats.get('word_count', 0)
        ld = linguistic_feats.get('lexical_density', 0)
        sed = linguistic_feats.get('spelling_error_density', 0)
        asl = linguistic_feats.get('avg_sentence_length', 0)
        dm = linguistic_feats.get('discourse_marker_count', 0)
        sc = max(linguistic_feats.get('sentence_count', 1), 1)
        awr = linguistic_feats.get('academic_word_ratio', 0)
        coherence = neural_feats.get('scalar_features', {}).get('semantic_coherence', 0.5)

        insights.append({
            'metric': 'Word Count',
            'value': f"{wc} words",
            'icon': '📝',
            'status': 'good' if wc >= 200 else 'warn' if wc >= 100 else 'poor',
            'note': "Good length" if wc >= 200 else "Essay is short — consider expanding your ideas."
        })

        insights.append({
            'metric': 'Lexical Diversity (TTR)',
            'value': f"{round(ld * 100, 1)}%",
            'icon': '🔤',
            'status': 'good' if ld >= 0.45 else 'warn' if ld >= 0.30 else 'poor',
            'note': "Rich vocabulary range" if ld >= 0.45 else "Vocabulary is repetitive — vary your word choices."
        })

        insights.append({
            'metric': 'Spelling Error Rate',
            'value': f"{sed:.1f} per 100 words",
            'icon': '✏️',
            'status': 'good' if sed < 1 else 'warn' if sed < 3 else 'poor',
            'note': "Very clean writing" if sed < 1 else "Proofread carefully to reduce spelling errors."
        })

        insights.append({
            'metric': 'Avg. Sentence Length',
            'value': f"{asl:.1f} words/sentence",
            'icon': '📐',
            'status': 'good' if 12 <= asl <= 25 else 'warn',
            'note': "Ideal range" if 12 <= asl <= 25 else
                    "Sentences are too short — elaborate more." if asl < 12 else
                    "Sentences are very long — consider breaking them up."
        })

        insights.append({
            'metric': 'Discourse Markers',
            'value': f"{dm} markers ({round(dm/sc*100, 1)} per 100 sentences)",
            'icon': '🔗',
            'status': 'good' if dm >= max(sc * 0.25, 3) else 'warn' if dm >= 2 else 'poor',
            'note': "Good use of transitions" if dm >= 3 else "Use more transitional phrases to connect ideas."
        })

        insights.append({
            'metric': 'Academic Vocabulary',
            'value': f"{round(awr * 100, 1)}% of words",
            'icon': '🎓',
            'status': 'good' if awr >= 0.07 else 'warn' if awr >= 0.04 else 'poor',
            'note': "Strong academic register" if awr >= 0.07 else "Incorporate more academic vocabulary."
        })

        insights.append({
            'metric': 'Semantic Coherence (AI)',
            'value': f"{round(coherence * 100, 1)}%",
            'icon': '🧠',
            'status': 'good' if coherence >= 0.65 else 'warn' if coherence >= 0.45 else 'poor',
            'note': "Ideas flow coherently" if coherence >= 0.65 else
                    "Improve logical flow between sentences." if coherence >= 0.45 else
                    "The essay lacks logical connection between ideas."
        })

        return insights
