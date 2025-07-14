# Improvements to AutoMISC Prompts and OQ-CQ Classification Performance
Evaluation based on 10 manually labeled transcripts.

## Version 1 (`v1.txt`)
**Accuracy (OQ-CQ):** 57.79%  
**Description:**  
Original baseline. The prompt included minimal guidance on distinguishing Open (OQ) vs. Closed (CQ) questions, leading to high ambiguity and poor classification.

## Version 2 (`v2.txt`)
**Accuracy (OQ-CQ):** 92.21%  
**Description:**  
Introduced a more precise T2 definition, "any utterance grammatically answerable by 'yes' or 'no', even if awkward or unlikely, must be labeled as CQ". Employed semi-few-shot prompting with patterns like "Can you...", "Do you...", "Is there...", and defaulted ambiguous cases to CQ. OQ was restricted to questions that structurally preclude binary responses. Some misclassifications remain for phrasings not covered by the few-shot examples (e.g., "Could you...", "Have you...").

## Version 3 (`v3.txt`)
**Accuracy (OQ-CQ):** 78.57%  
**Description:**  
Same prompt content as Version 2 but removed semi-few-shot prompting. Resulted in a noticeable performance drop, confirming the importance of example-driven prompting.

## Version 4 (`v4.txt`)
**Accuracy (OQ-CQ):** 69.48%  
**Description:**  
Enforced strict grammar-based rules across both T1 and T2 prompts, listing specific modal/auxiliary verbs (e.g., Can, Could, Do, Does...). This over-constrained the model and led to frequent misclassification of questions as neutral utterances (FA, FI, ST).

## Version 5 (`v5.txt`)
**Accuracy (OQ-CQ):** 89.61%  
**Description:**  
Reintroduced grammar-based CQ rules into the T1 prompt while retaining extensive semi-few-shot prompting in T2. Improved performance over Versions 3–4, but slightly underperformed compared to Version 2.

## Version 6 (`v6.txt`)
**Accuracy (OQ-CQ):** 94.81%  
**Description:**  
Best-performing version. Reverted the T1 prompt to a more general explanation with fewer syntactic constraints to increase the likelihood of selecting "Q". Maintained the strict and example-rich T2 prompt to handle the OQ vs. CQ classification. This version offers the highest accuracy so far and consistent output. 