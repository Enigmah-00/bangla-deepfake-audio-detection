# Judge Q&A Preparation

## Q1. Why handcrafted features and XGBoost instead of a large deep model?

This approach is efficient, CPU-friendly, small enough for practical deployment, and easier to explain through acoustic feature groups. It is a strong baseline for the available dataset. Future work includes comparison with larger speech representation models.

## Q2. Is 99.57% realistic?

It is the notebook random held-out split result. We do not claim it is universal field accuracy. Speaker-disjoint, cross-generator, and real-world validation are important next steps.

## Q3. How is AI central to the project?

The main function depends on acoustic feature extraction, learned scaling, and the trained XGBoost classifier. The UI, report, visualizations, and uncertainty logic are all built around model output.

## Q4. Why is this relevant to Bangladesh?

Bangla has fewer accessible detection tools, and voice messages are widely used in Bangladesh. Manipulated audio can affect scams, journalism, elections, public trust, and security communication.

## Q5. Does the system identify who is speaking?

No. It predicts whether acoustic patterns resemble real or fake classes in its training data. It does not perform speaker identification or identity verification.

## Q6. Can the result be used in court?

No. It is an AI-assisted screening tool. Legal or forensic use requires validated procedures, chain of custody, expert analysis, and stronger external validation.

## Q7. What happens to uploaded audio?

The app processes audio for the request. It is not intentionally retained, not used for retraining, and not included in the downloadable report.

## Q8. What is original about the project?

The project combines a Bangla-specific application context, national-security and misinformation relevance, deployable web workflow, acoustic explanation, uncertainty-aware communication, and privacy-conscious design.

