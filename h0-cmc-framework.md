---
name: h0-cmc-framework
description: "H0-CMC (Causal Multimodal Convergence) — the refined Track C validation framework: 4-stage evidence ladder, explicit-vs-implicit model test, social axis, 6 evidence tiers, counterfactual playback"
metadata: 
  node_type: memory
  type: project
  originSessionId: d6b69a37-ed66-436a-aafa-b9633f25df0f
---

**H0-CMC: Causal Multimodal Convergence** — the refined H0-Unified framework (user
directive, 2026-06-23). Supersedes "semantic meaning" with "candidate functional
interpretation."

**One-line:** *Animal vocalisations should be tested as structured acoustic cues whose
functional interpretation is validated by receiver responses under controlled social and
behavioral contexts — not decoded as words from acoustics alone.*

**4-stage evidence ladder:**
- L0: acoustic association (motif/fingerprint/calltype enrichment)
- L1: contextual prediction (acoustic + social predicts context > acoustic alone)
- L2: receiver response (playback elicits behavioral/vocal/physiological convergence)
- L3: causal specificity (manipulated stimulus changes receiver response predictively)

**Explicit-vs-implicit model test** (plugs directly into bat §9 A-G):
- Explicit acoustic code: A(acoustic) ≈ G(full) → signal is self-contained
- Socially grounded implicit: D(acoustic+social) ≫ A, D ≈ G → social context needed
- Bat result: D=4.3× ≈ G=4.7×, A=2.7× → **socially grounded implicit communication**

**Social axis separate from generic context** (bat §9 proved social is the true complement):
Context = social_participants + behavioral/environmental + individual_state. Social
(emitter/addressee/relationship/familiarity/audience) gets its own axis because §9 showed
it carries 91% of the multimodal synergy.

**6 evidence tiers:**
- T0: acoustic motif/enrichment → "recurrent acoustic structure"
- T1: acoustic + social/context prediction → "context-associated cue"
- T2: playback response (one modality) → "receiver detects distinction"
- T3: multimodal convergence → "candidate functional interpretation"
- T4: counterfactual playback + cross-context generalization → "causally supported signal"
- T5: field replication / fitness-relevant behavior → "ecologically validated function"

**Formula:** Evidence(M|V,C) = S_acoustic(V) + S_social(V,C) + R_behavioral + R_vocal +
R_physiological + I_causal(V,C). Candidate function supported when discovered structure +
social model + receiver convergence + causal playback all agree.

**Counterfactual controls:** natural / motif-preserved / order-shuffled / caller-swapped /
context-mismatched / onset-body-offset-swapped / amplitude-duration-matched.

**How to apply:** the computational pipeline (fingerprint + HECB + DUSTED + Track C)
operates at T0–T1. Playback experiments move to T2–T4. The pipeline's job is candidate
generation + context modeling; playback is receiver-side validation. `semantic_validation`
stays `none` until T3+ convergence. See [[cpc-policy]] for representation selection.
