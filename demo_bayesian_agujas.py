#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example demonstrating the three AGUJAS Bayesian inference capabilities
without requiring full framework dependencies
"""


def demonstrate_aguja_i():
    """
    AGUJA I: Adaptive Bayesian Prior for Causal Links

    Shows how causal link strength is computed using Bayesian inference
    instead of fixed values.
    """
    print("=" * 70)
    print("AGUJA I: El Prior Informado Adaptativo")
    print("=" * 70)

    # Simulate evidence for a causal link MP-001 → MR-001
    evidence_components = {
        "semantic_distance": 0.85,  # High similarity in embeddings
        "type_transition_prior": 0.80,  # producto → resultado is common
        "language_specificity": 0.70,  # Moderate causal language
        "temporal_coherence": 0.85,  # Logical verb sequence
        "financial_consistency": 0.60,  # Some budget alignment
        "textual_proximity": 0.75,  # Frequently mentioned together
    }

    # Weighted composite likelihood
    weights = {
        "semantic_distance": 0.25,
        "type_transition_prior": 0.20,
        "language_specificity": 0.20,
        "temporal_coherence": 0.15,
        "financial_consistency": 0.10,
        "textual_proximity": 0.10,
    }

    likelihood = sum(evidence_components[k] * weights[k] for k in weights)

    # Initialize prior (Beta distribution)
    prior_mean = 0.80  # Based on type transition
    prior_alpha = prior_mean * 4.0
    prior_beta = (1 - prior_mean) * 4.0

    # Bayesian update
    posterior_alpha = prior_alpha + likelihood
    posterior_beta = prior_beta + (1 - likelihood)

    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    posterior_var = (posterior_alpha * posterior_beta) / (
        (posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1)
    )
    posterior_std = posterior_var**0.5

    print("\nCausal Link: MP-001 → MR-001")
    print("\nEvidence Components:")
    for component, value in evidence_components.items():
        print(f"  - {component:25s}: {value:.2f} (weight: {weights[component]:.2f})")

    print(f"\nComposite Likelihood: {likelihood:.3f}")

    print("\nPrior Distribution:")
    print(f"  - Mean: {prior_mean:.3f}")
    print(f"  - Alpha: {prior_alpha:.3f}, Beta: {prior_beta:.3f}")

    print("\nPosterior Distribution:")
    print(f"  - Mean: {posterior_mean:.3f}")
    print(f"  - Std Dev: {posterior_std:.3f}")
    print(
        f"  - 95% Credible Interval: [{posterior_mean - 1.96 * posterior_std:.3f}, {posterior_mean + 1.96 * posterior_std:.3f}]"
    )

    print("\n✨ Instead of fixed 0.8, we have:")
    print(f"   P(causal_link) = {posterior_mean:.3f} ± {posterior_std:.3f}")
    print()


def demonstrate_aguja_ii():
    """
    AGUJA II: Hierarchical Bayesian Mechanism Inference

    Shows how mechanism types are inferred and validated.
    """
    print("=" * 70)
    print("AGUJA II: El Modelo Generativo de Mecanismos")
    print("=" * 70)

    # Hyperprior: Domain-level mechanism type distribution
    hyperprior = {
        "administrativo": 0.30,
        "técnico": 0.25,
        "financiero": 0.20,
        "político": 0.15,
        "mixto": 0.10,
    }

    # Observed verbs in product node MP-001
    observed_verbs = ["planificar", "coordinar", "supervisar"]

    # Typical verbs by mechanism type
    typical_verbs = {
        "administrativo": ["planificar", "coordinar", "gestionar", "supervisar"],
        "técnico": ["diagnosticar", "diseñar", "implementar", "evaluar"],
        "financiero": ["asignar", "ejecutar", "auditar", "reportar"],
        "político": ["concertar", "negociar", "aprobar", "promulgar"],
    }

    print("\nNode: MP-001 (Producto)")
    print(f"Observed Verbs: {observed_verbs}")

    # Bayesian update for each mechanism type
    posterior = {}
    for mech_type, prior_prob in hyperprior.items():
        if mech_type == "mixto":
            posterior[mech_type] = prior_prob
            continue

        typical = set(typical_verbs[mech_type])
        observed = set(observed_verbs)
        overlap = len(observed & typical)
        total = len(typical)

        # Likelihood: proportion of typical verbs observed (with Laplace smoothing)
        likelihood = (overlap + 1) / (total + 2)

        # Bayesian update
        posterior[mech_type] = prior_prob * likelihood

    # Normalize
    total = sum(posterior.values())
    posterior = {k: v / total for k, v in posterior.items()}

    print("\nMechanism Type Posterior:")
    for mech_type, prob in sorted(posterior.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"  {mech_type:15s}: {prob:.3f} {bar}")

    # Calculate uncertainty (entropy)
    import math

    entropy = -sum(p * math.log(p + 1e-10) for p in posterior.values() if p > 0)
    max_entropy = math.log(len(posterior))
    uncertainty = entropy / max_entropy

    print("\nUncertainty Metrics:")
    print(f"  - Entropy: {entropy:.3f}")
    print(f"  - Normalized Uncertainty: {uncertainty:.3f}")
    print(f"  - Confidence: {1 - uncertainty:.3f}")

    # Sufficiency test
    has_entity = True  # Assume entity is specified
    has_activities = len(observed_verbs) >= 2
    has_resources = True  # Assume budget is allocated

    sufficiency = (
        (0.4 if has_entity else 0.0)
        + (0.4 if has_activities else 0.0)
        + (0.2 if has_resources else 0.0)
    )

    print("\nSufficiency Test:")
    # Since has_entity and has_resources are always True in this demo,
    # we display them as constants
    print("  - Has Entity: True (✓)")
    print(
        f"  - Has Activities (≥2): {has_activities} ({'✓' if has_activities else '✗'})"
    )
    print("  - Has Resources: True (✓)")
    print(
        f"  - Sufficiency Score: {sufficiency:.2f} ({'SUFFICIENT' if sufficiency >= 0.6 else 'INSUFFICIENT'})"
    )
    print()


def _calculate_severity(p_failure):
    """Calculate severity level based on failure probability"""
    if p_failure > 0.15:
        return "CRITICAL"
    elif p_failure > 0.10:
        return "HIGH"
    else:
        return "MEDIUM"


def _print_omission_audit(omissions):
    """Print direct evidence audit for omissions"""
    print("\nDirect Evidence Audit (Layer 1):")
    for omission, p_failure, effort in omissions:
        severity = _calculate_severity(p_failure)
        print(f"  - Missing {omission:12s}: P(failure) = {p_failure:.3f} [{severity}]")


def _print_causal_implications(node):
    """Print causal implications of missing attributes"""
    print("\nCausal Implications (Layer 2):")
    if not node["has_baseline"]:
        print("  - Missing baseline → P(target_miscalibrated) = 0.73")
        print("    'Without baseline, target likely poorly calibrated'")

    if not node["has_entity"]:
        if node.get("budget_high", False):
            print("  - Missing entity + high budget → P(implementation_failure) = 0.89")
            print("    'Large budget without clear responsibility = high risk'")
        else:
            print("  - Missing entity → P(implementation_failure) = 0.65")
            print("    'Unclear responsibility threatens execution'")


def _calculate_remediations(omissions):
    """Calculate and prioritize remediations based on expected value"""
    remediations = []
    for omission, p_failure, effort in omissions:
        # EVI = Expected value of fixing / Effort
        impact = p_failure * 1.5  # Assume some cascading effect
        evi = impact / effort
        remediations.append((omission, p_failure, effort, impact, evi))

    remediations.sort(key=lambda x: x[4], reverse=True)
    return remediations


def _print_remediation_priority(remediations):
    """Print prioritized remediation table"""
    print("\nOptimal Remediation (Layer 3):")
    print("\n  Priority | Omission   | P(failure) | Effort | Impact | EVI")
    print("  ---------|------------|------------|--------|--------|-------")
    for i, (omission, p_fail, effort, impact, evi) in enumerate(remediations, 1):
        print(
            f"     {i}     | {omission:10s} | {p_fail:10.3f} | {effort:6d} | {impact:6.3f} | {evi:.3f}"
        )


def _print_systemic_assessment(total_omissions, total_possible):
    """Print system-wide success probability assessment"""
    completeness = 1.0 - (total_omissions / total_possible)
    success_probability = 0.70 * completeness  # Base rate * completeness

    print("\nSystemic Assessment:")
    print(f"  - Documentation Completeness: {completeness:.1%}")
    print(f"  - Estimated Success Probability: {success_probability:.1%}")
    print(
        f"  - Recommendation: {'FIX CRITICAL OMISSIONS' if success_probability < 0.60 else 'Acceptable'}"
    )


def demonstrate_aguja_iii():
    """
    AGUJA III: Bayesian Counterfactual Auditing

    Shows how omissions are detected and prioritized using causal reasoning.
    """
    print("=" * 70)
    print("AGUJA III: El Auditor Contrafactual Bayesiano")
    print("=" * 70)

    # Historical success rates
    historical_priors = {
        "entity_presence": 0.94,
        "baseline_presence": 0.89,
        "target_presence": 0.92,
        "budget_presence": 0.78,
        "mechanism_presence": 0.65,
    }

    # Example node with some omissions
    node = {
        "id": "MP-001",
        "has_baseline": False,
        "has_target": True,
        "has_entity": False,
        "has_budget": True,
        "has_mechanism": True,
    }

    print(f"\nNode: {node['id']}")
    print("Attributes:")
    for attr in ["baseline", "target", "entity", "budget", "mechanism"]:
        has = node[f"has_{attr}"]
        print(f"  - {attr:12s}: {'✓ Present' if has else '✗ MISSING'}")

    # Calculate failure probabilities for omissions
    omissions = []
    if not node["has_baseline"]:
        p_failure = 1.0 - historical_priors["baseline_presence"]
        omissions.append(("baseline", p_failure, 3))  # (name, failure_prob, effort)

    if not node["has_entity"]:
        p_failure = 1.0 - historical_priors["entity_presence"]
        omissions.append(("entity", p_failure, 2))

    _print_omission_audit(omissions)

    _print_causal_implications(node)

    # Calculate Expected Value of Information for prioritization
    remediations = _calculate_remediations(omissions)

    _print_remediation_priority(remediations)

    # System-wide success probability
    total_omissions = len(omissions)
    total_possible = 5  # 5 key attributes
    _print_systemic_assessment(total_omissions, total_possible)
    print()


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 70)
    print("FARFAN 2.0 - Bayesian Inference Demonstration")
    print("Three AGUJAS (Needles) Implementation")
    print("=" * 70)
    print()

    demonstrate_aguja_i()
    input("Press Enter to continue to AGUJA II...")
    print()

    demonstrate_aguja_ii()
    input("Press Enter to continue to AGUJA III...")
    print()

    demonstrate_aguja_iii()

    print("=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print(
        "  1. AGUJA I: Causal links now have probability distributions, not fixed values"
    )
    print("  2. AGUJA II: Mechanisms are inferred with uncertainty quantification")
    print("  3. AGUJA III: Omissions are prioritized by expected impact")
    print(
        "\nThe framework has evolved from deterministic to probabilistic reasoning! 🎯"
    )
    print()


if __name__ == "__main__":
    main()
