"""Deterministic synthetic non-BIRD content for the L0-PAD token-matched control.

The L0-PAD condition tests whether the L0→L4 BEX uplift is driven by *semantic*
metadata or merely by *prompt length*. To do that cleanly we need an L0 prompt
padded to roughly the L4 token count with content that:
  1. has the same surface character (looks like a domain-knowledge or
     data-dictionary section the model would expect),
  2. contains zero BIRD-relevant signal (so any uplift is attributable to length,
     not to accidental cue-leakage),
  3. is fully reproducible (fixed seed, no network access, no per-run randomness).

This module returns a `dict[topic, content]` shape that
`gyzasql.semantic_layer.context.populate_domain_knowledge` accepts directly.
The generator picks topics from a pool of non-BIRD enterprise domains
(logistics, hospitality, ad-tech, etc.) so the model is unlikely to
hallucinate that any of these tables exist in the actual schema.

Token-matching: the engine's prompt-builder doesn't expose its token count
to us pre-call, so we approximate via a 4-chars-per-token ratio. This could be
refined against tiktoken estimates of the real L4 prompts per database.
"""

from __future__ import annotations

import hashlib
import random


# Pool of non-BIRD enterprise topics. Each topic has a short list of ASCII-only
# narrative paragraphs; the generator concatenates and trims to hit target_tokens.
_TOPICS: dict[str, list[str]] = {
    "warehouse_inventory_basics": [
        "Inventory units flow from inbound receiving through put-away into bin "
        "storage. The system tracks lot numbers, expiration windows, and SKU "
        "rotation policies. FEFO (first-expired-first-out) and FIFO ordering "
        "produce different pick lists; the warehouse rule engine selects the "
        "policy per item class.",
        "A receipt event creates a putaway-pending row; once an operator scans "
        "the destination bin, the row transitions to in-stock with a quantity, "
        "lot reference, and bin coordinate. Cycle counts reconcile the system "
        "quantity against the physical count and produce variance entries that "
        "downstream auditing reports aggregate by SKU class.",
    ],
    "fleet_dispatch_routing": [
        "Each shift the dispatcher generates a route plan from outstanding "
        "delivery tickets. The optimizer accepts vehicle capacity, driver "
        "hours-of-service constraints, and customer time windows. Output is a "
        "sequence of stops with estimated arrival times. Real-time GPS pings "
        "update arrival predictions as the day progresses.",
        "Vehicle telematics streams report engine hours, fuel level, and brake "
        "events. Maintenance thresholds trigger a service order when engine "
        "hours exceed the next scheduled interval. Drivers with three or more "
        "harsh-braking events in a 30-day window receive a coaching flag in "
        "the safety dashboard.",
    ],
    "loyalty_tier_calculation": [
        "Customers earn points on qualifying spend. Tier eligibility is "
        "evaluated quarterly: a rolling 12-month spend total over the silver "
        "threshold elevates the customer to silver; gold and platinum follow "
        "the same pattern with their own thresholds. Tier downgrades are "
        "delayed one quarter to smooth seasonal spend volatility.",
        "Promotion engines target by tier: silver members receive 10% off "
        "their second qualifying purchase per quarter, gold members receive "
        "free expedited shipping plus the silver benefit, and platinum "
        "members add a dedicated concierge contact channel. All promo rules "
        "are stored in a versioned campaign table with an effective_from "
        "and effective_to timestamp.",
    ],
    "ad_campaign_pacing": [
        "Each campaign has a daily budget and a delivery curve. The bidder "
        "monitors spend by hour against the curve; if spend lags by more than "
        "two standard deviations of historical hourly spend, bid floors are "
        "lifted to accelerate delivery. If spend leads, floors are dropped "
        "to slow delivery and preserve budget for the remaining day.",
        "Creative rotation balances exploration and exploitation. New creatives "
        "receive a fixed share of impressions for the first three days to "
        "gather statistical signal; afterward, allocation is proportional to "
        "click-through rate scaled by recency. Underperforming creatives drop "
        "out at the end of week two unless flagged as evergreen.",
    ],
    "ticket_audit_workflow": [
        "Every issued ticket flows through three audit stages: automatic "
        "validation, manager review, and an exception queue for cases the "
        "rules flagged but the manager did not resolve. Stage transitions "
        "produce an audit-trail event with actor, timestamp, and decision "
        "code. The audit ledger is append-only and signed daily.",
        "Exception cases are routed by reason code: pricing mismatch, "
        "duplicate ticket, eligibility failure, or vendor override. Each "
        "reason has a different SLA: pricing mismatches resolve within "
        "24 hours, duplicates within 4 hours, eligibility within 48 hours, "
        "vendor overrides on a manual basis with a trailing weekly report.",
    ],
    "concert_seat_inventory": [
        "Venues are modeled as a tree of section, row, and seat nodes. "
        "Each seat has a price tier and a release schedule that controls "
        "when it becomes available. Holds (e.g. for fan-club presale, "
        "promoter, or accessible-seating reserves) are scoped to a "
        "release window and decrement the available-pool count.",
        "Pricing is dynamic per section: the engine moves prices in $5 "
        "increments based on recent sales velocity. A floor and ceiling "
        "guard against overcorrection. Past-show data anchors the initial "
        "curve; ongoing sales feed back into a per-show velocity index.",
    ],
}


_SEPARATOR = "\n\n"


def _hash_seed(dataset_name: str, seed: int) -> int:
    """Combine `seed` with the dataset name into a deterministic 32-bit seed."""
    h = hashlib.sha256(f"{dataset_name}|{seed}".encode()).hexdigest()
    return int(h[:8], 16)


def get_pad_domain_knowledge(
    dataset_name: str,
    target_tokens: int = 4000,
    seed: int = 42,
) -> dict[str, str]:
    """Return a dict[topic_label, content] suitable for `populate_domain_knowledge`.

    The output's combined character length targets `target_tokens × 4` chars
    (rough English word→token ratio). The shape matches what
    `_load_domain_knowledge()` returns so the engine treats it as a normal
    Domain Knowledge section.

    Args:
        dataset_name: BIRD database name (e.g. ``"financial"``). Combined with
            ``seed`` to produce reproducible per-database padding.
        target_tokens: approximate token budget for the pad content. Default
            is 4000; it could be tuned per-database by computing the real L4
            prompt token count first.
        seed: fixed seed for the deterministic shuffle. Default 42.

    Returns:
        Mapping from a synthetic topic label to a paragraph of non-BIRD prose.
        Each call with the same ``(dataset_name, target_tokens, seed)`` returns
        an identical dict; tested in a dry-run smoke.
    """
    rng = random.Random(_hash_seed(dataset_name, seed))
    topic_keys = list(_TOPICS.keys())
    rng.shuffle(topic_keys)

    target_chars = target_tokens * 4
    out: dict[str, str] = {}
    total_chars = 0
    cycle = 0
    while total_chars < target_chars and cycle < 8:
        for k in topic_keys:
            paragraphs = list(_TOPICS[k])
            rng.shuffle(paragraphs)
            content = _SEPARATOR.join(paragraphs)
            label = f"{k}_v{cycle}" if cycle else k
            out[label] = content
            total_chars += len(content) + len(label)
            if total_chars >= target_chars:
                break
        cycle += 1
    return out


__all__ = ["get_pad_domain_knowledge"]
