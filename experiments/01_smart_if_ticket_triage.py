"""Experiment 1: smart if-statements in a real-time support workflow.

One incoming ticket, one schema with every output type, one batched forward pass. The
branching below uses probabilities, not just discrete answers, which is the
style of control flow the Jev post describes. We then check that a full
multi-field decision fits inside a real-time latency budget.
"""

from common import JEV_FAST_MS, SINGLE_CALL_MS, check, finish, fmt, header, latency

from decisionmodel import Bool, Enum, Extract, Float, Int, DecisionModel, MultiLabel

TICKETS = [
    "Hi, my order #48213 arrived with a cracked screen. I want a refund of $259.99 today or I will dispute the "
    "charge with my bank. You can reach me at dana.kim@example.org. Dana",
    "Hey team, just wanted to say the new dashboard is fantastic. Setup took two minutes. Thanks! Marcus",
    "I've been locked out of my account since this morning and the password reset email never arrives. "
    "I have a client demo in an hour, please help.",
    "Do you offer a discount for nonprofits? We are a team of 40 and are comparing plans for next year.",
]

SCHEMA = {
    "urgent": Bool("Does this need a response within the hour?"),
    "team": Enum("Which team should handle this ticket?", ["billing and refunds", "technical support", "sales", "customer success"]),
    "tags": MultiLabel("Which of these apply to the message?",
                       ["refund request", "damaged product", "login problem", "positive feedback", "pricing question", "threat to leave or escalate"]),
    "anger": Int("How angry is the customer, from 1 (calm) to 5 (furious)?", 1, 5),
    "negativity": Float("How negative is this text, from 0 to 1?", 0.0, 1.0),
    "damaged": Bool("Did the product arrive damaged?"),
    "order_id": Extract("What is the order number?"),
    "email": Extract("What is the customer's email address?"),
}


def route(d) -> str:
    """Plain Python control flow over typed, probabilistic values."""
    if d.urgent.p() > 0.7 and d.anger.expected >= 3.5:
        return f"PAGE on-call lead for '{d.team.value}'"
    if d.damaged.p() > 0.6 and d.order_id.value:  # a probability is a score you can threshold however you like
        return f"open a replacement claim for order {d.order_id.value}, then queue for '{d.team.value}'"
    if d.team.confidence < 0.5:
        return "send to human triage queue (model is unsure about the team)"
    if "positive feedback" in d.tags.value and d.anger.expected < 2:
        return "auto-reply with thanks, forward to product team"
    return f"queue for '{d.team.value}'"


def main():
    dm = DecisionModel.load()
    header("Experiment 1: smart if-statements for ticket triage")
    for t in TICKETS:
        d = dm.decide(t, SCHEMA)
        print(f"\nTICKET: {t[:100]}...")
        print(f"  urgent      {d.urgent.value!s:6} P(yes)={d.urgent.p():.2f}")
        print(f"  team        {d.team.value}  (confidence {d.team.confidence:.2f})")
        print(f"  tags        {d.tags.value}")
        print(f"  anger       {d.anger.value}  (expected {d.anger.expected:.2f})")
        print(f"  negativity  {d.negativity.value:.2f}")
        print(f"  damaged     {d.damaged.value!s:6} P(yes)={d.damaged.p():.2f}")
        print(f"  order_id    {d.order_id.value!r}  (confidence {d.order_id.confidence:.2f})")
        print(f"  email       {d.email.value!r}  (confidence {d.email.confidence:.2f})")
        print(f"  ACTION   -> {route(d)}")

    print("\nChecks:")
    d0, d1, d2, d3 = (dm.decide(t, SCHEMA) for t in TICKETS)
    check("ticket 1 routed to billing and refunds", d0.team.value == "billing and refunds")
    check("ticket 1 order id extracted verbatim", d0.order_id.value is not None and "48213" in d0.order_id.value, repr(d0.order_id.value))
    check("ticket 1 email extracted verbatim", d0.email.value == "dana.kim@example.org", repr(d0.email.value))
    check("ticket 2 has no order id (returns None, not a guess)", d1.order_id.value is None)
    check("ticket 2 tagged as positive feedback", "positive feedback" in d1.tags.value)
    check("only ticket 1 clears the 0.6 action threshold for a damaged product",
          d0.damaged.p() > 0.6 and all(d.damaged.p() <= 0.6 for d in (d1, d2, d3)),
          ", ".join(f"{d.damaged.p():.2f}" for d in (d0, d1, d2, d3)))
    check("ticket 1 tagged as a refund request", "refund request" in d0.tags.value)
    check("ticket 2 is the least negative", d1.negativity.value == min(d.negativity.value for d in (d0, d1, d2, d3)))
    check("ticket 2 calmer than ticket 1", d1.anger.expected < d0.anger.expected, f"{d1.anger.expected:.2f} < {d0.anger.expected:.2f}")
    check("ticket 3 is urgent and technical", d2.urgent.value and d2.team.value == "technical support")
    check("ticket 4 routed to sales", d3.team.value == "sales")
    check("extracted values are substrings of the input", all(
        d[f].value is None or d[f].value in t for t, d in zip(TICKETS, (d0, d1, d2, d3)) for f in ("order_id", "email")))

    lat = latency(lambda: dm.decide(TICKETS[0], SCHEMA), runs=200)
    print(f"\n  latency for all 8 fields at once: {fmt(lat)}")
    check(f"8-field decision p95 under {JEV_FAST_MS:.0f} ms (fast end of Jev's 70 to 500 ms range)", lat["p95"] < JEV_FAST_MS)
    lat1 = latency(lambda: dm.check(TICKETS[2], "Is the customer locked out of their account?"), runs=200)
    print(f"  latency for a single smart if:    {fmt(lat1)}")
    check(f"single predicate p95 under {SINGLE_CALL_MS:.0f} ms", lat1["p95"] < SINGLE_CALL_MS)
    finish()


if __name__ == "__main__":
    main()
