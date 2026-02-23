"""
data_generator.py
=================
Simulates realistic SaaS/E-commerce user behavioral data.

Product Thinking:
- Users don't convert immediately; there's a natural funnel drop-off
- Acquisition channels have different quality (organic > paid in LTV)
- Device type influences conversion (desktop converts better)
- A/B groups are randomly assigned at signup
- Revenue follows a power-law distribution (few high-value users)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
N_USERS       = 20_000
START_DATE    = datetime(2023, 1, 1)
END_DATE      = datetime(2024, 6, 30)
DATE_RANGE    = (END_DATE - START_DATE).days

CHANNELS      = ["organic", "paid_search", "social", "email", "referral"]
CHANNEL_W     = [0.30, 0.25, 0.20, 0.15, 0.10]   # acquisition weights

DEVICES       = ["desktop", "mobile", "tablet"]
DEVICE_W      = [0.45, 0.42, 0.13]

# Funnel step conversion probabilities (conditional on previous step)
FUNNEL_PROBS  = {
    "login":        0.82,
    "view_product": 0.70,
    "add_to_cart":  0.45,
    "purchase":     0.38,
}

# Channel multipliers on purchase probability
CHANNEL_MULT  = {
    "organic":     1.20,
    "paid_search": 0.95,
    "social":      0.85,
    "email":       1.10,
    "referral":    1.15,
}

# Device multipliers on purchase probability
DEVICE_MULT   = {
    "desktop": 1.15,
    "mobile":  0.88,
    "tablet":  0.97,
}


def _random_date(start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    return start + timedelta(days=int(np.random.randint(0, delta)))


def generate_users() -> pd.DataFrame:
    """
    Create a user-level DataFrame with demographic/acquisition attributes.
    Each user is assigned a signup date, channel, device, and A/B group.
    """
    user_ids   = np.arange(1, N_USERS + 1)
    signup_dates = pd.to_datetime([
        START_DATE + timedelta(days=int(d))
        for d in np.random.randint(0, DATE_RANGE, N_USERS)
    ])
    channels   = np.random.choice(CHANNELS, N_USERS, p=CHANNEL_W)
    devices    = np.random.choice(DEVICES,  N_USERS, p=DEVICE_W)
    ab_groups  = np.random.choice(["A", "B"], N_USERS)   # 50/50 split

    return pd.DataFrame({
        "user_id":           user_ids,
        "signup_date":       signup_dates,
        "acquisition_channel": channels,
        "device_type":       devices,
        "experiment_group":  ab_groups,
    })


def generate_events(users: pd.DataFrame) -> pd.DataFrame:
    """
    For each user, simulate a sequence of funnel events after signup.

    Product Thinking:
    - Events happen within 30 days of signup (engagement window)
    - Each funnel step is conditional on completing the previous one
    - A/B group B has a 10% uplift in purchase conversion (simulated experiment)
    - Revenue only generated on 'purchase' events
    - Revenue ~ LogNormal to mimic real spending distributions
    """
    records = []

    for _, user in users.iterrows():
        uid      = user["user_id"]
        signup   = user["signup_date"]
        channel  = user["acquisition_channel"]
        device   = user["device_type"]
        ab       = user["experiment_group"]

        # ── Signup event ─────────────────────────────────────────────────────
        records.append({
            "user_id":            uid,
            "event_date":         signup,
            "event_type":         "signup",
            "revenue":            0.0,
            "device_type":        device,
            "acquisition_channel": channel,
            "experiment_group":   ab,
        })

        # ── Funnel events (conditional chain) ────────────────────────────────
        base_mult = CHANNEL_MULT[channel] * DEVICE_MULT[device]
        ab_mult   = 1.10 if ab == "B" else 1.0   # B variant improves purchase

        event_date = signup + timedelta(days=int(np.random.randint(1, 3)))

        funnel = ["login", "view_product", "add_to_cart", "purchase"]
        for step in funnel:
            prob = FUNNEL_PROBS[step] * base_mult
            if step == "purchase":
                prob *= ab_mult
            prob = min(prob, 0.98)   # cap at 98%

            if np.random.random() < prob:
                # Advance time by 0–3 days per step
                event_date += timedelta(days=int(np.random.randint(0, 4)))
                if event_date > END_DATE:
                    break

                revenue = 0.0
                if step == "purchase":
                    # LogNormal revenue: median ~$49, some high-value outliers
                    revenue = float(np.random.lognormal(mean=3.9, sigma=0.8))

                records.append({
                    "user_id":            uid,
                    "event_date":         event_date,
                    "event_type":         step,
                    "revenue":            revenue,
                    "device_type":        device,
                    "acquisition_channel": channel,
                    "experiment_group":   ab,
                })
            else:
                break   # Drop-off: user doesn't proceed further

    return pd.DataFrame(records)


def load_data() -> pd.DataFrame:
    """
    Entry point: generates and merges users + events into a single DataFrame.
    Returns a clean, analysis-ready events DataFrame.
    """
    print("⚙️  Generating user data...")
    users  = generate_users()
    print("⚙️  Simulating events (this may take ~5s)...")
    events = generate_events(users)
    events["event_date"]  = pd.to_datetime(events["event_date"])
    events["signup_date"] = events.groupby("user_id")["event_date"].transform("min")
    print(f"✅  Generated {len(events):,} events for {N_USERS:,} users.")
    return events
