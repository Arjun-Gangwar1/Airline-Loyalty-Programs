"""
Retention recommendation engine.

Converts churn probability scores into specific, actionable
next-best-actions per customer — with offer, channel, timing,
and revenue impact.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


OFFERS = {
    'Premium Loyalists': {
        'low':    {'action': 'Monitor + VIP newsletter',
                   'offer':  'Early access to new routes',
                   'channel': 'App notification', 'timing': 'Monthly'},
        'medium': {'action': 'Tier extension',
                   'offer':  'Tier status extended 6 months + 5K bonus miles',
                   'channel': 'Personalized email', 'timing': 'Within 14 days'},
        'high':   {'action': 'Executive intervention',
                   'offer':  'Tier extended 12m + 15K miles + lounge upgrade',
                   'channel': 'Personal call + email', 'timing': 'Within 48 hours'},
    },
    'Silent Drifters': {
        'low':    {'action': 'Gentle re-engagement',
                   'offer':  'We miss you — 2K bonus miles on next booking',
                   'channel': 'Email', 'timing': 'Within 21 days'},
        'medium': {'action': 'Win-back campaign',
                   'offer':  '5K bonus miles + tier extension 3 months',
                   'channel': 'Email + SMS', 'timing': 'Within 7 days'},
        'high':   {'action': 'Urgent reactivation',
                   'offer':  '8K bonus miles + waived change fees for 60 days',
                   'channel': 'SMS + Email + App push', 'timing': 'Within 48 hours'},
    },
    'Miles Hoarders': {
        'low':    {'action': 'Redemption nudge',
                   'offer':  'Use 5K miles this month, earn 1K bonus',
                   'channel': 'Email', 'timing': 'Within 30 days'},
        'medium': {'action': 'Redemption campaign',
                   'offer':  '"Double value" miles redemption event for 2 weeks',
                   'channel': 'Email + App', 'timing': 'Within 14 days'},
        'high':   {'action': 'Expiry warning + rescue offer',
                   'offer':  'Miles expiring soon + 10K bonus for first redemption',
                   'channel': 'Email + SMS', 'timing': 'Within 7 days'},
    },
    'Seasonal Travelers': {
        'low':    {'action': 'Seasonal newsletter',
                   'offer':  'Coming-season destination highlights',
                   'channel': 'Email', 'timing': '4 weeks before peak season'},
        'medium': {'action': 'Seasonal pre-booking offer',
                   'offer':  'Book now, earn 3X miles on summer flights',
                   'channel': 'Email + App', 'timing': '6 weeks before peak season'},
        'high':   {'action': 'Targeted seasonal win-back',
                   'offer':  'Exclusive package: flight + hotel + 8K bonus miles',
                   'channel': 'Email + SMS', 'timing': 'Immediately'},
    },
    'Rising Stars': {
        'low':    {'action': 'Tier progress nudge',
                   'offer':  'You are X miles from Silver status!',
                   'channel': 'App push', 'timing': 'Monthly'},
        'medium': {'action': 'Tier challenge',
                   'offer':  'Complete 3 flights in 60 days, earn Silver status',
                   'channel': 'Email + App', 'timing': 'Within 14 days'},
        'high':   {'action': 'Accelerator offer',
                   'offer':  'Double miles on all flights this month',
                   'channel': 'Email + SMS + App', 'timing': 'Within 7 days'},
    },
    'Budget Frequent Flyers': {
        'low':    {'action': 'Partner promotion',
                   'offer':  'Earn 2X miles with hotel partner this month',
                   'channel': 'Email', 'timing': 'Monthly'},
        'medium': {'action': 'Value bundle',
                   'offer':  'Discounted upgrade + bonus miles combo',
                   'channel': 'Email + App', 'timing': 'Within 14 days'},
        'high':   {'action': 'Loyalty lock-in',
                   'offer':  'Co-branded card offer: 10K signup bonus miles',
                   'channel': 'Email + Direct mail', 'timing': 'Within 7 days'},
    },
    'At-Risk VIPs': {
        'low':    {'action': 'VIP check-in',
                   'offer':  'Complimentary lounge visit invitation',
                   'channel': 'Personal email', 'timing': 'Within 21 days'},
        'medium': {'action': 'VIP rescue',
                   'offer':  'Tier extension + 10K miles + premium lounge access',
                   'channel': 'Personal call + email', 'timing': 'Within 7 days'},
        'high':   {'action': 'Executive outreach',
                   'offer':  'Dedicated account manager + 20K miles + tier guarantee',
                   'channel': 'Phone + Personal email', 'timing': 'Within 24 hours'},
    },
}

RETENTION_LIFT = {
    'high':   0.28,
    'medium': 0.18,
    'low':    0.08,
}


class RetentionEngine:
    """Turn churn predictions into specific retention actions."""

    def __init__(self):
        self.final_path   = Path("data/final")
        self.reports_path = Path("outputs/reports")
        self.fig_path     = Path("outputs/figures/retention")
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.fig_path.mkdir(parents=True, exist_ok=True)

    # ── Data loading ─────────────────────────────────────────────────────────

    def load_predictions(self):
        fp = self.final_path / 'all_customer_predictions.csv'
        if not fp.exists():
            fp = self.final_path / 'churn_predictions.csv'
        self.df = pd.read_csv(fp)
        print(f"Loaded {len(self.df):,} customer prediction records.")

    # ── Recommendations ───────────────────────────────────────────────────────

    def build_recommendations(self) -> pd.DataFrame:
        if 'segment_name' not in self.df.columns:
            self.df['segment_name'] = 'Silent Drifters'
        if 'risk_level' not in self.df.columns:
            self.df['risk_level'] = pd.cut(
                self.df['churn_probability'],
                bins=[0, 0.3, 0.6, 1.0], labels=['low', 'medium', 'high']
            )

        actions = []
        for _, row in self.df.iterrows():
            segment = row.get('segment_name', 'Silent Drifters')
            risk    = str(row.get('risk_level', 'medium')).lower()
            clv     = float(row.get('clv', 1000))
            prob    = float(row.get('churn_probability', 0.5))

            offer_map = OFFERS.get(segment, OFFERS['Silent Drifters'])
            offer     = offer_map.get(risk, offer_map['high'])
            lift      = RETENTION_LIFT.get(risk, 0.18)

            revenue_at_risk   = round(clv * prob, 2)
            potential_save    = round(revenue_at_risk * lift, 2)

            # Priority score: 40% probability + 40% CLV percentile + 20% lift
            actions.append({
                **{c: row[c] for c in row.index if c in
                   ['loyalty_number', 'segment_name', 'risk_level',
                    'churn_probability', 'clv', 'predicted_churn']},
                'recommended_action':  offer['action'],
                'offer':               offer['offer'],
                'channel':             offer['channel'],
                'timing':              offer['timing'],
                'retention_lift':      lift,
                'revenue_at_risk':     revenue_at_risk,
                'potential_save':      potential_save,
            })

        self.actions = pd.DataFrame(actions)

        # Priority score (normalised)
        if 'churn_probability' in self.actions.columns and 'clv' in self.actions.columns:
            p_norm   = self.actions['churn_probability']
            clv_norm = self.actions['clv'] / self.actions['clv'].max().clip(lower=1)
            lift_norm = self.actions['retention_lift'] / RETENTION_LIFT['high']
            self.actions['priority_score'] = (0.4 * p_norm + 0.4 * clv_norm + 0.2 * lift_norm).round(4)
            self.actions = self.actions.sort_values('priority_score', ascending=False)

        print(f"Built {len(self.actions):,} retention recommendations.")
        return self.actions

    # ── Business summary ──────────────────────────────────────────────────────

    def print_summary(self):
        if not hasattr(self, 'actions'):
            return
        total_at_risk  = self.actions['revenue_at_risk'].sum()
        total_saveable = self.actions['potential_save'].sum()
        at_risk_count  = (self.df.get('predicted_churn', self.df.get('churn', pd.Series())) == 1).sum()

        print(f"\n{'='*55}")
        print("  RETENTION ENGINE — BUSINESS SUMMARY")
        print(f"{'='*55}")
        print(f"  At-risk customers : {at_risk_count:,}")
        print(f"  Revenue at risk   : ${total_at_risk:,.0f}")
        print(f"  Potential savings : ${total_saveable:,.0f}")
        print(f"{'='*55}\n")

    # ── Visualisations ────────────────────────────────────────────────────────

    def plot_revenue_at_risk(self):
        if not hasattr(self, 'actions') or 'segment_name' not in self.actions.columns:
            return
        rev = self.actions.groupby('segment_name')['revenue_at_risk'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        rev.plot(kind='bar', ax=ax, color='coral', alpha=0.8)
        ax.set_title('Revenue at Risk by Segment')
        ax.set_xlabel('Segment')
        ax.set_ylabel('Revenue at Risk ($)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.fig_path / 'revenue_at_risk.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_churn_distribution(self):
        if 'churn_probability' not in self.df.columns:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        self.df['churn_probability'].hist(bins=30, ax=ax, color='steelblue', alpha=0.7)
        ax.axvline(0.3, color='orange', linestyle='--', label='Low threshold (0.3)')
        ax.axvline(0.6, color='red',    linestyle='--', label='High threshold (0.6)')
        ax.set_title('Churn Probability Distribution')
        ax.set_xlabel('Churn Probability')
        ax.set_ylabel('Customers')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.fig_path / 'churn_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self):
        self.actions.to_csv(self.reports_path / 'retention_actions.csv', index=False)
        high_priority = self.actions.head(100)
        high_priority.to_csv(self.reports_path / 'priority_action_list.csv', index=False)

        playbook = {seg: {risk: OFFERS.get(seg, {}).get(risk, {})
                          for risk in ['low', 'medium', 'high']}
                    for seg in OFFERS}
        with open(self.reports_path / 'retention_playbook.json', 'w') as f:
            json.dump(playbook, f, indent=2)

        print("Saved: retention_actions.csv, priority_action_list.csv, retention_playbook.json")

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        self.load_predictions()
        self.build_recommendations()
        self.print_summary()
        self.plot_revenue_at_risk()
        self.plot_churn_distribution()
        self.save()
        return self.actions
