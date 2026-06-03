"""
STAGE 7: RETENTION RECOMMENDATION ENGINE
=========================================

Turn churn predictions into SPECIFIC business actions.

Most projects stop at: "Customer has 78% churn probability"
YOU output:  "Send THIS offer to THIS customer via THIS channel NOW"

CHECKPOINT: After this stage, you'll have:
✅ Next-best-action for every at-risk customer
✅ Revenue at risk quantified ($)
✅ Segment-specific retention playbook
✅ Priority action table (ready for marketing)

Expected Time: 20-30 minutes
Expected Outputs:
- retention_actions.csv   (actionable recommendations)
- retention_playbook.json (strategy per segment)
- revenue_at_risk.png
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

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)


class RetentionEngine:
    """
    Prescriptive retention intelligence.

    For each at-risk customer, outputs:
      - WHY they are at risk (top drivers)
      - WHAT action to take
      - WHICH channel to use
      - WHEN to intervene
      - Expected revenue saved
    """

    # ── Offer Library ─────────────────────────────────────────────────────────
    OFFERS = {
        'Premium Loyalists': {
            'low':    {'action': 'Monitor + VIP newsletter',
                       'offer':  'Early access to new routes',
                       'channel':'App notification',
                       'timing': 'Monthly'},
            'medium': {'action': 'Tier extension',
                       'offer':  'Tier status extended 6 months + 5K bonus miles',
                       'channel':'Personalized email',
                       'timing': 'Within 14 days'},
            'high':   {'action': 'Executive intervention',
                       'offer':  'Tier extended 12m + 15K miles + lounge upgrade',
                       'channel':'Personal call + email',
                       'timing': 'Within 48 hours'},
        },
        'Silent Drifters': {
            'low':    {'action': 'Gentle re-engagement',
                       'offer':  'We miss you — 2K bonus miles on next booking',
                       'channel':'Email',
                       'timing': 'Within 21 days'},
            'medium': {'action': 'Win-back campaign',
                       'offer':  '5K bonus miles + tier extension 3 months',
                       'channel':'Email + SMS',
                       'timing': 'Within 7 days'},
            'high':   {'action': 'Urgent reactivation',
                       'offer':  '8K bonus miles + waived change fees for 60 days',
                       'channel':'SMS + Email + App push',
                       'timing': 'Within 48 hours'},
        },
        'Miles Hoarders': {
            'low':    {'action': 'Redemption nudge',
                       'offer':  'Reminder: your miles expire in 12 months',
                       'channel':'Email',
                       'timing': 'Within 30 days'},
            'medium': {'action': 'Redemption incentive',
                       'offer':  'Use 10K miles, earn 2K bonus — this month only',
                       'channel':'Email + App',
                       'timing': 'Within 14 days'},
            'high':   {'action': 'Urgent redemption push',
                       'offer':  'Triple bonus on next redemption + flight discount',
                       'channel':'Email + SMS',
                       'timing': 'Within 5 days'},
        },
        'Seasonal Travelers': {
            'low':    {'action': 'Seasonal campaign',
                       'offer':  'Holiday travel deals + 3K bonus miles',
                       'channel':'Email',
                       'timing': '4 weeks before peak season'},
            'medium': {'action': 'Seasonal activation',
                       'offer':  'Exclusive vacation bundle + bonus miles',
                       'channel':'Email + App',
                       'timing': '3 weeks before peak season'},
            'high':   {'action': 'Priority seasonal offer',
                       'offer':  'Personalised itinerary + 6K miles + seat upgrade',
                       'channel':'Email + Personal outreach',
                       'timing': '2 weeks before peak season'},
        },
        'Rising Stars': {
            'low':    {'action': 'Nurture campaign',
                       'offer':  'Tier upgrade progress tracker + 2K miles accelerator',
                       'channel':'App + Email',
                       'timing': 'Monthly'},
            'medium': {'action': 'Tier challenge',
                       'offer':  'Fly 3 more times this quarter → unlock Silver status',
                       'channel':'App push + Email',
                       'timing': 'Within 14 days'},
            'high':   {'action': 'Accelerated loyalty push',
                       'offer':  'Double miles for next 60 days + tier fast-track',
                       'channel':'Email + SMS',
                       'timing': 'Within 7 days'},
        },
        'Budget Frequent Flyers': {
            'low':    {'action': 'Value reinforcement',
                       'offer':  'Co-branded card offer — earn miles on daily spend',
                       'channel':'Email',
                       'timing': 'Within 30 days'},
            'medium': {'action': 'Partner bundle',
                       'offer':  'Hotel + car + miles package — save 20%',
                       'channel':'Email + App',
                       'timing': 'Within 14 days'},
            'high':   {'action': 'Competitive retention',
                       'offer':  'Price-match guarantee + 4K miles on next booking',
                       'channel':'Email + SMS',
                       'timing': 'Within 5 days'},
        },
        'At-Risk VIPs': {
            'low':    {'action': 'VIP re-engagement',
                       'offer':  'Complimentary lounge access for 3 visits',
                       'channel':'Personal email from account manager',
                       'timing': 'Within 14 days'},
            'medium': {'action': 'High-value intervention',
                       'offer':  'Tier extension 12m + 10K miles + priority support',
                       'channel':'Phone call + email',
                       'timing': 'Within 5 days'},
            'high':   {'action': 'CRITICAL — Maximum retention',
                       'offer':  'Tier extension 18m + 20K miles + dedicated concierge',
                       'channel':'CEO/Director personal call + email',
                       'timing': 'Within 24 hours'},
        },
        'DEFAULT': {
            'low':    {'action': 'Standard monitoring',
                       'offer':  'Quarterly engagement email',
                       'channel':'Email',
                       'timing': 'Quarterly'},
            'medium': {'action': 'Standard re-engagement',
                       'offer':  '3K bonus miles on next booking',
                       'channel':'Email',
                       'timing': 'Within 14 days'},
            'high':   {'action': 'Urgent retention',
                       'offer':  '5K bonus miles + tier extension 3 months',
                       'channel':'Email + SMS',
                       'timing': 'Within 7 days'},
        }
    }

    # Expected retention lift per (segment, risk) pair
    LIFT = {
        ('Premium Loyalists',   'low'):    0.05,
        ('Premium Loyalists',   'medium'): 0.30,
        ('Premium Loyalists',   'high'):   0.48,
        ('Silent Drifters',     'low'):    0.15,
        ('Silent Drifters',     'medium'): 0.35,
        ('Silent Drifters',     'high'):   0.28,
        ('Miles Hoarders',      'low'):    0.12,
        ('Miles Hoarders',      'medium'): 0.28,
        ('Miles Hoarders',      'high'):   0.22,
        ('Seasonal Travelers',  'low'):    0.10,
        ('Seasonal Travelers',  'medium'): 0.25,
        ('Seasonal Travelers',  'high'):   0.20,
        ('Rising Stars',        'low'):    0.08,
        ('Rising Stars',        'medium'): 0.32,
        ('Rising Stars',        'high'):   0.25,
        ('Budget Frequent Flyers','low'):  0.08,
        ('Budget Frequent Flyers','medium'):0.20,
        ('Budget Frequent Flyers','high'): 0.18,
        ('At-Risk VIPs',        'low'):    0.35,
        ('At-Risk VIPs',        'medium'): 0.45,
        ('At-Risk VIPs',        'high'):   0.40,
    }

    def __init__(self):
        self.final_path  = Path("data/final")
        self.output_path = Path("outputs/figures/retention")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.df = None

    # ─────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────

    def load_predictions(self):
        print("\n" + "=" * 60)
        print("STAGE 7: RETENTION RECOMMENDATION ENGINE")
        print("=" * 60)
        print("\n� Loading customer predictions from Stage 6...")

        try:
            fp = self.final_path / 'all_customer_predictions.csv'
            if not fp.exists():
                fp = self.final_path / 'churn_predictions.csv'

            self.df = pd.read_csv(fp)
            print(f"   ✓ Loaded: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns[:8])}...")

            # Detect required columns
            self.prob_col = self._detect('churn_probability', self.df.columns)
            self.risk_col = self._detect('risk_level', self.df.columns)
            self.seg_col  = self._detect('segment_name', self.df.columns)
            self.id_col   = self.df.columns[0]
            self.clv_col  = self._detect('clv', self.df.columns)

            print(f"\n   Probability column : {self.prob_col}")
            print(f"   Risk level column  : {self.risk_col}")
            print(f"   Segment column     : {self.seg_col}")
            print(f"   CLV column         : {self.clv_col}")

            if self.prob_col is None:
                print("❌ churn_probability column not found. Run Stage 6 first.")
                return False

            return True

        except FileNotFoundError as e:
            print(f"❌ {e}\n   Run Stage 6 first.")
            return False

    def _detect(self, keyword, columns):
        for col in columns:
            if keyword.lower() in col.lower():
                return col
        return None

    # ─────────────────────────────────────────────
    # BUILD RECOMMENDATIONS
    # ─────────────────────────────────────────────

    def build_recommendations(self):
        print("\n� Building personalized retention recommendations...")

        df = self.df.copy()

        # Ensure risk_level column
        if self.risk_col is None or df[self.risk_col].isnull().all():
            df['risk_level'] = pd.cut(
                df[self.prob_col],
                bins=[0, 0.30, 0.60, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            self.risk_col = 'risk_level'

        # Map risk level to short key
        risk_map = {
            'Low Risk':    'low',
            'Medium Risk': 'medium',
            'High Risk':   'high'
        }
        df['risk_key'] = df[self.risk_col].map(risk_map).fillna('medium')

        # Segment name
        if self.seg_col:
            df['seg_name'] = df[self.seg_col].fillna('DEFAULT')
        else:
            df['seg_name'] = 'DEFAULT'

        # Lookup offer
        actions, offers, channels, timings = [], [], [], []
        for _, row in df.iterrows():
            seg  = row['seg_name']
            risk = row['risk_key']
            lib  = self.OFFERS.get(seg, self.OFFERS['DEFAULT'])
            rec  = lib.get(risk, lib['medium'])
            actions.append(rec['action'])
            offers.append(rec['offer'])
            channels.append(rec['channel'])
            timings.append(rec['timing'])

        df['recommended_action'] = actions
        df['recommended_offer']  = offers
        df['recommended_channel']= channels
        df['intervention_timing']= timings

        # Retention lift
        df['expected_retention_lift'] = df.apply(
            lambda r: self.LIFT.get(
                (r['seg_name'], r['risk_key']),
                {'low': 0.10, 'medium': 0.22, 'high': 0.30}[r['risk_key']]
            ),
            axis=1
        )

        # Revenue estimates
        if self.clv_col and self.clv_col in df.columns:
            clv_raw = df[self.clv_col]
            # If log-transformed, convert back
            if clv_raw.max() < 20:
                clv_val = np.expm1(clv_raw)
            else:
                clv_val = clv_raw
        else:
            clv_val = pd.Series([5000] * len(df))  # fallback

        df['clv_estimate'] = clv_val
        df['revenue_at_risk']    = (df[self.prob_col] * df['clv_estimate']).round(2)
        df['potential_save']     = (df['revenue_at_risk'] * df['expected_retention_lift']).round(2)

        # Priority score (higher = act first)
        df['priority_score'] = (
            0.4 * df[self.prob_col] +
            0.4 * (df['clv_estimate'] / df['clv_estimate'].max()) +
            0.2 * df['expected_retention_lift']
        ).round(4)

        self.recommendations = df
        print(f"   ✓ Recommendations built for {len(df):,} customers")

    # ─────────────────────────────────────────────
    # BUSINESS SUMMARY
    # ─────────────────────────────────────────────

    def print_business_summary(self):
        df = self.recommendations

        print("\n" + "=" * 60)
        print("� RETENTION ENGINE — BUSINESS SUMMARY")
        print("=" * 60)

        total_at_risk = (df[self.risk_col].isin(['High Risk', 'Medium Risk'])).sum()
        high_risk     = (df[self.risk_col] == 'High Risk').sum()
        total_rev_risk = df['revenue_at_risk'].sum()
        total_save     = df['potential_save'].sum()

        print(f"\n   � Portfolio Overview:")
        print(f"      Total customers analyzed : {len(df):,}")
        print(f"      At-risk (Med + High)     : {total_at_risk:,} "
              f"({total_at_risk/len(df)*100:.1f}%)")
        print(f"      High-risk customers       : {high_risk:,}")

        print(f"\n   � Revenue Impact:")
        print(f"      Total revenue at risk     : ${total_rev_risk:,.0f}")
        print(f"      Potential revenue saved   : ${total_save:,.0f}")
        print(f"      (Assuming intervention success rates per segment)")

        # By risk level
        print(f"\n   � By Risk Level:")
        risk_summary = df.groupby(self.risk_col).agg(
            Customers   = (self.prob_col, 'count'),
            Avg_Prob    = (self.prob_col, 'mean'),
            Revenue_Risk= ('revenue_at_risk', 'sum'),
            Potential_Save=('potential_save', 'sum')
        ).round(2)
        print(risk_summary.to_string())

        # By segment
        if self.seg_col:
            print(f"\n   � By Segment:")
            seg_summary = df.groupby('seg_name').agg(
                Customers   = (self.prob_col, 'count'),
                Avg_Churn_Prob = (self.prob_col, 'mean'),
                Revenue_Risk = ('revenue_at_risk', 'sum'),
                Potential_Save = ('potential_save', 'sum')
            ).sort_values('Revenue_Risk', ascending=False).round(2)
            print(seg_summary.to_string())

        # Top 20 priority customers
        print(f"\n   � Top 10 Priority Customers (Act First):")
        top20_cols = [self.id_col, 'seg_name', self.risk_col,
                      self.prob_col, 'revenue_at_risk',
                      'recommended_action', 'intervention_timing']
        top20_cols = [c for c in top20_cols if c in df.columns]
        top20 = df.nlargest(10, 'priority_score')[top20_cols]
        print(top20.to_string(index=False))

    # ─────────────────────────────────────────────
    # GENERATE PLAYBOOK
    # ─────────────────────────────────────────────

    def generate_playbook(self):
        """Create full retention playbook document."""

        print("\n� Generating Retention Playbook...")

        playbook = {
            "title": "Airline Loyalty Retention Playbook",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "total_customers": len(self.recommendations),
                "high_risk_customers": int((self.recommendations[self.risk_col] == 'High Risk').sum()),
                "total_revenue_at_risk": float(self.recommendations['revenue_at_risk'].sum()),
                "potential_revenue_saved": float(self.recommendations['potential_save'].sum())
            },
            "segments": {}
        }

        for seg_name, offers in self.OFFERS.items():
            if seg_name == 'DEFAULT':
                continue
            seg_data = self.recommendations[
                self.recommendations['seg_name'] == seg_name
            ] if self.seg_col else pd.DataFrame()

            playbook['segments'][seg_name] = {
                "size": int(len(seg_data)),
                "avg_churn_probability": float(seg_data[self.prob_col].mean()) if len(seg_data) > 0 else 0,
                "revenue_at_risk": float(seg_data['revenue_at_risk'].sum()) if len(seg_data) > 0 else 0,
                "potential_save": float(seg_data['potential_save'].sum()) if len(seg_data) > 0 else 0,
                "interventions": {
                    risk: {
                        "action":  offers[risk]['action'],
                        "offer":   offers[risk]['offer'],
                        "channel": offers[risk]['channel'],
                        "timing":  offers[risk]['timing'],
                        "expected_lift": f"{self.LIFT.get((seg_name, risk), 0.20)*100:.0f}%"
                    }
                    for risk in ['low', 'medium', 'high']
                }
            }

        playbook_file = self.final_path / 'retention_playbook.json'
        with open(playbook_file, 'w') as f:
            json.dump(playbook, f, indent=2)
        print(f"   ✓ Saved: {playbook_file}")

        return playbook

    # ─────────────────────────────────────────────
    # VISUALIZATIONS
    # ─────────────────────────────────────────────

    def visualize_retention(self):
        print("\n� Generating retention visualizations...")

        df = self.recommendations
        risk_colors = {
            'Low Risk':    '#2ecc71',
            'Medium Risk': '#f39c12',
            'High Risk':   '#e74c3c'
        }

        # ── Revenue at Risk by Segment ─────────────
        if self.seg_col and 'seg_name' in df.columns:
            seg_rev = df.groupby('seg_name')['revenue_at_risk'].sum().sort_values(ascending=False)

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))

            seg_rev.plot(kind='barh', ax=axes[0], color='coral', alpha=0.85)
            axes[0].set_title('Revenue at Risk by Segment',
                               fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Revenue at Risk ($)')
            axes[0].grid(True, alpha=0.3, axis='x')
            for i, v in enumerate(seg_rev.values):
                axes[0].text(v + 500, i, f'${v:,.0f}', va='center', fontsize=9)

            seg_save = df.groupby('seg_name')['potential_save'].sum().sort_values(ascending=False)
            seg_save.plot(kind='barh', ax=axes[1], color='steelblue', alpha=0.85)
            axes[1].set_title('Potential Revenue Saved by Segment',
                               fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Potential Revenue Saved ($)')
            axes[1].grid(True, alpha=0.3, axis='x')
            for i, v in enumerate(seg_save.values):
                axes[1].text(v + 200, i, f'${v:,.0f}', va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(self.output_path / 'revenue_at_risk.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Saved: revenue_at_risk.png")

        # ── Churn Probability Distribution by Segment ─
        if self.seg_col and 'seg_name' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 7))
            segments = df['seg_name'].unique()
            colors   = plt.cm.Set2(np.linspace(0, 1, len(segments)))

            for seg, color in zip(segments, colors):
                seg_probs = df[df['seg_name'] == seg][self.prob_col]
                ax.hist(seg_probs, bins=30, alpha=0.5,
                        label=seg, color=color, density=True)

            ax.set_title('Churn Probability Distribution by Segment',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Churn Probability')
            ax.set_ylabel('Density')
            ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1))
            ax.axvline(x=0.5, color='black', linestyle='--',
                       alpha=0.5, label='Decision threshold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_path / 'churn_prob_by_segment.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Saved: churn_prob_by_segment.png")

        # ── Priority Score Scatter ─────────────────
        fig, ax = plt.subplots(figsize=(12, 7))
        colors_mapped = df[self.risk_col].map(risk_colors).fillna('#999999')
        scatter = ax.scatter(
            df[self.prob_col],
            df['clv_estimate'].clip(upper=df['clv_estimate'].quantile(0.99)),
            c=colors_mapped,
            alpha=0.3, s=15
        )
        ax.set_xlabel('Churn Probability', fontsize=12)
        ax.set_ylabel('Customer Lifetime Value ($)', fontsize=12)
        ax.set_title('Customer Risk Map: Churn Probability vs CLV',
                     fontsize=13, fontweight='bold')
        ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.4, label='High risk threshold')
        ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.4, label='Medium risk threshold')
        patches = [plt.Rectangle((0,0),1,1, color=c, alpha=0.7, label=l)
                   for l, c in risk_colors.items()]
        ax.legend(handles=patches, fontsize=10)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(self.output_path / 'customer_risk_map.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: customer_risk_map.png")

    # ─────────────────────────────────────────────
    # SAVE OUTPUTS
    # ─────────────────────────────────────────────

    def save_outputs(self):
        print("\n� Saving outputs...")

        # Full recommendations
        rec_file = self.final_path / 'retention_actions.csv'
        self.recommendations.to_csv(rec_file, index=False)
        print(f"   ✓ Saved: {rec_file}")

        # High-priority actions (ready for marketing)
        high_priority = self.recommendations[
            self.recommendations[self.risk_col].isin(['High Risk', 'Medium Risk'])
        ].sort_values('priority_score', ascending=False)

        key_cols = [self.id_col, 'seg_name', self.risk_col,
                    self.prob_col, 'revenue_at_risk', 'potential_save',
                    'recommended_action', 'recommended_offer',
                    'recommended_channel', 'intervention_timing',
                    'expected_retention_lift', 'priority_score']
        key_cols = [c for c in key_cols if c in high_priority.columns]

        priority_file = self.final_path / 'priority_action_list.csv'
        high_priority[key_cols].to_csv(priority_file, index=False)
        print(f"   ✓ Saved: {priority_file}")
        print(f"   ✓ Priority actions: {len(high_priority):,} customers")

        # Update progress
        progress_file = Path("checkpoints/progress.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            progress['current_stage'] = 7
            if 7 not in progress['completed_stages']:
                progress['completed_stages'].append(7)
            progress['last_updated'] = datetime.now().isoformat()
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            print("   ✓ Progress updated")

    # ─────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────

    def run_retention_engine(self):
        if not self.load_predictions():
            return False

        self.build_recommendations()
        self.print_business_summary()
        self.generate_playbook()
        self.visualize_retention()
        self.save_outputs()

        total_save = self.recommendations['potential_save'].sum()
        high_risk  = (self.recommendations[self.risk_col] == 'High Risk').sum()

        print("\n" + "=" * 60)
        print("✅ STAGE 7 COMPLETE — RETENTION ENGINE")
        print("=" * 60)
        print(f"\n   High-risk customers   : {high_risk:,}")
        print(f"   Potential revenue save : ${total_save:,.0f}")
        print(f"\n� Outputs: data/final/")
        print(f"   - retention_actions.csv")
        print(f"   - priority_action_list.csv")
        print(f"   - retention_playbook.json")
        print(f"\n� Visualizations: outputs/figures/retention/")
        print(f"\n� Final Step: Run 08_dashboard.py")
        return True


def main():
    engine = RetentionEngine()
    success = engine.run_retention_engine()
    if not success:
        print("\n❌ Stage 7 failed. Check errors above.")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())