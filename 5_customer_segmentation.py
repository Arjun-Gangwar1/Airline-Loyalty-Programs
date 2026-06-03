"""
STAGE 5: CUSTOMER SEGMENTATION
================================

Create 4-7 ACTIONABLE customer segments that marketing can use immediately.

NOT: "Cluster 1, 2, 3"
BUT: "Premium Loyalists", "Silent Drifters", "Miles Hoarders"

CHECKPOINT: After this stage, you'll have:
✅ 4-7 business-named customer segments
✅ Segment profiles (CLV, churn rate, behavior)
✅ Retention strategy per segment
✅ Segment assignment for all customers

Expected Time: 30-45 minutes
Expected Outputs:
- customer_segments.csv (segment assignment)
- segment_profiles.csv (segment statistics)
- segment_visualizations (PNG files)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class CustomerSegmentationEngine:
    """
    Build actionable behavioral customer segments.
    Each segment gets a business name + retention strategy.
    """

    # ── Segment archetypes ──────────────────────────────
    SEGMENT_ARCHETYPES = {
        'Premium Loyalists': {
            'description': 'High CLV, stable engagement, consistent flyers',
            'strategy': 'VIP treatment, exclusive benefits, personal relationship manager',
            'channel': 'Personal outreach + Premium app features',
            'offer': 'Lounge access, priority boarding, milestone rewards',
            'urgency': 'Low — maintain relationship',
            'color': '#2ecc71'
        },
        'Silent Drifters': {
            'description': 'Declining engagement, once-active customers going quiet',
            'strategy': 'Urgent re-engagement — act within 30 days',
            'channel': 'Email + SMS + In-app push',
            'offer': '5,000 bonus miles for next booking + tier extension',
            'urgency': '� HIGH — imminent churn risk',
            'color': '#e74c3c'
        },
        'Miles Hoarders': {
            'description': 'High points balance, low redemption — emotionally detached',
            'strategy': 'Redemption incentives to re-engage with the program',
            'channel': 'Email campaign',
            'offer': '"Use 10K miles, earn 2K bonus" limited-time offer',
            'urgency': 'Medium — engage before balance lapses',
            'color': '#f39c12'
        },
        'Seasonal Travelers': {
            'description': 'Periodic activity peaks (holiday/vacation flyers)',
            'strategy': 'Targeted campaigns before their typical travel window',
            'channel': 'Email 4 weeks before peak season',
            'offer': 'Vacation bundle + bonus miles on leisure bookings',
            'urgency': 'Low-Medium — seasonal activation',
            'color': '#3498db'
        },
        'Rising Stars': {
            'description': 'Growing engagement, increasing flight frequency',
            'strategy': 'Nurture to next tier — show path to elite status',
            'channel': 'App + Email',
            'offer': 'Tier upgrade challenge + bonus mile accelerator',
            'urgency': 'Low — accelerate growth trajectory',
            'color': '#9b59b6'
        },
        'Budget Frequent Flyers': {
            'description': 'High frequency but low CLV — price-sensitive travelers',
            'strategy': 'Volume rewards + partner benefits to increase spend',
            'channel': 'Email + App',
            'offer': 'Co-branded card offer, hotel/car partner promotions',
            'urgency': 'Medium — risk of switching to cheaper competitor',
            'color': '#1abc9c'
        },
        'At-Risk VIPs': {
            'description': 'Previously high-value customers showing sharp decline',
            'strategy': 'Maximum intervention — executive outreach justified',
            'channel': 'Personal call + Email + Premium offer',
            'offer': 'Tier extension 12 months + 15K bonus miles + lounge pass',
            'urgency': '� CRITICAL — high revenue at risk',
            'color': '#c0392b'
        }
    }

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.final_path = Path("data/final")
        self.output_path = Path("outputs/figures/segmentation")
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.features_df = None
        self.segments_df = None
        self.profiles = []

    # ─────────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────────

    def load_features(self):
        print("\n" + "=" * 60)
        print("STAGE 5: CUSTOMER SEGMENTATION")
        print("=" * 60)
        print("\n� Loading feature matrix from Stage 4...")

        try:
            self.features_df = pd.read_csv(self.final_path / 'customer_features.csv')
            print(f"   ✓ Features loaded: {self.features_df.shape}")

            # Detect ID and churn columns
            self.id_col = self.features_df.columns[0]
            self.has_churn = 'churn' in self.features_df.columns
            return True
        except FileNotFoundError:
            print("❌ ERROR: customer_features.csv not found.")
            print("   Please run 04_feature_engineering.py first.")
            return False

    # ─────────────────────────────────────────────
    # FEATURE SELECTION FOR CLUSTERING
    # ─────────────────────────────────────────────

    def select_clustering_features(self):
        """Choose the most informative features for segmentation."""

        print("\n� Selecting clustering features...")

        preferred = [
            'recency_days',
            'avg_flights_per_month',
            'clv_log',
            'clv_percentile',
            'consistency_score',
            'engagement_slope',
            'redemption_rate',
            'active_month_ratio',
            'loyalty_health_score',
            'zero_flight_month_pct',
            'flight_trend_3m_vs_6m',
            'seasonality_score',
            'earn_burn_ratio',
            'total_flights_historical',
            'activity_trajectory',
        ]

        available = [f for f in preferred if f in self.features_df.columns]
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = [self.id_col, 'churn']
        fallback = [c for c in numeric_cols if c not in exclude and c not in available]

        self.cluster_features = available + fallback[:max(0, 8 - len(available))]
        print(f"   ✓ {len(self.cluster_features)} clustering features selected:")
        for f in self.cluster_features:
            print(f"      • {f}")

    # ─────────────────────────────────────────────
    # DETERMINE OPTIMAL K
    # ─────────────────────────────────────────────

    def find_optimal_k(self, k_range=range(3, 9)):
        """Elbow method + silhouette score to find best K."""

        print("\n� Finding optimal number of clusters...")

        X = self.features_df[self.cluster_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        inertias, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels))
            print(f"   K={k} → Inertia={km.inertia_:,.0f}  Silhouette={sil_scores[-1]:.3f}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters (K)')
        axes[0].set_ylabel('Inertia')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(list(k_range), sil_scores, 'go-', linewidth=2, markersize=8)
        axes[1].set_title('Silhouette Scores (Higher = Better)',
                           fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Number of Clusters (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Good threshold')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_path / 'cluster_evaluation.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        best_k = list(k_range)[np.argmax(sil_scores)]
        print(f"\n   ✓ Best K by Silhouette: {best_k}")
        print(f"   → Using K={self.n_clusters} (balances stats + business interpretability)")

        self.scaler = scaler
        self.X_scaled = X_scaled

    # ─────────────────────────────────────────────
    # FIT FINAL SEGMENTATION MODEL
    # ─────────────────────────────────────────────

    def fit_segmentation(self):
        """Fit final K-Means model."""

        print(f"\n� Fitting K-Means with K={self.n_clusters}...")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=20,
            max_iter=500
        )
        self.features_df['segment_id'] = self.kmeans.fit_predict(self.X_scaled)

        sil = silhouette_score(self.X_scaled, self.features_df['segment_id'])
        print(f"   ✓ Final Silhouette Score: {sil:.3f}")

        dist = self.features_df['segment_id'].value_counts().sort_index()
        print(f"\n   Segment distribution:")
        for seg_id, cnt in dist.items():
            pct = cnt / len(self.features_df) * 100
            print(f"      Segment {seg_id}: {cnt:,} customers ({pct:.1f}%)")

    # ─────────────────────────────────────────────
    # PROFILE EACH SEGMENT
    # ─────────────────────────────────────────────

    def profile_segments(self):
        """Build rich profiles for each segment."""

        print("\n� Profiling segments...")

        profile_features = [
            'clv_log', 'clv_percentile',
            'recency_days', 'avg_flights_per_month',
            'consistency_score', 'engagement_slope',
            'redemption_rate', 'active_month_ratio',
            'loyalty_health_score', 'zero_flight_month_pct',
            'flight_trend_3m_vs_6m', 'total_flights_historical'
        ]
        profile_features = [f for f in profile_features
                            if f in self.features_df.columns]

        for seg_id in sorted(self.features_df['segment_id'].unique()):
            seg_data = self.features_df[self.features_df['segment_id'] == seg_id]
            size = len(seg_data)
            size_pct = size / len(self.features_df) * 100

            profile = {
                'segment_id': int(seg_id),
                'size': size,
                'size_pct': round(size_pct, 1),
            }

            for feat in profile_features:
                if feat in seg_data.columns:
                    profile[f'avg_{feat}'] = round(float(seg_data[feat].mean()), 4)

            if self.has_churn:
                churn_rate = seg_data['churn'].mean() * 100
                profile['churn_rate_pct'] = round(float(churn_rate), 2)
                profile['churned_count'] = int(seg_data['churn'].sum())
            else:
                profile['churn_rate_pct'] = None

            self.profiles.append(profile)

        print(f"   ✓ {len(self.profiles)} segment profiles created")

    # ─────────────────────────────────────────────
    # ASSIGN BUSINESS NAMES
    # ─────────────────────────────────────────────

    def assign_business_names(self):
        """
        Map statistical segments to business-meaningful archetypes.

        Logic:
        - High CLV + low churn + high health → Premium Loyalists
        - High CLV + high churn              → At-Risk VIPs
        - High recency + low engagement      → Silent Drifters
        - High points balance + low redemption → Miles Hoarders
        - Positive trend + young tenure      → Rising Stars
        - Low CLV + high frequency           → Budget Frequent Flyers
        - Everything else → Seasonal Travelers
        """

        print("\n�️  Assigning business names to segments...")

        profiles_df = pd.DataFrame(self.profiles)
        self.name_map = {}

        # Compute relative ranks for rule-based assignment
        def rank(col):
            if col not in profiles_df.columns:
                return pd.Series([0.5] * len(profiles_df), index=profiles_df.index)
            return profiles_df[col].rank(pct=True)

        high_clv = rank('avg_clv_percentile') >= 0.6
        low_clv = rank('avg_clv_percentile') < 0.4
        high_churn = rank('churn_rate_pct') >= 0.65
        low_churn = rank('churn_rate_pct') < 0.35
        high_recency = rank('avg_recency_days') >= 0.65    # many days = inactive
        low_engagement = rank('avg_active_month_ratio') < 0.4
        high_redemption = rank('avg_redemption_rate') >= 0.55
        low_redemption = rank('avg_redemption_rate') < 0.35
        high_freq = rank('avg_avg_flights_per_month') >= 0.6
        pos_trend = rank('avg_flight_trend_3m_vs_6m') >= 0.55
        high_health = rank('avg_loyalty_health_score') >= 0.6

        used_names = set()

        def pick_name(candidates):
            for name in candidates:
                if name not in used_names:
                    used_names.add(name)
                    return name
            # Fallback: numbered
            for i in range(1, 20):
                fb = f"Segment Group {i}"
                if fb not in used_names:
                    used_names.add(fb)
                    return fb

        for idx, row in profiles_df.iterrows():
            seg_id = row['segment_id']
            i = idx  # positional index

            if high_clv.iloc[i] and high_churn.iloc[i]:
                name = pick_name(['At-Risk VIPs'])
            elif high_clv.iloc[i] and high_health.iloc[i] and low_churn.iloc[i]:
                name = pick_name(['Premium Loyalists'])
            elif high_recency.iloc[i] and low_engagement.iloc[i]:
                name = pick_name(['Silent Drifters'])
            elif low_redemption.iloc[i] and not low_clv.iloc[i]:
                name = pick_name(['Miles Hoarders'])
            elif pos_trend.iloc[i] and low_churn.iloc[i]:
                name = pick_name(['Rising Stars'])
            elif high_freq.iloc[i] and low_clv.iloc[i]:
                name = pick_name(['Budget Frequent Flyers'])
            else:
                name = pick_name(['Seasonal Travelers', 'Premium Loyalists',
                                   'Rising Stars', 'Miles Hoarders'])

            self.name_map[seg_id] = name
            profiles_df.at[idx, 'segment_name'] = name
            archetype = self.SEGMENT_ARCHETYPES.get(name, {})
            profiles_df.at[idx, 'retention_strategy'] = archetype.get('strategy', 'Standard engagement')
            profiles_df.at[idx, 'recommended_offer'] = archetype.get('offer', 'Generic offer')
            profiles_df.at[idx, 'urgency'] = archetype.get('urgency', 'Low')
            profiles_df.at[idx, 'channel'] = archetype.get('channel', 'Email')

        self.profiles_df = profiles_df
        self.features_df['segment_name'] = self.features_df['segment_id'].map(self.name_map)

        print(f"\n   Segment Name Assignments:")
        for seg_id, name in self.name_map.items():
            size = len(self.features_df[self.features_df['segment_id'] == seg_id])
            churn = self.profiles_df[
                self.profiles_df['segment_id'] == seg_id]['churn_rate_pct'].values
            churn_str = f"{churn[0]:.1f}%" if len(churn) > 0 and churn[0] is not None else "N/A"
            print(f"      Segment {seg_id} → {name:<25} | {size:,} customers | Churn: {churn_str}")

    # ─────────────────────────────────────────────
    # PRINT SEGMENT PROFILES
    # ─────────────────────────────────────────────

    def print_segment_profiles(self):
        """Display rich segment profiles in the console."""

        print("\n" + "=" * 60)
        print("� SEGMENT PROFILES")
        print("=" * 60)

        for _, row in self.profiles_df.iterrows():
            name = row.get('segment_name', f"Segment {row['segment_id']}")
            archetype = self.SEGMENT_ARCHETYPES.get(name, {})

            print(f"\n{'─'*55}")
            print(f"  � {name.upper()}")
            print(f"{'─'*55}")
            print(f"  Size        : {row['size']:,} customers ({row['size_pct']}%)")
            if row.get('churn_rate_pct') is not None:
                print(f"  Churn Rate  : {row['churn_rate_pct']}%")
            if 'avg_clv_log' in row:
                clv_approx = np.expm1(row['avg_clv_log'])
                print(f"  Avg CLV     : ~${clv_approx:,.0f} (log-approx)")
            if 'avg_loyalty_health_score' in row:
                print(f"  Health Score: {row.get('avg_loyalty_health_score', 'N/A'):.1f}/100")
            if 'avg_recency_days' in row:
                print(f"  Recency     : {row.get('avg_recency_days', 0):.0f} days since last flight")
            if 'avg_avg_flights_per_month' in row:
                print(f"  Frequency   : {row.get('avg_avg_flights_per_month', 0):.2f} flights/month")
            print(f"\n  � Strategy : {row.get('retention_strategy', archetype.get('strategy', 'N/A'))}")
            print(f"  � Offer    : {row.get('recommended_offer', archetype.get('offer', 'N/A'))}")
            print(f"  � Channel  : {row.get('channel', archetype.get('channel', 'Email'))}")
            print(f"  ⚡ Urgency  : {row.get('urgency', archetype.get('urgency', 'Low'))}")

    # ─────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────

    def visualize_segments(self):
        """Generate comprehensive segment visualizations."""

        print("\n� Generating segment visualizations...")

        # Color map for segments
        color_map = {
            name: info['color'] for name, info in self.SEGMENT_ARCHETYPES.items()
        }
        default_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db',
                          '#9b59b6', '#1abc9c', '#c0392b']

        seg_colors = []
        for seg_id in sorted(self.features_df['segment_id'].unique()):
            name = self.name_map.get(seg_id, '')
            color = color_map.get(name, default_colors[seg_id % len(default_colors)])
            seg_colors.append(color)

        # ── 1. PCA 2D scatter ──────────────────────
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(self.X_scaled)

        fig, ax = plt.subplots(figsize=(12, 8))
        for seg_id in sorted(self.features_df['segment_id'].unique()):
            mask = self.features_df['segment_id'] == seg_id
            name = self.name_map.get(seg_id, f'Segment {seg_id}')
            color = color_map.get(name, default_colors[seg_id % len(default_colors)])
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                       c=color, label=name, alpha=0.5, s=20)

        ax.set_title('Customer Segments (PCA 2D Projection)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_path / 'segments_pca_scatter.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: segments_pca_scatter.png")

        # ── 2. Segment Size & Churn Bar ────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        seg_names = [self.name_map.get(r['segment_id'], f"Seg {r['segment_id']}")
                     for _, r in self.profiles_df.iterrows()]
        sizes = self.profiles_df['size'].values
        colors = [color_map.get(n, '#999') for n in seg_names]

        axes[0].barh(seg_names, sizes, color=colors, alpha=0.85)
        axes[0].set_title('Segment Sizes', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Number of Customers')
        axes[0].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(sizes):
            axes[0].text(v + 10, i, f'{v:,}', va='center', fontsize=9)

        if self.has_churn and 'churn_rate_pct' in self.profiles_df.columns:
            churn_rates = self.profiles_df['churn_rate_pct'].values
            bars = axes[1].barh(seg_names, churn_rates, color=colors, alpha=0.85)
            axes[1].set_title('Churn Rate by Segment',
                               fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Churn Rate (%)')
            axes[1].grid(True, alpha=0.3, axis='x')
            axes[1].axvline(
                x=self.features_df['churn'].mean() * 100,
                color='black', linestyle='--', alpha=0.5, label='Overall avg'
            )
            axes[1].legend(fontsize=9)
            for i, v in enumerate(churn_rates):
                if v is not None:
                    axes[1].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_path / 'segment_overview.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: segment_overview.png")

        # ── 3. Radar / Spider Chart ────────────────
        key_feats = [
            'avg_avg_flights_per_month', 'avg_consistency_score',
            'avg_loyalty_health_score', 'avg_redemption_rate',
            'avg_active_month_ratio'
        ]
        key_feats = [f for f in key_feats if f in self.profiles_df.columns]

        if len(key_feats) >= 3:
            labels = [f.replace('avg_', '').replace('avg_', '').replace('_', ' ').title()
                      for f in key_feats]
            N = len(labels)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(10, 10),
                                    subplot_kw=dict(polar=True))

            for _, row in self.profiles_df.iterrows():
                name = self.name_map.get(row['segment_id'], 'Unknown')
                color = color_map.get(name, '#999')
                raw = [row.get(f, 0) for f in key_feats]
                # Normalize 0-1 relative to range in profiles
                normed = []
                for i, f in enumerate(key_feats):
                    mn = self.profiles_df[f].min()
                    mx = self.profiles_df[f].max()
                    val = (raw[i] - mn) / max(mx - mn, 1e-6)
                    normed.append(val)
                normed += normed[:1]
                ax.plot(angles, normed, 'o-', linewidth=2, label=name, color=color)
                ax.fill(angles, normed, alpha=0.1, color=color)

            ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
            ax.set_title('Segment Behavioral Profiles',
                         size=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_path / 'segment_radar.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Saved: segment_radar.png")

    # ─────────────────────────────────────────────
    # SAVE OUTPUTS
    # ─────────────────────────────────────────────

    def save_outputs(self):
        print("\n� Saving outputs...")

        # Customer segment assignments
        segments_file = self.final_path / 'customer_segments.csv'
        keep_cols = [self.id_col, 'segment_id', 'segment_name']
        if self.has_churn:
            keep_cols.append('churn')
        self.features_df[keep_cols].to_csv(segments_file, index=False)
        print(f"   ✓ Saved: {segments_file}")

        # Full features + segments
        full_file = self.final_path / 'customer_features_segmented.csv'
        self.features_df.to_csv(full_file, index=False)
        print(f"   ✓ Saved: {full_file}")

        # Segment profiles
        profiles_file = self.final_path / 'segment_profiles.csv'
        self.profiles_df.to_csv(profiles_file, index=False)
        print(f"   ✓ Saved: {profiles_file}")

        # JSON summary
        summary = {
            'stage': 5,
            'completed_at': datetime.now().isoformat(),
            'n_clusters': self.n_clusters,
            'total_customers': len(self.features_df),
            'segments': [
                {
                    'id': int(r['segment_id']),
                    'name': r.get('segment_name', ''),
                    'size': int(r['size']),
                    'size_pct': float(r['size_pct']),
                    'churn_rate': float(r.get('churn_rate_pct', 0) or 0),
                    'strategy': r.get('retention_strategy', '')
                }
                for _, r in self.profiles_df.iterrows()
            ]
        }
        with open(self.final_path / 'segmentation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✓ Saved: segmentation_summary.json")

        # Update progress
        progress_file = Path("checkpoints/progress.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            progress['current_stage'] = 5
            if 5 not in progress['completed_stages']:
                progress['completed_stages'].append(5)
            progress['last_updated'] = datetime.now().isoformat()
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            print("   ✓ Progress updated")

    # ─────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────

    def run_segmentation(self):
        if not self.load_features():
            return False

        self.select_clustering_features()
        self.find_optimal_k()
        self.fit_segmentation()
        self.profile_segments()
        self.assign_business_names()
        self.print_segment_profiles()
        self.visualize_segments()
        self.save_outputs()

        print("\n" + "=" * 60)
        print("✅ STAGE 5 COMPLETE — CUSTOMER SEGMENTATION")
        print("=" * 60)
        print(f"\n   {self.n_clusters} actionable customer segments created")
        print(f"   All segments have: name, strategy, offer, channel, urgency")
        print(f"\n� Outputs: data/final/")
        print(f"   - customer_segments.csv")
        print(f"   - customer_features_segmented.csv")
        print(f"   - segment_profiles.csv")
        print(f"\n� Visualizations: outputs/figures/segmentation/")
        print(f"\n� Next Step: Run 06_baseline_models.py")
        return True


def main():
    engine = CustomerSegmentationEngine(n_clusters=5)
    success = engine.run_segmentation()
    if not success:
        print("\n❌ Stage 5 failed. Check errors above.")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())