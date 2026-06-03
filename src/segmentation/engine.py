"""
Customer segmentation engine — creates business-named behavioral segments.

Segments customers using K-Means clustering and maps clusters to
human-readable archetypes with retention strategies.
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


SEGMENT_ARCHETYPES = {
    'Premium Loyalists': {
        'description': 'High CLV, stable engagement, consistent flyers',
        'strategy':    'VIP treatment, exclusive benefits, personal relationship manager',
        'channel':     'Personal outreach + Premium app features',
        'offer':       'Lounge access, priority boarding, milestone rewards',
        'urgency':     'Low — maintain relationship',
        'color':       '#2ecc71',
    },
    'Silent Drifters': {
        'description': 'Declining engagement, once-active customers going quiet',
        'strategy':    'Urgent re-engagement — act within 30 days',
        'channel':     'Email + SMS + In-app push',
        'offer':       '5,000 bonus miles for next booking + tier extension',
        'urgency':     'HIGH — imminent churn risk',
        'color':       '#e74c3c',
    },
    'Miles Hoarders': {
        'description': 'High points balance, low redemption — emotionally detached',
        'strategy':    'Redemption incentives to re-engage with the program',
        'channel':     'Email campaign',
        'offer':       '"Use 10K miles, earn 2K bonus" limited-time offer',
        'urgency':     'Medium — engage before balance lapses',
        'color':       '#f39c12',
    },
    'Seasonal Travelers': {
        'description': 'Periodic activity peaks (holiday / vacation flyers)',
        'strategy':    'Targeted campaigns before their typical travel window',
        'channel':     'Email 4 weeks before peak season',
        'offer':       'Vacation bundle + bonus miles on leisure bookings',
        'urgency':     'Low-Medium — seasonal activation',
        'color':       '#3498db',
    },
    'Rising Stars': {
        'description': 'Growing engagement, increasing flight frequency',
        'strategy':    'Nurture to next tier — show path to elite status',
        'channel':     'App + Email',
        'offer':       'Tier upgrade challenge + bonus mile accelerator',
        'urgency':     'Low — accelerate growth trajectory',
        'color':       '#9b59b6',
    },
    'Budget Frequent Flyers': {
        'description': 'High frequency but low CLV — price-sensitive travelers',
        'strategy':    'Volume rewards + partner benefits to increase spend',
        'channel':     'Email + App',
        'offer':       'Co-branded card offer, hotel / car partner promotions',
        'urgency':     'Medium — risk of switching to cheaper competitor',
        'color':       '#1abc9c',
    },
    'At-Risk VIPs': {
        'description': 'Previously high-value customers showing sharp decline',
        'strategy':    'Maximum intervention — executive outreach justified',
        'channel':     'Personal call + priority email + lounge invitation',
        'offer':       'Tier retention + 20K miles + dedicated account manager',
        'urgency':     'CRITICAL — losing these customers is most costly',
        'color':       '#c0392b',
    },
}

PREFERRED_CLUSTER_FEATURES = [
    'recency_days', 'total_flights', 'avg_flights_mo', 'clv',
    'engagement_score', 'consistency_score', 'momentum_ratio',
    'redemption_rate', 'yoy_growth', 'seasonality_score',
]


class CustomerSegmentationEngine:
    """Cluster customers and assign business-meaningful segment names."""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters  = n_clusters
        self.final_path  = Path("data/final")
        self.fig_path    = Path("outputs/figures/segmentation")
        self.fig_path.mkdir(parents=True, exist_ok=True)
        self.scaler      = StandardScaler()

    # ── Data loading ─────────────────────────────────────────────────────────

    def load_features(self):
        self.df = pd.read_csv(self.final_path / 'customer_features.csv')
        print(f"Loaded feature matrix: {self.df.shape}")

    # ── Clustering ────────────────────────────────────────────────────────────

    def select_cluster_features(self) -> list:
        available = [c for c in PREFERRED_CLUSTER_FEATURES if c in self.df.columns]
        if len(available) < 3:
            available = [c for c in self.df.columns
                         if self.df[c].dtype in [np.float64, np.int64]
                         and c not in ('churn', 'loyalty_number')][:10]
        return available

    def find_optimal_k(self, k_range=range(3, 8)) -> int:
        features  = self.select_cluster_features()
        X         = self.scaler.fit_transform(self.df[features].fillna(0))
        best_k, best_score = self.n_clusters, -1
        for k in k_range:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            score  = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_k = score, k
        print(f"Optimal k={best_k} (silhouette={best_score:.3f})")
        self.n_clusters = best_k
        return best_k

    def fit(self):
        features = self.select_cluster_features()
        X        = self.scaler.fit_transform(self.df[features].fillna(0))
        km       = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.df['cluster']   = km.fit_predict(X)
        self.cluster_centers = km.cluster_centers_
        self.cluster_features = features
        print(f"Fitted {self.n_clusters}-cluster segmentation.")

    # ── Segment naming ────────────────────────────────────────────────────────

    def assign_segment_names(self):
        profiles = self.df.groupby('cluster')[self.cluster_features].mean()
        archetype_list = list(SEGMENT_ARCHETYPES.keys())
        name_map = {}

        for cluster_id, row in profiles.iterrows():
            clv_pct         = row.get('clv', 0)
            engagement_pct  = row.get('engagement_score', 50)
            recency         = row.get('recency_days', 180)
            momentum        = row.get('momentum_ratio', 1.0)
            redemption      = row.get('redemption_rate', 0.5)
            yoy             = row.get('yoy_growth', 0)
            seasonality     = row.get('seasonality_score', 0)

            clv_rank  = profiles['clv'].rank(pct=True).get(cluster_id, 0.5) if 'clv' in profiles else 0.5
            eng_rank  = profiles.get('engagement_score', pd.Series()).rank(pct=True).get(cluster_id, 0.5) \
                        if 'engagement_score' in profiles else 0.5

            if clv_rank > 0.75 and eng_rank > 0.6:
                name = 'Premium Loyalists'
            elif clv_rank > 0.75 and eng_rank < 0.4:
                name = 'At-Risk VIPs'
            elif eng_rank < 0.3 and recency > 200:
                name = 'Silent Drifters'
            elif redemption < 0.2 and clv_rank > 0.4:
                name = 'Miles Hoarders'
            elif momentum > 1.3:
                name = 'Rising Stars'
            elif seasonality > 0.5:
                name = 'Seasonal Travelers'
            else:
                name = 'Budget Frequent Flyers'

            # Avoid duplicates
            if name in name_map.values():
                remaining = [a for a in archetype_list if a not in name_map.values()]
                name = remaining[0] if remaining else f'Segment {cluster_id}'
            name_map[cluster_id] = name

        self.df['segment_name'] = self.df['cluster'].map(name_map)
        self.name_map = name_map
        print("Segment names assigned:", list(name_map.values()))

    # ── Visualisations ────────────────────────────────────────────────────────

    def plot_pca_scatter(self):
        X    = self.scaler.transform(self.df[self.cluster_features].fillna(0))
        pca  = PCA(n_components=2)
        Xr   = pca.fit_transform(X)
        colors = [SEGMENT_ARCHETYPES.get(n, {}).get('color', '#888888')
                  for n in self.df['segment_name']]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(Xr[:, 0], Xr[:, 1], c=colors, alpha=0.5, s=10)
        patches = [mpatches.Patch(color=SEGMENT_ARCHETYPES[n]['color'], label=n)
                   for n in self.name_map.values() if n in SEGMENT_ARCHETYPES]
        ax.legend(handles=patches, loc='upper right', fontsize=8)
        ax.set_title('Customer Segments — PCA Projection')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.tight_layout()
        plt.savefig(self.fig_path / 'segments_pca.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_segment_overview(self):
        counts = self.df['segment_name'].value_counts()
        colors = [SEGMENT_ARCHETYPES.get(n, {}).get('color', '#888888') for n in counts.index]
        fig, ax = plt.subplots(figsize=(10, 5))
        counts.plot(kind='bar', ax=ax, color=colors, alpha=0.85)
        ax.set_title('Customer Segment Distribution')
        ax.set_xlabel('Segment')
        ax.set_ylabel('Customers')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.fig_path / 'segment_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ── Save outputs ──────────────────────────────────────────────────────────

    def save(self):
        self.final_path.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.final_path / 'customer_features_segmented.csv', index=False)
        segment_col = ['segment_name'] + \
                      [c for c in self.df.columns if c not in ('segment_name', 'cluster')]
        self.df[['loyalty_number', 'segment_name', 'cluster'] if 'loyalty_number' in self.df.columns
                else ['segment_name', 'cluster']].to_csv(
            self.final_path / 'customer_segments.csv', index=False)

        profiles = self.df.groupby('segment_name')[self.cluster_features].mean().round(3)
        if 'churn' in self.df.columns:
            profiles['churn_rate'] = self.df.groupby('segment_name')['churn'].mean().round(3)
        profiles['size'] = self.df['segment_name'].value_counts()
        profiles.to_csv(self.final_path / 'segment_profiles.csv')
        print("Saved: customer_features_segmented.csv, customer_segments.csv, segment_profiles.csv")

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self):
        self.load_features()
        self.find_optimal_k()
        self.fit()
        self.assign_segment_names()
        self.plot_pca_scatter()
        self.plot_segment_overview()
        self.save()
        return self.df
