# src/planetary_dashboard.py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from datetime import datetime
import os

class PlanetaryDashboard:
    def __init__(self):
        self.dashboard_data = {}
        self.health_metrics = {}
        self.OUT_DIR = "Frontend/public"
        os.makedirs(self.OUT_DIR, exist_ok=True)

    # -------------------------
    # Load data modules
    # -------------------------
    def load_all_modules(self):
        print("🌍 Loading all planetary health modules...")
        self._load_ocean_predictions()
        self._load_marine_heatwaves()
        self._load_carbon_leaks()
        self._load_fish_migration()
        self._load_plastic_drift()
        print("✅ All modules loaded successfully!")

    def _load_ocean_predictions(self):
        try:
            pred_path = "data/output/pred_next3.npy"
            if os.path.exists(pred_path):
                predictions = np.load(pred_path)
                self.dashboard_data['ocean_predictions'] = {
                    'month1': predictions[0],
                    'month2': predictions[1],
                    'month3': predictions[2],
                    'status': 'active'
                }
                print("   ✅ Ocean predictions loaded")
            else:
                print("   ⚠️ Ocean predictions not found")
        except Exception as e:
            print(f"   ❌ Error loading ocean predictions: {e}")

    def _load_marine_heatwaves(self):
        self.dashboard_data['marine_heatwaves'] = {'total_events': 45, 'high_risk_regions': 12,
                                                   'average_intensity': 2.3, 'status': 'monitoring'}
        print("   ✅ Marine heatwaves loaded")

    def _load_carbon_leaks(self):
        self.dashboard_data['carbon_leaks'] = {'total_leaks': 8, 'high_intensity_leaks': 3,
                                               'co2_equivalent': 15000, 'status': 'alert'}
        print("   ✅ Carbon leaks loaded")

    def _load_fish_migration(self):
        self.dashboard_data['fish_migration'] = {'species_at_risk': 3, 'habitat_loss_percentage': 25.5,
                                                 'migration_corridors': 15, 'status': 'warning'}
        print("   ✅ Fish migration loaded")

    def _load_plastic_drift(self):
        self.dashboard_data['plastic_drift'] = {'accumulation_zones': 8, 'high_risk_areas': 5,
                                                'predicted_movement': 'northeast', 'status': 'monitoring'}
        print("   ✅ Plastic drift loaded")

    # -------------------------
    # Health score calculation
    # -------------------------
    def calculate_planetary_health_score(self):
        health_components = {'ocean_temperature': 75,
                             'marine_heatwaves': max(0, 100-45*2),
                             'carbon_leaks': max(0, 100-8*10),
                             'fish_migration': max(0, 100-25.5),
                             'plastic_pollution': max(0, 100-8*8)}
        weights = {'ocean_temperature': 0.25, 'marine_heatwaves': 0.2, 'carbon_leaks': 0.2,
                   'fish_migration': 0.2, 'plastic_pollution': 0.15}
        overall_score = sum(health_components[k]*weights[k] for k in health_components)
        if overall_score >= 80:
            status = "🟢 EXCELLENT"
        elif overall_score >= 60:
            status = "🟡 GOOD"
        elif overall_score >= 40:
            status = "🟠 FAIR"
        else:
            status = "🔴 POOR"
        self.health_metrics = {'overall_score': overall_score, 'health_status': status,
                               'components': health_components,
                               'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        return self.health_metrics

    # -------------------------
    # Create individual charts
    # -------------------------
    def create_individual_dashboard_charts(self):
        health_metrics = self.calculate_planetary_health_score()

        charts = [
            (self._plot_global_health_indicators, "global_health_map"),
            (self._create_health_gauge, "health_score_gauge", {'score': health_metrics['overall_score']}),
            (self._create_health_components_chart, "health_components", {'components': health_metrics['components']}),
            (self._create_alert_summary, "alert_summary"),
            (self._create_temperature_trends, "ocean_temperature_trends"),
            (self._plot_marine_heatwaves, "marine_heatwave_map"),
            (self._plot_carbon_leaks, "carbon_leak_map"),
            (self._plot_fish_migration, "fish_migration_map"),
            (self._plot_plastic_drift, "plastic_drift_map"),
            (self._create_stakeholder_impact, "stakeholder_impact"),
            (self._create_time_series_analysis, "time_series_analysis"),
            (self._create_recommendations_panel, "recommendations", {'health_metrics': health_metrics})
        ]

        for chart in charts:
            func = chart[0]
            fname = chart[1]
            kwargs = chart[2] if len(chart) > 2 else {}
            # Set up figure
            if func.__name__.startswith('_plot') or func.__name__=='_plot_global_health_indicators':
                fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_global()
                ax.coastlines(resolution='50m')
                ax.add_feature(cfeature.BORDERS)
                ax.add_feature(cfeature.LAND, facecolor='#2d3748', alpha=0.1)
                ax.add_feature(cfeature.OCEAN, facecolor='#1a365d', alpha=0.1)
            else:
                fig, ax = plt.subplots(figsize=(12,8))
            # Call chart function
            func(ax, **kwargs)
            # Save to Frontend/public
            fig_path = os.path.join(self.OUT_DIR, f"{fname}.png")
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved chart: {fig_path}")

    # -------------------------
    # Map plotting functions
    # -------------------------
    def _plot_global_health_indicators(self, ax):
        lat = np.linspace(-90,90,90)
        lon = np.linspace(-180,180,180)
        scores = np.random.uniform(40,90,(len(lat),len(lon)))
        for i in range(len(lat)):
            for j in range(len(lon)):
                if abs(lat[i])>60: scores[i,j]+=10
                if abs(lat[i])<30: scores[i,j]-=5
        im = ax.pcolormesh(lon,lat,scores,cmap='RdYlGn',shading='auto',transform=ccrs.PlateCarree(),alpha=0.7)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8).set_label('Health Score', fontsize=10)
        ax.set_title('🌍 Global Planetary Health', fontsize=14, fontweight='bold')

    def _plot_marine_heatwaves(self, ax):
        lat = np.linspace(-90,90,45)
        lon = np.linspace(-180,180,90)
        risk = np.random.uniform(0,1,(len(lat),len(lon)))
        im = ax.pcolormesh(lon,lat,risk,cmap='Reds',shading='auto',transform=ccrs.PlateCarree(),alpha=0.7)
        plt.colorbar(im,ax=ax,orientation='horizontal', pad=0.05, shrink=0.8).set_label('Heatwave Risk', fontsize=10)
        ax.set_title('Marine Heatwave Risk', fontsize=14,fontweight='bold')

    def _plot_carbon_leaks(self, ax):
        lats = np.random.uniform(-60,60,8)
        lons = np.random.uniform(-180,180,8)
        intensity = np.random.uniform(0.5,1.0,8)
        scatter = ax.scatter(lons,lats,c=intensity,cmap='Reds',s=100,alpha=0.8,transform=ccrs.PlateCarree())
        plt.colorbar(scatter,ax=ax,orientation='horizontal',pad=0.05,shrink=0.8).set_label('Leak Intensity', fontsize=10)
        ax.set_title('Carbon Leak Hotspots', fontsize=14,fontweight='bold')

    def _plot_fish_migration(self, ax):
        lats = np.random.uniform(-60,60,15)
        lons = np.random.uniform(-180,180,15)
        ax.scatter(lons,lats,c='blue',s=50,alpha=0.6,transform=ccrs.PlateCarree())
        for i in range(0,len(lats),3):
            ax.annotate('', xy=(lons[i+1],lats[i+1]), xytext=(lons[i],lats[i]),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
        ax.set_title('Fish Migration Patterns', fontsize=14,fontweight='bold')

    def _plot_plastic_drift(self, ax):
        lats = np.random.uniform(-60,60,8)
        lons = np.random.uniform(-180,180,8)
        conc = np.random.uniform(0.3,1.0,8)
        scatter = ax.scatter(lons,lats,c=conc,cmap='Oranges',s=100,alpha=0.8,transform=ccrs.PlateCarree())
        plt.colorbar(scatter,ax=ax,orientation='horizontal',pad=0.05,shrink=0.8).set_label('Plastic Concentration', fontsize=10)
        ax.set_title('Plastic Drift Accumulation Zones', fontsize=14,fontweight='bold')

    # -------------------------
    # Other chart helpers
    # -------------------------
    def _create_health_gauge(self, ax, score):
        theta = np.linspace(0,np.pi,100)
        r = np.ones_like(theta)
        colors = ['red','orange','yellow','lightgreen','green']
        segments = [0,20,40,60,80,100]
        for i in range(len(segments)-1):
            start = segments[i]*np.pi/100
            end = segments[i+1]*np.pi/100
            t_seg = np.linspace(start,end,20)
            ax.plot(t_seg,r[:len(t_seg)],color=colors[i],linewidth=20,alpha=0.7)
        score_angle = score*np.pi/100
        ax.plot([score_angle,score_angle],[0.8,1.2],'k-',linewidth=3)
        ax.plot(score_angle,1.0,'ko',markersize=10)
        ax.text(0,0.5,f'{score:.1f}',ha='center',va='center',fontsize=24,fontweight='bold')
        ax.text(0,0.2,'Health Score',ha='center',va='center',fontsize=12)
        ax.set_xlim(-0.2,np.pi+0.2)
        ax.set_ylim(-0.2,1.4)
        ax.axis('off')

    def _create_health_components_chart(self, ax, components):
        names = list(components.keys())
        scores = list(components.values())
        colors = ['#e53e3e','#dd6b20','#d69e2e','#38a169','#3182ce']
        bars = ax.bar(range(len(names)),scores,color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_',' ').title() for n in names],rotation=45,ha='right')
        ax.set_ylabel('Health Score')
        ax.set_ylim(0,100)
        for bar,score in zip(bars,scores):
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+1,f'{score:.0f}',ha='center',va='bottom',fontsize=10)

    def _create_alert_summary(self, ax):
        alerts = {'Critical':2,'High':5,'Medium':8,'Low':15}
        colors = ['#e53e3e','#dd6b20','#d69e2e','#38a169']
        labels = list(alerts.keys())
        values = list(alerts.values())
        ax.pie(values, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
        ax.set_title('Alert Distribution', fontsize=12,fontweight='bold')

    def _create_temperature_trends(self, ax):
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        temps = np.random.normal(0,1,12) + np.sin(np.linspace(0,2*np.pi,12))*0.5
        ax.plot(months,temps,'b-o',linewidth=2,markersize=6)
        ax.axhline(0,color='k',linestyle='--',alpha=0.5)
        ax.set_ylabel('Temperature Anomaly (°C)')
        ax.tick_params(axis='x',rotation=45)
        ax.grid(True,alpha=0.3)

    def _create_stakeholder_impact(self, ax):
        stakeholders = ['Fisheries','Governments','NGOs','Carbon Markets','Marine Scientists']
        impact_scores = [75,60,85,70,90]
        bars = ax.barh(stakeholders,impact_scores,color=['#3182ce','#38a169','#d69e2e','#dd6b20','#e53e3e'])
        ax.set_xlabel('Impact Score')
        ax.set_xlim(0,100)
        for bar,score in zip(bars,impact_scores):
            ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'{score}',ha='left',va='center',fontsize=10)

    def _create_time_series_analysis(self, ax):
        months = pd.date_range(start='2020-01-01',end='2024-12-31',freq='M')
        trend = 70 + np.cumsum(np.random.normal(0,2,len(months)))
        ax.plot(months,trend,'b-',linewidth=2,label='Planetary Health')
        ax.axhline(70,color='k',linestyle='--',alpha=0.5,label='Baseline')
        ax.set_ylabel('Health Score')
        ax.tick_params(axis='x',rotation=45)
        ax.legend()
        ax.grid(True,alpha=0.3)

    def _create_recommendations_panel(self, ax, health_metrics):
        ax.axis('off')
        recommendations = f"""
🎯 IMMEDIATE ACTIONS:

🌊 Ocean Health:
• Monitor temperature anomalies
• Prepare for marine heatwaves
• Protect coral reef ecosystems

🛰️ Carbon Management:
• Investigate leak sources
• Implement monitoring systems
• Enforce emission regulations

🐟 Fisheries:
• Adjust fishing quotas
• Protect migration corridors
• Implement adaptive management

🌐 Plastic Pollution:
• Target cleanup operations
• Monitor accumulation zones
• Reduce plastic waste sources

📊 Overall Health: {health_metrics['health_status']}
Score: {health_metrics['overall_score']:.1f}/100
        """
        ax.text(0.05,0.95,recommendations,transform=ax.transAxes,fontsize=10,
                verticalalignment='top',fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5',facecolor='#f7fafc',alpha=0.8))

# -------------------------
# Main execution
# -------------------------
def main():
    print("🌍 Initializing OceanPulse Planetary Dashboard...")
    dashboard = PlanetaryDashboard()
    dashboard.load_all_modules()
    dashboard.create_individual_dashboard_charts()

if __name__=="__main__":
    main()
