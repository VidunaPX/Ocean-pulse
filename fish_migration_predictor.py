# src/fish_migration_predictor.py
# Fish Migration Prediction based on Ocean Temperature Changes
# Part of OceanPulse Planetary Health Tracker

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
import os
from datetime import datetime, timedelta

class FishMigrationPredictor:
    """
    Advanced fish migration prediction system based on ocean temperature patterns
    """
    
    def __init__(self):
        self.fish_species = self._initialize_fish_species()
        self.migration_thresholds = self._define_migration_thresholds()
        
    def _initialize_fish_species(self):
        """
        Initialize fish species with their temperature preferences and migration patterns
        """
        species = {
            'tuna': {
                'preferred_temp': 22.0,  # °C
                'temp_tolerance': 5.0,    # °C
                'migration_speed': 50,    # km/day
                'habitat_depth': 'surface',
                'commercial_value': 'high'
            },
            'salmon': {
                'preferred_temp': 12.0,
                'temp_tolerance': 3.0,
                'migration_speed': 30,
                'habitat_depth': 'mixed',
                'commercial_value': 'high'
            },
            'cod': {
                'preferred_temp': 8.0,
                'temp_tolerance': 4.0,
                'migration_speed': 20,
                'habitat_depth': 'bottom',
                'commercial_value': 'high'
            },
            'mackerel': {
                'preferred_temp': 18.0,
                'temp_tolerance': 6.0,
                'migration_speed': 40,
                'habitat_depth': 'surface',
                'commercial_value': 'medium'
            },
            'sardine': {
                'preferred_temp': 16.0,
                'temp_tolerance': 4.0,
                'migration_speed': 25,
                'habitat_depth': 'surface',
                'commercial_value': 'medium'
            }
        }
        return species
    
    def _define_migration_thresholds(self):
        """
        Define temperature thresholds that trigger fish migration
        """
        return {
            'optimal': 0.5,      # °C deviation from preferred temp
            'stress': 2.0,        # °C deviation causing stress
            'migration': 3.0,     # °C deviation triggering migration
            'critical': 5.0       # °C deviation causing population decline
        }
    
    def predict_fish_migration(self, sst_data, sst_anomaly, lat, lon, time):
        """
        Predict fish migration patterns based on temperature changes
        """
        print("🐟 Predicting fish migration patterns...")
        
        migration_predictions = {}
        
        for species_name, species_info in self.fish_species.items():
            print(f"   Analyzing {species_name} migration...")
            
            # Calculate habitat suitability
            habitat_suitability = self._calculate_habitat_suitability(
                sst_data, species_info, lat, lon
            )
            
            # Predict migration directions
            migration_vectors = self._predict_migration_vectors(
                sst_anomaly, species_info, lat, lon
            )
            
            # Identify migration corridors
            migration_corridors = self._identify_migration_corridors(
                migration_vectors, lat, lon
            )
            
            # Calculate migration timing
            migration_timing = self._calculate_migration_timing(
                sst_anomaly, species_info, time
            )
            
            migration_predictions[species_name] = {
                'habitat_suitability': habitat_suitability,
                'migration_vectors': migration_vectors,
                'migration_corridors': migration_corridors,
                'migration_timing': migration_timing,
                'species_info': species_info
            }
        
        return migration_predictions
    
    def _calculate_habitat_suitability(self, sst_data, species_info, lat, lon):
        """
        Calculate habitat suitability based on temperature preferences
        """
        preferred_temp = species_info['preferred_temp']
        temp_tolerance = species_info['temp_tolerance']
        
        # Calculate suitability score (0-1)
        temp_deviation = np.abs(sst_data - preferred_temp)
        suitability = np.exp(-temp_deviation / temp_tolerance)
        
        # Normalize to 0-1 range
        suitability = np.clip(suitability, 0, 1)
        
        return suitability
    
    def _predict_migration_vectors(self, sst_anomaly, species_info, lat, lon):
        """
        Predict migration direction vectors based on temperature gradients
        """
        # Calculate temperature gradients
        grad_lat, grad_lon = np.gradient(sst_anomaly)
        
        # Migration vectors point towards cooler water (negative gradient)
        migration_lat = -grad_lat
        migration_lon = -grad_lon
        
        # Normalize vectors
        vector_magnitude = np.sqrt(migration_lat**2 + migration_lon**2)
        migration_lat = np.divide(migration_lat, vector_magnitude + 1e-8)
        migration_lon = np.divide(migration_lon, vector_magnitude + 1e-8)
        
        return migration_lat, migration_lon
    
    def _identify_migration_corridors(self, migration_vectors, lat, lon):
        """
        Identify major migration corridors using clustering
        """
        migration_lat, migration_lon = migration_vectors
        
        # Find points with significant migration potential
        migration_magnitude = np.sqrt(migration_lat**2 + migration_lon**2)
        significant_points = migration_magnitude > np.percentile(migration_magnitude, 80)
        
        # Get coordinates of significant migration points
        lat_indices, lon_indices = np.where(significant_points)
        migration_coords = np.column_stack([
            lat[lat_indices], 
            lon[lon_indices]
        ])
        
        # Cluster migration points to identify corridors
        if len(migration_coords) > 10:
            clustering = DBSCAN(eps=2.0, min_samples=5).fit(migration_coords)
            corridor_labels = clustering.labels_
        else:
            corridor_labels = np.zeros(len(migration_coords))
        
        return {
            'coordinates': migration_coords,
            'labels': corridor_labels,
            'magnitude': migration_magnitude[significant_points]
        }
    
    def _calculate_migration_timing(self, sst_anomaly, species_info, time):
        """
        Calculate optimal migration timing based on temperature trends
        """
        # Calculate temperature trends
        temp_trend = np.gradient(sst_anomaly, axis=0)
        
        # Find optimal migration periods (when temperature change is significant)
        migration_periods = np.abs(temp_trend) > 0.5  # °C per time step
        
        # Calculate migration urgency
        urgency = np.abs(temp_trend) / species_info['temp_tolerance']
        
        return {
            'trend': temp_trend,
            'periods': migration_periods,
            'urgency': urgency
        }
    
    def create_migration_visualization(self, migration_predictions, lat, lon, sst_anomaly):
        """
        Create comprehensive fish migration visualization
        """
        fig = plt.figure(figsize=(24, 16))
        
        # Create subplots for each species
        species_list = list(migration_predictions.keys())
        
        for i, species_name in enumerate(species_list):
            ax = plt.subplot(2, 3, i+1, projection=ccrs.PlateCarree())
            ax.set_global()
            ax.coastlines(resolution='50m', linewidth=0.8, color='#2d3748')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='#4a5568')
            ax.add_feature(cfeature.OCEAN, color='#1a365d', alpha=0.1)
            ax.add_feature(cfeature.LAND, color='#2d3748', alpha=0.1)
            
            # Plot habitat suitability
            suitability = migration_predictions[species_name]['habitat_suitability']
            im = ax.pcolormesh(lon, lat, suitability, cmap='RdYlGn', 
                             shading='auto', transform=ccrs.PlateCarree(), alpha=0.7)
            
            # Plot migration vectors
            migration_vectors = migration_predictions[species_name]['migration_vectors']
            migration_lat, migration_lon = migration_vectors
            
            # Sample vectors for visualization
            step = 10
            ax.quiver(lon[::step], lat[::step], migration_lon[::step, ::step], 
                     migration_lat[::step, ::step], 
                     transform=ccrs.PlateCarree(), color='blue', alpha=0.6, scale=20)
            
            # Plot migration corridors
            corridors = migration_predictions[species_name]['migration_corridors']
            if len(corridors['coordinates']) > 0:
                unique_labels = np.unique(corridors['labels'])
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                
                for label, color in zip(unique_labels, colors):
                    if label == -1:  # Noise points
                        continue
                    mask = corridors['labels'] == label
                    if np.any(mask):
                        ax.scatter(corridors['coordinates'][mask, 1], 
                                 corridors['coordinates'][mask, 0],
                                 c=[color], s=20, alpha=0.8, 
                                 transform=ccrs.PlateCarree())
            
            ax.set_title(f'{species_name.title()} Migration\n'
                        f'Preferred Temp: {migration_predictions[species_name]["species_info"]["preferred_temp"]}°C',
                        fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label('Habitat Suitability', fontsize=10)
        
        # Global migration summary
        ax_summary = plt.subplot(2, 3, 6)
        ax_summary.axis('off')
        
        # Calculate global migration metrics
        total_suitable_habitat = sum(
            np.sum(pred['habitat_suitability'] > 0.7) 
            for pred in migration_predictions.values()
        )
        
        total_migration_corridors = sum(
            len(pred['migration_corridors']['coordinates']) 
            for pred in migration_predictions.values()
        )
        
        # Species-specific impacts
        species_impacts = []
        for species_name, pred in migration_predictions.items():
            suitable_area = np.sum(pred['habitat_suitability'] > 0.7)
            total_area = pred['habitat_suitability'].size
            suitability_percentage = (suitable_area / total_area) * 100
            
            species_impacts.append({
                'species': species_name,
                'suitability': suitability_percentage,
                'commercial_value': pred['species_info']['commercial_value']
            })
        
        # Create impact summary
        impact_text = f"""
🐟 FISH MIGRATION IMPACT ASSESSMENT

📊 GLOBAL METRICS:
• Total Suitable Habitat: {total_suitable_habitat:,} grid cells
• Migration Corridors: {total_migration_corridors} identified
• Species Analyzed: {len(migration_predictions)}

🎯 SPECIES IMPACT:
"""
        
        for impact in species_impacts:
            status = "🟢 GOOD" if impact['suitability'] > 30 else "🟡 MODERATE" if impact['suitability'] > 15 else "🔴 POOR"
            impact_text += f"• {impact['species'].title()}: {impact['suitability']:.1f}% suitable ({status})\n"
        
        impact_text += f"""
🌊 FISHERIES IMPACT:
• High-Value Species: {sum(1 for i in species_impacts if i['commercial_value'] == 'high')} at risk
• Migration Pressure: {'HIGH' if total_migration_corridors > 100 else 'MODERATE'}
• Habitat Loss: {100 - np.mean([i['suitability'] for i in species_impacts]):.1f}% average

⚠️ RECOMMENDATIONS:
• Monitor temperature trends closely
• Adjust fishing quotas based on migration
• Protect critical migration corridors
• Implement adaptive management strategies
        """
        
        ax_summary.text(0.05, 0.95, impact_text, transform=ax_summary.transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7fafc', alpha=0.8))
        
        plt.suptitle('🐟 OceanPulse: Fish Migration Prediction System\n'
                    'Planetary Health Tracker | Fisheries Impact Assessment', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        return fig
    
    def generate_fisheries_recommendations(self, migration_predictions):
        """
        Generate actionable recommendations for fisheries management
        """
        recommendations = []
        
        for species_name, pred in migration_predictions.items():
            suitability = pred['habitat_suitability']
            high_suitability = np.sum(suitability > 0.7)
            total_area = suitability.size
            suitability_percentage = (high_suitability / total_area) * 100
            
            if suitability_percentage < 20:
                recommendations.append({
                    'species': species_name,
                    'priority': 'HIGH',
                    'action': 'Reduce fishing pressure immediately',
                    'reason': f'Only {suitability_percentage:.1f}% of habitat is suitable'
                })
            elif suitability_percentage < 40:
                recommendations.append({
                    'species': species_name,
                    'priority': 'MEDIUM',
                    'action': 'Monitor closely and prepare for quota adjustments',
                    'reason': f'{suitability_percentage:.1f}% of habitat is suitable'
                })
            else:
                recommendations.append({
                    'species': species_name,
                    'priority': 'LOW',
                    'action': 'Continue current management practices',
                    'reason': f'{suitability_percentage:.1f}% of habitat is suitable'
                })
        
        return recommendations

def main():
    """
    Main execution function for fish migration prediction
    """
    print("🐟 Initializing Fish Migration Predictor...")
    
    # Initialize predictor
    predictor = FishMigrationPredictor()
    
    # Load ocean temperature data (simulated)
    print("🌊 Loading ocean temperature data...")
    # This would normally load from actual datasets
    lat = np.linspace(-90, 90, 180)
    lon = np.linspace(-180, 180, 360)
    time = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
    
    # Generate synthetic SST data
    sst_data = np.random.normal(15, 10, (len(time), len(lat), len(lon)))
    sst_anomaly = np.random.normal(0, 2, (len(time), len(lat), len(lon)))
    
    # Predict fish migration
    migration_predictions = predictor.predict_fish_migration(
        sst_data[-1], sst_anomaly[-1], lat, lon, time
    )
    
    # Create visualizations
    print("📊 Creating migration visualizations...")
    os.makedirs("data/output", exist_ok=True)
    
    fig = predictor.create_migration_visualization(
        migration_predictions, lat, lon, sst_anomaly[-1]
    )
    plt.savefig("Frontend/public", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate recommendations
    recommendations = predictor.generate_fisheries_recommendations(migration_predictions)
    
    print("📋 Fisheries Recommendations:")
    for rec in recommendations:
        print(f"   {rec['species'].title()}: {rec['action']} ({rec['priority']} priority)")
    
    print("✅ Fish migration analysis completed!")
    print("📁 Output saved to: data/output/fish_migration_analysis.png")
    
    return predictor, migration_predictions, recommendations

if __name__ == "__main__":
    predictor, migration_predictions, recommendations = main()
