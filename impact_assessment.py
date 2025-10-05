# src/impact_assessment.py
# Stakeholder Impact Assessment and Early Warning System
# Part of OceanPulse Planetary Health Tracker

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ImpactMetric:
    """
    Data class for impact metrics
    """
    stakeholder: str
    metric_name: str
    current_value: float
    threshold: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    impact_description: str

class ImpactAssessment:
    """
    Comprehensive impact assessment system for all stakeholders
    """
    
    def __init__(self):
        self.stakeholders = self._initialize_stakeholders()
        self.impact_metrics = {}
        self.early_warnings = []
        self.alert_thresholds = self._define_alert_thresholds()
        
    def _initialize_stakeholders(self):
        """
        Initialize stakeholder categories and their priorities
        """
        return {
            'fisheries': {
                'priority_metrics': ['fish_migration', 'marine_heatwaves', 'ocean_temperature'],
                'impact_weight': 0.25,
                'alert_threshold': 0.7
            },
            'governments': {
                'priority_metrics': ['carbon_leaks', 'marine_heatwaves', 'ocean_temperature'],
                'impact_weight': 0.20,
                'alert_threshold': 0.8
            },
            'ngos': {
                'priority_metrics': ['plastic_pollution', 'marine_heatwaves', 'fish_migration'],
                'impact_weight': 0.20,
                'alert_threshold': 0.6
            },
            'carbon_markets': {
                'priority_metrics': ['carbon_leaks', 'ocean_temperature'],
                'impact_weight': 0.15,
                'alert_threshold': 0.9
            },
            'marine_scientists': {
                'priority_metrics': ['ocean_temperature', 'marine_heatwaves', 'fish_migration'],
                'impact_weight': 0.20,
                'alert_threshold': 0.5
            }
        }
    
    def _define_alert_thresholds(self):
        """
        Define alert thresholds for different impact levels
        """
        return {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
    
    def calculate_stakeholder_impacts(self, planetary_data):
        """
        Calculate impact metrics for all stakeholders
        """
        print("📊 Calculating stakeholder impacts...")
        
        # Initialize impact metrics
        self.impact_metrics = {}
        
        for stakeholder, config in self.stakeholders.items():
            print(f"   Analyzing {stakeholder} impacts...")
            
            # Calculate stakeholder-specific impact score
            impact_score = self._calculate_stakeholder_impact(stakeholder, planetary_data)
            
            # Generate impact metrics
            metrics = self._generate_impact_metrics(stakeholder, impact_score, planetary_data)
            
            # Assess risk level
            risk_level = self._assess_risk_level(impact_score, config['alert_threshold'])
            
            # Store results
            self.impact_metrics[stakeholder] = {
                'impact_score': impact_score,
                'risk_level': risk_level,
                'metrics': metrics,
                'recommendations': self._generate_recommendations(stakeholder, risk_level, metrics)
            }
        
        return self.impact_metrics
    
    def _calculate_stakeholder_impact(self, stakeholder, planetary_data):
        """
        Calculate impact score for a specific stakeholder
        """
        config = self.stakeholders[stakeholder]
        priority_metrics = config['priority_metrics']
        
        # Calculate weighted impact score
        total_impact = 0
        total_weight = 0
        
        for metric in priority_metrics:
            if metric in planetary_data:
                metric_value = planetary_data[metric]
                weight = 1.0 / len(priority_metrics)  # Equal weight for now
                
                # Normalize metric value (0-1 scale)
                normalized_value = self._normalize_metric_value(metric, metric_value)
                
                total_impact += normalized_value * weight
                total_weight += weight
        
        return total_impact / total_weight if total_weight > 0 else 0
    
    def _normalize_metric_value(self, metric, value):
        """
        Normalize metric value to 0-1 scale
        """
        # Define normalization ranges for different metrics
        normalization_ranges = {
            'ocean_temperature': (0, 5),  # °C anomaly
            'marine_heatwaves': (0, 100),  # number of events
            'carbon_leaks': (0, 50),  # number of leaks
            'fish_migration': (0, 100),  # percentage of habitat loss
            'plastic_pollution': (0, 20)  # number of accumulation zones
        }
        
        if metric in normalization_ranges:
            min_val, max_val = normalization_ranges[metric]
            normalized = np.clip((value - min_val) / (max_val - min_val), 0, 1)
            return normalized
        
        return 0.5  # Default neutral value
    
    def _generate_impact_metrics(self, stakeholder, impact_score, planetary_data):
        """
        Generate detailed impact metrics for a stakeholder
        """
        metrics = []
        
        if stakeholder == 'fisheries':
            metrics.extend([
                ImpactMetric(
                    stakeholder='fisheries',
                    metric_name='Fish Migration Risk',
                    current_value=planetary_data.get('fish_migration', 0),
                    threshold=30,
                    trend='increasing',
                    risk_level='high' if impact_score > 0.7 else 'medium',
                    impact_description='Fish species migrating due to temperature changes'
                ),
                ImpactMetric(
                    stakeholder='fisheries',
                    metric_name='Marine Heatwave Impact',
                    current_value=planetary_data.get('marine_heatwaves', 0),
                    threshold=20,
                    trend='increasing',
                    risk_level='critical' if impact_score > 0.8 else 'high',
                    impact_description='Marine heatwaves affecting fish populations'
                )
            ])
        
        elif stakeholder == 'governments':
            metrics.extend([
                ImpactMetric(
                    stakeholder='governments',
                    metric_name='Carbon Leak Incidents',
                    current_value=planetary_data.get('carbon_leaks', 0),
                    threshold=10,
                    trend='increasing',
                    risk_level='critical' if impact_score > 0.8 else 'high',
                    impact_description='Carbon leaks requiring regulatory action'
                ),
                ImpactMetric(
                    stakeholder='governments',
                    metric_name='Disaster Risk',
                    current_value=planetary_data.get('marine_heatwaves', 0),
                    threshold=25,
                    trend='increasing',
                    risk_level='high' if impact_score > 0.7 else 'medium',
                    impact_description='Marine heatwaves causing ecosystem disasters'
                )
            ])
        
        elif stakeholder == 'ngos':
            metrics.extend([
                ImpactMetric(
                    stakeholder='ngos',
                    metric_name='Plastic Pollution Zones',
                    current_value=planetary_data.get('plastic_pollution', 0),
                    threshold=5,
                    trend='increasing',
                    risk_level='high' if impact_score > 0.6 else 'medium',
                    impact_description='Plastic accumulation zones requiring cleanup'
                ),
                ImpactMetric(
                    stakeholder='ngos',
                    metric_name='Ecosystem Stress',
                    current_value=planetary_data.get('marine_heatwaves', 0),
                    threshold=15,
                    trend='increasing',
                    risk_level='critical' if impact_score > 0.7 else 'high',
                    impact_description='Marine ecosystems under stress from heatwaves'
                )
            ])
        
        elif stakeholder == 'carbon_markets':
            metrics.extend([
                ImpactMetric(
                    stakeholder='carbon_markets',
                    metric_name='Carbon Leak Volume',
                    current_value=planetary_data.get('carbon_leaks', 0),
                    threshold=5,
                    trend='increasing',
                    risk_level='critical' if impact_score > 0.9 else 'high',
                    impact_description='Carbon leaks affecting market integrity'
                ),
                ImpactMetric(
                    stakeholder='carbon_markets',
                    metric_name='Offset Verification Risk',
                    current_value=planetary_data.get('ocean_temperature', 0),
                    threshold=2,
                    trend='increasing',
                    risk_level='high' if impact_score > 0.8 else 'medium',
                    impact_description='Ocean temperature changes affecting carbon sinks'
                )
            ])
        
        elif stakeholder == 'marine_scientists':
            metrics.extend([
                ImpactMetric(
                    stakeholder='marine_scientists',
                    metric_name='Research Priority',
                    current_value=planetary_data.get('ocean_temperature', 0),
                    threshold=3,
                    trend='increasing',
                    risk_level='high' if impact_score > 0.5 else 'medium',
                    impact_description='Ocean temperature anomalies requiring research attention'
                ),
                ImpactMetric(
                    stakeholder='marine_scientists',
                    metric_name='Ecosystem Monitoring',
                    current_value=planetary_data.get('marine_heatwaves', 0),
                    threshold=10,
                    trend='increasing',
                    risk_level='critical' if impact_score > 0.6 else 'high',
                    impact_description='Marine heatwaves requiring intensive monitoring'
                )
            ])
        
        return metrics
    
    def _assess_risk_level(self, impact_score, alert_threshold):
        """
        Assess risk level based on impact score and threshold
        """
        if impact_score >= alert_threshold:
            return 'critical'
        elif impact_score >= alert_threshold * 0.8:
            return 'high'
        elif impact_score >= alert_threshold * 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, stakeholder, risk_level, metrics):
        """
        Generate actionable recommendations for stakeholders
        """
        recommendations = []
        
        if stakeholder == 'fisheries':
            if risk_level == 'critical':
                recommendations.extend([
                    "Immediately reduce fishing quotas by 50%",
                    "Implement emergency fishing closures in affected areas",
                    "Coordinate with marine scientists for real-time monitoring"
                ])
            elif risk_level == 'high':
                recommendations.extend([
                    "Prepare for quota adjustments based on migration patterns",
                    "Implement adaptive management strategies",
                    "Monitor fish migration corridors closely"
                ])
            else:
                recommendations.extend([
                    "Continue current management practices",
                    "Monitor temperature trends",
                    "Prepare contingency plans for future changes"
                ])
        
        elif stakeholder == 'governments':
            if risk_level == 'critical':
                recommendations.extend([
                    "Declare environmental emergency in affected regions",
                    "Implement immediate carbon leak investigation",
                    "Activate disaster response protocols"
                ])
            elif risk_level == 'high':
                recommendations.extend([
                    "Increase regulatory enforcement",
                    "Implement stricter emission monitoring",
                    "Prepare for potential disasters"
                ])
            else:
                recommendations.extend([
                    "Continue monitoring programs",
                    "Review and update regulations",
                    "Prepare for future climate impacts"
                ])
        
        elif stakeholder == 'ngos':
            if risk_level == 'critical':
                recommendations.extend([
                    "Launch emergency cleanup operations",
                    "Mobilize volunteer networks",
                    "Coordinate with international organizations"
                ])
            elif risk_level == 'high':
                recommendations.extend([
                    "Increase cleanup efforts in high-risk areas",
                    "Raise public awareness about environmental impacts",
                    "Coordinate with local communities"
                ])
            else:
                recommendations.extend([
                    "Continue regular monitoring",
                    "Maintain cleanup programs",
                    "Educate communities about environmental protection"
                ])
        
        elif stakeholder == 'carbon_markets':
            if risk_level == 'critical':
                recommendations.extend([
                    "Suspend trading in affected carbon credits",
                    "Implement immediate verification protocols",
                    "Coordinate with regulatory bodies"
                ])
            elif risk_level == 'high':
                recommendations.extend([
                    "Increase verification requirements",
                    "Implement stricter monitoring",
                    "Review carbon credit integrity"
                ])
            else:
                recommendations.extend([
                    "Continue standard verification",
                    "Monitor market integrity",
                    "Prepare for potential disruptions"
                ])
        
        elif stakeholder == 'marine_scientists':
            if risk_level == 'critical':
                recommendations.extend([
                    "Launch emergency research missions",
                    "Increase monitoring frequency",
                    "Coordinate with international research networks"
                ])
            elif risk_level == 'high':
                recommendations.extend([
                    "Increase research funding for affected areas",
                    "Expand monitoring networks",
                    "Collaborate with stakeholders"
                ])
            else:
                recommendations.extend([
                    "Continue regular research programs",
                    "Monitor long-term trends",
                    "Prepare for future research needs"
                ])
        
        return recommendations
    
    def generate_early_warnings(self, planetary_data):
        """
        Generate early warnings based on current conditions
        """
        print("⚠️ Generating early warnings...")
        
        warnings = []
        
        # Ocean temperature warnings
        if planetary_data.get('ocean_temperature', 0) > 3:
            warnings.append({
                'type': 'critical',
                'stakeholder': 'all',
                'message': 'Extreme ocean temperature anomalies detected',
                'action': 'Implement emergency response protocols',
                'timeline': 'immediate'
            })
        
        # Marine heatwave warnings
        if planetary_data.get('marine_heatwaves', 0) > 30:
            warnings.append({
                'type': 'high',
                'stakeholder': 'fisheries',
                'message': 'Multiple marine heatwaves detected',
                'action': 'Adjust fishing quotas and protect vulnerable species',
                'timeline': '1-2 weeks'
            })
        
        # Carbon leak warnings
        if planetary_data.get('carbon_leaks', 0) > 15:
            warnings.append({
                'type': 'critical',
                'stakeholder': 'governments',
                'message': 'High number of carbon leaks detected',
                'action': 'Launch immediate investigation and enforcement',
                'timeline': 'immediate'
            })
        
        # Fish migration warnings
        if planetary_data.get('fish_migration', 0) > 40:
            warnings.append({
                'type': 'high',
                'stakeholder': 'fisheries',
                'message': 'Significant fish migration patterns detected',
                'action': 'Implement adaptive management strategies',
                'timeline': '2-4 weeks'
            })
        
        # Plastic pollution warnings
        if planetary_data.get('plastic_pollution', 0) > 10:
            warnings.append({
                'type': 'medium',
                'stakeholder': 'ngos',
                'message': 'Multiple plastic accumulation zones identified',
                'action': 'Coordinate cleanup operations',
                'timeline': '1-3 months'
            })
        
        self.early_warnings = warnings
        return warnings
    
    def create_impact_dashboard(self, planetary_data):
        """
        Create comprehensive impact assessment dashboard
        """
        print("📊 Creating impact assessment dashboard...")
        
        # Calculate impacts
        impacts = self.calculate_stakeholder_impacts(planetary_data)
        warnings = self.generate_early_warnings(planetary_data)
        
        # Create visualization
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Stakeholder Impact Overview
        ax1 = plt.subplot(3, 4, 1)
        stakeholders = list(impacts.keys())
        impact_scores = [impacts[s]['impact_score'] for s in stakeholders]
        risk_levels = [impacts[s]['risk_level'] for s in stakeholders]
        
        colors = {'critical': '#e53e3e', 'high': '#dd6b20', 'medium': '#d69e2e', 'low': '#38a169'}
        bar_colors = [colors[risk] for risk in risk_levels]
        
        bars = ax1.bar(stakeholders, impact_scores, color=bar_colors)
        ax1.set_ylabel('Impact Score')
        ax1.set_title('Stakeholder Impact Overview', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, impact_scores):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Risk Level Distribution
        ax2 = plt.subplot(3, 4, 2)
        risk_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for risk in risk_levels:
            risk_counts[risk] += 1
        
        risk_labels = list(risk_counts.keys())
        risk_values = list(risk_counts.values())
        risk_colors = [colors[risk] for risk in risk_labels]
        
        wedges, texts, autotexts = ax2.pie(risk_values, labels=risk_labels, colors=risk_colors, 
                                          autopct='%1.0f', startangle=90)
        ax2.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        
        # 3. Early Warnings
        ax3 = plt.subplot(3, 4, 3)
        ax3.axis('off')
        
        warning_text = "⚠️ EARLY WARNINGS:\n\n"
        for i, warning in enumerate(warnings[:5]):  # Show first 5 warnings
            warning_text += f"{i+1}. {warning['message']}\n"
            warning_text += f"   Action: {warning['action']}\n"
            warning_text += f"   Timeline: {warning['timeline']}\n\n"
        
        if not warnings:
            warning_text += "No active warnings at this time."
        
        ax3.text(0.05, 0.95, warning_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef2f2', alpha=0.8))
        ax3.set_title('Active Warnings', fontsize=14, fontweight='bold')
        
        # 4. Impact Metrics by Stakeholder
        for i, (stakeholder, impact_data) in enumerate(impacts.items()):
            ax = plt.subplot(3, 4, 4+i)
            
            metrics = impact_data['metrics']
            if metrics:
                metric_names = [m.metric_name for m in metrics]
                metric_values = [m.current_value for m in metrics]
                metric_thresholds = [m.threshold for m in metrics]
                
                x = np.arange(len(metric_names))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, metric_values, width, label='Current', alpha=0.8)
                bars2 = ax.bar(x + width/2, metric_thresholds, width, label='Threshold', alpha=0.6)
                
                ax.set_ylabel('Value')
                ax.set_title(f'{stakeholder.title()} Metrics', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([name.replace(' ', '\n') for name in metric_names], fontsize=8)
                ax.legend()
                
                # Add value labels
                for bar, value in zip(bars1, metric_values):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Recommendations Summary
        ax_summary = plt.subplot(3, 4, 12)
        ax_summary.axis('off')
        
        recommendations_text = "🎯 KEY RECOMMENDATIONS:\n\n"
        for stakeholder, impact_data in impacts.items():
            if impact_data['risk_level'] in ['critical', 'high']:
                recommendations_text += f"{stakeholder.title()}:\n"
                for rec in impact_data['recommendations'][:2]:  # Show top 2 recommendations
                    recommendations_text += f"• {rec}\n"
                recommendations_text += "\n"
        
        ax_summary.text(0.05, 0.95, recommendations_text, transform=ax_summary.transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f9ff', alpha=0.8))
        ax_summary.set_title('Priority Recommendations', fontsize=14, fontweight='bold')
        
        plt.suptitle('🎯 OceanPulse: Stakeholder Impact Assessment\n'
                    'Early Warning System | Actionable Intelligence', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return fig

def main():
    """
    Main execution function for impact assessment
    """
    print("🎯 Initializing OceanPulse Impact Assessment...")
    
    # Initialize impact assessment
    assessment = ImpactAssessment()
    
    # Simulate planetary data
    planetary_data = {
        'ocean_temperature': 2.5,  # °C anomaly
        'marine_heatwaves': 35,   # number of events
        'carbon_leaks': 12,       # number of leaks
        'fish_migration': 45,     # percentage of habitat loss
        'plastic_pollution': 8    # number of accumulation zones
    }
    
    # Create impact dashboard
    print("📊 Creating impact assessment dashboard...")
    os.makedirs("data/output", exist_ok=True)
    
    fig = assessment.create_impact_dashboard(planetary_data)
    plt.savefig("Frontend/public", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Impact assessment completed!")
    print("📁 Output saved to: data/output/impact_assessment.png")
    
    return assessment

if __name__ == "__main__":
    assessment = main()
