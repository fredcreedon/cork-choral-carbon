from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import io
import base64
import os
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# European countries with viable train connections
TRAIN_VIABLE_COUNTRIES = [
    'austria', 'belgium', 'bulgaria', 'croatia', 'czech', 'denmark', 
    'estonia', 'finland', 'france', 'germany', 'greece', 'hungary',
    'ireland', 'italy', 'latvia', 'lithuania', 'luxembourg', 'netherlands',
    'poland', 'portugal', 'romania', 'slovakia', 'slovenia', 'spain',
    'sweden', 'uk', 'united kingdom', 'switzerland', 'norway'
]

# Emission factors (kg CO2 per passenger per km)
EMISSION_FACTORS = {
    'airplane_short': 0.255,
    'airplane_long': 0.195,
    'bus_public': 0.089,
    'bus_coach': 0.027,
    'car': 0.171,
    'train': 0.041,
}

# Approximate distances (km) - Expanded database
DISTANCE_DATABASE = {
    # Cork Airport routes
    'Cork_Airport_to_Cork_City': 8,
    'Cork_Airport_to_Maldron_Shandon': 8,
    'Cork_Airport_to_City_Centre': 8,
    
    # Cork City internal
    'Maldron_Shandon_to_Carrigawohill_Community_College': 25,
    'Carrigawohill_Community_College_to_Maldron_Shandon': 25,
    'Maldron_Shandon_to_Church_of_Christ_our_Light': 7,
    'Church_of_Christ_our_Light_to_Maldron_Shandon': 7,
    'Maldron_Shandon_to_Cork_Airport': 8,
    'Cork_City_to_Ballincollig': 10,
    'Cork_City_to_Carrigawohill': 25,
    
    # Major European cities to Cork
    'Germany_to_Cork': 1450,
    'Berlin_to_Cork': 1800,
    'Munich_to_Cork': 1600,
    'Hamburg_to_Cork': 1700,
    'Frankfurt_to_Cork': 1450,
    'London_to_Cork': 600,
    'Paris_to_Cork': 850,
    'Amsterdam_to_Cork': 1100,
    'Brussels_to_Cork': 1000,
    'Madrid_to_Cork': 1500,
    'Barcelona_to_Cork': 1700,
    'Rome_to_Cork': 2200,
    'Vienna_to_Cork': 1900,
    'Prague_to_Cork': 1800,
    'Warsaw_to_Cork': 2200,
    
    # Irish cities to Cork
    'Dublin_to_Cork': 265,
    'Galway_to_Cork': 210,
    'Limerick_to_Cork': 100,
    'Waterford_to_Cork': 125,
    'Killarney_to_Cork': 90,
    
    # Common venue types (average distances)
    'Airport_to_City_Hotel': 10,
    'Hotel_to_Concert_Hall': 5,
    'Hotel_to_Church': 7,
    'Hotel_to_Community_Centre': 15,
    'City_Centre_to_Suburb': 12,
}

class TravelAnalyzer:
    def __init__(self, team_name, num_people, journeys):
        self.team_name = team_name
        self.num_people = num_people
        self.journeys = journeys
    
    def is_in_europe(self, location):
        """Check if location is in train-viable European countries"""
        location_lower = location.lower()
        return any(country in location_lower for country in TRAIN_VIABLE_COUNTRIES)
    
    def get_optimal_transport(self, distance, start, end, current_mode):
        """Determine optimal transport mode for a journey"""
        # If already using optimal mode for distance, keep it
        if 'festival bus' in current_mode.lower() or 'coach' in current_mode.lower():
            if distance < 500:  # Short distance, bus is optimal
                return current_mode, None
        
        # Check if train is viable for flights
        if 'airplane' in current_mode.lower() or 'flight' in current_mode.lower():
            start_europe = self.is_in_europe(start)
            end_europe = self.is_in_europe(end)
            
            if start_europe and end_europe:
                if distance < 1000:
                    return 'train', 'Train connection likely available - research route options'
                elif distance < 2000:
                    return 'train', 'Multi-leg train journey possible via major hubs (research required)'
        
        # For ground transport, festival bus is optimal
        if distance > 5:  # Don't suggest bus for very short walks
            return 'festival bus', 'Most efficient option for this distance'
        
        return current_mode, None
        
    def estimate_distance(self, start, end):
        """Estimate distance between two locations using Nominatim geocoding and OSRM routing"""
        import requests
        import time
        
        # First try hardcoded database for speed
        start_clean = start.strip().replace(' ', '_')
        end_clean = end.strip().replace(' ', '_')
        
        key1 = f"{start_clean}_to_{end_clean}"
        key2 = f"{end_clean}_to_{start_clean}"
        
        if key1 in DISTANCE_DATABASE:
            return DISTANCE_DATABASE[key1]
        elif key2 in DISTANCE_DATABASE:
            return DISTANCE_DATABASE[key2]
        
        # DISABLED: API calls cause timeout on free tier
        # Try intelligent estimation using free APIs
        # try:
        #     # Geocode start location
        #     geocode_url = "https://nominatim.openstreetmap.org/search"
        #     ...
        # except Exception as e:
        #     pass
        
        # Fallback: Pattern matching for airports
        if 'airport' in start.lower() and ('hotel' in end.lower() or 'city' in end.lower()):
            return 10
        elif 'hotel' in start.lower() and 'airport' in end.lower():
            return 10
            
        # Venue type matching
        venue_patterns = {
            'concert_hall': 5,
            'church': 7,
            'community': 15,
            'centre': 15,
            'college': 15,
        }
        
        for pattern, distance in venue_patterns.items():
            if pattern in end.lower() and 'hotel' in start.lower():
                return distance
        
        # Check if it's international flight
        countries = ['germany', 'france', 'spain', 'italy', 'uk', 'poland', 
                    'czech', 'austria', 'netherlands', 'belgium', 'portugal']
        if any(country in start.lower() for country in countries):
            if 'cork' in end.lower() or 'ireland' in end.lower():
                return 1500  # Average European flight
        
        # Default estimates based on context
        if 'airport' in start.lower() or 'airport' in end.lower():
            return 10  # Airport transfers
        elif any(x in start.lower() or x in end.lower() for x in ['dublin', 'galway', 'limerick']):
            return 150  # Irish inter-city
        else:
            return 15  # Local area
    
    def calculate_emissions(self, distance, transport_mode):
        """Calculate CO2 emissions for a journey"""
        mode = transport_mode.lower()
        if 'airplane' in mode or 'flight' in mode:
            factor = EMISSION_FACTORS['airplane_short']
        elif 'public bus' in mode:
            factor = EMISSION_FACTORS['bus_public']
        elif 'festival bus' in mode or 'coach' in mode:
            factor = EMISSION_FACTORS['bus_coach']
        elif 'car' in mode:
            factor = EMISSION_FACTORS['car']
        elif 'train' in mode:
            factor = EMISSION_FACTORS['train']
        else:
            factor = EMISSION_FACTORS['bus_public']
        
        return distance * factor
    
    def generate_scenarios(self):
        """Generate Your Trip vs Optimised Trip comparison with efficiency rating"""
        df = pd.DataFrame(self.journeys)
        current_emissions = df['total_emissions'].sum()
        
        # Build optimised scenario
        optimised_journeys = []
        optimisation_notes = []
        
        for journey in self.journeys:
            optimal_mode, note = self.get_optimal_transport(
                journey['distance'],
                journey['start_point'],
                journey['arrival_point'],
                journey['transport_mode']
            )
            
            # Calculate optimal emissions
            optimal_emissions_per_person = self.calculate_emissions(journey['distance'], optimal_mode)
            optimal_total = optimal_emissions_per_person * self.num_people
            
            optimised_journeys.append({
                **journey,
                'optimal_mode': optimal_mode,
                'optimal_emissions': optimal_total,
                'note': note
            })
            
            if note and journey['transport_mode'].lower() != optimal_mode.lower():
                optimisation_notes.append({
                    'journey': f"{journey['start_point']} → {journey['arrival_point']}",
                    'current': journey['transport_mode'],
                    'optimal': optimal_mode,
                    'note': note,
                    'saving': journey['total_emissions'] - optimal_total
                })
        
        optimised_emissions = sum(j['optimal_emissions'] for j in optimised_journeys)
        
        # Calculate efficiency rating
        if optimised_emissions > 0:
            efficiency_score = (optimised_emissions / current_emissions) * 100
        else:
            efficiency_score = 100
        
        # Determine rating band
        if efficiency_score >= 95:
            rating = 'A+++'
            rating_label = 'Outstanding'
            rating_color = '#2ecc71'
        elif efficiency_score >= 85:
            rating = 'A'
            rating_label = 'Excellent'
            rating_color = '#27ae60'
        elif efficiency_score >= 70:
            rating = 'B'
            rating_label = 'Good'
            rating_color = '#f39c12'
        elif efficiency_score >= 50:
            rating = 'C'
            rating_label = 'Moderate'
            rating_color = '#e67e22'
        elif efficiency_score >= 30:
            rating = 'D'
            rating_label = 'Below Optimal'
            rating_color = '#e74c3c'
        else:
            rating = 'E'
            rating_label = 'High Impact'
            rating_color = '#c0392b'
        
        # Build what worked well list
        good_choices = []
        improvements = []
        
        for note in optimisation_notes:
            if note['saving'] > 0:
                improvements.append(f"Switch {note['journey']} from {note['current']} to {note['optimal']} (saves {note['saving']:.0f} kg CO₂)")
                if note['note']:
                    improvements.append(f"  → {note['note']}")
        
        # Identify what they did well
        festival_bus_count = sum(1 for j in self.journeys if 'festival' in j['transport_mode'].lower() or 'coach' in j['transport_mode'].lower())
        if festival_bus_count > 0:
            good_choices.append(f"Used festival buses/coaches for {festival_bus_count} journey(s)")
        
        train_count = sum(1 for j in self.journeys if 'train' in j['transport_mode'].lower())
        if train_count > 0:
            good_choices.append(f"Chose train for {train_count} journey(s)")
        
        if not improvements:
            good_choices.append("Already using optimal transport for all journeys!")
        
        scenarios = {
            'your_trip': {
                'name': 'Your Actual Trip',
                'emissions': current_emissions,
                'emissions_per_person': current_emissions / self.num_people,
                'description': f'The carbon footprint of your actual travel arrangements',
                'journeys': self.journeys
            },
            'optimised': {
                'name': 'Optimised Plan',
                'emissions': optimised_emissions,
                'emissions_per_person': optimised_emissions / self.num_people,
                'description': 'Absolute lowest carbon scenario using best available transport',
                'journeys': optimised_journeys,
                'notes': optimisation_notes
            },
            'efficiency': {
                'score': efficiency_score,
                'rating': rating,
                'rating_label': rating_label,
                'rating_color': rating_color,
                'savings_potential': current_emissions - optimised_emissions,
                'savings_pct': ((current_emissions - optimised_emissions) / current_emissions * 100) if current_emissions > 0 else 0,
                'good_choices': good_choices,
                'improvements': improvements[:5]  # Limit to top 5
            }
        }
        
        return scenarios
        return scenarios
    
    def create_comparison_chart(self):
        """Create comparison chart showing Your Trip vs Optimised Plan"""
        scenarios = self.generate_scenarios()
        your_trip = scenarios['your_trip']
        optimised = scenarios['optimised']
        efficiency = scenarios['efficiency']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Chart 1: Total emissions comparison
        categories = ['Your Actual Trip', 'Optimised Plan']
        emissions = [your_trip['emissions'], optimised['emissions']]
        colors = ['#3498db', '#2ecc71']
        
        bars = ax1.bar(categories, emissions, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=2)
        ax1.set_ylabel('Total CO₂ Emissions (kg)', fontsize=12, fontweight='bold')
        
        savings = your_trip['emissions'] - optimised['emissions']
        if savings > 0:
            title = f"Potential savings: {savings:.0f} kg ({efficiency['savings_pct']:.0f}%)"
        else:
            title = "Already at optimal efficiency!"
        ax1.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, emission in zip(bars, emissions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{emission:.0f} kg',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Chart 2: Per person comparison with efficiency rating
        per_person = [your_trip['emissions_per_person'], optimised['emissions_per_person']]
        bars2 = ax2.bar(categories, per_person, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=2)
        ax2.set_ylabel('CO₂ Emissions per Person (kg)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Efficiency Rating: {efficiency["rating"]} ({efficiency["rating_label"]})', 
                     fontsize=13, fontweight='bold', pad=15, color=efficiency['rating_color'])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, emission in zip(bars2, per_person):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{emission:.1f} kg',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.suptitle(f'{self.team_name}: Carbon Efficiency Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64 for web display
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        team_name = data.get('team_name', 'Team')
        num_people = int(data.get('num_people', 35))
        journeys_data = data.get('journeys', [])
        
        # Create temporary analyzer for distance estimation
        temp_analyzer = TravelAnalyzer(team_name, num_people, [])
        
        # Process journeys
        journeys = []
        for j in journeys_data:
            distance = j.get('distance', None)
            
            # If no distance provided, estimate it
            if not distance or distance == '' or float(distance) == 0:
                start = j.get('start_point', '')
                end = j.get('arrival_point', '')
                distance = temp_analyzer.estimate_distance(start, end)
            else:
                distance = float(distance)
            
            transport_mode = j.get('transport_mode', 'bus')
            
            emissions_per_person = temp_analyzer.calculate_emissions(distance, transport_mode)
            
            journey = {
                'event': j.get('event', ''),
                'start_point': j.get('start_point', ''),
                'arrival_point': j.get('arrival_point', ''),
                'date': j.get('date', ''),
                'transport_mode': transport_mode,
                'distance': distance,
                'emissions_per_person': emissions_per_person,
                'total_emissions': emissions_per_person * num_people
            }
            journeys.append(journey)
        
        analyzer = TravelAnalyzer(team_name, num_people, journeys)
        scenarios = analyzer.generate_scenarios()
        
        # Generate single comparison chart
        chart = analyzer.create_comparison_chart()
        
        # Calculate summary stats
        total_emissions = sum(j['total_emissions'] for j in journeys)
        total_distance = sum(j['distance'] for j in journeys)
        
        return jsonify({
            'success': True,
            'summary': {
                'team_name': team_name,
                'num_people': num_people,
                'total_emissions': round(total_emissions, 1),
                'emissions_per_person': round(total_emissions / num_people, 1),
                'total_distance': round(total_distance, 1),
                'car_equivalent': round(total_emissions / 0.171, 0),
                'trees_needed': round(total_emissions / 21, 0)
            },
            'scenarios': scenarios,
            'chart': chart
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read Excel file
        df = pd.read_excel(file)
        
        # Extract team info
        team_name = df.columns[0]  # First column header should be team name
        num_people = int(df.columns[1].split()[0])  # Extract number from second column
        
        # Create temporary analyzer for distance estimation
        temp_analyzer = TravelAnalyzer(team_name, num_people, [])
        
        # Process journeys
        journeys = []
        for idx, row in df.iterrows():
            event = row.iloc[0]
            start_point = row.iloc[1]
            arrival_point = row.iloc[2]
            date = row.iloc[3]
            transport_mode = row.iloc[4]
            
            # Try to get distance from file, or estimate it
            if len(row) > 5 and pd.notna(row.iloc[5]) and row.iloc[5] != '':
                distance = float(row.iloc[5])
            else:
                # Estimate distance using our intelligent function
                distance = temp_analyzer.estimate_distance(str(start_point), str(arrival_point))
            
            journey = {
                'event': event,
                'start_point': start_point,
                'arrival_point': arrival_point,
                'date': str(date),
                'transport_mode': transport_mode,
                'distance': distance
            }
            journeys.append(journey)
        
        return jsonify({
            'success': True,
            'team_name': team_name,
            'num_people': num_people,
            'journeys': journeys
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    # Support deployment platforms (Render, Railway, etc.)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False') == 'True'
    app.run(host='0.0.0.0', port=port, debug=debug)
