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
        
        # Try intelligent estimation using free APIs
        try:
            # Geocode start location
            geocode_url = "https://nominatim.openstreetmap.org/search"
            headers = {'User-Agent': 'CorkChoralCarbonCalculator/1.0'}
            
            start_response = requests.get(
                geocode_url,
                params={'q': start, 'format': 'json', 'limit': 1},
                headers=headers,
                timeout=5
            )
            time.sleep(1)  # Rate limiting - be nice to free service
            
            end_response = requests.get(
                geocode_url,
                params={'q': end, 'format': 'json', 'limit': 1},
                headers=headers,
                timeout=5
            )
            
            if start_response.status_code == 200 and end_response.status_code == 200:
                start_data = start_response.json()
                end_data = end_response.json()
                
                if start_data and end_data:
                    start_lat = float(start_data[0]['lat'])
                    start_lon = float(start_data[0]['lon'])
                    end_lat = float(end_data[0]['lat'])
                    end_lon = float(end_data[0]['lon'])
                    
                    # Use OSRM for routing distance
                    osrm_url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
                    route_response = requests.get(
                        osrm_url,
                        params={'overview': 'false'},
                        timeout=5
                    )
                    
                    if route_response.status_code == 200:
                        route_data = route_response.json()
                        if route_data.get('routes'):
                            # Distance in meters, convert to km
                            distance_km = route_data['routes'][0]['distance'] / 1000
                            return round(distance_km, 1)
        
        except Exception as e:
            # If API fails, fall back to pattern matching
            pass
        
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
        """Generate optimisation scenarios with specific, actionable recommendations"""
        df = pd.DataFrame(self.journeys)
        current_emissions = df['total_emissions'].sum()
        
        # Analyze current transport breakdown
        ground_journeys = df[~df['transport_mode'].str.lower().str.contains('airplane|flight')]
        flight_journeys = df[df['transport_mode'].str.lower().str.contains('airplane|flight')]
        
        ground_emissions = ground_journeys['total_emissions'].sum() if len(ground_journeys) > 0 else 0
        flight_emissions = flight_journeys['total_emissions'].sum() if len(flight_journeys) > 0 else 0
        
        # GREEN SCENARIO: Maximum use of festival buses
        green_df = df.copy()
        for idx, row in green_df.iterrows():
            if 'airplane' not in row['transport_mode'].lower():
                # Switch all ground transport to festival buses
                new_emissions = row['distance'] * EMISSION_FACTORS['bus_coach'] * self.num_people
                green_df.at[idx, 'total_emissions'] = new_emissions
        green_emissions = green_df['total_emissions'].sum()
        green_ground_savings = ground_emissions - (green_emissions - flight_emissions)
        
        # FLEXIBLE SCENARIO: Mix of cars and private transport
        flex_df = df.copy()
        for idx, row in flex_df.iterrows():
            if 'airplane' not in row['transport_mode'].lower():
                # Use cars instead of buses (higher emissions but more flexibility)
                new_emissions = row['distance'] * EMISSION_FACTORS['car'] * self.num_people
                flex_df.at[idx, 'total_emissions'] = new_emissions
        flex_emissions = flex_df['total_emissions'].sum()
        
        # BALANCED SCENARIO: Strategic mix
        balanced_df = df.copy()
        for idx, row in balanced_df.iterrows():
            if 'airplane' not in row['transport_mode'].lower():
                # Use festival bus for main transfers, keep current for others
                if row['distance'] > 10:  # Longer journeys use festival bus
                    new_emissions = row['distance'] * EMISSION_FACTORS['bus_coach'] * self.num_people
                else:  # Short journeys keep current arrangement
                    new_emissions = row['total_emissions']
                balanced_df.at[idx, 'total_emissions'] = new_emissions
        balanced_emissions = balanced_df['total_emissions'].sum()
        
        scenarios = {
            'current': {
                'name': 'Current Plan',
                'emissions': current_emissions,
                'emissions_per_person': current_emissions / self.num_people,
                'saving': 0,
                'saving_pct': 0,
                'cost_index': 100,
                'convenience': 100,
                'description': 'Your current travel arrangements',
                'actions': [
                    f"Continue with current plans",
                    f"Total emissions: {current_emissions:.0f} kg CO₂",
                    f"Flights account for {(flight_emissions/current_emissions*100):.0f}% of emissions" if flight_emissions > 0 else "No flight emissions",
                    f"Ground transport: {(ground_emissions/current_emissions*100):.0f}% of emissions" if ground_emissions > 0 else "No ground transport emissions"
                ]
            },
            'green': {
                'name': 'Green Option',
                'emissions': green_emissions,
                'emissions_per_person': green_emissions / self.num_people,
                'saving': current_emissions - green_emissions,
                'saving_pct': ((current_emissions - green_emissions) / current_emissions * 100),
                'cost_index': 85,
                'convenience': 90,
                'description': 'Use festival buses for ALL ground transport (airport, venues, hotels)',
                'actions': [
                    f"SPECIFIC CHANGES: Replace all private cars, taxis, and public buses with festival coaches",
                    f"Contact festival coordinator to book {len(ground_journeys)} bus journeys for your {self.num_people} people",
                    f"Share your performance schedule to align with bus timetable",
                    f"Saves {green_ground_savings:.0f} kg CO₂ on ground transport ({(green_ground_savings/ground_emissions*100):.0f}% reduction)" if ground_emissions > 0 else "Optimizes ground transport",
                    f"Estimated cost saving: 15% (shared coach vs multiple vehicles)"
                ]
            },
            'flexible': {
                'name': 'Flexible Option',
                'emissions': flex_emissions,
                'emissions_per_person': flex_emissions / self.num_people,
                'saving': current_emissions - flex_emissions,
                'saving_pct': ((current_emissions - flex_emissions) / current_emissions * 100),
                'cost_index': 140,
                'convenience': 110,
                'description': 'Use private cars/taxis for maximum scheduling freedom',
                'actions': [
                    f"SPECIFIC CHANGES: Book {max(1, self.num_people // 4)} hire cars or use taxi services",
                    f"Allows independent arrival/departure times for different groups",
                    f"No dependency on festival bus schedules - full flexibility",
                    f"Additional cost: ~40% more than festival buses",
                    f"Additional emissions: {abs(current_emissions - flex_emissions):.0f} kg CO₂ ({abs((current_emissions - flex_emissions)/current_emissions*100):.0f}% increase)" if flex_emissions > current_emissions else "Similar emissions to current plan"
                ]
            },
            'balanced': {
                'name': 'Balanced Approach',
                'emissions': balanced_emissions,
                'emissions_per_person': balanced_emissions / self.num_people,
                'saving': current_emissions - balanced_emissions,
                'saving_pct': ((current_emissions - balanced_emissions) / current_emissions * 100),
                'cost_index': 92,
                'convenience': 98,
                'description': 'Festival buses for main transfers, flexible transport for early/late events',
                'actions': [
                    f"SPECIFIC CHANGES: Use festival buses for {len([j for j in self.journeys if j['distance'] > 10])} long-distance trips (airport, main venues)",
                    f"Keep taxi/car flexibility for {len([j for j in self.journeys if j['distance'] <= 10])} short local trips (late performances, small venues)",
                    f"Book 1-2 backup taxis for emergencies or schedule conflicts",
                    f"Saves {(current_emissions - balanced_emissions):.0f} kg CO₂ while maintaining schedule flexibility",
                    f"Cost increase: ~8% more than green option, but 32% cheaper than all-private transport"
                ]
            }
        }
        
        return scenarios
    
    def create_comparison_chart(self, scenario_key):
        """Create simplified comparison chart for a specific scenario"""
        scenarios = self.generate_scenarios()
        current = scenarios['current']
        comparison = scenarios[scenario_key]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Chart 1: Emissions comparison
        categories = ['Current Plan', comparison['name']]
        emissions = [current['emissions'], comparison['emissions']]
        colors = ['#3498db', '#2ecc71' if comparison['saving'] > 0 else '#e74c3c']
        
        bars = ax1.bar(categories, emissions, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=2)
        ax1.set_ylabel('Total CO₂ Emissions (kg)', fontsize=12, fontweight='bold')
        
        if comparison['saving'] > 0:
            title = f"Reduces emissions by {comparison['saving']:.0f} kg ({comparison['saving_pct']:.0f}%)"
        else:
            title = f"Increases emissions by {abs(comparison['saving']):.0f} kg ({abs(comparison['saving_pct']):.0f}%)"
        ax1.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, emission in zip(bars, emissions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{emission:.0f} kg',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Chart 2: Per person comparison
        per_person = [current['emissions_per_person'], comparison['emissions_per_person']]
        bars2 = ax2.bar(categories, per_person, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=2)
        ax2.set_ylabel('CO₂ Emissions per Person (kg)', fontsize=12, fontweight='bold')
        ax2.set_title('Per Person Impact', fontsize=13, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, emission in zip(bars2, per_person):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{emission:.1f} kg',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.suptitle(f'{self.team_name}: {comparison["name"]} vs Current Plan',
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
        
        # Generate comparison charts
        charts = {}
        for scenario_key in ['green', 'flexible', 'balanced']:
            charts[scenario_key] = analyzer.create_comparison_chart(scenario_key)
        
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
            'charts': charts
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
