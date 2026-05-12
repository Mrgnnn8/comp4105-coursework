"""
Serves as a list which informs the agent of all legal move combinations available. 
"""

ATTRIBUTE_MOVES = {
    "continent": [
        "Asia", "Africa", "North America", "South America",
        "Antarctica", "Europe", "Oceania"
    ],

    "region": [
        "Northern Europe", "Western Europe", "Eastern Europe", "Southern Europe",
        "Western Africa", "Eastern Africa", "Middle Africa", "Northern Africa", "Southern Africa",
        "Central Asia", "Eastern Asia", "Southern Asia", "Southeast Asia", "Western Asia",
        "Caribbean", "Central America", "Northern America",
        "South America",
        "Australia and New Zealand", "Melanesia", "Micronesia", "Polynesia"
    ],

    "landlocked": ["Yes", "No"],
    "is_island": ["Yes", "No"],

    "which_ocean": [
        "Pacific", "Atlantic", "Indian", "Southern", "Arctic", "None"
    ],

    "number_of_land_borders": [
        "0", "1", "2", "3", "4", "5", "6", "7", "8",
        "9", "10", "11", "12", "13", "14"
    ],

    "largest_biome": [
        "tropical rainforest",
        "temperate forest",
        "boreal forest",
        "savanna",
        "grassland",
        "desert",
        "tundra",
        "mediterranean shrubland",
        "mountain/alpine"
    ],

    "has_desert": ["Yes", "No"],
    "has_rainforest": ["Yes", "No"],
    "has_mountains": ["Yes", "No"],

    "average_altitude_band": [
        "Lowland (< 200m)",
        "Low-to-moderate (200-500m)",
        "Moderate (500-1000m)",
        "High (1000-2000m)",
        "Very high (> 2000m)"
    ],

    "volcanic_activity": ["Yes", "No"],

    "climate_zone": [
        "Tropical", "Temperate", "Polar", "Arid",
        "Mediterranean", "Highland"
    ],

    "average_temperature_band": [
        "Very cold (< 0 C)",
        "Cold (0-10 C)",
        "Mild (10-20 C)",
        "Warm (20-25 C)",
        "Hot (> 25 C)"
    ],

    "has_tropical_region": ["Yes", "No"],
    "has_arctic_region": ["Yes", "No"],
    "monsoon_season": ["Yes", "No"],

    "average_rainfall_band": [
        "Arid (< 250 mm)",
        "Semi-arid (250-500 mm)",
        "Moderate (500-1000 mm)",
        "Wet (1000-2000 mm)",
        "Very wet (> 2000 mm)"
    ],

    "population_band": [
        "Micro (< 1M)",
        "Small (1M-10M)",
        "Medium (10M-50M)",
        "Large (50M-200M)",
        "Mega (> 200M)"
    ],

    "population_density_band": [
        "Very sparse (< 25)",
        "Sparse (25-100)",
        "Moderate (100-300)",
        "Dense (300-1000)",
        "Very dense (> 1000)"
    ],

    "official_language_family": [
        "Indo-European",
        "Sino-Tibetan",
        "Afro-Asiatic",
        "Niger-Congo",
        "Austronesian",
        "Turkic",
        "Austroasiatic",
        "Kra-Dai",
        "Koreanic",
        "Japonic",
        "Mongolic",
        "Kartvelian",
        "Dravidian",
        "Uralic"
    ],

    "number_of_official_languages": [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "11", "12", "13", "14", "15", "16", "17", "18",
        "19", "20", "21", "22"
    ],

    "dominant_religion": [
        "Christianity",
        "Islam",
        "Hinduism",
        "Buddhism",
        "Judaism",
        "Non-religious"
    ],

    "urbanisation_rate_band": [
        "Mostly rural (< 30%)",
        "Rural-leaning (30-50%)",
        "Mixed (50-70%)",
        "Urban-leaning (70-85%)",
        "Highly urban (> 85%)"
    ],

    "median_age_band": [
        "Very young (< 20)",
        "Young (20-25)",
        "Middle (25-35)",
        "Older (35-42)",
        "Aged (> 42)"
    ],

    "gdp_per_capita_band": [
        "Low income (< $1,000)",
        "Lower-middle ($1,000-$4,000)",
        "Upper-middle ($4,000-$15,000)",
        "High ($15,000-$40,000)",
        "Very high (> $40,000)"
    ],

    "primary_industry": [
        "agriculture",
        "manufacturing",
        "services",
        "tourism",
        "mining/extractives",
        "oil & gas",
        "finance",
        "fisheries"
    ],

    "major_export_category": [
        "machinery",
        "vehicles",
        "electronics",
        "oil & gas",
        "minerals",
        "agricultural products",
        "textiles",
        "services"
    ],

    "currency_type": [
        "Euro",
        "CFA Franc West",
        "CFA Franc Central",
        "Eastern Caribbean Dollar (ECCU)",
        "US Dollar (official use)",
        "pegged to USD",
        "pegged to Euro",
        "own free-floating currency",
        "own pegged currency (other)"
    ],

    "government_type": [
        "presidential republic",
        "parliamentary republic",
        "semi-presidential republic",
        "constitutional monarchy",
        "absolute monarchy",
        "one-party state",
        "military junta",
        "theocracy",
        "transitional/provisional government"
    ],

    "un_member": ["Yes", "No"],
    "nato_member": ["Yes", "No"],
    "commonwealth_member": ["Yes", "No"],
    "eu_member": ["Yes", "No"],
    "gained_independence_after_1900": ["Yes", "No"],
    "drives_on_left": ["Yes", "No"],
    "metric_system": ["Yes", "No"],
    "colonial_history": ["Yes", "No"],
}