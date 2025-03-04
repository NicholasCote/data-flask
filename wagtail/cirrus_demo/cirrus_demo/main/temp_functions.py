import numpy as np
from matplotlib import pyplot as plt

def plot_temperature(temp_values, time_values, location_name=None):
    """
    Plot temperature values on a line plot.
    
    Args:
        temp_values: Temperature values in Kelvin
        time_values: Corresponding time values
        location_name: Name of the location for the title
    """
    # Convert from Kelvin to Celsius
    temp_celsius = temp_values - 273.15
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot the temperature line
    ax.plot(time_values, temp_celsius, color='red', linewidth=1.5)
    
    title = '2-meter Temperature'
    if location_name:
        title += f' for {location_name}'
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Improve x-axis formatting
    plt.xticks(rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add some annotations
    mean_temp = np.mean(temp_celsius)
    max_temp = np.max(temp_celsius)
    min_temp = np.min(temp_celsius)
    
    ax.axhline(y=mean_temp, color='blue', linestyle='--', alpha=0.7)
    ax.text(time_values[5], mean_temp + 0.5, f'Mean: {mean_temp:.1f}°C', 
            color='blue', fontsize=10)
    
    # Mark the max temperature
    max_idx = np.argmax(temp_celsius)
    ax.scatter(time_values[max_idx], max_temp, color='darkred', s=50, zorder=5)
    ax.text(time_values[max_idx], max_temp + 0.5, f'Max: {max_temp:.1f}°C', 
            color='darkred', fontsize=10, ha='right')
    
    # Mark the min temperature
    min_idx = np.argmin(temp_celsius)
    ax.scatter(time_values[min_idx], min_temp, color='darkblue', s=50, zorder=5)
    ax.text(time_values[min_idx], min_temp - 0.5, f'Min: {min_temp:.1f}°C', 
            color='darkblue', fontsize=10, ha='left')
    
    plt.tight_layout()
    
    return fig