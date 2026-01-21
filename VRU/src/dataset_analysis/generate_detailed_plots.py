import json
import matplotlib.pyplot as plt
import os
import numpy as np
import textwrap

def generate_detailed_plots():
    json_path = 'dimension_keyword_analysis.json'
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Scientific style settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'ytick.labelsize': 10,  # Important for horizontal bar labels
        'xtick.labelsize': 10,
        'figure.autolayout': True,
        'figure.dpi': 300
    })
    
    # Use a color palette
    # We will assign a color to each "Category Group" (Major Category) to distinguish the sub-items
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    dimensions_data = data['dimensions']
    
    for dim_name, dim_info in dimensions_data.items():
        # Collection phase
        # We want a list of items: (Label, Count, GroupName)
        plot_items = []
        
        categories = dim_info['categories']
        
        # Iterate over Major Groups (e.g., "Time", "Sky")
        for group_name, group_data in categories.items():
            keywords = group_data.get('keywords', {})
            for sub_label, sub_stats in keywords.items():
                count = sub_stats['count']
                if count > 0: # Only plot non-zero items? Or all defined ones? Non-zero is safer for clutter.
                    plot_items.append({
                        'label': sub_label,
                        'count': count,
                        'group': group_name
                    })
        
        # Add Others
        others_count = dim_info['others']['count']
        if others_count > 0:
            plot_items.append({
                'label': 'Unclassified (Others)',
                'count': others_count,
                'group': 'Others'
            })

        if not plot_items:
            continue

        # Sort items
        # Strategy: Group by 'group', and within group sort by 'count' descending?
        # Or just sort by Group Name to keep related items together?
        # Let's preserve the order defined in the config (which usually groups logical things) 
        # but sort major groups by total count if we wanted dynamic ordering.
        # Here we rely on the insertion order in plot_items, which follows the JSON loop order.
        # But we might want to reverse it for barh (so first item is at top).
        
        plot_items.reverse() # For barh, bottom is index 0.

        labels = [item['label'] for item in plot_items]
        counts = [item['count'] for item in plot_items]
        groups = [item['group'] for item in plot_items]
        
        # Determine colors based on groups
        unique_groups = list(set(groups))
        # Ensure 'Others' is last or gray if possible.
        if 'Others' in unique_groups:
            unique_groups.remove('Others')
            unique_groups.append('Others')
            
        group_color_map = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}
        group_color_map['Others'] = '#999999' # Distinct gray for others
        
        bar_colors = [group_color_map[g] for g in groups]

        # Plotting
        fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4))) # Dynamic height
        
        rects = ax.barh(np.arange(len(labels)), counts, color=bar_colors, edgecolor='none', height=0.6)
        
        # Add Legend for Groups
        handles = [plt.Rectangle((0,0),1,1, color=group_color_map[g]) for g in unique_groups]
        ax.legend(handles, unique_groups, title="Category Groups", loc='lower right')

        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Frequency')
        ax.set_title(f'Feature Distribution: {dim_name}')
        
        # Add value labels
        ax.bar_label(rects, padding=3, fmt='%d')
        
        # Improve grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False) # Clean look
        
        safe_name = dim_name.replace(' ', '_').replace('&', 'and').replace('/', '_')
        output_path = os.path.join(output_dir, f'{safe_name}_detailed_dist.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Generated {output_path}")

    # --- Summary Plot (Overall Extraction) ---
    # Keep the overall plot as it is useful for context
    # ...

if __name__ == "__main__":
    generate_detailed_plots()
