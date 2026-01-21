import json
import matplotlib.pyplot as plt
import os
import numpy as np
import textwrap

def generate_plots():
    json_path = 'dimension_keyword_analysis.json'
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Use a style suitable for scientific publications
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('bmh') # Fallback
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.autolayout': True
    })

    dimensions_data = data['dimensions']
    
    # --- Plot 1: Overall Extraction Rates ---
    dim_names = []
    total_q = []
    extracted_q = []
    
    for name, info in dimensions_data.items():
        dim_names.append(name.replace(' & ', '\n& ')) # Wrap text for labels
        total_q.append(info['total_questions'])
        extracted_q.append(info['extracted_options_count'])

    x = np.arange(len(dim_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, total_q, width, label='Total Questions', color='#B0C4DE')
    rects2 = ax.bar(x + width/2, extracted_q, width, label='Extracted Answers', color='#4682B4')

    ax.set_ylabel('Number of Questions')
    ax.set_title('Keyword Extraction Coverage by Dimension')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, rotation=0)
    ax.legend()
    ax.grid(axis='x') # vertical grid is distracting

    # Add counts
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_coverage.png'), dpi=300)
    plt.close()

    # --- Plot 2...N: Per Dimension Category Distribution ---
    for dim_name, info in dimensions_data.items():
        categories = info['categories']
        others_count = info['others']['count']
        
        # Prepare data: Category Counts including Others
        labels = []
        counts = []
        
        # Collect regular categories
        for cat_name, cat_data in categories.items():
            labels.append(cat_name)
            counts.append(cat_data['total_count'])
            
        # Add Others
        if others_count > 0:
            labels.append('Unclassified\n(Others)')
            counts.append(others_count)

        # Sort by count desc
        if counts:
            sorted_indices = np.argsort(counts)[::-1]
            labels = [labels[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]

        # Colors
        colors = plt.cm.GnBu(np.linspace(0.4, 0.9, len(counts)))

        fig, ax = plt.subplots(figsize=(8, 6))
        rects = ax.bar(labels, counts, color=colors)
        
        ax.set_ylabel('Frequency')
        ax.set_title(f'Category Distribution: {dim_name}')
        ax.set_xticks(np.arange(len(labels)))
        
        # Wrap long labels
        wrapped_labels = ['\n'.join(textwrap.wrap(l, 15)) for l in labels]
        ax.set_xticklabels(wrapped_labels, rotation=45, ha='right')
        
        ax.bar_label(rects, padding=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        safe_name = dim_name.replace(' ', '_').replace('&', 'and').replace('/', '_')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe_name}_distribution.png'), dpi=300)
        plt.close()

    print(f"Comparison plots generated in {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    generate_plots()
