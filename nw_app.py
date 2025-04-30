import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import io
import base64
import os
import re
import sys

# Add the current directory to the path to import the NeedlemanWunsch class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the NeedlemanWunsch class
from needleman_wunsch import NeedlemanWunsch, visualize_alignment_enhanced


# Add the enhanced visualization methods to the NeedlemanWunsch class
def add_enhanced_visualizations():
    """Add all the visualization methods to the NeedlemanWunsch class"""
    # Add the existing scalable methods
    NeedlemanWunsch.visualize_matrix_scalable = visualize_matrix_scalable
    NeedlemanWunsch.visualize_alignment_scalable = visualize_alignment_scalable

    # Add the new enhanced methods
    NeedlemanWunsch.visualize_matrix_interactive = visualize_matrix_interactive
    NeedlemanWunsch.visualize_alignment_enhanced = visualize_alignment_enhanced
    NeedlemanWunsch._plot_alignment_segment = _plot_alignment_segment
    NeedlemanWunsch._plot_alignment_segment_with_conservation = _plot_alignment_segment_with_conservation


# Define the scalable visualization functions
def visualize_matrix_scalable(self, seq1, seq2, title="Score Matrix", max_size=20):
    """
    Visualize the score matrix as a heatmap with automatic scaling for larger sequences.

    Parameters:
    - seq1, seq2: The sequences being aligned
    - title: The title for the plot
    - max_size: Maximum full-detail visualization size (for each dimension)
    """
    if self.score_matrix is None:
        raise ValueError("Run align() first to generate the score matrix.")

    m, n = len(seq1) + 1, len(seq2) + 1

    # Check if sequences are too long for detailed visualization
    if m > max_size + 1 or n > max_size + 1:
        # For larger sequences, use a different visualization approach
        plt.figure(figsize=(12, 10))

        # Create enhanced heatmap for larger matrices
        ax = sns.heatmap(self.score_matrix, cmap="YlGnBu",
                         xticklabels=False, yticklabels=False)

        # Add more informative title
        plt.title(f"{title} ({m - 1}x{n - 1} matrix)")

        # Add axis labels
        plt.xlabel(f"Sequence 2 ({n - 1} bases)")
        plt.ylabel(f"Sequence 1 ({m - 1} bases)")

        # Add corner annotations to help with orientation
        # Add a colorbar with better labeling
        cbar = ax.collections[0].colorbar
        cbar.set_label('Alignment Score')

        # Add optimal score annotation
        optimal_score = self.score_matrix[m - 1, n - 1]
        plt.annotate(f'Optimal Score: {optimal_score:.1f}',
                     xy=(0.5, 0.05), xycoords='figure fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     ha='center', fontsize=12)
    else:
        # For smaller sequences, use the original detailed visualization
        plt.figure(figsize=(10, 8))

        # Create labels with sequence characters
        row_labels = [''] + list(seq1)
        col_labels = [''] + list(seq2)

        # Create heatmap with annotations
        ax = sns.heatmap(self.score_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                         xticklabels=col_labels, yticklabels=row_labels)

        plt.title(f"{title} - Optimal Score: {self.score_matrix[m - 1, n - 1]:.1f}")

    plt.tight_layout()
    return plt


def visualize_alignment_scalable(self, aligned_seq1, aligned_seq2, alignment_visual, max_width=80):
    """
    Visualize the sequence alignment with highlighting, handling longer sequences.

    Parameters:
    - aligned_seq1, aligned_seq2: The aligned sequences
    - alignment_visual: Visual representation of the alignment
    - max_width: Maximum number of characters to show in a single view
    """
    total_length = len(aligned_seq1)

    if total_length <= max_width:
        # For shorter alignments, use the original visualization
        plt.figure(figsize=(min(total_length / 6 + 2, 20), 4))

        # Convert alignment to color coding
        colors = []
        for char in alignment_visual:
            if char == '|':  # Match
                colors.append('green')
            elif char == '.':  # Mismatch
                colors.append('red')
            else:  # Gap
                colors.append('grey')

        # Calculate position for text
        y_positions = [2, 1, 0]  # For seq1, alignment markers, seq2

        # Plot sequences and alignment
        for i, (seq, y) in enumerate(zip([aligned_seq1, alignment_visual, aligned_seq2], y_positions)):
            for j, (char, color) in enumerate(zip(seq, colors)):
                plt.text(j, y, char, ha='center', va='center', color='black',
                         backgroundcolor=color if i != 1 else 'white',
                         alpha=0.3 if i != 1 else 1.0, fontfamily='monospace', fontsize=15)

        # Add a legend
        legend_elements = [
            Patch(facecolor='green', alpha=0.3, label='Match'),
            Patch(facecolor='red', alpha=0.3, label='Mismatch'),
            Patch(facecolor='grey', alpha=0.3, label='Gap')
        ]
        plt.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), ncol=3)

        # Set plot properties
        plt.xlim(-1, len(aligned_seq1))
        plt.ylim(-1, 3)
        plt.axis('off')
        plt.title('Sequence Alignment Visualization')
    else:
        # For longer alignments, create a chunked visualization
        # We'll show the first chunk and stats about the whole alignment
        plt.figure(figsize=(16, 8))

        # First section: Summary statistics
        plt.subplot(2, 1, 1)

        # Calculate alignment statistics
        matches = alignment_visual.count('|')
        mismatches = alignment_visual.count('.')
        gaps = alignment_visual.count(' ')

        match_percent = (matches / total_length) * 100
        mismatch_percent = (mismatches / total_length) * 100
        gap_percent = (gaps / total_length) * 100

        # Create a bar chart of alignment statistics
        categories = ['Matches', 'Mismatches', 'Gaps']
        values = [match_percent, mismatch_percent, gap_percent]
        colors = ['green', 'red', 'grey']

        plt.bar(categories, values, color=colors, alpha=0.6)
        plt.title(f'Alignment Statistics (Total Length: {total_length} bp)')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)

        # Add text labels
        for i, v in enumerate(values):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center')

        # Second section: First chunk of the alignment
        plt.subplot(2, 1, 2)

        # Only display the first chunk
        display_length = min(max_width, total_length)

        # Convert alignment to color coding for the displayed chunk
        colors = []
        for char in alignment_visual[:display_length]:
            if char == '|':  # Match
                colors.append('green')
            elif char == '.':  # Mismatch
                colors.append('red')
            else:  # Gap
                colors.append('grey')

        # Calculate position for text
        y_positions = [2, 1, 0]  # For seq1, alignment markers, seq2

        # Plot sequences and alignment for the displayed chunk
        for i, (seq, y) in enumerate(zip([aligned_seq1[:display_length],
                                          alignment_visual[:display_length],
                                          aligned_seq2[:display_length]], y_positions)):
            for j, (char, color) in enumerate(zip(seq, colors)):
                plt.text(j, y, char, ha='center', va='center', color='black',
                         backgroundcolor=color if i != 1 else 'white',
                         alpha=0.3 if i != 1 else 1.0, fontfamily='monospace', fontsize=12)

        # Set plot properties
        plt.xlim(-1, display_length)
        plt.ylim(-1, 3)
        plt.axis('off')
        plt.title(f'First {display_length} positions of alignment (showing {display_length}/{total_length})')

        # Add a legend
        legend_elements = [
            Patch(facecolor='green', alpha=0.3, label='Match'),
            Patch(facecolor='red', alpha=0.3, label='Mismatch'),
            Patch(facecolor='grey', alpha=0.3, label='Gap')
        ]
        plt.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.tight_layout()
    return plt


# New enhanced visualization methods
def visualize_matrix_interactive(self, seq1, seq2, title="Score Matrix", max_size=20):
    """
    Advanced visualization of the score matrix with interactive features for larger sequences.

    Parameters:
    - seq1, seq2: The sequences being aligned
    - title: The title for the plot
    - max_size: Maximum full-detail visualization size (for each dimension)
    """
    if self.score_matrix is None:
        raise ValueError("Run align() first to generate the score matrix.")

    m, n = len(seq1) + 1, len(seq2) + 1

    # For very large matrices, create a multi-scale view
    if m > max_size * 3 or n > max_size * 3:
        # Create a figure with subplots for different views
        fig = plt.figure(figsize=(15, 12))

        # 1. Overview of entire matrix (top left)
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        sns.heatmap(self.score_matrix, cmap="YlGnBu",
                    xticklabels=False, yticklabels=False, ax=ax1)
        ax1.set_title("Overview")

        # 2. Start region (top right)
        start_size = min(max_size, m - 1, n - 1)
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        start_matrix = self.score_matrix[:start_size + 1, :start_size + 1]
        row_labels = [''] + list(seq1[:start_size])
        col_labels = [''] + list(seq2[:start_size])
        sns.heatmap(start_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                    xticklabels=col_labels, yticklabels=row_labels, ax=ax2)
        ax2.set_title("Starting Region")

        # 3. End region (bottom left)
        end_size = min(max_size, m - 1, n - 1)
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        end_matrix = self.score_matrix[m - end_size - 1:, n - end_size - 1:]
        row_labels = list(seq1[-end_size:])
        col_labels = list(seq2[-end_size:])
        sns.heatmap(end_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                    xticklabels=col_labels, yticklabels=row_labels, ax=ax3)
        ax3.set_title("Ending Region")

        # 4. Diagonal region around optimal path (bottom right)
        mid_m, mid_n = m // 2, n // 2
        half_size = min(max_size // 2, m // 4, n // 4)
        mid_matrix = self.score_matrix[mid_m - half_size:mid_m + half_size,
                     mid_n - half_size:mid_n + half_size]

        mid_row_labels = list(seq1[mid_m - half_size - 1:mid_m + half_size - 1])
        mid_col_labels = list(seq2[mid_n - half_size - 1:mid_n + half_size - 1])

        ax4 = plt.subplot2grid((2, 2), (1, 1))
        sns.heatmap(mid_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                    xticklabels=mid_col_labels, yticklabels=mid_row_labels, ax=ax4)
        ax4.set_title("Middle Region")

        plt.suptitle(f"{title} ({m - 1}x{n - 1} matrix) - Multi-scale View", fontsize=16)

    elif m > max_size or n > max_size:
        # For moderately large matrices, use the existing scalable approach
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(self.score_matrix, cmap="YlGnBu",
                         xticklabels=False, yticklabels=False)
        plt.title(f"{title} ({m - 1}x{n - 1} matrix)")
        plt.xlabel(f"Sequence 2 ({n - 1} bases)")
        plt.ylabel(f"Sequence 1 ({m - 1} bases)")

        # Add optimal score annotation
        optimal_score = self.score_matrix[m - 1, n - 1]
        plt.annotate(f'Optimal Score: {optimal_score:.1f}',
                     xy=(0.5, 0.05), xycoords='figure fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     ha='center', fontsize=12)
    else:
        # For small matrices, use detailed visualization
        plt.figure(figsize=(10, 8))
        row_labels = [''] + list(seq1)
        col_labels = [''] + list(seq2)
        ax = sns.heatmap(self.score_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                         xticklabels=col_labels, yticklabels=row_labels)
        plt.title(f"{title} - Optimal Score: {self.score_matrix[m - 1, n - 1]:.1f}")

    plt.tight_layout()
    return plt


def _plot_alignment_segment(self, seq1, seq2, alignment_visual, title="Sequence Alignment"):
    """Helper function to plot a segment of the alignment."""
    # Convert alignment to color coding
    colors = []
    for char in alignment_visual:
        if char == '|':  # Match
            colors.append('green')
        elif char == '.':  # Mismatch
            colors.append('red')
        else:  # Gap
            colors.append('grey')

    # Calculate position for text
    y_positions = [2, 1, 0]  # For seq1, alignment markers, seq2

    # Plot sequences and alignment
    for i, (seq, y) in enumerate(zip([seq1, alignment_visual, seq2], y_positions)):
        for j, (char, color) in enumerate(zip(seq, colors)):
            plt.text(j, y, char, ha='center', va='center', color='black',
                     backgroundcolor=color if i != 1 else 'white',
                     alpha=0.3 if i != 1 else 1.0, fontfamily='monospace', fontsize=12)

    # Set plot properties
    plt.xlim(-1, len(seq1))
    plt.ylim(-1, 3)
    plt.axis('off')
    plt.title(title)

    # Add a legend
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Match'),
        Patch(facecolor='red', alpha=0.3, label='Mismatch'),
        Patch(facecolor='grey', alpha=0.3, label='Gap')
    ]
    plt.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=3)


def _plot_alignment_segment_with_conservation(self, seq1, seq2, alignment_visual, title="Sequence Alignment"):
    """Plots alignment segment with conservation highlighting."""
    # Calculate conservation in a sliding window to identify conserved and variable regions
    window_size = min(5, len(alignment_visual) // 5) if len(alignment_visual) > 15 else 3
    conservation_scores = []

    for i in range(len(alignment_visual)):
        # For each position, look at surrounding window
        start = max(0, i - window_size // 2)
        end = min(len(alignment_visual), i + window_size // 2 + 1)
        window = alignment_visual[start:end]

        # Calculate conservation score (percentage of matches)
        conservation = window.count('|') / len(window)
        conservation_scores.append(conservation)

    # Normalize scores
    max_score = max(conservation_scores)
    min_score = min(conservation_scores)
    range_score = max_score - min_score if max_score > min_score else 1
    normalized_scores = [(score - min_score) / range_score for score in conservation_scores]

    # Create base colors for alignment
    base_colors = []
    for char in alignment_visual:
        if char == '|':  # Match
            base_colors.append('green')
        elif char == '.':  # Mismatch
            base_colors.append('red')
        else:  # Gap
            base_colors.append('grey')

    # Apply conservation gradient to background of sequences
    y_positions = [2, 1, 0]  # For seq1, alignment markers, seq2

    # Plot sequences and alignment with conservation highlighting
    for i, (seq, y) in enumerate(zip([seq1, alignment_visual, seq2], y_positions)):
        for j, (char, color) in enumerate(zip(seq, base_colors)):
            # For the alignment row, just use the basic colors
            if i == 1:
                bg_color = 'white'
                alpha = 1.0
            else:
                # For sequence rows, add conservation gradient overlay
                bg_color = color
                # Higher conservation = more opaque
                alpha = 0.2 + (normalized_scores[j] * 0.5)

            plt.text(j, y, char, ha='center', va='center', color='black',
                     backgroundcolor=bg_color, alpha=alpha,
                     fontfamily='monospace', fontsize=12)

    # Add conservation gradient line above the alignment
    for j, score in enumerate(normalized_scores):
        # Use a color gradient from blue (low conservation) to yellow (high conservation)
        # This creates a visual 'conservation profile'
        color = plt.cm.viridis(score)
        plt.plot([j, j + 1], [3, 3], color=color, linewidth=5)

    # Set plot properties
    plt.xlim(-1, len(seq1))
    plt.ylim(-1, 4)  # Extra space for conservation bar
    plt.axis('off')
    plt.title(title)

    # Add legends
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Match'),
        Patch(facecolor='red', alpha=0.3, label='Mismatch'),
        Patch(facecolor='grey', alpha=0.3, label='Gap')
    ]
    plt.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=3)


# Add these interactive features to the Streamlit UI
def extend_streamlit_ui(st, tab3, results, nw, max_size, max_alignment_width):
    """
    Adds enhanced visualization options to the Streamlit UI.

    Parameters:
    - st: Streamlit module
    - tab3: The visualization tab
    - results: The alignment results
    - nw: NeedlemanWunsch object
    - max_size: Maximum size for detailed visualization
    - max_alignment_width: Maximum width for alignment display
    """
    if results:
        # Add new visualization options
        viz_type = st.radio(
            "Select Visualization Type",
            ["Basic Score Matrix", "Interactive Score Matrix", "Basic Alignment", "Enhanced Alignment"],
            horizontal=True
        )

        # For alignment visualizations, add conservation analysis option
        if "Alignment" in viz_type:
            highlight_regions = st.checkbox("Highlight conserved regions", value=True)

        # Add slider for adjusting detail level in interactive views
        if "Interactive" in viz_type or "Enhanced" in viz_type:
            custom_max_size = st.slider(
                "Detail Level",
                min_value=10,
                max_value=50,
                value=max_size,
                help="Higher values show more details but may increase rendering time."
            )
        else:
            custom_max_size = max_size

        # Create visualization based on selection
        try:
            if viz_type == "Basic Score Matrix":
                fig = nw.visualize_matrix_scalable(
                    results['seq1'],
                    results['seq2'],
                    title="Score Matrix",
                    max_size=custom_max_size
                )
            elif viz_type == "Interactive Score Matrix":
                # Ensure the method exists on the nw object
                if not hasattr(nw, 'visualize_matrix_interactive'):
                    nw.visualize_matrix_interactive = visualize_matrix_interactive.__get__(nw)

                fig = nw.visualize_matrix_interactive(
                    results['seq1'],
                    results['seq2'],
                    title="Interactive Score Matrix",
                    max_size=custom_max_size
                )
            elif viz_type == "Basic Alignment":
                fig = nw.visualize_alignment_scalable(
                    results['aligned_seq1'],
                    results['aligned_seq2'],
                    results['alignment_visual'],
                    max_width=max_alignment_width
                )
            else:  # Enhanced Alignment
                # Ensure the methods exist on the nw object
                if not hasattr(nw, 'visualize_alignment_enhanced'):
                    nw.visualize_alignment_enhanced = visualize_alignment_enhanced.__get__(nw)
                    nw._plot_alignment_segment = _plot_alignment_segment.__get__(nw)
                    nw._plot_alignment_segment_with_conservation = _plot_alignment_segment_with_conservation.__get__(nw)

                fig = nw.visualize_alignment_enhanced(
                    results['aligned_seq1'],
                    results['aligned_seq2'],
                    results['alignment_visual'],
                    max_width=max_alignment_width,
                    highlight_regions=highlight_regions if "Alignment" in viz_type else True
                )

            # Display the plot
            st.pyplot(fig)

            # Add export options with different resolutions
            export_col1, export_col2 = st.columns(2)

            with export_col1:
                img_data_standard = io.BytesIO()
                plt.savefig(img_data_standard, format='png', dpi=100, bbox_inches='tight')
                img_data_standard.seek(0)

                st.download_button(
                    label=f"Download {viz_type} (Standard)",
                    data=img_data_standard,
                    file_name=f"nw_{viz_type.lower().replace(' ', '_')}.png",
                    mime="image/png",
                )

            with export_col2:
                img_data_hires = io.BytesIO()
                plt.savefig(img_data_hires, format='png', dpi=300, bbox_inches='tight')
                img_data_hires.seek(0)

                st.download_button(
                    label=f"Download {viz_type} (High Resolution)",
                    data=img_data_hires,
                    file_name=f"nw_{viz_type.lower().replace(' ', '_')}_hires.png",
                    mime="image/png",
                )

            # Close the current figure properly
            plt.close()

        except Exception as e:
            st.error(f"Visualization failed: {str(e)}")
            st.exception(e)
    else:
        st.info("Run an alignment to see visualizations")


# Function to integrate this code into the main application
def integrate_with_main_app():
    """
    This code shows how to integrate the enhanced visualization functions
    with your main Streamlit application.

    Add this to your main() function in nw_app.py
    """
    # Inside your main() function, modify the Tab 3 (Visualizations) section:

    # Replace the current tab3 content with:
    """
    # Tab 3: Visualizations
    with tab3:
        st.header("Visualizations")

        # Use the enhanced UI extensions
        extend_streamlit_ui(
            st, 
            tab3, 
            st.session_state.alignment_results, 
            results['nw'] if 'results' in locals() and 'nw' in results else None,
            max_size, 
            max_alignment_width
        )
    """

    # Make sure you also add these methods to the NeedlemanWunsch class:
    """
    # Add new visualization methods to the NeedlemanWunsch class
    def add_enhanced_visualizations():
        # Add the improved visualization methods to the NeedlemanWunsch class
        NeedlemanWunsch.visualize_matrix_scalable = visualize_matrix_scalable
        NeedlemanWunsch.visualize_alignment_scalable = visualize_alignment_scalable
        NeedlemanWunsch.visualize_matrix_interactive = visualize_matrix_interactive
        NeedlemanWunsch.visualize_alignment_enhanced = visualize_alignment_enhanced
        NeedlemanWunsch._plot_alignment_segment = _plot_alignment_segment
        NeedlemanWunsch._plot_alignment_segment_with_conservation = _plot_alignment_segment_with_conservation

    # Call this function in main() before setting up the Streamlit page config
    add_enhanced_visualizations()
    """


# Example of sequence conservation analysis
def analyze_sequence_conservation(aligned_seq1, aligned_seq2, alignment_visual, window_size=5):
    """
    Analyzes sequence conservation patterns and identifies conserved and variable regions.

    Parameters:
    - aligned_seq1, aligned_seq2: The aligned sequences
    - alignment_visual: Visual representation of the alignment
    - window_size: Size of sliding window for conservation analysis

    Returns:
    - Dict containing conservation analysis results
    """
    total_length = len(alignment_visual)

    # Calculate conservation scores using sliding window
    conservation_scores = []
    for i in range(total_length):
        # For each position, look at surrounding window
        start = max(0, i - window_size // 2)
        end = min(total_length, i + window_size // 2 + 1)
        window = alignment_visual[start:end]

        # Calculate conservation score (percentage of matches)
        conservation = window.count('|') / len(window)
        conservation_scores.append(conservation)

    # Identify conserved regions (high conservation scores)
    conserved_regions = []
    variable_regions = []

    in_conserved = False
    conserved_start = 0

    # Threshold for considering a region conserved
    threshold = 0.7
    min_region_size = 3

    for i, score in enumerate(conservation_scores):
        if score > threshold and not in_conserved:
            # Start of a new conserved region
            conserved_start = i
            in_conserved = True
        elif score <= threshold and in_conserved:
            # End of a conserved region
            if i - conserved_start >= min_region_size:
                conserved_regions.append((conserved_start, i))
            in_conserved = False

    # Handle the case where the last region extends to the end
    if in_conserved and total_length - conserved_start >= min_region_size:
        conserved_regions.append((conserved_start, total_length))

    # Identify variable regions (gaps between conserved regions)
    prev_end = 0
    for start, end in conserved_regions:
        if start - prev_end >= min_region_size:
            variable_regions.append((prev_end, start))
        prev_end = end

    # Add the last variable region if it exists
    if prev_end < total_length and total_length - prev_end >= min_region_size:
        variable_regions.append((prev_end, total_length))

    # Calculate average conservation score and other statistics
    avg_conservation = sum(conservation_scores) / len(conservation_scores)
    max_conservation = max(conservation_scores)
    min_conservation = min(conservation_scores)

    return {
        'conservation_scores': conservation_scores,
        'conserved_regions': conserved_regions,
        'variable_regions': variable_regions,
        'avg_conservation': avg_conservation,
        'max_conservation': max_conservation,
        'min_conservation': min_conservation
    }