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

# Import the NeedlemanWunsch class and extend it with our scalable visualizations
from needleman_wunsch import NeedlemanWunsch


# Add the scalable visualization methods to the NeedlemanWunsch class
def add_scalable_visualizations():
    # Add the improved visualization methods to the NeedlemanWunsch class
    NeedlemanWunsch.visualize_matrix_scalable = visualize_matrix_scalable
    NeedlemanWunsch.visualize_alignment_scalable = visualize_alignment_scalable


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


def main():
    # Add the scalable visualization methods to the NeedlemanWunsch class
    add_scalable_visualizations()

    st.set_page_config(
        page_title="Needleman-Wunsch Sequence Alignment",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Application title and description
    st.title("Needleman-Wunsch Sequence Alignment Tool")
    st.markdown("""
    This tool implements the Needleman-Wunsch algorithm for global sequence alignment of DNA, RNA, or protein sequences.
    """)

    # Create sidebar for parameters and options
    with st.sidebar:
        st.header("Parameters")

        # Scoring parameters
        st.subheader("Scoring Parameters")
        match_score = st.slider("Match Score", 0.0, 5.0, 1.0, 0.5)
        mismatch_penalty = st.slider("Mismatch Penalty", -5.0, 0.0, -1.0, 0.5)
        gap_penalty = st.slider("Gap Penalty", -5.0, 0.0, -2.0, 0.5)

        # Sequence type selection
        st.subheader("Sequence Type")
        seq_type = st.radio("Select Sequence Type", ["DNA/RNA", "Protein"])

        # Visualization parameters
        st.subheader("Visualization Settings")
        max_size = st.slider("Max Size for Detailed View", 10, 50, 20, 5,
                             help="Maximum sequence length for detailed visualization. Larger sequences will use simplified views.")
        max_alignment_width = st.slider("Max Alignment Display Width", 40, 200, 80, 10,
                                        help="Maximum number of characters to show in the alignment visualization")

        # Load example sequences
        st.subheader("Example Sequences")
        if st.button("Load Short DNA Example"):
            st.session_state.seq1 = "ATGGTGCATCTGACTCCTGA"
            st.session_state.seq2 = "ATGGTGCATCTGACTCCTGT"

        if st.button("Load Medium DNA Example"):
            st.session_state.seq1 = "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTG"
            st.session_state.seq2 = "ATGGTGCATCTGACTCCTGTGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTG"

        if st.button("Load Protein Example"):
            st.session_state.seq1 = "MVHLTPEEKSAVTALWGKV"
            st.session_state.seq2 = "MVHLTPEEKTAVTALWGKV"

    # Create tabs for input, results, and visualization
    tab1, tab2, tab3 = st.tabs(["Sequence Input", "Alignment Results", "Visualizations"])

    # Initialize session state for sequences if not exists
    if 'seq1' not in st.session_state:
        st.session_state.seq1 = ""
    if 'seq2' not in st.session_state:
        st.session_state.seq2 = ""
    if 'alignment_results' not in st.session_state:
        st.session_state.alignment_results = None

    # Tab 1: Sequence Input
    with tab1:
        st.header("Input Sequences")

        # Display sequence length warning if needed
        if len(st.session_state.seq1) > 200 or len(st.session_state.seq2) > 200:
            st.warning("""
            ⚠️ **Large Sequence Warning**: One or both of your sequences is quite long. 
            The alignment will work but detailed visualizations will be simplified.
            """)

        # Text areas for sequence input
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sequence 1")
            seq1 = st.text_area(
                "Enter or paste sequence 1",
                value=st.session_state.seq1,
                height=200
            )

            # File uploader for sequence 1
            uploaded_file1 = st.file_uploader("Or upload a file for sequence 1", type=["txt", "fasta", "fa"])
            if uploaded_file1 is not None:
                content = uploaded_file1.read().decode()
                # Process FASTA if needed
                if content.startswith('>'):
                    lines = content.strip().split('\n')
                    seq1 = ''.join(lines[1:])
                else:
                    seq1 = ''.join(content.split())
                st.session_state.seq1 = seq1

        with col2:
            st.subheader("Sequence 2")
            seq2 = st.text_area(
                "Enter or paste sequence 2",
                value=st.session_state.seq2,
                height=200
            )

            # File uploader for sequence 2
            uploaded_file2 = st.file_uploader("Or upload a file for sequence 2", type=["txt", "fasta", "fa"])
            if uploaded_file2 is not None:
                content = uploaded_file2.read().decode()
                # Process FASTA if needed
                if content.startswith('>'):
                    lines = content.strip().split('\n')
                    seq2 = ''.join(lines[1:])
                else:
                    seq2 = ''.join(content.split())
                st.session_state.seq2 = seq2

        # Update session state with current sequences
        st.session_state.seq1 = seq1
        st.session_state.seq2 = seq2

        # Display sequence information
        st.markdown("### Sequence Information")
        seq_info_col1, seq_info_col2 = st.columns(2)

        with seq_info_col1:
            st.metric("Sequence 1 Length", len(seq1) if seq1 else 0)

        with seq_info_col2:
            st.metric("Sequence 2 Length", len(seq2) if seq2 else 0)

        # Alignment button
        st.markdown("---")
        align_btn = st.button("Run Alignment", use_container_width=True, type="primary")

        if align_btn:
            # Validate sequences
            valid1, msg1 = validate_sequence(seq1, seq_type)
            valid2, msg2 = validate_sequence(seq2, seq_type)

            if not valid1:
                st.error(msg1)
            elif not valid2:
                st.error(msg2)
            elif not seq1 or not seq2:
                st.error("Please enter both sequences")
            else:
                # Run alignment with progress tracking
                with st.spinner("Running alignment..."):
                    try:
                        # Initialize the NeedlemanWunsch object with parameters
                        nw = NeedlemanWunsch(
                            match_score=match_score,
                            mismatch_penalty=mismatch_penalty,
                            gap_penalty=gap_penalty
                        )

                        # Clean the sequences
                        seq1_clean = seq1.upper().strip()
                        seq2_clean = seq2.upper().strip()

                        # Show progress message for larger sequences
                        if len(seq1_clean) > 100 or len(seq2_clean) > 100:
                            progress_bar = st.progress(0)
                            st.info("Aligning large sequences - this may take a moment...")

                            # Perform alignment
                            aligned_seq1, aligned_seq2, alignment_visual, score = nw.align(seq1_clean, seq2_clean)

                            # Update progress
                            progress_bar.progress(100)
                        else:
                            # Perform alignment for smaller sequences
                            aligned_seq1, aligned_seq2, alignment_visual, score = nw.align(seq1_clean, seq2_clean)

                        # Store results in session state
                        st.session_state.alignment_results = {
                            'seq1': seq1_clean,
                            'seq2': seq2_clean,
                            'aligned_seq1': aligned_seq1,
                            'aligned_seq2': aligned_seq2,
                            'alignment_visual': alignment_visual,
                            'score': score,
                            'nw': nw  # Store the NeedlemanWunsch object for visualizations
                        }

                        # Automatically switch to results tab
                        st.rerun()
                    except Exception as e:
                        st.error(f"Alignment failed: {str(e)}")
                        st.exception(e)  # Show full traceback in debug mode

    # Tab 2: Alignment Results
    with tab2:
        st.header("Alignment Results")

        if st.session_state.alignment_results:
            results = st.session_state.alignment_results

            # Display alignment score
            st.subheader(f"Alignment Score: {results['score']:.2f}")

            # Display original sequences
            st.markdown("### Original Sequences")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Sequence 1 ({len(results['seq1'])} characters):**")
                st.code(results['seq1'])
            with col2:
                st.markdown(f"**Sequence 2 ({len(results['seq2'])} characters):**")
                st.code(results['seq2'])

            # Display alignment
            st.markdown("### Alignment")

            # Use monospace font for alignment display
            st.markdown("""
            <style>
            .alignment-text {
                font-family: monospace;
                white-space: pre;
                overflow-x: auto;
            }
            </style>
            """, unsafe_allow_html=True)

            # For longer alignments, show a truncated view with option to expand
            alignment_length = len(results['aligned_seq1'])
            display_limit = 100

            if alignment_length > display_limit:
                st.info(f"Showing first {display_limit} characters of alignment (total length: {alignment_length})")

                alignment_html = f"""
                <div class="alignment-text">
                Seq1: {results['aligned_seq1'][:display_limit]}...<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{results['alignment_visual'][:display_limit].replace('|', '┃').replace('.', '×').replace(' ', '&nbsp;')}...<br>
                Seq2: {results['aligned_seq2'][:display_limit]}...
                </div>
                """
                st.markdown(alignment_html, unsafe_allow_html=True)

                # Show full alignment in expander
                with st.expander("Show full alignment"):
                    full_alignment_html = f"""
                    <div class="alignment-text">
                    Seq1: {results['aligned_seq1']}<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{results['alignment_visual'].replace('|', '┃').replace('.', '×').replace(' ', '&nbsp;')}<br>
                    Seq2: {results['aligned_seq2']}
                    </div>
                    """
                    st.markdown(full_alignment_html, unsafe_allow_html=True)
            else:
                # For shorter alignments, show everything directly
                alignment_html = f"""
                <div class="alignment-text">
                Seq1: {results['aligned_seq1']}<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{results['alignment_visual'].replace('|', '┃').replace('.', '×').replace(' ', '&nbsp;')}<br>
                Seq2: {results['aligned_seq2']}
                </div>
                """
                st.markdown(alignment_html, unsafe_allow_html=True)

            # Alignment statistics
            matches = results['alignment_visual'].count('|')
            mismatches = results['alignment_visual'].count('.')
            gaps = results['alignment_visual'].count(' ')

            st.markdown("### Alignment Statistics")
            stat_col1, stat_col2, stat_col3 = st.columns(3)

            with stat_col1:
                st.metric("Matches", matches, f"{(matches / len(results['alignment_visual'])) * 100:.1f}%")

            with stat_col2:
                st.metric("Mismatches", mismatches, f"{(mismatches / len(results['alignment_visual'])) * 100:.1f}%")

            with stat_col3:
                st.metric("Gaps", gaps, f"{(gaps / len(results['alignment_visual'])) * 100:.1f}%")

            # Download alignment results
            st.markdown("### Download Results")

            result_text = f"# Needleman-Wunsch Alignment Results\n\n"
            result_text += f"# Parameters:\n"
            result_text += f"# Match Score: {match_score}\n"
            result_text += f"# Mismatch Penalty: {mismatch_penalty}\n"
            result_text += f"# Gap Penalty: {gap_penalty}\n\n"
            result_text += f"# Original sequences:\n"
            result_text += f"Sequence 1: {results['seq1']}\n"
            result_text += f"Sequence 2: {results['seq2']}\n\n"
            result_text += f"# Alignment score: {results['score']:.2f}\n\n"
            result_text += f"# Alignment:\n"
            result_text += f"Seq1: {results['aligned_seq1']}\n"
            result_text += f"      {results['alignment_visual']}\n"
            result_text += f"Seq2: {results['aligned_seq2']}\n"

            st.download_button(
                label="Download Alignment Results",
                data=result_text,
                file_name="alignment_results.txt",
                mime="text/plain",
            )
        else:
            st.info("Run an alignment to see results")

    # Tab 3: Visualizations
    with tab3:
        st.header("Visualizations")

        if st.session_state.alignment_results:
            results = st.session_state.alignment_results
            nw = results['nw']

            # Select visualization type
            viz_type = st.radio(
                "Select Visualization",
                ["Score Matrix", "Alignment Visualization"],
                horizontal=True
            )

            # Show a warning for larger sequences
            seq_len1, seq_len2 = len(results['seq1']), len(results['seq2'])
            if seq_len1 > max_size or seq_len2 > max_size:
                st.info(f"""
                ℹ️ One or both sequences exceed the maximum size ({max_size}) for detailed visualization.
                Using simplified visualization mode. Adjust the 'Max Size for Detailed View' in settings if needed.
                """)

            # Create figure based on selection
            try:
                if viz_type == "Score Matrix":
                    fig = nw.visualize_matrix_scalable(
                        results['seq1'],
                        results['seq2'],
                        title="Score Matrix",
                        max_size=max_size
                    )
                else:  # Alignment Visualization
                    fig = nw.visualize_alignment_scalable(
                        results['aligned_seq1'],
                        results['aligned_seq2'],
                        results['alignment_visual'],
                        max_width=max_alignment_width
                    )

                # Display the plot
                st.pyplot(fig)

                # Download button for visualization
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png', dpi=300, bbox_inches='tight')
                img_data.seek(0)

                st.download_button(
                    label=f"Download {viz_type}",
                    data=img_data,
                    file_name=f"nw_{viz_type.lower().replace(' ', '_')}.png",
                    mime="image/png",
                )

                # Close the current figure properly
                plt.close()
            except Exception as e:
                st.error(f"Visualization failed: {str(e)}")
                st.exception(e)  # Show full traceback in debug mode
        else:
            st.info("Run an alignment to see visualizations")

    # Add footer with documentation and help
    st.markdown("---")
    with st.expander("About the Needleman-Wunsch Algorithm"):
        st.markdown("""
        ### Needleman-Wunsch Algorithm

        The Needleman-Wunsch algorithm is a dynamic programming algorithm used for global sequence alignment. It was developed by Saul B. Needleman and Christian D. Wunsch in 1970.

        #### How it works:

        1. **Matrix Initialization**: Create a scoring matrix with dimensions (m+1) × (n+1), where m and n are the lengths of the two sequences.

        2. **Scoring System**: Define scores for:
           - Match: When characters match
           - Mismatch: When characters don't match
           - Gap: When a gap must be inserted

        3. **Matrix Filling**: Fill the matrix using dynamic programming:
           - Calculate scores for diagonal (match/mismatch), vertical (gap in sequence 2), and horizontal (gap in sequence 1) moves
           - Choose the maximum score and store it in the current cell
           - Keep track of which move was chosen for traceback

        4. **Traceback**: Starting from the bottom-right cell, follow the traceback pointers to reconstruct the optimal alignment, working backwards to the top-left cell.

        #### Applications:

        - DNA/RNA sequence comparison
        - Protein sequence alignment
        - Evolutionary relationships between species
        - Mutation analysis
        - Gene finding and annotation
        """)

    with st.expander("Large Sequence Handling"):
        st.markdown("""
        ### Working with Larger Sequences

        This application has been optimized to handle sequences of various sizes:

        #### Visualization Scaling

        - **Small sequences** (under the "Max Size" threshold): Full detailed visualization with cell values, arrows, and character labels
        - **Larger sequences**: Simplified visualization showing the overall structure and path

        #### Performance Considerations

        - The Needleman-Wunsch algorithm has O(mn) time and space complexity, where m and n are the sequence lengths
        - Very large sequences (over ~1000 bases) may become slow or run out of memory
        - For production use with large sequences, consider specialized bioinformatics tools

        #### Tips for Large Sequences

        - Adjust the visualization parameters in the sidebar
        - For sequences over 500 bases, consider using a subset for initial exploration
        - The alignment results are shown even when visualizations are simplified
        """)


def validate_sequence(sequence, seq_type):
    """Validate sequence characters based on type"""
    sequence = sequence.upper().strip()
    if not sequence:
        return True, sequence  # Empty sequence is technically valid

    if seq_type == "DNA/RNA":
        # DNA/RNA sequence validation
        valid_chars = set("ACGTU")
        invalid_chars = [c for c in sequence if c not in valid_chars and c != '-']
        if invalid_chars:
            return False, f"Invalid characters in DNA/RNA sequence: {''.join(set(invalid_chars))}"
    else:  # protein
        # Protein sequence validation
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
        invalid_chars = [c for c in sequence if c not in valid_chars and c != '-']
        if invalid_chars:
            return False, f"Invalid characters in protein sequence: {''.join(set(invalid_chars))}"

    return True, sequence


if __name__ == "__main__":
    main()