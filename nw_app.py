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

# Import the NeedlemanWunsch class from the original file
from needleman_wunsch import NeedlemanWunsch


def main():
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

        # Load example sequences
        st.subheader("Example Sequences")
        if st.button("Load DNA Example"):
            st.session_state.seq1 = "GCATGCU"
            st.session_state.seq2 = "GATTACA"

        if st.button("Load Protein Example"):
            st.session_state.seq1 = "HEAGAWGHEE"
            st.session_state.seq2 = "PAWHEAE"

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
                # Run alignment
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

                        # Perform alignment
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
                ["Score Matrix", "Traceback Matrix", "Alignment Visualization"],
                horizontal=True
            )

            # Create figure based on selection
            try:
                plt.figure(figsize=(10, 8))

                if viz_type == "Score Matrix":
                    fig = nw.visualize_matrix(
                        results['seq1'],
                        results['seq2'],
                        title="Score Matrix"
                    )
                elif viz_type == "Traceback Matrix":
                    fig = nw.visualize_traceback(
                        results['seq1'],
                        results['seq2']
                    )
                else:  # Alignment Visualization
                    fig = nw.visualize_alignment(
                        results['aligned_seq1'],
                        results['aligned_seq2'],
                        results['alignment_visual']
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

                plt.close()
            except Exception as e:
                st.error(f"Visualization failed: {str(e)}")
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