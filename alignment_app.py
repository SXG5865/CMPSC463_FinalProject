import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import io
import os
import re
import sys

# Add the current directory to the path to import the SequenceAlignment class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the SequenceAlignment class
from sequence_alignment import SequenceAlignment


def main():
    st.set_page_config(
        page_title="Sequence Alignment Toolkit",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Application title and description
    st.title("Sequence Alignment Toolkit")
    st.markdown("""
    This application implements multiple sequence alignment algorithms to compare DNA, RNA, or protein sequences:

    - **Global Alignment** (Needleman-Wunsch): Aligns entire sequences from end to end
    - **Semi-Global Alignment**: Allows free end gaps for one or both sequences

    Choose your alignment method and parameters to analyze sequences of interest.
    """)

    # Create sidebar for parameters and options
    with st.sidebar:
        st.header("Parameters")

        # Alignment algorithm selection
        st.subheader("Alignment Method")
        alignment_mode = st.radio(
            "Select Alignment Algorithm",
            ["Global (Needleman-Wunsch)", "Semi-Global"],
            help="""
            Global: Aligns entire sequences from end to end (best for similar sequences)
            Semi-Global: Allows free end gaps (best for overlapping sequences)
            """
        )

        # Semi-global options (show only when semi-global is selected)
        if alignment_mode == "Semi-Global":
            semi_global_option = st.radio(
                "End Gap Penalties",
                ["None (both ends free)", "Only Seq1 ends", "Only Seq2 ends", "Both ends (standard)"],
                help="""
                Specifies which sequence ends should have gap penalties:
                - None: No penalties for gaps at either sequence end
                - Only Seq1: Penalize gaps at ends of Seq1 only
                - Only Seq2: Penalize gaps at ends of Seq2 only
                - Both: Standard global alignment with all end gaps penalized
                """
            )
            # Map radio options to parameter values
            semi_global_map = {
                "None (both ends free)": "none",
                "Only Seq1 ends": "seq1",
                "Only Seq2 ends": "seq2",
                "Both ends (standard)": "both"
            }
            penalize_ends = semi_global_map[semi_global_option]

        # Scoring parameters
        st.subheader("Scoring Parameters")
        match_score = st.slider("Match Score", 0.0, 5.0, 2.0, 0.5)
        mismatch_penalty = st.slider("Mismatch Penalty", -5.0, 0.0, -1.0, 0.5)
        gap_penalty = st.slider("Gap Penalty", -5.0, 0.0, -2.0, 0.5)

        # Sequence type
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

        if st.button("Load Conserved Region Example"):
            st.session_state.seq1 = "ACGTACGTACGTAGCTAGCATCGATCGTAGCATCGAT"
            st.session_state.seq2 = "TACGTTTTACGTAGCATTTGACGATCGTA"

        if st.button("Load Multiple Conserved Regions"):
            st.session_state.seq1 = "ACGTACGTTTCGAGTCAGGCTTAAGCTAGCATTTACGATCGTAGCTTCGAGTCGAT"
            st.session_state.seq2 = "TTTTACGTTTTTTTTTTTGCTAGCATTTTTTTTTTTTCGAGTCGGG"

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
                with st.spinner(
                        f"Running {'global' if alignment_mode == 'Global (Needleman-Wunsch)' else 'semi-global'} alignment..."):
                    try:
                        # Initialize the SequenceAlignment object with parameters
                        sa = SequenceAlignment(
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

                        # Perform alignment based on selected mode
                        if alignment_mode == "Global (Needleman-Wunsch)":
                            aligned_seq1, aligned_seq2, alignment_visual, score = sa.align(
                                seq1_clean, seq2_clean, mode='global'
                            )
                            # Store results in session state
                            st.session_state.alignment_results = {
                                'mode': 'global',
                                'seq1': seq1_clean,
                                'seq2': seq2_clean,
                                'aligned_seq1': aligned_seq1,
                                'aligned_seq2': aligned_seq2,
                                'alignment_visual': alignment_visual,
                                'score': score,
                                'sa': sa  # Store the SequenceAlignment object for visualizations
                            }
                        else:  # Semi-Global
                            aligned_seq1, aligned_seq2, alignment_visual, score = sa.semi_global_align(
                                seq1_clean, seq2_clean, penalize_ends=penalize_ends
                            )
                            # Store results in session state
                            st.session_state.alignment_results = {
                                'mode': 'semi-global',
                                'seq1': seq1_clean,
                                'seq2': seq2_clean,
                                'aligned_seq1': aligned_seq1,
                                'aligned_seq2': aligned_seq2,
                                'alignment_visual': alignment_visual,
                                'score': score,
                                'penalize_ends': penalize_ends,
                                'sa': sa  # Store the SequenceAlignment object for visualizations
                            }

                        # Update progress for large sequences
                        if len(seq1_clean) > 100 or len(seq2_clean) > 100:
                            progress_bar.progress(100)

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
            mode = results['mode']

            # Display alignment score
            if mode == 'global':
                st.subheader(f"Global Alignment Score: {results['score']:.2f}")
            else:  # semi-global
                st.subheader(f"Semi-Global Alignment Score: {results['score']:.2f}")
                st.info(f"End gap penalty mode: {results['penalize_ends']}")

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
            if mode == 'global':
                st.markdown("### Global Alignment")
            else:
                st.markdown("### Semi-Global Alignment")

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

            result_text = f"# Sequence Alignment Results\n\n"
            result_text += f"# Alignment mode: {mode}\n"
            result_text += f"# Parameters:\n"
            result_text += f"# Match Score: {match_score}\n"
            result_text += f"# Mismatch Penalty: {mismatch_penalty}\n"
            result_text += f"# Gap Penalty: {gap_penalty}\n"

            if mode == 'semi-global':
                result_text += f"# End gap penalty mode: {results['penalize_ends']}\n"

            result_text += f"\n# Original sequences:\n"
            result_text += f"Sequence 1: {results['seq1']}\n"
            result_text += f"Sequence 2: {results['seq2']}\n\n"
            result_text += f"# Alignment score: {results['score']:.2f}\n"

            result_text += f"\n# Alignment:\n"
            result_text += f"Seq1: {results['aligned_seq1']}\n"
            result_text += f"      {results['alignment_visual']}\n"
            result_text += f"Seq2: {results['aligned_seq2']}\n"

            st.download_button(
                label="Download Alignment Results",
                data=result_text,
                file_name=f"{mode}_alignment_results.txt",
                mime="text/plain",
            )
        else:
            st.info("Run an alignment to see results")

    # Tab 3: Visualizations
    with tab3:
        st.header("Visualizations")

        if st.session_state.alignment_results:
            results = st.session_state.alignment_results
            sa = results['sa']
            mode = results['mode']

            # Select visualization type
            viz_type = st.radio(
                "Select Visualization",
                ["Score Matrix", "Traceback Matrix", "Alignment Visualization"],
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
                    # Generate appropriate title based on alignment mode
                    if mode == 'global':
                        title = "Needleman-Wunsch Score Matrix"
                    else:  # semi-global
                        title = f"Semi-Global Alignment Score Matrix (mode: {results['penalize_ends']})"

                    fig = sa.visualize_matrix_scalable(
                        results['seq1'],
                        results['seq2'],
                        title=title,
                        max_size=max_size
                    )
                elif viz_type == "Traceback Matrix":
                    # Generate appropriate title based on alignment mode
                    if mode == 'global':
                        title = "Needleman-Wunsch Traceback Matrix"
                    else:  # semi-global
                        title = f"Semi-Global Alignment Traceback Matrix (mode: {results['penalize_ends']})"

                    fig = sa.visualize_traceback(
                        results['seq1'],
                        results['seq2'],
                        title=title
                    )
                else:  # Alignment Visualization
                    fig = sa.visualize_alignment_scalable(
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
                    file_name=f"{mode}_{viz_type.lower().replace(' ', '_')}.png",
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
    with st.expander("About Sequence Alignment Algorithms"):
        st.markdown("""
        ### Sequence Alignment Algorithms

        This application implements two types of sequence alignment algorithms:

        #### 1. Global Alignment (Needleman-Wunsch)

        - **Purpose**: Align entire sequences from end to end
        - **Best for**: Sequences that are similar throughout their length
        - **Applications**: Comparing closely related genes, proteins, or organisms
        - **Characteristics**:
          - Forced to include every position in both sequences
          - Penalizes gaps at the beginning and end
          - Provides a complete picture of overall similarity

        #### 2. Semi-Global Alignment

        - **Purpose**: Allow free gaps at the ends of one or both sequences
        - **Best for**: Overlapping sequences or when terminal gaps shouldn't be penalized
        - **Applications**: Sequence assembly, primer alignment, overlap detection
        - **Characteristics**:
          - Compromise between global and local alignment
          - Flexible handling of end gaps
          - Multiple modes available (free gaps at different ends)

        ### Algorithm Details

        Both algorithms use dynamic programming, filling a scoring matrix based on the following values:

        - **Match score**: Reward for matching characters
        - **Mismatch penalty**: Penalty for mismatched characters
        - **Gap penalty**: Penalty for inserting a gap in either sequence

        The main differences are in how the matrix is initialized and how the traceback is performed.
        """)

    with st.expander("Choosing the Right Algorithm"):
        st.markdown("""
        ### Which Algorithm Should I Use?

        #### Use Global Alignment (Needleman-Wunsch) when:

        - Your sequences have similar lengths
        - You expect similarity throughout the sequences
        - You need to compare entire sequences
        - You're working with closely related organisms
        - You want to evaluate overall similarity

        #### Use Semi-Global Alignment when:

        - You're aligning a fragment to a complete sequence
        - Working with sequences that should overlap
        - You want to find where one sequence is contained within another
        - You're doing sequence assembly or primer design
        - You want to ignore terminal gaps in your alignment score

        ### End Gap Penalty Options for Semi-Global Alignment

        - **None (both ends free)**: Gaps at the beginning and end of both sequences are not penalized
        - **Only Seq1 ends**: Gaps at the beginning and end of Sequence 1 are penalized, but not for Sequence 2
        - **Only Seq2 ends**: Gaps at the beginning and end of Sequence 2 are penalized, but not for Sequence 1
        - **Both ends (standard)**: All gaps are penalized (equivalent to global alignment)
        """)

    with st.expander("Working with Larger Sequences"):
        st.markdown("""
        ### Working with Larger Sequences

        This application has been optimized to handle sequences of various sizes:

        #### Visualization Scaling

        - **Small sequences** (under the "Max Size" threshold): Full detailed visualization with cell values and character labels
        - **Larger sequences**: Simplified visualization showing the overall structure and highest-scoring regions

        #### Performance Considerations

        - All algorithms have O(mn) time and space complexity, where m and n are the sequence lengths
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