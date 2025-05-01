import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


class SequenceAlignment:
    """
    A class implementing Needleman-Wunsch (global) sequence alignment and semi-global
    alignment algorithms.
    """

    def __init__(self, match_score=1, mismatch_penalty=-1, gap_penalty=-2, gap_open_penalty=None,
                 gap_extend_penalty=None):
        """
        Initialize the sequence alignment with scoring parameters.

        Parameters:
        - match_score: Score for matching characters (default: 1)
        - mismatch_penalty: Penalty for mismatched characters (default: -1)
        - gap_penalty: Penalty for introducing a gap (default: -2)
        - gap_open_penalty: Penalty for opening a gap (default: None, uses gap_penalty if None)
        - gap_extend_penalty: Penalty for extending a gap (default: None, uses gap_penalty if None)
        """
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.gap_open_penalty = gap_open_penalty if gap_open_penalty is not None else gap_penalty
        self.gap_extend_penalty = gap_extend_penalty if gap_extend_penalty is not None else gap_penalty

        # Set during alignment
        self.score_matrix = None
        self.traceback = None
        self.max_score = 0
        self.max_pos = (0, 0)
        self.alignment_mode = None  # Will be set to 'global' or 'semi-global' during alignment

    def _initialize_matrices(self, seq1, seq2, mode='global'):
        """
        Initialize score and traceback matrices for global alignment.

        Parameters:
        - seq1, seq2: Sequences to align
        - mode: 'global' for Needleman-Wunsch
        """
        self.alignment_mode = mode
        m, n = len(seq1) + 1, len(seq2) + 1

        # Initialize score matrix
        self.score_matrix = np.zeros((m, n))

        # Initialize traceback matrix with empty strings
        self.traceback = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                self.traceback[i, j] = ""

        # For global alignment, initialize first row and column with gap penalties
        for i in range(m):
            self.score_matrix[i, 0] = i * self.gap_penalty
            if i > 0:
                self.traceback[i, 0] = "↑"  # Up arrow for vertical movement

        for j in range(n):
            self.score_matrix[0, j] = j * self.gap_penalty
            if j > 0:
                self.traceback[0, j] = "←"  # Left arrow for horizontal movement

        # Reset max score and position tracking
        self.max_score = 0
        self.max_pos = (0, 0)

    def _fill_matrices(self, seq1, seq2, mode='global', affine_gaps=False):
        """
        Fill the score and traceback matrices using dynamic programming.

        Parameters:
        - seq1, seq2: Sequences to align
        - mode: 'global' for Needleman-Wunsch
        - affine_gaps: Whether to use affine gap penalties
        """
        m, n = len(seq1) + 1, len(seq2) + 1

        for i in range(1, m):
            for j in range(1, n):
                # Calculate match/mismatch score
                match = (self.match_score if seq1[i - 1] == seq2[j - 1]
                         else self.mismatch_penalty)
                diagonal = self.score_matrix[i - 1, j - 1] + match

                # Calculate gap scores
                if affine_gaps:
                    up = self.score_matrix[i - 1, j] + self.gap_penalty
                    left = self.score_matrix[i, j - 1] + self.gap_penalty
                else:
                    up = self.score_matrix[i - 1, j] + self.gap_penalty
                    left = self.score_matrix[i, j - 1] + self.gap_penalty

                # Choose best score
                # In global alignment, we take the max of the three options
                self.score_matrix[i, j] = max(diagonal, up, left)

                # Record the traceback direction(s)
                tb = ""
                if diagonal == self.score_matrix[i, j]:
                    tb += "↖"  # Diagonal arrow
                if up == self.score_matrix[i, j]:
                    tb += "↑"  # Up arrow
                if left == self.score_matrix[i, j]:
                    tb += "←"  # Left arrow

                self.traceback[i, j] = tb

                # Track the maximum score
                if self.score_matrix[i, j] > self.max_score:
                    self.max_score = self.score_matrix[i, j]
                    self.max_pos = (i, j)

    def _get_global_alignment(self, seq1, seq2):
        """
        Perform traceback for global alignment (Needleman-Wunsch).

        Returns:
        - aligned_seq1: First sequence with gaps inserted
        - aligned_seq2: Second sequence with gaps inserted
        - alignment_visual: Visual representation of the alignment
        """
        i, j = len(seq1), len(seq2)
        aligned_seq1, aligned_seq2 = "", ""

        while i > 0 or j > 0:
            tb = self.traceback[i, j]

            if i > 0 and j > 0 and "↖" in tb:
                # Diagonal move (match or mismatch)
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                i -= 1
                j -= 1
            elif i > 0 and "↑" in tb:
                # Vertical move (gap in seq2)
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = "-" + aligned_seq2
                i -= 1
            elif j > 0 and "←" in tb:
                # Horizontal move (gap in seq1)
                aligned_seq1 = "-" + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                j -= 1
            else:
                # This shouldn't happen if matrices are correctly filled
                break

        # Create a visual representation of the alignment
        alignment_visual = self._create_alignment_visual(aligned_seq1, aligned_seq2)

        return aligned_seq1, aligned_seq2, alignment_visual

    def _create_alignment_visual(self, aligned_seq1, aligned_seq2):
        """Create a visual representation of the alignment."""
        alignment_visual = ""
        for i in range(len(aligned_seq1)):
            if aligned_seq1[i] == aligned_seq2[i]:
                alignment_visual += "|"  # Match
            elif aligned_seq1[i] == "-" or aligned_seq2[i] == "-":
                alignment_visual += " "  # Gap
            else:
                alignment_visual += "."  # Mismatch

        return alignment_visual

    def align(self, seq1, seq2, mode='global', affine_gaps=False):
        """
        Align two sequences using global alignment.

        Parameters:
        - seq1: First sequence to align
        - seq2: Second sequence to align
        - mode: Must be 'global'
        - affine_gaps: Whether to use affine gap penalties

        Returns:
        - (aligned_seq1, aligned_seq2, alignment_visual, score)
        """
        if mode != 'global':
            raise ValueError("Mode must be 'global'")

        # Initialize matrices
        self._initialize_matrices(seq1, seq2, mode)

        # Fill matrices using dynamic programming
        self._fill_matrices(seq1, seq2, mode, affine_gaps)

        # Get the alignment through traceback
        aligned_seq1, aligned_seq2, alignment_visual = self._get_global_alignment(seq1, seq2)
        score = self.score_matrix[len(seq1), len(seq2)]
        return aligned_seq1, aligned_seq2, alignment_visual, score

    def visualize_matrix(self, seq1, seq2, title=None):
        """Visualize the score matrix as a heatmap."""
        if self.score_matrix is None:
            raise ValueError("Run align() first to generate the score matrix.")

        if title is None:
            if self.alignment_mode == 'global':
                title = "Needleman-Wunsch Score Matrix"
            else:
                title = "Semi-Global Score Matrix"

        plt.figure(figsize=(10, 8))

        # Create labels with sequence characters
        row_labels = [''] + list(seq1)
        col_labels = [''] + list(seq2)

        # Create heatmap
        ax = sns.heatmap(self.score_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                         xticklabels=col_labels, yticklabels=row_labels)

        plt.title(f"{title} - Optimal Score: {self.score_matrix[len(seq1), len(seq2)]:.1f}")

        plt.tight_layout()
        return plt

    def visualize_matrix_scalable(self, seq1, seq2, title=None, max_size=20):
        """
        Visualize the score matrix as a heatmap with automatic scaling for larger sequences.

        Parameters:
        - seq1, seq2: The sequences being aligned
        - title: The title for the plot
        - max_size: Maximum full-detail visualization size (for each dimension)
        """
        if self.score_matrix is None:
            raise ValueError("Run align() first to generate the score matrix.")

        if title is None:
            if self.alignment_mode == 'global':
                title = "Needleman-Wunsch Score Matrix"
            else:
                title = "Semi-Global Score Matrix"

        m, n = len(seq1) + 1, len(seq2) + 1

        # Check if sequences are too long for detailed visualization
        if m > max_size + 1 or n > max_size + 1:
            # For larger sequences, use a different visualization approach
            plt.figure(figsize=(12, 10))

            # Create enhanced heatmap for larger matrices
            ax = sns.heatmap(self.score_matrix, cmap="YlGnBu",
                             xticklabels=False, yticklabels=False)

            # Add axis labels
            plt.xlabel(f"Sequence 2 ({n - 1} bases)")
            plt.ylabel(f"Sequence 1 ({m - 1} bases)")

            # Add a colorbar with better labeling
            cbar = ax.collections[0].colorbar
            cbar.set_label('Alignment Score')

            # Add title and score information
            optimal_score = self.score_matrix[m - 1, n - 1]
            plt.title(f"{title} ({m - 1}x{n - 1} matrix)")
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

            # Add title and score information
            plt.title(f"{title} - Optimal Score: {self.score_matrix[len(seq1), len(seq2)]:.1f}")

        plt.tight_layout()
        return plt

    def visualize_alignment_scalable(self, aligned_seq1, aligned_seq2, alignment_visual,
                                     seq1=None, seq2=None, start_pos=None, end_pos=None, max_width=80):
        """
        Visualize the sequence alignment with highlighting, handling longer sequences.

        Parameters:
        - aligned_seq1, aligned_seq2: The aligned sequences
        - alignment_visual: Visual representation of the alignment
        - seq1, seq2, start_pos, end_pos: Not used in global alignment
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

            if self.alignment_mode == 'global':
                plt.title('Global Sequence Alignment')
            else:
                plt.title('Semi-Global Sequence Alignment')
        else:
            # For longer alignments, create a chunked visualization
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

            if self.alignment_mode == 'global':
                plt.title(f'Global Alignment Statistics (Total Length: {total_length} bp)')
            else:
                plt.title(f'Semi-Global Alignment Statistics (Total Length: {total_length} bp)')

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

    def semi_global_align(self, seq1, seq2, penalize_ends='both'):
        """
        Perform semi-global alignment (free end gaps for one or both sequences).

        Parameters:
        - seq1: First sequence to align
        - seq2: Second sequence to align
        - penalize_ends: Which sequence ends to penalize:
            - 'both': standard global alignment
            - 'none': no end gap penalties (like local alignment)
            - 'seq1': don't penalize gaps at the end of seq2
            - 'seq2': don't penalize gaps at the end of seq1

        Returns:
        - Alignment results like global alignment
        """
        m, n = len(seq1) + 1, len(seq2) + 1

        # Initialize matrices
        self.alignment_mode = 'semi-global'
        self._initialize_matrices(seq1, seq2, 'global')

        # Modify initialization based on which ends to penalize
        if penalize_ends == 'none' or penalize_ends == 'seq1':
            # Don't penalize gaps at the beginning of seq1
            for j in range(n):
                self.score_matrix[0, j] = 0
                self.traceback[0, j] = ""

        if penalize_ends == 'none' or penalize_ends == 'seq2':
            # Don't penalize gaps at the beginning of seq2
            for i in range(m):
                self.score_matrix[i, 0] = 0
                self.traceback[i, 0] = ""

        # Fill matrix
        self._fill_matrices(seq1, seq2, 'global')

        # For semi-global alignment, find best score along the edges
        best_score = float('-inf')
        best_pos = (0, 0)

        if penalize_ends == 'none' or penalize_ends == 'seq1':
            # Check bottom row (gaps at the end of seq1)
            for j in range(n):
                if self.score_matrix[m - 1, j] > best_score:
                    best_score = self.score_matrix[m - 1, j]
                    best_pos = (m - 1, j)

        if penalize_ends == 'none' or penalize_ends == 'seq2':
            # Check rightmost column (gaps at the end of seq2)
            for i in range(m):
                if self.score_matrix[i, n - 1] > best_score:
                    best_score = self.score_matrix[i, n - 1]
                    best_pos = (i, n - 1)

        # If penalize_ends is 'both', use the standard global alignment
        if penalize_ends == 'both':
            best_score = self.score_matrix[m - 1, n - 1]
            best_pos = (m - 1, n - 1)

        # Start traceback from the best position
        self.max_pos = best_pos
        self.max_score = best_score

        # Perform modified traceback
        i, j = best_pos
        aligned_seq1, aligned_seq2 = "", ""

        while i > 0 or j > 0:
            if (penalize_ends == 'seq2' and j == 0) or (penalize_ends == 'seq1' and i == 0) or (
                    penalize_ends == 'none' and (i == 0 or j == 0)):
                # We've reached a free end, stop traceback
                break

            tb = self.traceback[i, j]

            if i > 0 and j > 0 and "↖" in tb:
                # Diagonal move (match or mismatch)
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                i -= 1
                j -= 1
            elif i > 0 and "↑" in tb:
                # Vertical move (gap in seq2)
                aligned_seq1 = seq1[i - 1] + aligned_seq1
                aligned_seq2 = "-" + aligned_seq2
                i -= 1
            elif j > 0 and "←" in tb:
                # Horizontal move (gap in seq1)
                aligned_seq1 = "-" + aligned_seq1
                aligned_seq2 = seq2[j - 1] + aligned_seq2
                j -= 1
            else:
                # This shouldn't happen if matrices are correctly filled
                break

        # Add any remaining prefix gaps if not at the start of both sequences
        while i > 0 and (penalize_ends != 'seq2' and penalize_ends != 'none'):
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            i -= 1

        while j > 0 and (penalize_ends != 'seq1' and penalize_ends != 'none'):
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            j -= 1

        # Create alignment visual
        alignment_visual = self._create_alignment_visual(aligned_seq1, aligned_seq2)

        return aligned_seq1, aligned_seq2, alignment_visual, best_score


# Example usage
def example_alignments():
    """Run example alignments using different algorithms and visualize the results."""
    # Initialize the sequence alignment object
    sa = SequenceAlignment(match_score=2, mismatch_penalty=-1, gap_penalty=-2)

    # Define sequences
    seq1 = "ACGTACGTACGTAGCTAGCATCGATCGTAGCATCGAT"
    seq2 = "TACGTTTTACGTAGCATTTGACGATCGTA"

    print("=== Original Sequences ===")
    print(f"Seq1: {seq1}")
    print(f"Seq2: {seq2}")
    print("\n")

    # Global alignment (Needleman-Wunsch)
    print("=== Global Alignment (Needleman-Wunsch) ===")
    aligned_seq1, aligned_seq2, alignment_visual, score = sa.align(seq1, seq2, mode='global')

    print(f"Alignment score: {score}")
    print(f"Aligned Seq1: {aligned_seq1}")
    print(f"Alignment:    {alignment_visual}")
    print(f"Aligned Seq2: {aligned_seq2}")
    print("\n")

    sa.visualize_matrix_scalable(seq1, seq2, "Global Alignment Score Matrix")
    plt.savefig('global_score_matrix.png')

    sa.visualize_alignment_scalable(aligned_seq1, aligned_seq2, alignment_visual)
    plt.savefig('global_alignment.png')

    # Semi-global alignment
    print("=== Semi-Global Alignment (free end gaps) ===")
    aligned_seq1, aligned_seq2, alignment_visual, score = sa.semi_global_align(seq1, seq2, penalize_ends='none')

    print(f"Alignment score: {score}")
    print(f"Aligned Seq1: {aligned_seq1}")
    print(f"Alignment:    {alignment_visual}")
    print(f"Aligned Seq2: {aligned_seq2}")

    sa.visualize_matrix_scalable(seq1, seq2, "Semi-Global Alignment Score Matrix")
    plt.savefig('semi_global_score_matrix.png')

    sa.visualize_alignment_scalable(aligned_seq1, aligned_seq2, alignment_visual)
    plt.savefig('semi_global_alignment.png')

    plt.show()


if __name__ == "__main__":
    print("Sequence Alignment Algorithms")
    print("----------------------------")

    # Run the example alignments
    example_alignments()