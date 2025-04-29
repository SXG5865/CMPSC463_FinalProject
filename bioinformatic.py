import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns


class NeedlemanWunsch:
    def __init__(self, match_score=1, mismatch_penalty=-1, gap_penalty=-2):
        """
        Initialize the Needleman-Wunsch algorithm with scoring parameters.

        Parameters:
        - match_score: Score for matching characters (default: 1)
        - mismatch_penalty: Penalty for mismatched characters (default: -1)
        - gap_penalty: Penalty for introducing a gap (default: -2)
        """
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.score_matrix = None
        self.traceback = None

    def _initialize_matrices(self, seq1, seq2):
        """Initialize score and traceback matrices."""
        # Add 1 to lengths for the empty prefix column/row
        m, n = len(seq1) + 1, len(seq2) + 1

        # Initialize score matrix with zeros
        self.score_matrix = np.zeros((m, n))

        # Initialize traceback matrix with empty strings
        self.traceback = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                self.traceback[i, j] = ""

        # Fill the first row and column with gap penalties
        for i in range(m):
            self.score_matrix[i, 0] = i * self.gap_penalty
            if i > 0:
                self.traceback[i, 0] = "↑"  # Up arrow for vertical movement

        for j in range(n):
            self.score_matrix[0, j] = j * self.gap_penalty
            if j > 0:
                self.traceback[0, j] = "←"  # Left arrow for horizontal movement

    def _fill_matrices(self, seq1, seq2):
        """Fill the score and traceback matrices using dynamic programming."""
        m, n = len(seq1) + 1, len(seq2) + 1

        for i in range(1, m):
            for j in range(1, n):
                # Calculate scores for three possible moves
                match = (self.match_score if seq1[i - 1] == seq2[j - 1]
                         else self.mismatch_penalty)
                diagonal = self.score_matrix[i - 1, j - 1] + match
                up = self.score_matrix[i - 1, j] + self.gap_penalty
                left = self.score_matrix[i, j - 1] + self.gap_penalty

                # Choose the best score and record the move
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

    def _get_alignment(self, seq1, seq2):
        """
        Perform traceback to find the optimal alignment.

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
        alignment_visual = ""
        for i in range(len(aligned_seq1)):
            if aligned_seq1[i] == aligned_seq2[i]:
                alignment_visual += "|"  # Match
            elif aligned_seq1[i] == "-" or aligned_seq2[i] == "-":
                alignment_visual += " "  # Gap
            else:
                alignment_visual += "."  # Mismatch

        return aligned_seq1, aligned_seq2, alignment_visual

    def align(self, seq1, seq2):
        """
        Align two sequences using the Needleman-Wunsch algorithm.

        Parameters:
        - seq1: First sequence to align
        - seq2: Second sequence to align

        Returns:
        - A tuple containing (aligned_seq1, aligned_seq2, alignment_visual, score)
        """
        # Initialize matrices
        self._initialize_matrices(seq1, seq2)

        # Fill matrices using dynamic programming
        self._fill_matrices(seq1, seq2)

        # Get the alignment through traceback
        aligned_seq1, aligned_seq2, alignment_visual = self._get_alignment(seq1, seq2)

        # Return the alignment and the optimal score
        return (
            aligned_seq1,
            aligned_seq2,
            alignment_visual,
            self.score_matrix[len(seq1), len(seq2)]
        )

    def visualize_matrix(self, seq1, seq2, title="Score Matrix"):
        """Visualize the score matrix as a heatmap."""
        if self.score_matrix is None:
            raise ValueError("Run align() first to generate the score matrix.")

        plt.figure(figsize=(10, 8))

        # Create labels with sequence characters
        row_labels = [''] + list(seq1)
        col_labels = [''] + list(seq2)

        # Create heatmap
        ax = sns.heatmap(self.score_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                         xticklabels=col_labels, yticklabels=row_labels)

        plt.title(title)
        plt.tight_layout()
        return plt

    def visualize_traceback(self, seq1, seq2):
        """Visualize the traceback matrix."""
        if self.traceback is None:
            raise ValueError("Run align() first to generate the traceback matrix.")

        m, n = len(seq1) + 1, len(seq2) + 1
        plt.figure(figsize=(10, 8))

        # Create a custom colormap for different directions
        cmap = plt.cm.get_cmap('Blues', 4)

        # Create a grid of arrows based on the traceback matrix
        arrow_grid = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if "↖" in self.traceback[i, j]:
                    arrow_grid[i, j] = 3  # Diagonal
                elif "↑" in self.traceback[i, j]:
                    arrow_grid[i, j] = 2  # Up
                elif "←" in self.traceback[i, j]:
                    arrow_grid[i, j] = 1  # Left

        # Create labels with sequence characters
        row_labels = [''] + list(seq1)
        col_labels = [''] + list(seq2)

        # Plot heatmap
        ax = sns.heatmap(arrow_grid, cmap=cmap, xticklabels=col_labels,
                         yticklabels=row_labels, annot=self.traceback, fmt='',
                         cbar=False)

        plt.title('Traceback Matrix')
        plt.tight_layout()
        return plt

    def visualize_alignment(self, aligned_seq1, aligned_seq2, alignment_visual):
        """Visualize the sequence alignment with highlighting."""
        plt.figure(figsize=(12, 4))

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
        plt.tight_layout()
        return plt


# Example usage
def example_alignment():
    """Run an example alignment and visualize the results."""
    # Initialize the algorithm
    nw = NeedlemanWunsch(match_score=1, mismatch_penalty=-1, gap_penalty=-2)

    # Define sequences
    seq1 = "GCATGCU"
    seq2 = "GATTACA"

    print(f"Aligning sequences:\nSeq1: {seq1}\nSeq2: {seq2}\n")

    # Perform alignment
    aligned_seq1, aligned_seq2, alignment_visual, score = nw.align(seq1, seq2)

    # Print results
    print(f"Alignment score: {score}")
    print(f"Aligned Seq1: {aligned_seq1}")
    print(f"Alignment:    {alignment_visual}")
    print(f"Aligned Seq2: {aligned_seq2}")

    # Visualize the score matrix
    nw.visualize_matrix(seq1, seq2)
    plt.savefig('score_matrix.png')

    # Visualize the traceback matrix
    nw.visualize_traceback(seq1, seq2)
    plt.savefig('traceback_matrix.png')

    # Visualize the alignment
    nw.visualize_alignment(aligned_seq1, aligned_seq2, alignment_visual)
    plt.savefig('alignment_visualization.png')

    plt.show()


def align_custom_sequences():
    """Allow user to input custom sequences for alignment."""
    print("\n--- Custom Sequence Alignment ---")
    seq1 = input("Enter first sequence: ").upper()
    seq2 = input("Enter second sequence: ").upper()

    # Validate input - only allow valid nucleotides or amino acids
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")  # Amino acids
    if not all(c in valid_chars for c in seq1 + seq2):
        valid_chars = set("ACGTU")  # Nucleotides
        if not all(c in valid_chars for c in seq1 + seq2):
            print("Invalid characters in sequences. Please use valid nucleotides or amino acids.")
            return

    # Get scoring parameters
    try:
        match = int(input("Enter match score (default 1): ") or "1")
        mismatch = int(input("Enter mismatch penalty (default -1): ") or "-1")
        gap = int(input("Enter gap penalty (default -2): ") or "-2")
    except ValueError:
        print("Invalid input for scoring parameters. Using defaults.")
        match, mismatch, gap = 1, -1, -2

    # Initialize and run the algorithm
    nw = NeedlemanWunsch(match_score=match, mismatch_penalty=mismatch, gap_penalty=gap)
    aligned_seq1, aligned_seq2, alignment_visual, score = nw.align(seq1, seq2)

    # Print results
    print(f"\nAlignment score: {score}")
    print(f"Aligned Seq1: {aligned_seq1}")
    print(f"Alignment:    {alignment_visual}")
    print(f"Aligned Seq2: {aligned_seq2}")

    # Visualize the alignment
    nw.visualize_matrix(seq1, seq2)
    plt.savefig('custom_score_matrix.png')

    nw.visualize_traceback(seq1, seq2)
    plt.savefig('custom_traceback_matrix.png')

    nw.visualize_alignment(aligned_seq1, aligned_seq2, alignment_visual)
    plt.savefig('custom_alignment_visualization.png')

    plt.show()


if __name__ == "__main__":
    print("Needleman-Wunsch Global Sequence Alignment")
    print("------------------------------------------")

    # Run the example alignment
    example_alignment()

    # Ask if user wants to try custom sequences
    if input("\nDo you want to try custom sequences? (y/n): ").lower().startswith('y'):
        align_custom_sequences()

    print("\nThank you for using the Needleman-Wunsch alignment tool!")