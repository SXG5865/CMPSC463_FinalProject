import numpy as np
import matplotlib.pyplot as plt
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

    def analyze_sequence_conservation(self, aligned_seq1, aligned_seq2, alignment_visual, window_size=5):
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


    def visualize_alignment_enhanced(self, aligned_seq1, aligned_seq2, alignment_visual, max_width=80,
                                     highlight_regions=True):
        """
        Enhanced visualization for sequence alignment with region highlighting and statistics.

        Parameters:
        - aligned_seq1, aligned_seq2: The aligned sequences
        - alignment_visual: Visual representation of the alignment
        - max_width: Maximum number of characters to show in a single view
        - highlight_regions: Whether to highlight conserved and variable regions
        """
        total_length = len(aligned_seq1)

        # For very long alignments, create a multi-scale visualization
        if total_length > max_width * 3:
            plt.figure(figsize=(16, 12))

            # Section 1: Alignment Statistics (top left)
            plt.subplot(2, 2, 1)
            matches = alignment_visual.count('|')
            mismatches = alignment_visual.count('.')
            gaps = alignment_visual.count(' ')

            match_percent = (matches / total_length) * 100
            mismatch_percent = (mismatches / total_length) * 100
            gap_percent = (gaps / total_length) * 100

            categories = ['Matches', 'Mismatches', 'Gaps']
            values = [match_percent, mismatch_percent, gap_percent]
            colors = ['green', 'red', 'grey']

            plt.bar(categories, values, color=colors, alpha=0.6)
            plt.title(f'Alignment Statistics')
            plt.ylabel('Percentage (%)')
            plt.ylim(0, 100)

            # Add text labels
            for i, v in enumerate(values):
                plt.text(i, v + 2, f"{v:.1f}%", ha='center')

            # Section 2: Start of alignment (top right)
            plt.subplot(2, 2, 2)
            start_display = min(max_width, total_length)
            self._plot_alignment_segment(aligned_seq1[:start_display],
                                         aligned_seq2[:start_display],
                                         alignment_visual[:start_display],
                                         title="Starting Region")

            # Section 3: Middle of alignment (bottom left)
            plt.subplot(2, 2, 3)
            if total_length > max_width * 2:
                mid_point = total_length // 2
                half_window = max_width // 2
                mid_start = max(0, mid_point - half_window)
                mid_end = min(total_length, mid_point + half_window)

                self._plot_alignment_segment(
                    aligned_seq1[mid_start:mid_end],
                    aligned_seq2[mid_start:mid_end],
                    alignment_visual[mid_start:mid_end],
                    title="Middle Region"
                )
            else:
                # If not long enough, use this for another purpose
                # Find the region with the most mismatches to highlight
                window_size = min(max_width, total_length // 2)
                max_mismatch_count = 0
                max_mismatch_pos = 0

                for i in range(0, total_length - window_size):
                    mismatch_count = alignment_visual[i:i + window_size].count('.')
                    if mismatch_count > max_mismatch_count:
                        max_mismatch_count = mismatch_count
                        max_mismatch_pos = i

                self._plot_alignment_segment(
                    aligned_seq1[max_mismatch_pos:max_mismatch_pos + window_size],
                    aligned_seq2[max_mismatch_pos:max_mismatch_pos + window_size],
                    alignment_visual[max_mismatch_pos:max_mismatch_pos + window_size],
                    title="Region with Most Mismatches"
                )

            # Section 4: End of alignment (bottom right)
            plt.subplot(2, 2, 4)
            end_start = max(0, total_length - max_width)
            self._plot_alignment_segment(
                aligned_seq1[end_start:],
                aligned_seq2[end_start:],
                alignment_visual[end_start:],
                title="Ending Region"
            )

            plt.suptitle(f"Multi-region Alignment View (Total Length: {total_length} positions)", fontsize=16)

        elif total_length > max_width:
            # For moderate length sequences, use the existing chunked view approach
            plt.figure(figsize=(16, 8))

            # First section: Summary statistics
            plt.subplot(2, 1, 1)
            matches = alignment_visual.count('|')
            mismatches = alignment_visual.count('.')
            gaps = alignment_visual.count(' ')

            match_percent = (matches / total_length) * 100
            mismatch_percent = (mismatches / total_length) * 100
            gap_percent = (gaps / total_length) * 100

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
            display_length = min(max_width, total_length)

            # Add sequence conservation analysis if enabled
            if highlight_regions and total_length > 10:
                # Calculate conservation score for highlighting regions
                self._plot_alignment_segment_with_conservation(
                    aligned_seq1[:display_length],
                    aligned_seq2[:display_length],
                    alignment_visual[:display_length],
                    title=f"First {display_length} positions of alignment (showing {display_length}/{total_length})"
                )
            else:
                self._plot_alignment_segment(
                    aligned_seq1[:display_length],
                    aligned_seq2[:display_length],
                    alignment_visual[:display_length],
                    title=f"First {display_length} positions of alignment (showing {display_length}/{total_length})"
                )

        else:
            # For shorter alignments, show everything directly
            plt.figure(figsize=(min(total_length / 5 + 2, 20), 4))

            if highlight_regions and total_length > 10:
                self._plot_alignment_segment_with_conservation(aligned_seq1, aligned_seq2, alignment_visual)
            else:
                self._plot_alignment_segment(aligned_seq1, aligned_seq2, alignment_visual)

        plt.tight_layout()
        return plt