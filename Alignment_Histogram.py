#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
from Bio import SeqIO
import pysam
import numpy as np
import matplotlib.pyplot as plt
import traceback # Import traceback for better error reporting

# --- Helper Functions (convert_fastq_to_fasta, run_magicblast, filter_mapped_reads) ---
# These remain the same as in your provided code.

def convert_fastq_to_fasta(fastq_file, fasta_file, min_read_length):
    """
    Convert a FASTQ file to FASTA, filtering reads below a minimum length.
    """
    records_written = 0
    try:
        with open(fastq_file, "r") as fq_in, open(fasta_file, "w") as fa_out:
            for record in SeqIO.parse(fq_in, "fastq"):
                if len(record.seq) >= min_read_length:
                    SeqIO.write(record, fa_out, "fasta")
                    records_written += 1
        print(f"Converted {fastq_file} to {fasta_file}, {records_written} reads written (min length: {min_read_length})")
    except FileNotFoundError:
        print(f"Error: Input FASTQ file not found at {fastq_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during FASTQ to FASTA conversion: {e}")
        sys.exit(1)


def run_magicblast(fasta_file, database, sam_output):
    """
    Run magicblast to align sequences from the FASTA file against the given database.
    """
    command = ["magicblast", "-query", fasta_file, "-db", database, "-out", sam_output]
    print("Running magicblast:")
    print(" ".join(command))
    try:
        # Using stderr=subprocess.PIPE to capture potential errors from magicblast
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stderr:
            print("Magicblast stderr:\n", result.stderr) # Print warnings or info
        print("magicblast completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running magicblast (return code {e.returncode}):")
        print("Command:", " ".join(e.cmd))
        print("Stderr:", e.stderr)
        print("Stdout:", e.stdout)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: 'magicblast' command not found. Is it installed and in your PATH?")
        sys.exit(1)

def filter_mapped_reads(sam_input, mapped_sam_output):
    """
    Use samtools to filter the SAM file, keeping only mapped reads.
    Includes the header.
    """
    # Using samtools view -h -F 4: -h includes header, -F 4 excludes unmapped reads.
    command = f"samtools view -h -F 4 {sam_input} > {mapped_sam_output}"
    print("Filtering mapped reads with the command:")
    print(command)
    try:
        # Using shell=True here is convenient for the redirection '>'.
        # Ensure sam_input and mapped_sam_output are properly handled/escaped if needed.
        subprocess.run(command, shell=True, check=True, executable='/bin/bash') # Be explicit about shell
        print("Mapped reads saved to", mapped_sam_output)
    except subprocess.CalledProcessError as e:
        print(f"Error filtering mapped reads (samtools view, return code {e.returncode}):")
        print(f"Command failed: {command}")
        sys.exit(1)
    except FileNotFoundError:
         print(f"Error: 'samtools' command not found. Is it installed and in your PATH?")
         sys.exit(1)

# --- NEW Histogram Function for Aligned Read Lengths ---

def create_aligned_length_histogram(mapped_sam, output_prefix):
    """
    Creates a histogram of the lengths of reads that were successfully aligned.
    Reads the mapped SAM file, extracts read lengths, and plots the distribution.
    Saves the plot to a file AND displays it interactively.

    Args:
        mapped_sam (str): Path to the SAM file containing *only* mapped reads.
        output_prefix (str): Prefix for the output plot file name.
    """
    samfile = None
    read_lengths = []
    print(f"Reading {mapped_sam} to collect lengths of mapped reads...")
    try:
        # Use 'with' statement for automatic closing
        with pysam.AlignmentFile(mapped_sam, "r") as samfile:
            processed_reads = 0
            counted_reads = 0
            for read in samfile.fetch(until_eof=True): # Iterate through all records
                processed_reads += 1
                # We expect all reads here to be mapped because of the filter step,
                # but double-check and only count primary alignments.
                # Exclude secondary (0x100) and supplementary (0x800) alignments
                # to avoid counting the same original read multiple times.
                if not read.is_unmapped and not read.is_secondary and not read.is_supplementary:
                    read_lengths.append(read.query_length) # Length of the read sequence
                    counted_reads += 1

        print(f"Finished iterating. Processed {processed_reads} alignment records.")
        print(f"Collected lengths for {counted_reads} primary mapped reads.")

        if not read_lengths:
            print("Warning: No primary mapped reads found. Cannot create length histogram.")
            return # Exit the function gracefully

        # --- Create the histogram plot ---
        plt.figure(figsize=(10, 6))
        # Let matplotlib determine good bins, or specify manually, e.g., bins=50
        counts, bin_edges, patches = plt.hist(read_lengths, bins=50, color='lightcoral', edgecolor='black')
        plt.xlabel("Aligned Read Length (bp)")
        plt.ylabel("Number of Reads")
        plt.title(f"Histogram of Aligned Read Lengths")
        plt.grid(axis='y', alpha=0.75)
        # plt.yscale('log') # Optional: use log scale if counts vary widely
        plt.tight_layout()
        # --- End Plot Creation ---

        # --- Save the plot to a file ---
        output_file = f"{output_prefix}_aligned_length_histogram.png"
        plt.savefig(output_file)
        print(f"Aligned read length histogram saved to {output_file}")
        # --- End Save ---

        # --- Display the plot ---
        print("Displaying aligned length histogram plot...")
        plt.show() # Pauses script until plot window is closed
        # --- End Display ---

        # Optionally print summary statistics
        if read_lengths:
            print("\nAligned Read Length Statistics:")
            print(f"  Min length: {np.min(read_lengths)}")
            print(f"  Max length: {np.max(read_lengths)}")
            print(f"  Avg length: {np.mean(read_lengths):.2f}")
            print(f"  Median length: {np.median(read_lengths)}")

    except FileNotFoundError:
        print(f"Error: Mapped SAM file not found at {mapped_sam}")
        sys.exit(1)
    except ValueError as e: # Catches potential errors from pysam/numpy
         print(f"Error processing SAM file {mapped_sam} for length histogram: {e}")
         traceback.print_exc()
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during length histogram creation:")
        traceback.print_exc()
        sys.exit(1)
    # No 'finally' needed here as 'with' handles closing samfile

# --- Existing Histogram Function (Read Start Positions) ---
# This function (create_read_start_histogram) remains unchanged from your version.
def create_read_start_histogram(mapped_sam, output_prefix, bin_size=1000):
    """
    Create a histogram of the number of reads starting in each bin
    across the reference genome. Works with unindexed SAM files.
    Saves the plot to a file AND displays it interactively.

    Args:
        mapped_sam (str): Path to the SAM file containing mapped reads.
        output_prefix (str): Prefix for the output plot file name.
        bin_size (int): Size of each bin in the histogram (default: 1000).
    """
    samfile = None # Initialize to None
    try:
        samfile = pysam.AlignmentFile(mapped_sam, "r")

        # Robust Header Check (same as before)
        if not samfile.header or not samfile.references or not samfile.lengths or len(samfile.lengths) == 0:
             if not samfile.header: print(f"Error: SAM file {mapped_sam} seems to be missing its header entirely.")
             elif not samfile.references: print(f"Error: SAM file {mapped_sam} header lacks reference sequence names (likely missing @SQ lines).")
             elif not samfile.lengths or len(samfile.lengths) == 0: print(f"Error: SAM file {mapped_sam} header lacks reference sequence lengths (likely missing LN tags in @SQ lines or malformed header).")
             else: print(f"Error: Could not obtain valid reference sequence information from header in {mapped_sam}.")
             print("Cannot determine reference dimensions for histogram. Please ensure the input SAM file has a valid header with @SQ lines.")
             sys.exit(1)

        # Assume single reference sequence (same as before)
        if len(samfile.references) > 1:
            print(f"Warning: Multiple reference sequences found ({samfile.nreferences}). Using the first one ('{samfile.references[0]}') for the start position histogram.")
        ref_name = samfile.references[0]
        ref_length = samfile.lengths[0]
        target_tid = samfile.get_tid(ref_name)
        if target_tid == -1:
             print(f"Error: Could not find reference name '{ref_name}' in SAM header TIDs.")
             sys.exit(1)

        # Calculate bins and initialize counts (same as before)
        num_bins = (ref_length + bin_size - 1) // bin_size
        read_starts_per_bin = np.zeros(num_bins, dtype=int)

        # Iterate through SAM file and count reads (same as before)
        processed_reads = 0
        counted_reads = 0
        print(f"Iterating through alignments in {mapped_sam} to count starts for reference '{ref_name}'...")
        # Use fetch(until_eof=True) for potentially unindexed SAM
        for read in samfile.fetch(until_eof=True):
            processed_reads += 1
            # Only count primary alignments for the start position histogram
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            # Ensure alignment is to the target reference sequence
            if read.reference_id != target_tid:
                continue
            start = read.reference_start
            if start is None:
                # This shouldn't happen for mapped, primary reads, but check just in case
                print(f"Warning: Read {read.query_name} is mapped but has no reference_start. Skipping.")
                continue
            bin_index = start // bin_size
            if 0 <= bin_index < num_bins:
                read_starts_per_bin[bin_index] += 1
                counted_reads += 1
            else:
                 print(f"Warning: Read {read.query_name} starts at {start} (ref_length={ref_length}), outside expected bin range [0, {num_bins-1}]. Skipping.")

        print(f"Finished iterating. Processed {processed_reads} alignment records.")
        print(f"Total reads counted with start position on '{ref_name}': {counted_reads}")
        if counted_reads == 0:
            print(f"Warning: No primary alignments found starting on reference '{ref_name}'. Start position histogram will be empty.")

        # --- Create the histogram plot (same plotting code as before) ---
        plt.figure(figsize=(12, 6))
        bin_positions = np.arange(num_bins) * bin_size
        plt.bar(bin_positions, read_starts_per_bin, width=bin_size, align='edge', color='skyblue', edgecolor='black')
        plt.xlabel(f"Genome Position on '{ref_name}' (Bin Start)")
        plt.ylabel("Number of Reads Starting in Bin")
        plt.title(f"Histogram of Read Start Positions (Bin Size: {bin_size} bp)")
        plt.grid(axis='y', alpha=0.75)
        plt.xlim(0, ref_length)
        # Adjust ticks (same logic as before)
        tick_step = max(1, (num_bins // 20) if num_bins > 0 else 1) # Avoid division by zero
        tick_positions = bin_positions[::tick_step]
        # Ensure the last position is included if it's not already
        if num_bins > 0 and bin_positions[-1] not in tick_positions:
             tick_positions = np.append(tick_positions, bin_positions[-1])
        plt.xticks(tick_positions, [f"{int(x):,}" for x in tick_positions], rotation=45, ha="right")

        plt.tight_layout()
        # --- End Plot Creation ---

        # --- Save the plot to a file ---
        output_file = f"{output_prefix}_{ref_name}_read_start_histogram_bin{bin_size}.png" # Added bin size to filename
        plt.savefig(output_file)
        print(f"Read start histogram saved to {output_file}")
        # --- End Save ---

        # --- Display the plot ---
        print("Displaying start position histogram plot...")
        plt.show() # Pauses script until plot window is closed
        # --- End Display ---

        # Print the bin information to the console (same as before)
        print("\nBin Information (Reads Starting Per Bin, excluding empty bins):")
        print("Bin_Start\tBin_End\tReads_in_Bin")
        for i, read_count in enumerate(read_starts_per_bin):
            if read_count > 0:
                bin_start = i * bin_size
                bin_end = min((i + 1) * bin_size - 1, ref_length - 1)
                print(f"{bin_start}\t{bin_end}\t{read_count}")

    except FileNotFoundError:
        print(f"Error: Mapped SAM file not found at {mapped_sam}")
        sys.exit(1)
    except ValueError as e:
         print(f"Error processing SAM file {mapped_sam} for start position histogram: {e}")
         traceback.print_exc()
         sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during start position histogram creation:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if samfile:
            samfile.close()

# --- Main Function (Updated) ---

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: Convert FASTQ to FASTA, align with magicblast, filter mapped reads, "
                    "create histogram of aligned read lengths, and create histogram of read start positions. " # Updated description
                    "Saves and displays plots."
    )
    parser.add_argument("--fastq", required=True, help="Input FASTQ file")
    parser.add_argument("--database", required=True, help="Magicblast database path prefix to align against")
    parser.add_argument("--output_dir", default=".", help="Directory to store output files (default: current directory)")
    parser.add_argument("--prefix", default="result", help="Prefix for output file names (default: 'result')")
    parser.add_argument("--min_read_length", type=int, default=0,
                        help="Minimum read length to include in alignment (default: 0)")
    parser.add_argument("--bin_size", type=int, default=1000,
                        help="Size of each bin in the start position histogram (in base pairs, default: 1000)")
    args = parser.parse_args()

    # Input Validation (same as before)
    if not os.path.isfile(args.fastq):
        print(f"Error: Input FASTQ file not found: {args.fastq}")
        sys.exit(1)
    db_extensions = ['.nhr', '.nin', '.nsq', '.nal']
    db_exists = any(os.path.isfile(args.database + ext) for ext in db_extensions)
    if not db_exists:
         print(f"Warning: Cannot find typical BLAST index files ({', '.join(db_extensions)}) for database prefix '{args.database}'. Make sure the path is correct.")
    if args.bin_size <= 0:
        print(f"Error: --bin_size must be a positive integer (got {args.bin_size})")
        sys.exit(1)

    # Create output directory (same as before)
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Using output directory: {os.path.abspath(args.output_dir)}")
    except OSError as e:
        print(f"Error creating output directory '{args.output_dir}': {e}")
        sys.exit(1)

    # Define file paths (same as before)
    output_dir_abs = os.path.abspath(args.output_dir)
    fasta_file = os.path.join(output_dir_abs, f"{args.prefix}.fasta")
    sam_file = os.path.join(output_dir_abs, f"{args.prefix}_magicblast.sam")
    mapped_sam = os.path.join(output_dir_abs, f"{args.prefix}_mapped.sam")
    histogram_prefix = os.path.join(output_dir_abs, args.prefix) # Prefix for plot files

    # --- Pipeline steps ---
    print("--- Step 1: Convert FASTQ to FASTA ---")
    convert_fastq_to_fasta(args.fastq, fasta_file, args.min_read_length)

    print("\n--- Step 2: Run Magicblast Alignment ---")
    run_magicblast(fasta_file, args.database, sam_file)

    print("\n--- Step 3: Filter Mapped Reads (samtools) ---")
    filter_mapped_reads(sam_file, mapped_sam)

    print("\n--- Step 4: Create Aligned Read Length Histogram ---") # NEW STEP
    create_aligned_length_histogram(mapped_sam, histogram_prefix)

    print("\n--- Step 5: Create Read Start Position Histogram ---") # Renumbered step
    create_read_start_histogram(mapped_sam, histogram_prefix, bin_size=args.bin_size)

    print("\n--- Pipeline finished ---")


if __name__ == '__main__':
    main()
