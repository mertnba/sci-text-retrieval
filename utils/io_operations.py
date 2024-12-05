import csv
import pandas as pd
from collections import defaultdict

def load_queries(file_path):
    """
    Loads queries from a TSV file.

    Args:
        file_path (str): Path to the queries TSV file.

    Returns:
        dict: A dictionary where keys are query IDs and values are queries.
    """
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        next(tsv_reader, None)  # Skip the header row if present
        for row in tsv_reader:
            qid = row[0]
            query_text = row[2]
            queries[qid] = query_text
    return queries

def load_candidate_passages(file_path):
    """
    Loads candidate passages from a TSV file.

    Args:
        file_path (str): Path to the candidate passages TSV file.

    Returns:
        defaultdict: A dictionary where keys are query IDs and values are lists of passage IDs.
    """
    candidate_passages = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        next(tsv_reader, None)  # Skip the header row if present
        for row in tsv_reader:
            qid = row[0]
            pid = row[1]
            candidate_passages[qid].append(pid)
    return candidate_passages

def load_passages(file_path):
    """
    Loads passages from a TSV file.

    Args:
        file_path (str): Path to the passages TSV file.

    Returns:
        dict: A dictionary where keys are passage IDs and values are passage texts.
    """
    passages = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        next(tsv_reader, None)  # Skip the header row if present
        for row in tsv_reader:
            pid = row[1]
            passage_text = row[3]
            passages[pid] = passage_text
    return passages

def write_results_to_file(output_filename, results, columns):
    """
    Writes results to a CSV file.

    Args:
        output_filename (str): Path to the output CSV file.
        results (list of dict): List of dictionaries containing the result data.
        columns (list of str): Column names for the output CSV file.
    """
    with open(output_filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

def write_ranked_passages_to_txt(input_file, output_file):
    """
    Converts a CSV file of ranked passages into a plain text file format.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output plain text file.
    """
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    current_qid = None
    current_rank = 1
    with open(output_file, 'w') as outfile:
        for line in lines:
            parts = line.strip().split(' ')
            qid = parts[0]
            if qid == current_qid:
                current_rank += 1
            else:
                current_qid = qid
                current_rank = 1
            parts[3] = str(current_rank)
            outfile.write(' '.join(parts) + '\n')
