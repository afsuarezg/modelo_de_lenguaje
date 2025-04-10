"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary
of a text to a configurable number of symbols, with only a small increase in the number of tokens.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from functools import lru_cache

import regex as re
from tqdm import tqdm

from cs336_basics.tokenizer.tas_tokenization_utils import SPLIT_PATTERN, bytes_to_unicode

logger = logging.getLogger(__name__)


def find_sublist_indices(main_list, sublist):
    indices = []
    sublist_len = len(sublist)
    i = 0

    while i <= len(main_list) - sublist_len:
        if main_list[i : i + sublist_len] == sublist:
            indices.append(i)
            # Skip past this match to avoid overlapping matches
            i += sublist_len
        else:
            i += 1
    return indices


def update_pair_counts_and_index(
    pair_to_merge: tuple[str, str],
    changed,
    pairs_to_corpus_frequency: Counter[tuple[str, str]],
    pairs_to_vocab_index_and_num_occurrences: dict[tuple[str, str], Counter[int]],
):
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    for j, word, old_word, freq in changed:
        # Update the statistics to reflect the removal of the original pair
        for old_word_pair_start_index in find_sublist_indices(old_word, pair_to_merge):
            if old_word_pair_start_index > 0:
                # Case 1: suppose that `old_word` is the sequence ("A", "B", "C"). If ("B", "C") are merged,
                # then we need to reduce the frequency of ("A", "B").
                preceding_pair = old_word[old_word_pair_start_index - 1 : old_word_pair_start_index + 1]
                if pairs_to_corpus_frequency[preceding_pair] > freq:
                    pairs_to_corpus_frequency[preceding_pair] -= freq
                    pairs_to_vocab_index_and_num_occurrences[preceding_pair][j] -= 1
                else:
                    # Remove pairs if they go to 0 count, so we don't unnecessarily
                    # waste time in max()
                    pairs_to_corpus_frequency.pop(preceding_pair)
                    pairs_to_vocab_index_and_num_occurrences.pop(preceding_pair)
            if old_word_pair_start_index < len(old_word) - 2:
                # Case 2: suppose that `old_word` is the sequence ("A", "B", "C", "B"). If ("B", "C") are merged,
                # then we need to reduce the frequency of ("C", "B").
                # However, we want to skip this if `old_word` is the sequence
                # ("A", "B", "C", "B", "C"), because the frequency of ("C", "B") will be reduced in case 1.
                if (old_word[old_word_pair_start_index + 2 : old_word_pair_start_index + 4]!= pair_to_merge):
                    following_pair = old_word[old_word_pair_start_index + 1 : old_word_pair_start_index + 3]
                    if pairs_to_corpus_frequency[following_pair] > freq:
                        pairs_to_corpus_frequency[following_pair] -= freq
                        pairs_to_vocab_index_and_num_occurrences[following_pair][j] -= 1
                    else:
                        # Remove pairs if they go to 0 count, so we don't unnecessarily
                        # waste time in max()
                        pairs_to_corpus_frequency.pop(following_pair)
                        pairs_to_vocab_index_and_num_occurrences.pop(following_pair)
        # Update statistics for addition of new pair
        merged_pair: str = pair_to_merge[0] + pair_to_merge[1]
        new_word_merged_pair_indices = [idx for idx, piece in enumerate(word) if piece == merged_pair]
        for new_word_merged_pair_index in new_word_merged_pair_indices:
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if new_word_merged_pair_index > 0:
                # Case 1: suppose that `word` is the sequence ("A", "BC", "D"), since we're merging ("B", "C").
                # Then, we'd need to increase the frequency of ("A", "BC")
                preceding_pair = word[new_word_merged_pair_index - 1 : new_word_merged_pair_index + 1]
                pairs_to_corpus_frequency[preceding_pair] += freq
                pairs_to_vocab_index_and_num_occurrences[preceding_pair][j] += 1
            if (new_word_merged_pair_index < len(word) - 1 and word[new_word_merged_pair_index + 1] != merged_pair):
                # Case 2: suppose that `word` is the sequence ("A", "BC", "B"), since we're merging ("B", "C").
                # Then, we'd need to increase the frequency of ("BC", "B").
                # However, we want to skip this if `word` is the sequence
                # ("A", "BC", "BC"), because the frequency of ("BC", "BC") will be increased in case 1.
                following_pair = word[new_word_merged_pair_index : new_word_merged_pair_index + 2]
                pairs_to_corpus_frequency[following_pair] += freq
                pairs_to_vocab_index_and_num_occurrences[following_pair][j] += 1


def count_and_index_segment_pairs(sorted_vocab_with_corpus_frequency: list[tuple[tuple[str, ...], int]]):
    """
    Count the frequency of the segment pairs across all words and create an index from
    segment pair to words that they occur in and times they occur in those words.
    """
    # Map pairs to their corpus frequency (the number of times they occur in the input text)
    pairs_to_corpus_frequency: Counter[tuple[str, str]] = Counter()

    # Map pairs to a dictionary with the indices of vocab items that contain this pair (keys)
    # and the number of times the pair occurs in the vocab item (values)
    pairs_to_vocab_index_and_num_occurrences: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)

    for i, (segmented_word, word_corpus_frequency) in enumerate(sorted_vocab_with_corpus_frequency):
        # Iterate over pairs of consecutive characters in the word, populating the
        # pair frequencies  and the index from pairs to words.
        for j in range(len(segmented_word) - 1):
            prev_char, char = segmented_word[j], segmented_word[j + 1]
            pairs_to_corpus_frequency[prev_char, char] += word_corpus_frequency
            pairs_to_vocab_index_and_num_occurrences[prev_char, char][i] += 1
    return pairs_to_corpus_frequency, pairs_to_vocab_index_and_num_occurrences


def merge_pair_in_vocab(
    pair_to_merge: tuple[str, str],
    sorted_vocab_with_corpus_frequency: list[tuple[tuple[str, ...], int]],
    pairs_to_vocab_index_and_num_occurrences: dict[tuple[str, str], Counter[int]],
):
    """
    Edit the vocab to replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'
    """
    changes = []
    escaped_search_for = re.escape(" ".join(pair_to_merge))
    # Make sure that `escaped_search_for` is standalone. It should be surrounded by whitespace or
    # located at the start or end of the string, without being immediately adjacent to other
    # non-whitespace characters.
    pattern = rf"(?<!\S){escaped_search_for}(?!\S)"
    # Replace each backslash (\) with double backslashes (\\) to escape them
    # in the merged replacement string
    replacement = "".join(pair_to_merge).replace("\\", "\\\\")

    for (vocab_index, num_pair_occurences_in_word) in pairs_to_vocab_index_and_num_occurrences[pair_to_merge].items():
        if num_pair_occurences_in_word < 1:
            continue
        segmented_word, segmented_word_corpus_freq = sorted_vocab_with_corpus_frequency[vocab_index]
        segmented_word_with_merged_pair = tuple(re.sub(pattern, replacement, " ".join(segmented_word)).split(" "))
        sorted_vocab_with_corpus_frequency[vocab_index] = (segmented_word_with_merged_pair, segmented_word_corpus_freq)
        changes.append(
            (
                vocab_index,
                segmented_word_with_merged_pair,
                segmented_word,
                segmented_word_corpus_freq,
            )
        )

    return changes


def train_bpe(
    input_path,
    vocab_size,
    special_tokens,
    vocab_outpath=None,
    merges_outpath=None,
    break_ties_with_gpt2_remapped_tokens=False,
):
    """
    Learn a BPE vocabulary of size `vocab_size`, and write the vocab to
    `vocab_outpath` and `merges_outpath`.
    """
    gpt2_byte_encoder = bytes_to_unicode()
    gpt2_byte_decoder = {v: k for k, v in gpt2_byte_encoder.items()}

    @lru_cache(maxsize=None)
    def decode_bytes_from_gpt2_encoded(gpt2_encoded: str):
        return bytes([gpt2_byte_decoder[token] for token in gpt2_encoded])

    # Initialize the vocabulary with the special tokens and
    # all the bytes.
    vocabulary = {}
    for special_token in special_tokens:
        vocabulary[special_token] = len(vocabulary)
    for _, byte_str_repr in gpt2_byte_encoder.items():
        vocabulary[byte_str_repr] = len(vocabulary)

    if vocab_size <= len(vocabulary):
        raise ValueError(
            f"Asked for vocab size of {vocab_size}, but special tokens and singular bytes already "
            f"take up {len(vocabulary)} vocab items. Please increase the vocab size"
        )
    num_merges = vocab_size - len(vocabulary)

    # Get the counts of the unique words in the input file.
    # e.g., {'the': 10, 'apple': 7, ...}
    logger.info("Pre-tokenizing input and counting tokens")
    token_counts: Counter[str] = Counter()
    with open(input_path, encoding='utf-8') as infile:
        infile_content = infile.read()
        token_matches = re.finditer(SPLIT_PATTERN, infile_content)
        for token_match in tqdm(token_matches):
            token = token_match.group()
            if token:
                # encode the token as a bytes (b'') object
                token_bytes = token.encode("utf-8")
                # translate all bytes to their unicode string representation and flatten
                token_translated = "".join(gpt2_byte_encoder[b] for b in token_bytes)
                token_counts[token_translated] += 1
    logger.info("Finished pre-tokenizing input and counting tokens")

    # Segment the tokens into characters.
    # e.g., {('t', 'h', 'e'): 10, ('a', 'p', 'p', 'l', 'e'): 7, ...}
    segmented_token_counts: Counter[tuple[str, ...]] = Counter({tuple(token): count for (token, count) in token_counts.items()})
    # Create a vocab (int index <-> word mapping) by sorting the segmented words by corpus frequency.
    # e.g., [(('t', 'h', 'e'), 10), (('a', 'p', 'p', 'l', 'e'), 7), ...]
    sorted_vocab_with_corpus_frequency: list[tuple[tuple[str, ...], int]] = sorted(segmented_token_counts.items(), key=lambda x: x[1], reverse=True)

    (pairs_to_corpus_frequency, pairs_to_vocab_index_and_num_occurrences) = count_and_index_segment_pairs(sorted_vocab_with_corpus_frequency)

    merges = []
    for _ in tqdm(range(num_merges)):
        if break_ties_with_gpt2_remapped_tokens:
            # Match HF behavior: break ties on alphabetical order in the
            # GPT-2 remapped token space.
            pair_to_merge = max(pairs_to_corpus_frequency, key=lambda x: (pairs_to_corpus_frequency[x], x))
        else:
            # Match behavior in our handout: break ties on alphabetical order
            # with the original bytestring.
            pair_to_merge = max(pairs_to_corpus_frequency, key=lambda x: (pairs_to_corpus_frequency[x],(decode_bytes_from_gpt2_encoded(x[0]), decode_bytes_from_gpt2_encoded(x[1]))))

        changes = merge_pair_in_vocab(
            pair_to_merge,
            sorted_vocab_with_corpus_frequency,
            pairs_to_vocab_index_and_num_occurrences)

        update_pair_counts_and_index(
            pair_to_merge,
            changes,
            pairs_to_corpus_frequency,
            pairs_to_vocab_index_and_num_occurrences)

        merges.append((pair_to_merge[0], pair_to_merge[1]))

        # Remove the merged pair from the counts, since we won't need them anymore
        pairs_to_corpus_frequency.pop(pair_to_merge)
        pairs_to_vocab_index_and_num_occurrences.pop(pair_to_merge)

    # Write the merges out to the file
    if merges_outpath:
        with open(merges_outpath, "w", encoding="utf-8") as merges_outfile:
            for merge in merges:
                merges_outfile.write(f"{merge[0]} {merge[1]}\n")

    # Add the merges to the vocabulary
    for merge in merges:
        vocabulary["".join(merge)] = len(vocabulary)

    if vocab_outpath:
        with open(vocab_outpath, "w", encoding="utf-8") as vocab_outfile:
            json.dump(vocabulary, vocab_outfile, indent=4, ensure_ascii=False)

    # Return vocab and merges for tests.
    vocab = {gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item]) for gpt2_vocab_item, gpt2_vocab_index in vocabulary.items()}
    merges = [(decode_bytes_from_gpt2_encoded(merge[0]), decode_bytes_from_gpt2_encoded(merge[1])) for merge in merges]

    return vocab, merges


