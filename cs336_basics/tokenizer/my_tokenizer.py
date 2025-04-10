from collections.abc import Iterable, Iterator
from typing import List, Tuple, Set
import pickle
import regex as re
import json

from .tas_train_bpe import train_bpe
from tests.common import gpt2_bytes_to_unicode

class Tokenizer():
    def __init__(self, 
                 vocab:dict[int,bytes],
                 merges:list[tuple[bytes,bytes]],
                 special_tokens:list[str]|None=None,
                 errors: str='replace',
                 pretokenizer_pattern:str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):

        self.vocab=vocab #dict[int,bytes]
        self.vocab_inv={v:k for k,v in vocab.items()}#dict[bytes,int]
        # self.merges={(self.vocab_inv[first], self.vocab_inv[second]):i for i, (first, second) in enumerate(merges, start=256)}#dict[(int,int):new_index]
        self.byte_encoder= gpt2_bytes_to_unicode() #dict[int, str]
        self.byte_decoder= {v:k for k,v in self.byte_encoder.items()} #dict[str, int]
        self.merges=merges
        self.merges_ranks={(first, second): i for i, (first,second) in enumerate(merges, start=256)}#dict[tuple[bytes, bytes]: merge_index]
        self.pretokenizer_pattern = pretokenizer_pattern
        self.errors=errors
        self.cache={}

        if special_tokens:
            self.special_tokens=special_tokens
            escaped_special_tokens=[re.escape(d) for d in sorted(special_tokens, reverse=True, key=len)]
            self.special_tokens_pattern=f"({'|'.join(escaped_special_tokens)})"
        else:
            self.special_tokens=None
            self.special_tokens_pattern=None

    @classmethod
    def from_files(cls,
                   vocab_filepath:str,
                   merges_filepath:str,
                   special_tokens:list[str]|None=None,
                   errors: str='replace',
                   pretokenizer_pattern:str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):
        
        # bpe token encoder/decoder
        with open(vocab_filepath, encoding='utf-8') as f:
            vocab = json.load(f)
        merges = []
        with open(merges_filepath, encoding='utf-8') as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(cleaned_line.split(" ")))
        
        return cls(vocab=vocab, 
                   merges=merges, 
                   special_tokens=special_tokens, 
                   errors=errors,
                   pretokenizer_pattern=pretokenizer_pattern)


    def _bpe(self, pretoken_bytes: bytes) -> list[int]:
        """
        input:
            - pretoken_bytes_list: list[str]

        output: 
            - list[ints]

       """
        pretoken_bytes_flattened=[bytes([b]) for b in pretoken_bytes]

        while len(pretoken_bytes_flattened)>=2:

            pairs=self.get_unique_pairs_bytes(pretoken_bytes_flattened)

            min_pair=min(pairs, key=lambda pair:self.merges_ranks.get(pair, float('inf')))

            if min_pair not in self.merges_ranks:
                break 

            else:
                pretoken_bytes_flattened=self.merge_bytes(bytes_list=pretoken_bytes_flattened, pair=min_pair)
            
        pretoken_bytes_2_ints=[self.vocab_inv[b] for b in pretoken_bytes_flattened]

        return pretoken_bytes_2_ints
    

    def chunk_string_on_linebreak(self, text:str, min_chars:int=5000):
        """
        Lazily yields chunks of the input string with at least `min_chars` characters,
        ending at the next newline character ('\n'). The next chunk starts after that newline.
        """
        index = 0
        length = len(text)

        while index < length:
            # Tentatively look ahead to the minimum number of characters
            next_index = index + min_chars

            # If we're already at the end, yield the rest
            if next_index >= length:
                yield text[index:]
                break

            # If the char at next_index is a newline, split there
            if text[next_index] == '\n':
                yield text[index:next_index]
                index = next_index + 1
                continue

            # Otherwise, find the next newline
            newline_index = text.find('\n', next_index)
            if newline_index == -1:
                # No newline found, yield the rest
                yield text[index:]
                break
            else:
                yield text[index:newline_index]
                index = newline_index + 1

    def chunk_string_on_linebreak(self, text_iter: Iterable, min_chars: int = 1000):
        """
        Lazily yields chunks from an iterable of strings with at least `min_chars` characters,
        ensuring chunks end at the next newline (including the newline character) and resume just after it.
        """
        buffer = ""
        for piece in text_iter:
            buffer += piece
            while True:
                if len(buffer) < min_chars:
                    break  # Wait for more content

                # Find position where we can safely split (next newline after min_chars)
                split_pos = buffer.find('\n', min_chars)
                if split_pos == -1:
                    break  # No newline yet, need more content

                # Include the newline character in the yielded chunk
                yield buffer[:split_pos + 1]
                buffer = buffer[split_pos + 1:]  # Resume after the newline

        # Final chunk if any content is left
        if buffer:
            yield buffer


    def encode(self, text:str)-> list[int]:

        output_index=[]

        chunks = self.split_string_by_special_tokens(text, self.special_tokens_pattern)

        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                pretokens=[chunk]
            
            else:
                pretokens=(pretoken_match.group() for pretoken_match in re.finditer(self.pretokenizer_pattern, chunk)) #list[str]

            for pretoken in pretokens:#cada pretoken es una string
                
                if self.special_tokens and pretoken in self.special_tokens:#self.special_tokens es una lista de strings
                        output_index.append(self.vocab_inv[pretoken.encode('utf-8')])#hay que agregar a la lista definitiva de indices el índice que corresponde al special_token. Como vocab_inv es dict[bytes, ints] e incluye todos los elementos del diccionary, incluyendo los special_tokens, entonces hay encodificar la string a bytes y buscar esos bytes en vocab_inv.
                        # En caso de que el pretoken no coincida con ninguno de los special_tokens, el pretoken es una string que se debe procesar a través de la combinación de los ints que más se repiten en el orden en el que se hicieron los merges. 
                else:
                    pretoken_bytes=pretoken.encode("utf-8")              
                    pretoken_all_merges=self._bpe(pretoken_bytes)
                    output_index.extend(pretoken_all_merges)
                    
        return output_index


    def encode_iterable(self,  iterable:Iterable[str]) -> Iterator[int]:
        for chunk in self.chunk_string_on_linebreak(iterable):
                yield from self.encode(chunk)
        


    def decode(self, ids: List[int])-> str:
        bytes_list = list(map(self.vocab.get, ids))
        flattened_bytes_unicode_printable = b"".join(bytes_list)
        flattened_strings_unicode_printable = [chr(byte) for byte in flattened_bytes_unicode_printable]
        # flattened_bytes_unicode_original=bytearray([self.byte_decoder[elem] for elem in flattened_strings_unicode_printable])
        
        text=flattened_bytes_unicode_printable.decode("utf-8", errors=self.errors)
        return text


    def get_unique_pairs(self, pretoken_indices: list[int]) -> Set[Tuple[int,int]]:
        
        pairs = set()
        for pair in zip(pretoken_indices, pretoken_indices[1:]):
            pairs.add(pair)

        return pairs


    def get_unique_pairs_bytes(self, bytes_list) -> set[tuple[bytes, bytes]]:
        pairs = set()
        for i in range(len(bytes_list) - 1):
            pairs.add((bytes_list[i], bytes_list[i + 1]))
        return pairs


    def lazy_read_chunks(self, file_handle, min_length=1000):
        buffer = ""
        # with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = file_handle.read(min_length)
            if not chunk:
                break
            buffer += chunk

            # Keep reading until we have at least one space after the min_length
            while True:
                next_char = file_handle.read(1)
                if not next_char or next_char.isspace():
                    buffer += next_char
                    break
                buffer += next_char

            # Now split and keep the last (incomplete) word for the next round
            last_space_index = buffer.rfind(' ')
            if last_space_index == -1:
                # no spaces found (very rare), yield everything
                yield buffer
                buffer = ""
            else:
                yield buffer[:last_space_index]
                buffer = buffer[last_space_index + 1:]

        # Yield remaining buffer
        if buffer.strip():
            yield buffer


    def lazy_read_chunks2(self, file_handle, min_length=1000):
        while True:
            start_pos = file_handle.tell()
            chunk = file_handle.read(min_length)
            if not chunk:
                break

            # Read forward until we find a space or EOF (we allow space as the first char of next chunk)
            extension = ''
            while True:
                char = file_handle.read(1)
                if not char:  # EOF
                    break
                extension += char
                if char.isspace():
                    break

            full_chunk = chunk + extension

            # Find last space BEFORE min_length mark (not in the extended part)
            split_index = full_chunk.rfind(' ', 0, len(chunk) + len(extension))
            if split_index == -1:
                # no space found, yield whole thing
                yield full_chunk
            else:
                yield full_chunk[:split_index]
                # Move file pointer back to where the next chunk should start
                file_handle.seek(start_pos + split_index)


    def lazy_read_chunks(self, file_handle, min_length=1000):
        """
        Lazily reads strings from a file, ensuring words are not cut in half,
        while resuming exactly from the previous chunk's end.

        Args:
            filepath (str): The path to the file.
            min_length (int): The minimum length of each string chunk.

        Yields:
            str: String chunks from the file.
        """
        buffer = ""
        while True:
            chunk = file_handle.read(4096)  # Read in chunks of 4KB
            if not chunk:
                if buffer: #yield the last bit if any
                    yield buffer
                break

            buffer += chunk

            if len(buffer) < min_length:
                continue  # Not enough data yet

            last_space = buffer.rfind(' ', 0, min_length)

            if last_space == -1: #no spaces found
                yield buffer[:min_length]
                buffer = buffer[min_length:]
            else:
                yield buffer[:last_space]
                buffer = buffer[last_space + 1:] # +1 to skip the space


    def lazy_read_strings(file_handle, min_length=200):
        """
        Lazily reads strings from a file, ensuring words are not cut in half,
        while resuming exactly from the previous chunk's end.

        Args:
            filepath (str): The path to the file.
            min_length (int): The minimum length of each string chunk.

        Yields:
            str: String chunks from the file.
        """
        buffer = ""
        while True:
            chunk = file_handle.read(4096)  # Read in chunks of 4KB
            if not chunk:
                if buffer: #yield the last bit if any
                    yield buffer
                break

            buffer += chunk

            if len(buffer) < min_length:
                continue  # Not enough data yet

            last_space = buffer.rfind(' ', 0, min_length + 1) #+1 to handle edge case of space being at min_length

            if last_space == -1: #no spaces found
                yield buffer[:min_length]
                buffer = buffer[min_length:]
            else:
                yield buffer[:last_space]
                buffer = buffer[last_space + 1:] # +1 to skip the space


    def lazy_read_strings_no_min(self, file_handle):
        """
        Lazily reads strings from a file, ensuring words are not cut in half,
        resuming exactly from the previous chunk's end, without enforcing a minimum length.

        Args:
            filepath (str): The path to the file.

        Yields:
            str: String chunks from the file.
        """
        buffer = ""
        while True:
            chunk = file_handle.read(4096)  # Read in chunks of 4KB
            if not chunk:
                if buffer:
                    yield buffer
                break

            buffer += chunk

            last_space = buffer.rfind(' ')

            if last_space == -1:
                yield buffer
                buffer = ""
            else:
                yield buffer[:last_space]
                buffer = buffer[last_space + 1:]  # +1 to skip the space


    def lazy_read_file(self, file_handle, chunk_size=4096):
        """
        Lazily read strings from a file in chunks, ensuring words are not cut in half
        at the end of each chunk, while making sure the next chunk starts exactly
        where the previous one ended.
        
        Args:
            file_path (str): Path to the file to read
            chunk_size (int, optional): Base size of chunks to read. Defaults to 4096.
        
        Yields:
            str: Chunks of text with complete words
        """
        # Position in the file
        position = 0
        
        while True:
            # Store the current position
            start_position = position
            
            # Read a chunk
            file_handle.seek(position)
            chunk = file_handle.read(chunk_size)
            
            # If no more content, break
            if not chunk:
                break
            
            # If we're not at the end of the file and the chunk doesn't end with whitespace
            if len(chunk) == chunk_size and not chunk[-1].isspace():
                # Look for the last space or newline in the chunk
                last_space = max(
                    chunk.rfind(' '), 
                    chunk.rfind('\n'), 
                    chunk.rfind('\t')
                )
                
                if last_space != -1:
                    # Adjust the chunk to end at the last space
                    position = start_position + last_space + 1
                    chunk = chunk[:last_space + 1]
                else:
                    # No spaces found in the entire chunk, which is unusual
                    # but we'll handle it by returning the whole chunk
                    position = start_position + len(chunk)
            else:
                # We're either at the end of file or the chunk ends with whitespace
                position = start_position + len(chunk)
            
            yield chunk


    def merge(self, ids:tuple[int], pair:tuple[int,int], idx:int):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        newids = []
        i = 0
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    

    def merge_bytes(self, bytes_list: list[bytes], pair= Tuple[bytes, bytes]) -> list[bytes]:
        """
        Input:
            - list[bytes]
            - Tuple[bytes,bytes]
        Output:
            - list[bytes]
        """
        new_bytes=[]
        i=0
        while i<len(bytes_list):
            if i <len(bytes_list)-1 and bytes_list[i] == pair[0] and  bytes_list[i+1]==pair[1]:
                new_bytes.append(pair[0]+pair[1])
                i+=2
            else:
                new_bytes.append(bytes_list[i])
                i+=1

        return new_bytes
    

    def find_earliest_delimiter(self, buffer: str):
        """Find the earliest occurrence of any token in the buffer."""
        earliest = None
        for token in self.special_tokens:
            pos = buffer.find(token)
            if pos != -1 and (earliest is None or pos < earliest[0]):
                earliest = (pos, token)
        return earliest
        

    def split_iterable_by_tokens(self, iterable:str):
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            earliest = self.find_earliest_delimiter(buffer)

            while earliest:
                pos, token = earliest
                # Yield up to and including the earliest delimiter
                yield buffer[: pos + len(token)]
                # Remove the yielded part from the buffer
                buffer = buffer[pos + len(token):]
                # Look for the next delimiter
                earliest = self.find_earliest_delimiter(buffer)

        # Yield any remaining text in the buffer
        if buffer:
            yield buffer


    def split_string_by_special_tokens(self, s:str, delimiters):
        if not delimiters:
            # If we aren't given any delimiters, then return the string as one chunk.
            return [s]

        # Split the string using the compiled pattern, while keeping delimiters intact
        return [segment for segment in re.split(self.special_tokens_pattern, s) if segment]


