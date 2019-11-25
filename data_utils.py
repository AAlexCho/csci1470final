import os
import re
import sys
import time
from glob import glob

from gensim import corpora
from tensorflow.python.platform import gfile
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

# download necessary packages
nltk.download('stopwords')

# Regular expressions used to tokenize.
# split on punctuations
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
# matching digits
_DIGIT_RE = re.compile(r"(^| )\d+")

_ENTITY = "@entity"
_BAR = "_BAR"
_UNK = "_UNK"
BAR_ID = 0
UNK_ID = 1
_START_VOCAB = [_BAR, _UNK]

tokenizer = RegexpTokenizer(r'@?\w+')
english_stopwords_set = set(stopwords.words("english"))


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = tokenizer.tokenize(sentence)
    for word in words:
        if word not in english_stopwords_set:
            yield word


def create_vocabulary(vocabulary_path, context, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
        vocabulary_path: path where the vocabulary will be created.
        context: space delimited corpora
        max_vocabulary_size: limit on the size of the created vocabulary.
        tokenizer: a function to use to tokenize each data sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        t0 = time.time()
        print("Creating vocabulary at %s" % (vocabulary_path))
        texts = [word for word in context.lower().split()
                 if word not in english_stopwords_set]
        dictionary = corpora.Dictionary([texts], prune_at=max_vocabulary_size)
        print("Tokenize : %.4fs" % (t0 - time.time()))
        dictionary.save(vocabulary_path)


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].
    Args:
      vocabulary_path: path to the file containing the vocabulary.
    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).
    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        vocab = corpora.Dictionary.load(vocabulary_path)
        return vocab.token2id, vocab.token2id.keys()
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.
    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    Args:
        sentence: a string, the sentence to convert to token-ids.
        vocabulary: a dictionary mapping tokens to integers.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    Returns:
        a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)

    for word in words:
        if normalize_digits:
            # Normalize digits by 0 before looking words up in the vocabulary.
            word = re.sub(_DIGIT_RE, ' ', word)
        yield vocabulary.get(word, UNK_ID)


def data_to_token_ids(data_path, target_path, vocab,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.
    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.
    Args:
        data_path: path to the data file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    with open(data_path, mode="r") as data_file:
        counter = 0
        results = []
        for line in data_file:
            if counter == 0:
                results.append(line)
            elif counter == 4:
                entity, ans = line.split(":", 1)
                try:
                    results.append(f"{vocab[entity[:]]}:{ans}")
                except:
                    continue
            else:
                token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                  normalize_digits)
                results.append(" ".join(str(tok) for tok in token_ids) + "\n")
            if line == "\n":
                counter += 1

        try:
            len_d, len_q = len(results[2].split()), len(results[4].split())
        except:
            return

        with open(f"{target_path}_{len_d+len_q}", mode="w") as tokens_file:
            tokens_file.writelines(results)


def get_toplevel_files(dir_name, suffix):
    for _, _, filenames in os.walk(dir_name):
        for filename in filenames:
            if filename.endswith(suffix):
                yield os.path.join(dir_name, filename)


def get_all_questions(dir_name):
    """
    Get all question file paths
    """
    yield from get_toplevel_files(dir_name, ".question")


def get_all_context(dir_name, context_fname):
    """
    Combine all questions into a context

    A question is divided into 5 sections
        where the second secion contains the qeustion
        the last section contains identity to actual name mapping
    """
    with open(context_fname, 'w') as context:
        for fname in tqdm(get_all_questions(dir_name)):
            with open(fname) as f:
                # skip first section
                f.readline()
                f.readline()
                context.write(f'{f.readline().rstrip()} ')
                # read entity mapping
                for line in f:
                    if line[0] == '@' and ':' in line:
                        context.write(f"{line.replace(':', ' ').rstrip()} ")


def questions_to_token_ids(data_path, vocab_fname, vocab_size):
    vocab, _ = initialize_vocabulary(vocab_fname)
    for fname in tqdm(get_all_questions(data_path)):
        data_to_token_ids(fname, fname + f".ids{vocab_size}", vocab)


def get_vocab_fname(data_dir, dataset_name):
    return os.path.join(data_dir, dataset_name,
                        f'{dataset_name}.vocab{vocab_size}')


def prepare_data(data_dir, dataset_name, vocab_size):
    train_path = os.path.join(data_dir, dataset_name, 'questions', 'training')

    # where to find the context
    context_fname = os.path.join(data_dir, dataset_name,
                                 f'{dataset_name}.context')
    # where to find the vocabulary
    vocab_fname = get_vocab_fname(data_dir, dataset_name)

    if not os.path.exists(context_fname):
        print(f" [*] Combining all contexts for {dataset_name} in {train_path} ...")
        context = get_all_context(train_path, context_fname)
    else:
        context = gfile.GFile(context_fname, mode="r").read()
        print(" [*] Skip combining all contexts")

    if not os.path.exists(vocab_fname):
        print(f" [*] Create vocab from {context_fname} to {vocab_fname} ...")
        create_vocabulary(vocab_fname, context, vocab_size)
    else:
        print(" [*] Skip creating vocab")
    print(f" [*] Convert data in {train_path} into vocab indicies...")
    questions_to_token_ids(train_path, vocab_fname, vocab_size)


def load_vocab(data_dir, dataset_name, vocab_size):
    vocab_fname = get_vocab_fname(data_dir, dataset_name)
    print(" [*] Loading vocab from %s ..." % vocab_fname)
    return initialize_vocabulary(vocab_fname)


def load_dataset(data_dir, dataset_name, vocab_size):
    train_files = glob(os.path.join(
        data_dir, dataset_name, "questions", "training",
        f"*.question.ids{vocab_size}_*"))
    max_idx = len(train_files)
    for idx, fname in enumerate(train_files):
        with open(fname) as f:
            yield f.read().split("\n\n"), idx, max_idx


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(" [*] usage: python3 data_utils.py DATA_DIR DATASET_NAME VOCAB_SIZE")
    else:
        data_dir = sys.argv[1]
        dataset_name = sys.argv[2]
        if len(sys.argv) > 3:
            vocab_size = int(sys.argv[3])
        else:
            vocab_size = 100000
        prepare_data(data_dir, dataset_name, vocab_size)