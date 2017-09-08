import os


def load_spelling_data(min_word_len, max_word_len):
    cmu_data = load_data("cmudict-0.7b", "ISO-8859-1").split("\n")

    words = list()
    pronunciations = list()
    ignore = True
    for l in cmu_data:
        if (len(l) > 0 and l[0] == 'A'):
            ignore = False
        if not ignore and len(l) > 0:
            word, pronounce = l.split("  ")
            pseq = pronounce.split()
            if len(pseq) >= min_word_len and len(pseq) <= max_word_len:
                words.append(word.lower())
                pronunciations.append(pseq)

    return pronunciations, words
      

def extract_symbol_vocab(data):
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']
    set_words = set([character for line in data for character in line])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

def load_sort_letter_data():
    source_path = 'data/letters_source.txt'
    target_path = 'data/letters_target.txt'

    source_sentences = load_data(source_path, 'utf-8').split("\n")
    target_sentences = load_data(target_path, 'utf-8').split("\n")

    return source_sentences, target_sentences

def load_data(path, encode):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding=encode, errors='ignore') as f:
        data = f.read()

    return data


def extract_vocab(data):
    special_words = ['<pad>', '<unk>', '<s>',  '<\s>']

    set_words = set([word for line in data.split('\n') for word in line.split()])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


def pad_id_sequences(source_ids, source_vocab_to_int, target_ids, target_vocab_to_int, sequence_length):
    new_source_ids = [list(reversed(sentence + [source_vocab_to_int['<pad>']] * (sequence_length - len(sentence)))) \
                      for sentence in source_ids]
    new_target_ids = [sentence + [target_vocab_to_int['<pad>']] * (sequence_length - len(sentence)) \
                      for sentence in target_ids]

    return new_source_ids, new_target_ids


def batch_data(source, target, batch_size):
    """
    Batch source and target together
    """
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield source_batch, target_batch
