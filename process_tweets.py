# -*- coding: utf-8 -*-
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.manifold import TSNE
from collections import Counter
from six.moves import cPickle
import gensim.models.word2vec as w2v
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing
import os
import sys
import io
import re
import json

def try_load_or_process(filename, processor_fn, function_arg):
    load_fn = None
    save_fn = None
    if filename.endswith("json"):
        load_fn = load_json
        save_fn = save_json
    else:
        load_fn = load_bin
        save_fn = save_bin
    if os.path.exists(filename):
        return load_fn(filename)
    else:
        ret = processor_fn(function_arg)
        save_fn(ret, filename)
        return ret

def print_progress(current, maximum):
    sys.stdout.write("\r")
    sys.stdout.flush()
    sys.stdout.write(str(current) + "/" + str(maximum))
    sys.stdout.flush()

def save_bin(item, filename):
    with open(filename, "wb") as f:
        cPickle.dump(item, f)

def load_bin(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return cPickle.load(f)

def save_json(variable, filename):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(unicode(json.dumps(variable, indent=4, ensure_ascii=False)))

def load_json(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with io.open(filename, "r", encoding="utf-8") as f:
                ret = json.load(f)
        except:
            pass
    return ret

def process_raw_data(input_file):
    valid = u"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#@.:/ äöåÄÖÅ"
    url_match = "(https?:\/\/[0-9a-zA-Z\-\_]+\.[\-\_0-9a-zA-Z]+\.?[0-9a-zA-Z\-\_]*\/?.*)"
    name_match = "\@[\_0-9a-zA-Z]+\:?"
    lines = []
    print("Loading raw data from: " + input_file)
    if os.path.exists(input_file):
        with io.open(input_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
    num_lines = len(lines)
    ret = []
    for count, text in enumerate(lines):
        if count % 50 == 0:
            print_progress(count, num_lines)
        text = re.sub(url_match, u"", text)
        text = re.sub(name_match, u"", text)
        text = re.sub("\&amp\;?", u"", text)
        text = re.sub("[\:\.]{1,}$", u"", text)
        text = re.sub("^RT\:?", u"", text)
        text = u''.join(x for x in text if x in valid)
        text = text.strip()
        if len(text.split()) > 5:
                if text not in ret:
                    ret.append(text)
    return ret

def tokenize_sentences(sentences):
    ret = []
    max_s = len(sentences)
    print("Got " + str(max_s) + " sentences.")
    for count, s in enumerate(sentences):
        tokens = []
        words = re.split(r'(\s+)', s)
        if len(words) > 0:
            for w in words:
                if w is not None:
                    w = w.strip()
                    w = w.lower()
                    if w.isspace() or w == "\n" or w == "\r":
                        w = None
                    if len(w) < 1:
                        w = None
                    if w is not None:
                        tokens.append(w)
        if len(tokens) > 0:
            ret.append(tokens)
        if count % 50 == 0:
            print_progress(count, max_s)
    return ret

def clean_sentences(tokens):
    all_stopwords = load_json("stopwords-iso.json")
    extra_stopwords = ["ssä", "lle", "h.", "oo", "on", "muk", "kov", "km", "ia", "täm", "sy", "but", ":sta", "hi", "py", "xd", "rr", "x:", "smg", "kum", "uut", "kho", "k", "04n", "vtt", "htt", "väy", "kin", "#8", "van", "tii", "lt3", "g", "ko", "ett", "mys", "tnn", "hyv", "tm", "mit", "tss", "siit", "pit", "viel", "sit", "n", "saa", "tll", "eik", "nin", "nii", "t", "tmn", "lsn", "j", "miss", "pivn", "yhn", "mik", "tn", "tt", "sek", "lis", "mist", "tehd", "sai", "l", "thn", "mm", "k", "ku", "s", "hn", "nit", "s", "no", "m", "ky", "tst", "mut", "nm", "y", "lpi", "siin", "a", "in", "ehk", "h", "e", "piv", "oy", "p", "yh", "sill", "min", "o", "va", "el", "tyn", "na", "the", "tit", "to", "iti", "tehdn", "tlt", "ois", ":", "v", "?", "!", "&"]
    stopwords = None
    if all_stopwords is not None:
        stopwords = all_stopwords["fi"]
        stopwords += extra_stopwords
    ret = []
    max_s = len(tokens)
    for count, sentence in enumerate(tokens):
        if count % 50 == 0:
            print_progress(count, max_s)
        cleaned = []
        for token in sentence:
            if len(token) > 0:
                if stopwords is not None:
                    for s in stopwords:
                        if token == s:
                            token = None
                if token is not None:
                    if re.search("^[0-9\.\-\s\/]+$", token):
                        token = None
                if token is not None:
                    cleaned.append(token)
        if len(cleaned) > 0:
            ret.append(cleaned)
    return ret

def get_word_frequencies(corpus):
    frequencies = Counter()
    for sentence in corpus:
        for word in sentence:
            frequencies[word] += 1
    freq = frequencies.most_common()
    return freq

def get_word2vec(sentences):
    num_workers = multiprocessing.cpu_count()
    num_features = 200
    epoch_count = 10
    sentence_count = len(sentences)
    w2v_file = os.path.join(save_dir, "word_vectors.w2v")
    word2vec = None
    if os.path.exists(w2v_file):
        print("w2v model loaded from " + w2v_file)
        word2vec = w2v.Word2Vec.load(w2v_file)
    else:
        word2vec = w2v.Word2Vec(sg=1,
                                seed=1,
                                workers=num_workers,
                                size=num_features,
                                min_count=min_frequency_val,
                                window=5,
                                sample=0)

        print("Building vocab...")
        word2vec.build_vocab(sentences)
        print("Word2Vec vocabulary length:", len(word2vec.wv.vocab))
        print("Training...")
        word2vec.train(sentences, total_examples=sentence_count, epochs=epoch_count)
        print("Saving model...")
        word2vec.save(w2v_file)
    return word2vec

def create_embeddings(word2vec):
    all_word_vectors_matrix = word2vec.wv.syn0
    num_words = len(all_word_vectors_matrix)
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    dim = word2vec.wv[vocab[0]].shape[0]
    embedding = np.empty((num_words, dim), dtype=np.float32)
    metadata = ""
    for i, word in enumerate(vocab):
        embedding[i] = word2vec.wv[word]
        metadata += word + "\n"
    metadata_file = os.path.join(save_dir, "metadata.tsv")
    with io.open(metadata_file, "w", encoding="utf-8") as f:
        f.write(metadata)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    X = tf.Variable([0.0], name='embedding')
    place = tf.placeholder(tf.float32, shape=embedding.shape)
    set_x = tf.assign(X, place, validate_shape=False)
    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: embedding})

    summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding:0'
    embedding_conf.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(summary_writer, config)

    save_file = os.path.join(save_dir, "model.ckpt")
    print("Saving session...")
    saver = tf.train.Saver()
    saver.save(sess, save_file)

def most_similar(input_word, num_similar):
    sim = word2vec.wv.most_similar(input_word, topn=num_similar)
    output = []
    found = []
    for item in sim:
        w, n = item
        found.append(w)
    output = [input_word, found]
    return output

def test_word2vec(test_words):
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    output = []
    associations = {}
    test_items = test_words
    for count, word in enumerate(test_items):
        if word in vocab:
            print("[" + str(count+1) + "] Testing: " + word)
            if word not in associations:
                associations[word] = []
            similar = most_similar(word, num_similar)
            t_sne_scatterplot(word)
            output.append(similar)
            for s in similar[1]:
                if s not in associations[word]:
                    associations[word].append(s)
        else:
            print("Word " + word + " not in vocab")
    filename = os.path.join(save_dir, "word2vec_test.json")
    save_json(output, filename)
    filename = os.path.join(save_dir, "associations.json")
    save_json(associations, filename)
    filename = os.path.join(save_dir, "associations.csv")
    handle = io.open(filename, "w", encoding="utf-8")
    handle.write(u"Source,Target\n")
    for w, sim in associations.iteritems():
        for s in sim:
            handle.write(w + u"," + s + u"\n")
    return output

def t_sne_scatterplot(word):
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    dim0 = word2vec.wv[vocab[0]].shape[0]

    arr = np.empty((0, dim0), dtype='f')
    w_labels = [word]
    nearby = word2vec.wv.similar_by_word(word, topn=num_similar)
    arr = np.append(arr, np.array([word2vec[word]]), axis=0)
    for n in nearby:
        w_vec = word2vec[n[0]]
        w_labels.append(n[0])
        arr = np.append(arr, np.array([w_vec]), axis=0)

    tsne = TSNE(n_components=2, random_state=1)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    plt.rc("font", size=16)
    plt.figure(figsize=(16, 12), dpi=80)
    plt.scatter(x_coords[0], y_coords[0], s=800, marker="o", color="blue")
    plt.scatter(x_coords[1:], y_coords[1:], s=200, marker="o", color="red")

    for label, x, y in zip(w_labels, x_coords, y_coords):
        plt.annotate(label.upper(), xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()-50, x_coords.max()+50)
    plt.ylim(y_coords.min()-50, y_coords.max()+50)
    filename = os.path.join(plot_dir, word + "_tsne.png")
    plt.savefig(filename)
    plt.close()

def calculate_t_sne():
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    arr = np.empty((0, dim0), dtype='f')
    labels = []
    vectors_file = os.path.join(save_dir, "vocab_vectors.npy")
    labels_file = os.path.join(save_dir, "labels.json")
    if os.path.exists(vectors_file) and os.path.exists(labels_file):
        print("Loading pre-saved vectors from disk")
        arr = load_bin(vectors_file)
        labels = load_json(labels_file)
    else:
        print("Creating an array of vectors for each word in the vocab")
        for count, word in enumerate(vocab):
            if count % 50 == 0:
                print_progress(count, vocab_len)
            w_vec = word2vec[word]
            labels.append(word)
            arr = np.append(arr, np.array([w_vec]), axis=0)
        save_bin(arr, vectors_file)
        save_json(labels, labels_file)

    x_coords = None
    y_coords = None
    x_c_filename = os.path.join(save_dir, "x_coords.npy")
    y_c_filename = os.path.join(save_dir, "y_coords.npy")
    if os.path.exists(x_c_filename) and os.path.exists(y_c_filename):
        print("Reading pre-calculated coords from disk")
        x_coords = load_bin(x_c_filename)
        y_coords = load_bin(y_c_filename)
    else:
        print("Computing T-SNE for array of length: " + str(len(arr)))
        tsne = TSNE(n_components=2, random_state=1, verbose=1)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)
        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        print("Saving coords.")
        save_bin(x_coords, x_c_filename)
        save_bin(y_coords, y_c_filename)
    return x_coords, y_coords, labels, arr

def show_cluster_locations(results, labels, x_coords, y_coords):
    for item in results:
        name = item[0]
        print("Plotting graph for " + name)
        similar = item[1]
        in_set_x = []
        in_set_y = []
        out_set_x = []
        out_set_y = []
        name_x = 0
        name_y = 0
        for count, word in enumerate(labels):
            xc = x_coords[count]
            yc = y_coords[count]
            if word == name:
                name_x = xc
                name_y = yc
            elif word in similar:
                in_set_x.append(xc)
                in_set_y.append(yc)
            else:
                out_set_x.append(xc)
                out_set_y.append(yc)
        plt.figure(figsize=(16, 12), dpi=80)
        plt.scatter(name_x, name_y, s=400, marker="o", c="blue")
        plt.scatter(in_set_x, in_set_y, s=80, marker="o", c="red")
        plt.scatter(out_set_x, out_set_y, s=8, marker=".", c="black")
        if 'plot_lims' in globals():
            plt.xlim(plot_lims["xmin"], plot_lims["xmax"])
            plt.ylim(plot_lims["ymin"], plot_lims["ymax"])
        filename = os.path.join(big_plot_dir, name + "_tsne.png")
        plt.savefig(filename)
        plt.close()


if __name__ == '__main__':
    input_dir = "data"
    save_dir = "analysis"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Preprocessing raw data")
    raw_input_file = os.path.join(input_dir, "tweets.txt")
    filename = os.path.join(save_dir, "data.json")
    processed = try_load_or_process(filename, process_raw_data, raw_input_file)
    print("Unique sentences: " + str(len(processed)))

    print("Tokenizing sentences")
    filename = os.path.join(save_dir, "tokens.json")
    tokens = try_load_or_process(filename, tokenize_sentences, processed)

    print("Cleaning tokens")
    filename = os.path.join(save_dir, "cleaned.json")
    cleaned = try_load_or_process(filename, clean_sentences, tokens)

    print("Getting word frequencies")
    filename = os.path.join(save_dir, "frequencies.json")
    frequencies = try_load_or_process(filename, get_word_frequencies, cleaned)
    vocab_size = len(frequencies)
    print("Unique words: " + str(vocab_size))

    trimmed_vocab = []
    min_frequency_val = 6
    for item in frequencies:
        if item[1] >= min_frequency_val:
            trimmed_vocab.append(item[0])
    trimmed_vocab_size = len(trimmed_vocab)
    print("Trimmed vocab length: " + str(trimmed_vocab_size))
    filename = os.path.join(save_dir, "trimmed_vocab.json")
    save_json(trimmed_vocab, filename)

    print
    print("Instantiating word2vec model")
    word2vec = get_word2vec(cleaned)
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    print("word2vec vocab contains " + str(vocab_len) + " items.")
    dim0 = word2vec.wv[vocab[0]].shape[0]
    print("word2vec items have " + str(dim0) + " features.")

    print("Creating tensorboard embeddings")
    create_embeddings(word2vec)

    print("Calculating T-SNE for word2vec model")
    x_coords, y_coords, labels, arr = calculate_t_sne()

    plot_dir = os.path.join(save_dir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    num_similar = 40
    test_words = []
    for item in frequencies[:50]:
        test_words.append(item[0])
    results = test_word2vec(test_words)

    big_plot_dir = os.path.join(save_dir, "big_plots")
    if not os.path.exists(big_plot_dir):
        os.makedirs(big_plot_dir)
    show_cluster_locations(results, labels, x_coords, y_coords)

