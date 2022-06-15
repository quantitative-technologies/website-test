---
title: "My First Post"
date: 2022-06-09T20:29:58+07:00
draft: false
---

Code:

```python
def preprocess_data(flags, sequence_lengths=False):
    """Load data, shuffle it, process the vocabulary and save to DATA_FILENAME, if not done already.
       Returns processed data. NOTE: If the max_doc_len changes from a previous run,
       then DATA_FILENAME should be deleted so that it can be properly recreated."""

    preprocessed_path = path.join(flags.model_dir, DATA_FILENAME)
    if path.isfile(preprocessed_path):
        with open(preprocessed_path, 'rb') as f:
            train_raw, x_train, y_train, x_test, y_test, \
                train_lengths, test_lengths, classes = pickle.load(f)
    else:
        # Get the raw data, downloading it if necessary.
        train_raw, test_raw, classes = get_data(flags.data_dir)

        # Seeding is necessary for reproducibility.
        np.random.seed(flags.np_seed)
        # Shuffle data to make the distribution of classes roughly stratified for each mini-batch.
        # This is not necessary for full batch training, but is essential for mini-batch training.
        train_raw = shuffle(train_raw)
        test_raw = shuffle(test_raw)
        train_sentences, y_train, test_sentences, y_test = extract_data(train_raw, test_raw)
        # Encode the raw data as integer vectors.
        x_train, x_test, train_lengths, test_lengths, _, _ = process_vocabulary(
            train_sentences, test_sentences, flags,
            reuse=True, sequence_lengths=sequence_lengths)

        # Save the processed data to avoid re-processing.
        saved = False
        with open(preprocessed_path, 'wb') as f:
            try:
                pickle.dump([train_raw, x_train, y_train, x_test, y_test,
                             train_lengths, test_lengths, classes], f)
                saved = True
            except (OverflowError, MemoryError):
                # Can happen if max-doc-len is large.
                pass

        if not saved:
            remove(preprocessed_path)

    return train_raw, x_train, y_train, x_test, y_test, train_lengths, test_lengths, classes
```

Some text after the code block.