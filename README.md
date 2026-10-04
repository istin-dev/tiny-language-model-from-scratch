# My First Language Model

This is a small next-word prediction project I built to understand the basic idea behind language models. Instead of using a machine-learning library, I created the logic from scratch with Python so I could see each step clearly.

The program reads a short collection of sentences, breaks the text into words, and records which word follows each word. It then converts those observations into probabilities and uses them to generate a new sentence.

This is a **bigram language model**: it predicts the next word using only the current word. Modern large language models are far more advanced and consider much more context, but this project demonstrates the same core idea of learning patterns from text and using probabilities to make predictions.

## How it works

1. A small text corpus is converted to lowercase and split into tokens.
2. A vocabulary is created from the unique tokens.
3. Neighboring word pairs are counted.
4. Counts are converted into next-word probabilities.
5. Words are sampled according to those probabilities to generate text.

For example, the training text contains `i like` three times. After `like`, the words `python`, `ml`, and `ai` each appear once, so each one has a probability of roughly 0.33.

## Run the project

You only need Python 3; there are no external dependencies.

```bash
python first_language_model.py
```

The program prints the tokens, vocabulary, learned word transitions, example probabilities, and generated text.

## What I learned

Through this project, I learned:

- how raw text is cleaned and split into tokens;
- how to build a vocabulary from unique words;
- how bigrams represent relationships between neighboring words;
- how frequency counts can be converted into probabilities;
- how weighted random sampling generates different text from learned patterns;
- why a random seed makes experiments reproducible;
- why a small bigram model has limited memory and cannot understand meaning like a modern LLM.

## Limitations and next steps

Because the dataset is tiny and the model looks back only one word, its output can be repetitive or grammatically awkward. Good next steps would be to:

- train it on a larger text file;
- support trigrams so the model uses two previous words;
- add special start and end tokens for cleaner sentences;
- handle unknown words and punctuation;
- compare different random seeds and generated outputs.

## Built with

- Python 3
- `collections.Counter` and `defaultdict`
- `random.choices` for weighted sampling
