from collections import Counter, defaultdict
import random

# A tiny training dataset. The model learns which words usually follow
# one another by reading these sentences.
corpus = """
i like python
i like ML
i like AI
python is easy
python is powerful
ML is interesting
AI is powerful
"""

tokens = corpus.lower().split()

print("Tokens:")
print(tokens)

vocabulary = sorted(set(tokens))

print("\nVocabulary:")
print(vocabulary)
print("\nVocabulary Size:", len(vocabulary))

# Count every pair of neighboring words. For example, if "i like" appears
# three times, the count for "like" after "i" will be three.
next_word_counts = defaultdict(Counter)

for current_word, next_word in zip(tokens, tokens[1:]):
    next_word_counts[current_word][next_word] += 1

print("\nWHAT THE MODEL LEARNED:")

for current_word, possible_words in next_word_counts.items():
    print(current_word, "->", dict(possible_words))

# Turn the counts after one example word into probabilities.

word = "like"
possible_words = next_word_counts[word]
total = sum(possible_words.values())

print(f"\nPOSSIBLE WORDS AFTER '{word}':")

for candidate, count in possible_words.items():
    probability = count / total

    print(
        candidate,
        "count:",
        count,
        "probability:",
        round(probability, 2)
    )

# Use a fixed seed so the example produces the same result on every run.
random.seed(10)

generated_words = ["i"]

for _ in range(10):
    current_word = generated_words[-1]

    if current_word not in next_word_counts:
        break

    possible_words = list(next_word_counts[current_word].keys())
    frequencies = list(next_word_counts[current_word].values())

    selected_word = random.choices(possible_words, weights=frequencies)[0]

    generated_words.append(selected_word)

generated_sentence = " ".join(generated_words)

print("\nGENERATED TEXT:")
print(generated_sentence)
