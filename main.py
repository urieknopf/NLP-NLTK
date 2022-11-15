# ESU NLP - NLTK - Demonstration of an understanding of using Python NLTK
# In this programming assignment I use NLTK to explore inaugural address corpus.


import nltk.corpus
nltk.download('inaugural')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
raw = nltk.corpus.inaugural.raw()

# Convert raw text into tokens, and convert tokens into text with `nltk.Text`
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)
print("\nData: ")
print(tokens[:10])
print(text)


def question_one(all_tokens):
    unique_tokens = len(set(all_tokens))
    lexical_diversity = len(set(all_tokens)) / float(len(all_tokens))

    print("Total Tokens: " + str(len(tokens)))
    print("Unique tokens: " + str(unique_tokens))
    print("The lexical diversity of the given text input is: " + str(lexical_diversity) + "")
    return


print("\n\nQuestion One:")
question_one(tokens)


def question_two(all_tokens):
    frequency_dictionary = nltk.FreqDist(all_tokens)
    first_20 = frequency_dictionary.most_common(20)

    print("The 20 most frequently occurring (unique) tokens in the text are: ")
    for key, value in first_20:
        print("{0:<5}".format(key), "{0:>10}".format('Frequency:'), value)
    return


print("\n\nQuestion Two:")
question_two(tokens)


def question_three(all_tokens):
    frequency_dictionary = nltk.FreqDist(all_tokens)
    words = list(frequency_dictionary.keys())
    sorted_dictionary_list = [x for x in words if len(x) > 5 and frequency_dictionary[x] > 150]

    print('Tokens with a length greater than 5 and frequency more than 150: ')
    for word in range(len(sorted_dictionary_list)):
        print(sorted_dictionary_list[word])
    return


print("\n\nQuestion Three:")
question_three(tokens)


def question_four(corpus, all_tokens):
    sentence_tokens = nltk.sent_tokenize(corpus)
    avg = len(all_tokens)/len(sentence_tokens)
    print("The average number of tokens per sentence: " + str(avg))
    return


print("\n\nQuestion Four:")
question_four(raw, tokens)


def question_five(all_tokens):
    frequency_dictionary = nltk.FreqDist(all_tokens)
    word_frequency = [x for x in list(frequency_dictionary.keys()) if x.isalpha() and frequency_dictionary[x] > 2000]
    new_frequency_dictionary = {key: frequency_dictionary[key] for key in word_frequency}
    return sorted(new_frequency_dictionary.items(), key=lambda kv: kv[1], reverse=True)


print("\n\nQuestion Five: \n", question_five(tokens))


def question_six(all_tokens):
    corpus_tags = nltk.pos_tag(all_tokens)
    frequency_dictionary = nltk.FreqDist([tag for (word, tag) in corpus_tags])
    return frequency_dictionary.most_common(5)


print("\n\nQuestion Six: \n", question_six(tokens))



