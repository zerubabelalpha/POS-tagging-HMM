import os
import math 
import numpy as np 


#dataset loader.
def load_dataset(data_path):

    if os.path.exists(data_path):
        with open(data_path, 'r') as fs:
            content = fs.read()
            return content

    else : raise FileNotFoundError("file_not_found")


#parse dataset
def parse_dataset(data_path):
    raw_data = load_dataset(data_path)
    dataset =[]
    current_sentense = []

    lines = raw_data.strip().split("\n")

    for line in lines:
        line = line.strip()

        if line =="":
            if current_sentense:
                dataset.append(current_sentense)
                current_sentense =[]
            continue

        if "\t" not in line:
            continue

        word, tag = line.split("\t")
        current_sentense.append((word, tag))

    if current_sentense:
        dataset.append(current_sentense)

    return dataset


#vocabulary ,these will collect unique words and tags
def build_vocabulary(dataset):
    vocabulary = set()
    tag_set = set()

    for sentense in dataset:
        for word, tag in sentense:
            vocabulary.add(word)
            tag_set.add(tag)

    tag_set.add("START")
    tag_set.add("STOP")

    return vocabulary, tag_set


# counts
def compute_counts(dataset):

    initial_counts = {}
    transition_counts = {}
    emission_counts = {}
    tag_counts = {}

    total_sentences = len(dataset)

    for sentence in dataset:

        #initial counts
        first_word, first_tag = sentence[0]

        initial_counts[first_tag] = initial_counts.get(first_tag, 0) + 1

        if "START" not in transition_counts:
            transition_counts["START"] = {}
        transition_counts["START"][first_tag] = (
            transition_counts["START"].get(first_tag, 0) + 1
        )

        #loop through sentense
        for i in range(len(sentence)):

            word, tag = sentence[i]

            #tag counts
            tag_counts[tag] = ( tag_counts.get(tag, 0) + 1)

            #emission counts
            if tag not in emission_counts:
                emission_counts[tag] = {}

            emission_counts[tag][word] = (emission_counts[tag].get(word, 0) + 1)

            #transition counts
            if i > 0:

                prev_word, prev_tag = sentence[i - 1]

                if prev_tag not in transition_counts:
                    transition_counts[prev_tag] = {}

                transition_counts[prev_tag][tag] = (
                    transition_counts[prev_tag].get(tag, 0) + 1
                )

            if i == len(sentence) - 1:
                if tag not in transition_counts:
                    transition_counts[tag] = {}
                transition_counts[tag]["STOP"] = (
                    transition_counts[tag].get("STOP", 0) + 1
                )

    tag_counts["START"] = total_sentences
    tag_counts["STOP"] = total_sentences

    return {
        "initial_counts": initial_counts,
        "transition_counts": transition_counts,
        "emission_counts": emission_counts,
        "tag_counts": tag_counts,
        "total_sentences": total_sentences
    }


#probablities
def compute_probabilities(counts, vocabulary, tag_set):

    initial_probs = {}
    transition_probs = {}
    emission_probs = {}

    V = len(vocabulary)
    regular_tags = tag_set - {"START", "STOP"}
    T = len(regular_tags)

    initial_counts = counts["initial_counts"]
    transition_counts = counts["transition_counts"]
    emission_counts = counts["emission_counts"]
    tag_counts = counts["tag_counts"]
    total_sentences = counts["total_sentences"]

    
    # initial probab
    for tag in regular_tags:
        count = transition_counts.get("START", {}).get(tag, 0)
        initial_probs[tag] = (count + 1) / (total_sentences + T)

    
    # transition probab
    for prev_tag in tag_set:
        if prev_tag == "STOP":
            continue  
            
        transition_probs[prev_tag] = {}
        
        
        if prev_tag == "START":
            total = total_sentences
        else:
            total = tag_counts.get(prev_tag, 0)
        
        # Determine possible next tags
        if prev_tag == "START":
            next_tags = regular_tags
        else:
            next_tags = tag_set - {"START"}  
        
        for next_tag in next_tags:
            count = 0
            if prev_tag in transition_counts:
                count = transition_counts[prev_tag].get(next_tag, 0)
            
            transition_probs[prev_tag][next_tag] = (
                count + 1
            ) / (
                total + len(next_tags)
            )

    
    # emmission probab
    for tag in regular_tags:
        emission_probs[tag] = {}
        total = tag_counts.get(tag, 0)
        
        for word in vocabulary:
            count = 0
            if tag in emission_counts:
                count = emission_counts[tag].get(word, 0)
            
            emission_probs[tag][word] = (
                count + 1
            ) / (
                total + V
            )

    return initial_probs, transition_probs, emission_probs


#unknowen word probab
def unknown_word_probability(tag, tag_counts, vocabulary_size):

    return 1 / (tag_counts.get(tag, 0) + vocabulary_size)


#viteribi decoder
def viterbi(
    sentence,
    tag_set,
    vocabulary,
    initial_probs,
    transition_probs,
    emission_probs,
    tag_counts
):
    regular_tags = sorted(list(tag_set - {"START", "STOP"}))
    n_tags = len(regular_tags)
    T = len(sentence)

    # DP TABLES
    viterbi_table = np.zeros((n_tags, T))
    backpointer = np.zeros((n_tags, T), dtype=int)

    # INITIALIZATION
    first_word = sentence[0]

    for s, tag in enumerate(regular_tags):
        if first_word in vocabulary:
            emission = emission_probs[tag].get(first_word, 1e-10)
        else:
            emission = unknown_word_probability(
                tag,
                tag_counts,
                len(vocabulary)
            )

        # Use transition from START
        start_transition = transition_probs.get("START", {}).get(tag, 1e-10)
        
        viterbi_table[s, 0] = (
            math.log(start_transition)
            + math.log(emission)
        )

    # RECURSION
    for t in range(1, T):
        current_word = sentence[t]

        for s, current_tag in enumerate(regular_tags):
            if current_word in vocabulary:
                emission = emission_probs[current_tag].get(current_word, 1e-10)
            else:
                emission = unknown_word_probability(
                    current_tag,
                    tag_counts,
                    len(vocabulary)
                )

            best_score = -float("inf")
            best_state = 0

            for ps, previous_tag in enumerate(regular_tags):
                transition = transition_probs[previous_tag][current_tag]

                score = (
                    viterbi_table[ps, t - 1]
                    + math.log(transition)
                    + math.log(emission)
                )

                if score > best_score:
                    best_score = score
                    best_state = ps

            viterbi_table[s, t] = best_score
            backpointer[s, t] = best_state

    # TERMINATION 
    best_score = -float("inf")
    best_last_state = 0
    
    for s, tag in enumerate(regular_tags):
        stop_prob = transition_probs.get(tag, {}).get("STOP", 1e-10)
        final_score = viterbi_table[s, T - 1] + math.log(stop_prob)
        
        if final_score > best_score:
            best_score = final_score
            best_last_state = s

    # TRACEBACK
    best_path = [best_last_state]

    for t in range(T - 1, 0, -1):
        best_last_state = backpointer[best_last_state, t]
        best_path.insert(0, best_last_state)

    predicted_tags = [regular_tags[state] for state in best_path]

    return predicted_tags, viterbi_table


# Split dataset into train and test
def train_test_split(dataset, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    indices = np.random.permutation(len(dataset))
    split_idx = int(len(dataset) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]
    
    return train_data, test_data


def evaluate(test_dataset,
             tag_set,
             vocabulary,
             initial_probs,
             transition_probs,
             emission_probs,
             tag_counts):
    total = 0
    correct = 0
    sentence_accuracy = []
    
    # Tags for confusion matrix
    regular_tags = sorted(list(tag_set - {"START", "STOP"}))
    confusion = {tag: {tag2: 0 for tag2 in regular_tags} for tag in regular_tags}

    for sentence in test_dataset:
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]

        predicted_tags, _ = viterbi(
            words,
            tag_set,
            vocabulary,
            initial_probs,
            transition_probs,
            emission_probs,
            tag_counts
        )

        sent_correct = 0
        for true, pred in zip(true_tags, predicted_tags):
            total += 1
            confusion[true][pred] += 1
            
            if true == pred:
                correct += 1
                sent_correct += 1
        
        sentence_accuracy.append(sent_correct / len(sentence))

    accuracy = correct / total if total > 0 else 0
    avg_sent_accuracy = np.mean(sentence_accuracy) if sentence_accuracy else 0

    return accuracy, avg_sent_accuracy, confusion


# confusion matrix
def print_confusion_matrix(confusion, tags):
    print("\nConfusion Matrix:")
    
    
    # Header
    header = "True\\Pred".ljust(12)
    for tag in tags:
        header += tag.ljust(10)
    print(header)
    print("-" * 60)
    
    # Rows
    for true_tag in tags:
        row = true_tag.ljust(12)
        for pred_tag in tags:
            row += str(confusion[true_tag][pred_tag]).ljust(10)
        print(row)


# main
if __name__ == "__main__":

    data_path = "./brown-universal.txt"
   
    dataset = parse_dataset(data_path)

    train_data, test_data = train_test_split(dataset, test_size=0.2)
   
    vocabulary, tag_set = build_vocabulary(train_data)

    counts = compute_counts(train_data)

    initial_probs, transition_probs, emission_probs = (
        compute_probabilities(
            counts,
            vocabulary,
            tag_set
        )
    )

    # TEST SENTENCE
    test_sentence = ["Mr.", "Podger", "had", "thanked", "him"]

    predicted_tags, table = viterbi(
        test_sentence,
        tag_set,
        vocabulary,
        initial_probs,
        transition_probs,
        emission_probs,
        counts["tag_counts"]
    )

    print("\nSentence:\n")
    print(test_sentence)

    print("\nPredicted Tags:\n")
    print(predicted_tags)

    # EVALUATION
    accuracy, avg_sent_accuracy, confusion = evaluate(
        test_data,
        tag_set,
        vocabulary,
        initial_probs,
        transition_probs,
        emission_probs,
        counts["tag_counts"]
    )
    
    print(f"\nOverall Accuracy: {round(accuracy * 100, 2)}%")
    print(f"Average Sentence Accuracy: {round(avg_sent_accuracy * 100, 2)}%")
    
    # Print confusion matrix for most common tags
    regular_tags = sorted(list(tag_set - {"START", "STOP"}))
    print_confusion_matrix(confusion, regular_tags)