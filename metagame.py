import dask_ml.cluster
import glob
from matplotlib.pyplot import pie, axis, show
import math
import seaborn as sns
import matplotlib.pyplot as plt

deck_names = glob.glob("modern_decks/*")
decks = []

NUM_CLUSTERS = 8

for filename in deck_names:
    with open(filename, 'r') as deck:
        decks.append(deck.read())

for i, deck in enumerate(decks):
    decks[i] = deck.split('\n')[1:]

# print(decks)

for deck in decks:
    for i, card in enumerate(deck):
        if card is not '':
            quantity = int(card.split(' ')[0])
            card_name = ' '.join(card.split(' ')[1:])
            deck[i] = (quantity, card_name)
        else:
            deck.remove(card)
print(decks)

def card_names(a_deck):
    return [card[1] for card in a_deck]


all_card_names = []
for deck_card_names in [card_names(deck) for deck in decks]:
    all_card_names += deck_card_names

print('Total number of card names: ' + str(len(all_card_names)))
all_card_names = set(all_card_names)
print('Number of unique card names: ' + str(len(all_card_names)))
all_card_names = list(all_card_names)
print("\n")


def deck_to_vector(deck):
    v = [0] * len(all_card_names)
    for i, name in enumerate(all_card_names):
        for number, card_name in deck:
            if card_name == name:
                v[i] += number
    return v


deck_vectors = [deck_to_vector(deck) for deck in decks]

km = dask_ml.cluster.KMeans(n_clusters=NUM_CLUSTERS, oversampling_factor=5)
km.fit(deck_vectors)

labels = list(km.labels_.compute())

decks_labels = list(zip(decks, labels))


def most_common_cards(deck, k):
    deck.sort(key=lambda deck: deck[0], reverse=True)
    return [card[1] for card in deck[:k]]


def decks_by_label(a_label):
    return [(deck, label) for (deck, label) in decks_labels if label == a_label]


# Determine and print clusters

for LABEL in range(NUM_CLUSTERS):
    label_set = set(most_common_cards(decks_by_label(LABEL)[0][0], 40))
    for deck, label in decks_by_label(LABEL):
        label_set.intersection(set(most_common_cards(deck, 40)))
    label_set = set(label_set)
    print("Cluster number {}:".format(LABEL))
    print(label_set)
    print("\n")


def apparition_ratio(a_card):
    label_count = [0] * NUM_CLUSTERS
    for deck, label in decks_labels:
        if a_card in [card_name for _, card_name in deck]:
            label_count[label] += 1
    total_apps = sum(label_count)
    return [count / total_apps for count in label_count], total_apps


def distance(x, y):
    dist = 0.0
    for z, elem in enumerate(x):
        dist += (elem - y[z]) * (elem - y[z])
    return math.sqrt(dist)


def closest_cards(a_card, b):
    this_card = apparition_ratio(a_card)[0]
    distances = []
    for name in all_card_names:
        dist = distance(apparition_ratio(name)[0], this_card)
        distances.append((name, dist))
    distances.sort(key=lambda x: x[1])
    distances = [(name, dist) for name, dist in distances if name != a_card]
    return [name for name, _ in distances[:b]]


def versatile_cards(k):
    variances = []
    for name in all_card_names:
        versatility = sum([1 if x > 0 else 0 for x in apparition_ratio(name)[0]])
        variances.append((name, versatility))
    variances.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in variances[:k]]


# analyze and print data
print("Most Common Cards in different versions of cluster ")
for deck, label in decks_by_label(2)[:10]:
    print(str(most_common_cards(deck, 7)) + " " + str(label))

print("\nMost versatile cards:\n")
for card in versatile_cards(30):
    if card not in ['Island', 'Forest', 'Mountain', 'Swamp', 'Plains']:
        print(card)

print("\nCards commonly found with Thoughtseize\n" + str(closest_cards("Thoughtseize", 10)))

print("\nCards commonly found with Llanowar Elves\n" + str(closest_cards("Llanowar Elves", 10)))

print("\nApparition ratio for Scalding Tarn\n" + str(apparition_ratio("Scalding Tarn")))

print("\nApparition ratio for Serum Visions\n" + str(apparition_ratio("Serum Visions")))

# graph data
plt.rc('font', size=14)
label_counts = [(label, len(decks_by_label(label))) for label in range(NUM_CLUSTERS)]
counts = [count for _, count in label_counts]
points = {
    'cluster': [label for label, _ in label_counts],
    'count': [count for _, count in label_counts],
}
sns.barplot(x="cluster", y="count", data=points).set_title("# Decks by Cluster")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('# Decks by Cluster')
show()

plt.rcParams['figure.facecolor'] = "slateblue"
plt.rcParams['text.color'] = "w"
card_names = ["Lightning Bolt", "Mutavault", "Path to Exile"]
for card_name in card_names:
    df = apparition_ratio(card_name)[0]
    label_list = list(range(NUM_CLUSTERS))
    i = 0
    while i < len(label_list):
        if '0.00' in str("{0:.2f}".format(float(df[i]))):
            df.pop(i)
            label_list.pop(i)
        else:
            i += 1
    pie(df, labels=label_list, autopct=lambda i: "{0:.2f}".format(float(i)))
    plt.title(card_name + " % by cluster")
    plt.savefig(card_name + '_distribution')
    show()