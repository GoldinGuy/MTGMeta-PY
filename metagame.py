import dask_ml.cluster
from matplotlib.pyplot import pie, axis, show
import math
import seaborn as sns
import matplotlib.pyplot as plt
import json

decks = []
NUM_CLUSTERS = 20
FORMAT = 'modern'

with open('decks_json/decks-' + FORMAT + '.json') as f:
    decks_json = json.load(f)

for deck in decks_json:
    try:
        if deck['main'][0]['name'] is not None:
            cards_in_deck = []
            # deck_name = str(deck['name']).replace("/", "_")
            for card in deck['main']:
                card_name = str(card['name']).replace("'", "\\'")
                quantity = int(card['quantity'])
                cards_in_deck.append([quantity, card_name])
            decks.append(cards_in_deck)
    except Exception as e:
        print(e)
        decks_json.remove(deck)


def card_names(_deck):
    return [card[1] for card in _deck]


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

    cluster_name = 'Unknown'
    x = 0
    for deck in decks_json:
        y = 0
        try:
            for card in deck['main']:
                if str(card['name']).replace("'", "\\'") in label_set:
                    y += 1
            if y > x:
                x = y
                cluster_name = str(deck['name'])
        except Exception as e:
            print(e)

    print("Cluster # {} (" + cluster_name + ") :".format(LABEL))
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

# cards to analyze
cards_to_analyze = ['Thoughtseize', 'Llanowar Elves', 'Scalding Tarn', 'Serum Visions']

print("\nCards commonly found with " + cards_to_analyze[0] + "\n" + str(closest_cards(cards_to_analyze[0], 10)))
print("\nCards commonly found with " + cards_to_analyze[1] + "\n" + str(closest_cards(cards_to_analyze[1], 10)))
print("\nApparition ratio for " + cards_to_analyze[2] + "\n" + str(apparition_ratio(cards_to_analyze[2])))
print("\nApparition ratio for " + cards_to_analyze[3] + "\n" + str(apparition_ratio(cards_to_analyze[3])))

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
plt.savefig('graphs/# Decks by Cluster')
show()

# cards to analyze % by cluster (pie chart)
card_names = ["Lightning Bolt", "Mutavault", "Path to Exile"]

plt.rcParams['figure.facecolor'] = "slateblue"
plt.rcParams['text.color'] = "w"
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
    pie(df, labels=label_list, autopct=lambda i: "{0:.2f}".format(float(i)), normalize=False)
    plt.title(card_name + " % by cluster")
    plt.savefig('graphs/' + card_name + '_distribution')
    show()
