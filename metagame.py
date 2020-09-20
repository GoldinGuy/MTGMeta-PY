import dask_ml.cluster
import math
import json
from matplotlib.pyplot import pie, axis, show
import seaborn as sns
import matplotlib.pyplot as plt


def card_names(_deck):
    return [card[1] for card in _deck]


def most_common_cards(deck, k):
    deck.sort(key=lambda deck: deck[0], reverse=True)
    return [card[1] for card in deck[:k]]


def decks_by_label(a_label):
    return [(deck, label) for (deck, label) in decks_labels if label == a_label]


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


# CUSTOMIZABLE GLOBAL VARS
NUM_CLUSTERS = 20
NUM_TOP_VERS = 15
# FORMATS = ['pauper', 'legacy', 'modern', 'standard']
FORMATS = ['standard']

IGNORE = ['Island', 'Forest', 'Mountain', 'Swamp', 'Plains']

for FORMAT in FORMATS:
    # DEFINE OUTPUT JSON
    decks = []
    unique_cards = []
    format_json = {
        'archetypes': [],
        'format_top_cards': [],
        'format_versatile_cards': [],
        'total_cards_parsed': 0,
        'unique_cards_parsed': 0
    }

    with open('decks_json/decks-' + FORMAT + '.json') as f:
        decks_json = json.load(f)

    # CONVERT JSON FILE TO LIST W/ PROPER FORMATTING
    json_index = 0
    for deck in decks_json:
        json_index += 1
        try:
            # if deck['main'][0]['name'] is not None:
            cards_in_deck = []
            for card in deck['main']:
                card_name = str(card.get('name')).replace("'", "\\'")
                quantity = int(card['quantity'])
                cards_in_deck.append([quantity, card_name])
                if any(card_name in d for d in unique_cards):
                    unique_cards[next((i for i,d in enumerate(unique_cards) if card_name in d), None)][card_name] += quantity
                elif card_name not in IGNORE:
                    unique_cards.append({card_name: quantity})
            decks.append(cards_in_deck)
        except Exception as e:
            decks_json.pop(json_index)


    def deck_to_vector(input_deck):
        v = [0] * len(all_card_names)
        for x, name in enumerate(all_card_names):
            for number, card_name in input_deck:
                if card_name == name:
                    v[x] += number
        return v

    # DETERMINE TOTAL PARSE DATA
    all_card_names = []
    for deck_card_names in [card_names(deck) for deck in decks]:
        all_card_names += deck_card_names
    format_json['total_cards_parsed'] = len(all_card_names)
    all_card_names_no_basics_len = len([i for i in all_card_names if i not in IGNORE])
    all_card_names = set(all_card_names)
    format_json['unique_cards_parsed'] = len(all_card_names)
    all_card_names = list(all_card_names)
    all_card_names_no_basics = [i for i in all_card_names if i not in IGNORE]

    # K-MEANS CLUSTERING
    deck_vectors = [deck_to_vector(deck) for deck in decks]

    km = dask_ml.cluster.KMeans(n_clusters=NUM_CLUSTERS, oversampling_factor=5)
    km.fit(deck_vectors)

    labels = list(km.labels_.compute())
    decks_labels = list(zip(decks, labels))

    card_counts = [(card_label, len(decks_by_label(card_label))) for card_label in range(NUM_CLUSTERS)]
    total_instances = sum([count for _, count in card_counts])

    # FOR EACH ARCHETPYE IN FORMAT
    for CLUSTER_ID in range(NUM_CLUSTERS):
        # DETERMINE MOST COMMON CARDS IN CLUSTER
        card_set = set(most_common_cards(decks_by_label(CLUSTER_ID)[0][0], 40))
        for deck, card in decks_by_label(CLUSTER_ID):
            card_set.intersection(set(most_common_cards(deck, 40)))
        card_set = set(card_set)
        # DETERMINE BEST-FIT NAME/DECK FOR CLUSTER
        cluster_name = 'Unknown'
        best_fit_deck = []
        best_fit_deck_sb = []
        max_similar_cards = 0
        for deck in decks_json:
            num_similar_cards = 0
            for card in deck['main']:
                if str(card.get('name')).replace("'", "\\'") in card_set:
                    num_similar_cards += 1
            if num_similar_cards > max_similar_cards:
                max_similar_cards = num_similar_cards
                cluster_name = str(deck['name'])
                best_fit_deck = deck['main']
                best_fit_deck_sb = deck['sb']
        # PRINT CLUSTER
        print("Cluster #" + str(CLUSTER_ID) + " (" + cluster_name + ") :")
        print(card_set)
        print("\n")

        instances = (CLUSTER_ID, len(decks_by_label(CLUSTER_ID)))[1]

        deck_archetype = {
            'archetype_name': cluster_name,
            'top_cards': list(card_set),
            'metagame_percentage': "{:.2%}".format(instances / total_instances),
            'best_fit_deck': {'main': best_fit_deck, 'sb': best_fit_deck_sb}
        }

        format_json['archetypes'].append(deck_archetype)

    unique_cards = sorted(unique_cards, key=lambda s: s[next(iter(s))], reverse=True)


    def closest_cards(a_card, b):
        this_card = apparition_ratio(a_card)[0]
        distances = []
        for name in all_card_names_no_basics:
            dist = distance(apparition_ratio(name)[0], this_card)
            distances.append((name, dist))
        distances.sort(key=lambda x: x[1])
        distances = [(name, dist) for name, dist in distances if name != a_card]
        return [name for name, _ in distances[:b]]


    def versatile_cards(k):
        variances = []
        for name in all_card_names_no_basics:
            versatility = sum([1 if x > 0 else 0 for x in apparition_ratio(name)[0]])
            variances.append((name, versatility))
        variances.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in variances[:k]]


    def common_decks(card):
        common_decks = []
        apps = apparition_ratio(card)
        i = 0
        while i < NUM_CLUSTERS:
            if (apps[0][i] * 100) > 15:
                common_decks.append(format_json['archetypes'][i]['archetype_name'])
            i += 1
        return common_decks

    # DETERMINE TOP CARDS IN FORMAT
    for card_dict in unique_cards[:NUM_TOP_VERS]:
        card_name = list(card_dict.keys())[0]
        quantity = list(card_dict.values())[0]
        top_card = {
            'card_name': card_name,
            'common_archetypes': common_decks(card_name),
            'cards_found_with': closest_cards(card_name, 8),
            'total_instances': quantity,
            'percentage_of_total_cards': "{:.2%}".format(quantity / all_card_names_no_basics_len),
        }

        format_json['format_top_cards'].append(top_card)

    # DETERMINE VERSATILE CARDS IN FORMAT
    for card in versatile_cards(NUM_TOP_VERS):
        versatile_card = {
            'card_name': card,
            'common_archetypes': common_decks(card),
            'cards_found_with': closest_cards(card, 8),
            'total_instances': apparition_ratio(card)[1]
        }

        format_json['format_versatile_cards'].append(versatile_card)

    # SORT JSON
    format_json['archetypes'].sort(key=lambda s: s['metagame_percentage'].strip('%'), reverse=True)
    format_json['format_versatile_cards'].sort(key=lambda s: s['total_instances'], reverse=True)

    with open(FORMAT + '.json', 'w') as outfile:
        json.dump(format_json, outfile, indent=4)

# ANALYZE/PRINT DATA (OPTIONAL)
# print("Most Common Cards in different versions of cluster ")
# for deck, label in decks_by_label(2)[:10]:
#     print(str(most_common_cards(deck, 7)) + " " + str(label))


# cards to analyze
# cards_to_analyze = ['Thoughtseize', 'Llanowar Elves', 'Scalding Tarn', 'Serum Visions']
#
# print("\nCards commonly found with " + cards_to_analyze[0] + "\n" + str(closest_cards(cards_to_analyze[0], 10)))
# print("\nCards commonly found with " + cards_to_analyze[1] + "\n" + str(closest_cards(cards_to_analyze[1], 10)))
# print("\nApparition ratio for " + cards_to_analyze[2] + "\n" + str(apparition_ratio(cards_to_analyze[2])))
# print("\nApparition ratio for " + cards_to_analyze[3] + "\n" + str(apparition_ratio(cards_to_analyze[3])))

    # graph data
    # plt.rc('font', size=14)
    # label_counts = [(label, len(decks_by_label(label))) for label in range(NUM_CLUSTERS)]
    # counts = [count for _, count in label_counts]
    # points = {
    #     'cluster': [label for label, _ in label_counts],
    #     'count': [count for _, count in label_counts],
    # }
    # sns.barplot(x="cluster", y="count", data=points, order=points['cluster']).set_title("# Decks by Cluster in " + FORMAT)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.savefig('graphs/' + FORMAT + ' # Decks by Cluster')
    # show()


# cards to analyze % by cluster (pie chart)
# card_names = ["Lightning Bolt", "Mutavault", "Path to Exile"]
#
# plt.rcParams['figure.facecolor'] = "slateblue"
# plt.rcParams['text.color'] = "w"
# for card_name in card_names:
#     df = apparition_ratio(card_name)[0]
#     label_list = list(range(NUM_CLUSTERS))
#     i = 0
#     while i < len(label_list):
#         if '0.00' in str("{0:.2f}".format(float(df[i]))):
#             df.pop(i)
#             label_list.pop(i)
#         else:
#             i += 1
#     pie(df, labels=label_list, autopct=lambda i: "{0:.2f}".format(float(i)), normalize=False)
#     plt.title(card_name + " % by cluster")
#     plt.savefig('graphs/' + card_name + '_distribution')
#     show()
