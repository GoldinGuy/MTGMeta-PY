import json

formats = ['brawl', 'commander', 'legacy', 'modern', 'pauper', 'pioneer']

for format in formats:
    with open('decks_json/decks-' + format + '.json') as f:
        decks_json = json.load(f)

    # print(decks_json)
    for deck in decks_json:
        try:
            if deck['main'][0]['name'] is not None:
                deck_name = str(deck['name']).replace("/", "_")
                deckFile = open(deck['format'] + '_decks/' + deck_name, "w")
                card_lines = ['\n']
                for card in deck['main']:
                    card_name = str(card['name']).replace("'", "\\'")
                    card_lines.append(str(card['quantity']) + ' ' + card_name + '\n')
                deckFile.writelines(card_lines)
                deckFile.close()
        except Exception as e:
            print(e)