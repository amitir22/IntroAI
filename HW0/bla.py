SYMBOLS = ["hearts", "spades", "diamonds", "clovers"]
VALUES = list(range(1, 14))


class Card(object):
    # symbol is one of [diamond, heart, ...]
    def __init__(self, filename, symbol, value):
        self.filename = filename
        self.symbol = symbol
        self.value = value


def generate_card_stack():
    cards = []
    for symbol in SYMBOLS:
        for value in VALUES:
            cards.append(Card('some_file.extension', symbol, value))

    return cards


def print_cards(cards):
    for card in cards:
        print(card.symbol + ' ' + str(card.value))


def main():
    print_cards(generate_card_stack())


if __name__ == '__main__':
    main()
