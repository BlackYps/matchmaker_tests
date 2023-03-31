import requests
import csv
import random


FILENAME = "first-games.csv"


def get_ladder_opponents(num_entries):
    response = requests.get(
        "https://api.faforever.com/data/game?sort=-id&include=playerStats"
        "&filter=(featuredMod.id==6;playerStats.beforeMean==1500)"
        f"&page%5Bnumber%5D=1&page%5Bsize%5D={num_entries}"
    )
    if response.status_code != 200:
        print("Something when wrong fetching ladder leaderboards.")
        return

    return response.json()["included"]


def save_samples(num_games):
    print("Fetching data from api")
    opponents = get_ladder_opponents(num_games)
    print("Fetched data")
    with open(FILENAME, mode="w") as f:
        writer = csv.writer(f)
        losses = 0
        wins = 0
        writer.writerow(
            [
                "beforeMean",
                "beforeDeviation",
                "rating",
                "result",
            ]
        )
        for opponent in opponents:
            if opponent["attributes"]["beforeMean"] == 1500:
                continue

            rating = opponent["attributes"]["beforeMean"] - 3 * opponent["attributes"]["beforeDeviation"]
            result = opponent["attributes"]["result"]
            if result == "DEFEAT":
                wins += 1
            elif result == "VICTORY":
                losses += 1

            writer.writerow(
                [
                    opponent["attributes"]["beforeMean"],
                    opponent["attributes"]["beforeDeviation"],
                    rating,
                    result,
                ]
            )

        games = wins + losses
        print(f"Counted games: {games}")
        winrate = wins / games
        print(f"Winrate of first match: {winrate}")
        writer.writerow(
            [
                "Counted games:",
                games,
                "New player winrate:",
                winrate,
            ]
        )


if __name__ == "__main__":
    save_samples(100)
