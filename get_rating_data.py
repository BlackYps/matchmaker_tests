import requests
import csv
import random


FILENAME = "ratings.csv"


def get_global_ratings(num_entries=42500):
    response = requests.get(
        "https://api.faforever.com/leaderboards/global?"
        f"page%5Bnumber%5D=1&page%5Bsize%5D={num_entries}"
    )
    if response.status_code != 200:
        print("Something when wrong fetching global leaderboards.")
        return

    return response.json()["data"]


def get_ladder_ratings(num_entries=7232):
    response = requests.get(
        "https://api.faforever.com/leaderboards/ladder1v1?"
        f"page%5Bnumber%5D=1&page%5Bsize%5D={num_entries}"
    )
    if response.status_code != 200:
        print("Something when wrong fetching ladder leaderboards.")
        return

    return response.json()["data"]


def get_ladder_rating(player_id):
    response = requests.get(
        f"https://api.faforever.com/leaderboards/ladder1v1/{player_id}"
    )
    if response.status_code != 200 or "data" not in response.json():
        return False, {}

    attributes = response.json()["data"]["attributes"]

    return True, attributes


def get_global_rating(player_id):
    response = requests.get(
        f"https://api.faforever.com/leaderboards/global/{player_id}"
    )
    if response.status_code != 200 or "data" not in response.json():
        return False, {}

    attributes = response.json()["data"]["attributes"]

    return True, attributes


def save_samples(num_samples):
    ladder_players = get_ladder_ratings()

    with open(FILENAME, mode="w") as f:
        writer = csv.writer(f)
        sample_count = 0
        while sample_count < num_samples:
            if sample_count % 10 == 0:
                print(f"Sampling {sample_count}/{num_samples}")

            player = random.choice(ladder_players)
            ladder = player["attributes"]
            player_id = player["id"]

            success, globalR = get_global_rating(player_id)

            if not success:
                continue

            writer.writerow(
                [
                    player_id,
                    globalR["mean"],
                    globalR["deviation"],
                    globalR["numGames"],
                    ladder["mean"],
                    ladder["deviation"],
                    ladder["numGames"],
                ]
            )
            sample_count += 1


if __name__ == "__main__":
    save_samples(20)
