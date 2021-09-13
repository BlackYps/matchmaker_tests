import requests
import csv
import random


FILENAME = "first-games.csv"


def get_latest_ladder_ids(num_entries):
    response = requests.get(
        "https://api.faforever.com/data/ladder1v1Rating?sort=-id"
        "&filter=(mean!=1500;deviation<150)"
        f"&page%5Bnumber%5D=1&page%5Bsize%5D={num_entries}"
    )
    if response.status_code != 200:
        print("Something when wrong fetching ladder leaderboards.")
        return

    return response.json()["data"]


def get_match_history(player_id, num_games):
    response = requests.get(
        f"https://api.faforever.com/data/gamePlayerStats?include=game"
        f"&filter=(player.id=={player_id};game.featuredMod.id==6)"
        f"&sort=scoreTime&page%5Bnumber%5D=1&page%5Bsize%5D={num_games}"
    )
    if response.status_code != 200 or "data" not in response.json():
        return False, {}

    attributes = response.json()["data"]

    return True, attributes


def save_samples(num_players, num_games):
    ladder_players = get_latest_ladder_ids(num_players)

    with open(FILENAME, mode="w") as f:
        writer = csv.writer(f)
        player_count = 0
        lost_first_matches = 0
        won_first_matches = 0
        while player_count < num_players:
            if player_count % 5 == 0:
                print(f"Sampling {player_count}/{num_players}")

            player = ladder_players[player_count]
            player_id = player["id"]
            player_count += 1

            success, matches = get_match_history(player_id, num_games)

            if not success:
                continue
            if matches[0]["attributes"]["beforeMean"] != 1500:
                continue
            
            writer.writerow([])
            writer.writerow([])
            writer.writerow(
                [
                    player_id,
                ]
            )
            num = 0
            while num < num_games:
                match = matches[num]["attributes"]
                rating = match["beforeMean"] - 3 * match["beforeDeviation"]
                
                if num == 1 and rating > 0:
                    won_first_matches += 1
                if num == 1 and rating < 0:
                    lost_first_matches += 1
                
                writer.writerow(
                    [
                        "",
                        num,
                        match["beforeMean"],
                        match["beforeDeviation"],
                        rating,
                    ]
                )
                num += 1
                
                    
        print(f"Counted players: {won_first_matches + lost_first_matches}")
        print(f"Winrate of first match: {won_first_matches / (won_first_matches + lost_first_matches)}")


if __name__ == "__main__":
    save_samples(500, 2)
