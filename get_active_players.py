import requests
import csv
import matplotlib.pyplot as plt
import statistics

FILENAME = "active-tmm-players-rating.csv"


def get_game_infos(num_entries):
    response = requests.get(
        "https://api.faforever.com/data/leaderboardRatingJournal?sort=-id"
        "&filter=leaderboard.id==4"
        f"&page%5Bnumber%5D=1&page%5Bsize%5D={num_entries}"
    )
    if response.status_code != 200:
        print("Something when wrong fetching ladder leaderboards.")
        return

    return response.json()["data"]


def save_samples(num_games):
    print("Fetching data. This can take a while")
    dataset = get_game_infos(num_games)
    with open(FILENAME, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mean",
                "deviation",
            ]
        )
        for data in dataset:
            writer.writerow(
                [
                    data["attributes"]["meanBefore"],
                    data["attributes"]["deviationBefore"],
                ]
            )


if __name__ == "__main__":
    samples = 1000
    save_samples(samples)

    ratings = []
    newbie_ratings = []
    newbie_min_games = 10

    with open(FILENAME, mode="r") as f:
        reader = csv.reader(f)
        reader.__next__()
        for row in reader:
            ratings.append(float(row[0]) - 3 * float(row[1]))
        avg = statistics.mean(ratings)

    bins = [-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
    fig, ax = plt.subplots()
    ax.hist(ratings, bins, density=False, label=f"{len(ratings)} total players, avg {avg} rating")
    ax.grid(axis='x')
    ax.legend()
    plt.savefig(f"active-tmm-players-ratings {samples} samples.png")
    plt.show()
