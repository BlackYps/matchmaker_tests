import logging
import random
import statistics
import numpy
import matplotlib.pyplot as plt
import pytest

from matplotlib.ticker import MultipleLocator
from sortedcontainers import SortedList

from server import config
from server.matchmaker.algorithm.team_matchmaker import TeamMatchMaker
from server.matchmaker.search import Match, Search
from tests.conftest import make_player


@pytest.fixture
def player_factory():
    player_id_counter = 0

    def make(
        mean: int = 1500,
        deviation: int = 500,
        ladder_games: int = config.NEWBIE_MIN_GAMES+1,
        name=None
    ):
        nonlocal player_id_counter
        """Make a player with the given ratings"""
        player = make_player(
            ladder_rating=(mean, deviation),
            ladder_games=ladder_games,
            login=name,
            lobby_connection_spec=None,
            player_id=player_id_counter
        )
        player_id_counter += 1
        return player
    return make


def calculate_game_quality(match: Match):
    ratings = []
    for team in match:
        for search in team.get_original_searches():
            ratings.append(search.average_rating)

    rating_disparity = abs(match[0].cumulative_rating - match[1].cumulative_rating)
    fairness = max((config.MAXIMUM_RATING_IMBALANCE - rating_disparity) / config.MAXIMUM_RATING_IMBALANCE, 0)
    deviation = statistics.pstdev(ratings)
    uniformity = max((config.MAXIMUM_RATING_DEVIATION - deviation) / config.MAXIMUM_RATING_DEVIATION, 0)

    quality = fairness * uniformity
    return quality, rating_disparity, deviation


def get_random_searches_list(player_factory, min_size=0, max_size=10, max_players=4):
    searches = []
    for _ in range(random.randint(min_size, max_size)):
        # With this distribution the number of players is 1.4 * len(list)
        num_players = min(int(random.paretovariate(2.0)), max_players)
        players = [player_factory(
            mean=int(random.triangular(-200, 2200, 900)),
            deviation=int(random.triangular(10, 200, 80)),
            ladder_games=random.randint(0, 200),
            name=f"p{i}"
        ) for i in range(num_players)]
        searches.append(Search(players))

    return searches


def test_matchmaker(caplog, player_factory):
    # Disable debug logging for performance
    caplog.set_level(logging.INFO)

    matchmaker = TeamMatchMaker()
    qualities = []
    rating_disparities = []
    deviations = []
    queue_len_before_pop = []
    created_games = []
    queue_len_after_pop = []
    wait_time = SortedList()
    newbie_wait_time = []
    queue = []
    iteration = []
    for i in range(2000):
        queue.extend(get_random_searches_list(player_factory, 0, 4, 4))
        queue_len_before_pop.append(sum(len(search.players) for search in queue))

        matches, unmatched = matchmaker.find(queue, 4)

        created_games.append(len(matches))
        queue_len_after_pop.append(sum(len(search.players) for search in unmatched))
        for search in unmatched:
            search.register_failed_matching_attempt()
        for match in matches:
            quality_without_bonuses, rating_disparity, deviation = calculate_game_quality(match)
            qualities.append(quality_without_bonuses)
            rating_disparities.append(rating_disparity)
            deviations.append(deviation)
            if i % 10 == 0:
                print(f"{repr(match[0].get_original_searches())} tot. rating: {match[0].cumulative_rating} vs "
                      f"{repr(match[1].get_original_searches())} tot. rating: {match[1].cumulative_rating} "
                      f"Quality: {quality_without_bonuses}")
            wait_time.update(search.failed_matching_attempts
                             for team in match for search in team.get_original_searches())
            newbie_wait_time.extend(search.failed_matching_attempts
                                    for team in match for search in team.get_original_searches() if search.has_newbie())
        queue = unmatched
        iteration.append(i)

    wait_time_90_percentile = numpy.percentile(wait_time, 90)
    max_wait_time = wait_time.pop()
    avg_wait_time = statistics.mean(wait_time)
    med_wait_time = statistics.median(wait_time)
    newbie_wait_time_90_percentile = numpy.percentile(newbie_wait_time, 90)
    newbie_max_wait_time = max(newbie_wait_time)
    newbie_avg_wait_time = statistics.mean(newbie_wait_time)
    newbie_med_wait_time = statistics.median(newbie_wait_time)
    min_length = min(queue_len_after_pop)
    max_length = max(queue_len_after_pop)
    avg_length = statistics.mean(queue_len_after_pop)
    med_length = statistics.median(queue_len_after_pop)
    best_quality = max(qualities)
    worst_quality = min(qualities)
    avg_quality = statistics.mean(qualities)
    med_quality = statistics.median(qualities)
    quality_percentile = numpy.percentile(qualities, 75)
    rating_disparity_90_percentile = numpy.percentile(rating_disparities, 90)
    max_rating_disparity = max(rating_disparities)
    avg_rating_disparity = statistics.mean(rating_disparities)
    med_rating_disparity = statistics.median(rating_disparities)
    deviations_90_percentile = numpy.percentile(deviations, 90)
    max_deviations = max(deviations)
    avg_deviations = statistics.mean(deviations)
    med_deviations = statistics.median(deviations)

    print()
    print(f"quality was between {worst_quality} and {best_quality} "
          f"with average {avg_quality} and 75th percentile {quality_percentile}")
    print(f"rating disparity was on average {avg_rating_disparity}, median {med_rating_disparity}, "
          f"90th percentile {rating_disparity_90_percentile} and max {max_rating_disparity}")
    print(f"rating deviation was on average {avg_deviations}, median {med_deviations}, "
          f"90th percentile {deviations_90_percentile} and max {max_deviations}")
    print(f"number of unmatched players was between {min_length} and {max_length} "
          f"with average {avg_length} and median {med_length}")
    print(f"matched {len(wait_time)} searches total")
    print(f"wait time was on average {avg_wait_time}, median {med_wait_time}, "
          f"90th percentile {wait_time_90_percentile} and max {max_wait_time} cycles")
    print(f"matched {len(newbie_wait_time)} newbie searches")
    print(f"newbie wait time was on average {newbie_avg_wait_time}, median {newbie_med_wait_time}, "
          f"90th percentile {newbie_wait_time_90_percentile} and max {newbie_max_wait_time} cycles")
    print()
    print(f"{worst_quality},{best_quality},{avg_quality},{med_quality},{quality_percentile}")
    print(f" ,{max_rating_disparity},{avg_rating_disparity},{med_rating_disparity},{rating_disparity_90_percentile}")
    print(f" ,{max_deviations},{avg_deviations},{med_deviations},{deviations_90_percentile}")
    print(f"{min_length},{max_length},{avg_length},{med_length},")
    print(f" ,{max_wait_time},{avg_wait_time},{med_wait_time},{wait_time_90_percentile}")
    print(f" ,{newbie_max_wait_time},{newbie_avg_wait_time},{newbie_med_wait_time},{newbie_wait_time_90_percentile}")

    fig, ax = plt.subplots()
    ax.plot(iteration, queue_len_before_pop, label='length before pop')
    ax.plot(iteration, created_games, label='created games')
    ax.plot(iteration, queue_len_after_pop, label='length after pop')
    ax.grid()
    ax.legend()
    ax.yaxis.set_major_locator(MultipleLocator(4))
    plt.show()


def test_player_generation(player_factory):
    searches = get_random_searches_list(player_factory, min_size=1000, max_size=1000, max_players=4)
    searches_by_size = TeamMatchMaker()._searches_by_size(searches)
    for i in range(5):
        print(f"{len(searches_by_size[i])} searches with {i} players")

    x = [search.average_rating for search in searches]
    bins = [-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
    fig, ax = plt.subplots()
    ax.hist(x, bins, density=True)
    ax.grid(axis='x')
    plt.show()
