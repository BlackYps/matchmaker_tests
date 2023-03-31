import logging
import random
import statistics
from math import sqrt

import numpy
import matplotlib.pyplot as plt
import pytest

from matplotlib.ticker import MultipleLocator

from server import config
from server.matchmaker.algorithm.bucket_teams import BucketTeamMatchmaker
from server.matchmaker.algorithm.team_matchmaker import TeamMatchMaker
from server.matchmaker.search import Match, Search, CombinedSearch
from tests.conftest import make_player
from tests.unit_tests.test_matchmaker_algorithm_team_matchmaker import make_searches


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
    # This should be the same as assign_game_quality() in team_matchmaker.py
    # just without the time bonuses, because for the analysis we don't want
    # them to skew the metrics.
    # There is probably a better way to do this to ensure this is stays in sync
    # with the original function
    ratings = []
    for team in match:
        for search in team.get_original_searches():
            ratings.append(search.average_rating)

    rating_disparity = abs(match[0].cumulative_rating - match[1].cumulative_rating)
    unfairness = rating_disparity / config.MAXIMUM_RATING_IMBALANCE
    deviation = statistics.pstdev(ratings)
    rating_variety = deviation / config.MAXIMUM_RATING_DEVIATION

    quality = 1 - sqrt(unfairness ** 2 + rating_variety ** 2)
    return quality, rating_disparity, deviation


def calculate_capped_game_quality(match: Match):
    # legacy version of computing game quality
    ratings = []
    for team in match:
        for search in team.get_original_searches():
            ratings.append(search.average_rating)

    rating_disparity = abs(match[0].cumulative_rating - match[1].cumulative_rating)
    fairness = max(1 - (rating_disparity / config.MAXIMUM_RATING_IMBALANCE), 0)
    deviation = statistics.pstdev(ratings)
    uniformity = max(1 - (deviation / config.MAXIMUM_RATING_DEVIATION), 0)

    quality = fairness * uniformity
    return quality, rating_disparity, deviation


def get_random_searches_list(player_factory, min_size=0, max_size=10, max_players=4):
    searches = []
    for _ in range(random.randint(min_size, max_size)):
        # This is pretty close to the actual team sizes for 4v4
        num_players = min(int(random.paretovariate(2.5)), max_players)

        players = []
        for i in range(num_players):
            if random.random() < 0.05:
                mean = random.triangular(-500, 1300, 0)
                ladder_games = random.randint(0, config.NEWBIE_MIN_GAMES)
            else:
                mean = min(max(random.gauss(1000, 400), -200), 2300)
                ladder_games = random.randint(config.NEWBIE_MIN_GAMES + 2, 200)
            player = player_factory(
                mean=int(mean),
                deviation=33,
                ladder_games=ladder_games,
                name=f"p{i}",
            )
            players.append(player)
        searches.append(Search(players))
    return searches


def test_matchmaker_performance(caplog, player_factory):

    # Disable debug logging for performance
    caplog.set_level(logging.INFO)
    print()

    rating_disparities = []
    newbie_rating_disparities = []
    deviations = []
    avg_ratings = []
    newbie_avg_ratings = []
    skill_differences = []
    newbie_skill_differences = []
    queue_len_before_pop = []
    created_games = []
    queue_len_after_pop = []
    wait_time = []
    newbie_wait_time = []
    queue = []
    search_ratings = []
    search_newbie_ratings = []

    # Set options for this run here
    matchmaker = TeamMatchMaker()
    # matchmaker = BucketTeamMatchmaker()
    team_size = 4
    min_influx = 0  # Min number of searches joining the queue per matching round
    max_influx = 10
    iterations = int(40000 / (max_influx + min_influx))

    for i in range(iterations):
        queue.extend(get_random_searches_list(player_factory, min_influx, max_influx, team_size))
        queue_len_before_pop.append(sum(len(search.players) for search in queue))

        matches, unmatched = matchmaker.find(queue, team_size, 1000)

        created_games.append(len(matches))
        queue_len_after_pop.append(sum(len(search.players) for search in unmatched))
        for search in unmatched:
            search.register_failed_matching_attempt()
        for match in matches:
            quality_without_bonuses, rating_disparity, deviation = calculate_game_quality(match)
            deviations.append(deviation)
            ratings = [search.average_rating for team in match for search in team.get_original_searches()]
            ratings.sort()
            min_rating = ratings[0]
            max_rating = ratings.pop()
            avg_rating = statistics.mean(ratings)  # this only breaks down to searches but shouldn't matter
            if match[0].has_newbie() or match[1].has_newbie():
                newbie_avg_ratings.append(avg_rating)
                newbie_skill_differences.append(max_rating - min_rating)
                newbie_rating_disparities.append(rating_disparity)
            else:
                avg_ratings.append(avg_rating)
                skill_differences.append(max_rating - min_rating)
                rating_disparities.append(rating_disparity)
            wait_time.extend(search.failed_matching_attempts
                                  for team in match for search in team.get_original_searches())
            newbie_wait_time.extend(search.failed_matching_attempts
                                  for team in match for search in team.get_original_searches() if search.has_newbie())
            search_ratings.extend(search.average_rating
                                  for team in match for search in team.get_original_searches())
            search_newbie_ratings.extend(search.average_rating
                                  for team in match for search in team.get_original_searches() if search.has_newbie())
            # I use this to give me some info on particular edge cases. Set this to whatever interests you,
            # but be careful, you can get a lot of log output.
            if quality_without_bonuses < -0.5 and any(search.has_newbie()
                                              for team in match for search in team.get_original_searches()):
                print(f"{repr(match[0].get_original_searches())} tot. rating: {match[0].cumulative_rating} vs "
                      f"{repr(match[1].get_original_searches())} tot. rating: {match[1].cumulative_rating} "
                      f"Quality: {quality_without_bonuses}")

        queue = unmatched

    avg_wait_time = statistics.mean(wait_time)
    med_wait_time = statistics.median(wait_time)
    wait_time_percentile = numpy.percentile(wait_time, 75)
    wait_time_90_percentile = numpy.percentile(wait_time, 90)
    max_wait_time = max(wait_time)
    newbie_avg_wait_time = statistics.mean(newbie_wait_time)
    newbie_med_wait_time = statistics.median(newbie_wait_time)
    newbie_wait_time_percentile = numpy.percentile(newbie_wait_time, 75)
    newbie_wait_time_90_percentile = numpy.percentile(newbie_wait_time, 90)
    newbie_max_wait_time = max(newbie_wait_time)
    avg_length = statistics.mean(queue_len_after_pop)
    med_length = statistics.median(queue_len_after_pop)
    length_percentile = numpy.percentile(queue_len_after_pop, 75)
    length_90_percentile = numpy.percentile(queue_len_after_pop, 90)
    max_length = max(queue_len_after_pop)
    avg_rating_disparity = statistics.mean(rating_disparities)
    med_rating_disparity = statistics.median(rating_disparities)
    rating_disparity_percentile = numpy.percentile(rating_disparities, 75)
    rating_disparity_90_percentile = numpy.percentile(rating_disparities, 90)
    max_rating_disparity = max(rating_disparities)
    avg_deviations = statistics.mean(deviations)
    med_deviations = statistics.median(deviations)
    deviations_percentile = numpy.percentile(deviations, 75)
    deviations_90_percentile = numpy.percentile(deviations, 90)
    max_deviations = max(deviations)
    avg_skill_difference = statistics.mean(skill_differences)
    med_skill_difference = statistics.median(skill_differences)
    skill_difference_percentile = numpy.percentile(skill_differences, 75)
    skill_difference_90_percentile = numpy.percentile(skill_differences, 90)
    max_skill_difference = max(skill_differences)

    print()
    print(f"rating disparity was on average {avg_rating_disparity:.2f}, median {med_rating_disparity:.2f}, "
          f"90th percentile {rating_disparity_90_percentile:.2f} and max {max_rating_disparity}")
    print(f"rating deviation was on average {avg_deviations:.2f}, median {med_deviations:.2f}, "
          f"90th percentile {deviations_90_percentile:.2f} and max {max_deviations:.2f}")
    print(f"skill difference was on average {avg_skill_difference:.2f}, median {med_skill_difference:.2f}, "
          f"90th percentile {skill_difference_90_percentile:.2f} and max {max_skill_difference:.2f}")
    print(f"number of unmatched players was on average {avg_length:.2f}, median {med_length} and max {max_length} ")
    print(f"matched {len(wait_time)} searches total")
    print(f"wait time was on average {avg_wait_time:.2f}, median {med_wait_time}, "
          f"90th percentile {wait_time_90_percentile} and max {max_wait_time} cycles")
    print(f"matched {len(newbie_wait_time)} newbie searches")
    print(f"newbie wait time was on average {newbie_avg_wait_time:.2f}, median {newbie_med_wait_time}, "
          f"90th percentile {newbie_wait_time_90_percentile} and max {newbie_max_wait_time} cycles")
    print()
    # This is here, so I can copy the values in my spreadsheet easily.
    print(f"{avg_rating_disparity:.2f},{med_rating_disparity:.2f},{rating_disparity_percentile:.2f},{rating_disparity_90_percentile:.2f},{max_rating_disparity:.2f}")
    print(f"{avg_deviations:.2f},{med_deviations:.2f},{deviations_percentile:.2f},{deviations_90_percentile:.2f},{max_deviations:.2f}")
    print(f"{avg_skill_difference:.2f},{med_skill_difference:.2f},{skill_difference_percentile:.2f},{skill_difference_90_percentile:.2f},{max_skill_difference:.2f}")
    print(f"{avg_length:.2f},{med_length:.2f},{length_percentile:.2f},{length_90_percentile:.2f},{max_length:.2f}")
    print(f"{avg_wait_time:.1f},{med_wait_time:.1f},{wait_time_percentile:.1f},{wait_time_90_percentile:.1f},{max_wait_time:.1f}")
    print(f"{newbie_avg_wait_time:.1f},{newbie_med_wait_time:.1f},{newbie_wait_time_percentile:.1f},{newbie_wait_time_90_percentile:.1f},{newbie_max_wait_time:.1f}")

    fig, axs = plt.subplots(2, 3, figsize=(12, 9))
    fig.suptitle(f"{team_size}v{team_size} influx: {min_influx}-{max_influx} iterations: {iterations}\n "
                 f"time bonus: {config.TIME_BONUS}, max: {config.MAXIMUM_TIME_BONUS}, "
                 f"newbie bonus: {config.NEWBIE_TIME_BONUS}, max: {config.MAXIMUM_NEWBIE_TIME_BONUS}, "
                 f"minority bonus: {config.MINORITY_BONUS}\n"
                 f"min quality: {config.MINIMUM_GAME_QUALITY}, max imbalance: {config.MAXIMUM_RATING_IMBALANCE}, "
                 f"max deviation: {config.MAXIMUM_RATING_DEVIATION}")
    axs[0, 1].scatter(search_ratings, wait_time, label='wait time', marker=".")
    axs[0, 1].scatter(search_newbie_ratings, newbie_wait_time, label='newbie wait time', marker=".")
    axs[0, 1].set_ylim((0, 80))
    axs[0, 1].set(xlabel="rating")

    axs[0, 2].scatter(avg_ratings, rating_disparities, label="rating disparity between teams", marker=".")
    axs[0, 2].scatter(newbie_avg_ratings, newbie_rating_disparities, label="games with newbies", marker="1")
    axs[0, 2].set(xlabel="game rating")
    axs[0, 2].set_ylim(bottom=0, top=440)

    axs[1, 2].scatter(avg_ratings, skill_differences, label="skill differences between players", marker=".")
    axs[1, 2].scatter(newbie_avg_ratings, newbie_skill_differences, label="games with newbies", marker="1")
    axs[1, 2].set(xlabel="game rating")
    axs[1, 2].set_ylim(bottom=0, top=1800)

    rating_disparities.extend(newbie_rating_disparities)
    rating_disparities.sort()
    deviations.sort()
    skill_differences.extend(newbie_skill_differences)
    skill_differences.sort()
    wait_time.sort()
    newbie_wait_time.sort()
    axs[0, 0].plot(rating_disparities, label='rating disparity')
    axs[0, 0].plot(deviations, label='rating deviation')
    axs[0, 0].plot(skill_differences, label='skill differences')
    axs[0, 0].set_ylim((0, 2000))
    axs[0, 0].set(xlabel="game number")
    axs[1, 0].plot(wait_time, label='wait time')
    axs[1, 0].plot(newbie_wait_time)
    axs[1, 0].text(0.9, 0.78, f"average: {avg_wait_time:.1f}\nmedian: {med_wait_time}\nnewbie average: {newbie_avg_wait_time:.1f}\nnewbie median: {newbie_med_wait_time}",
                   horizontalalignment='right', transform=axs[1, 0].transAxes)
    axs[1, 0].set_ylim((0, 50))
    axs[1, 0].set(xlabel="search number")
    bins = [-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
            1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
    axs[1, 1].hist(search_ratings, bins, density=False, label="search average rating")
    axs[1, 1].hist(search_newbie_ratings, bins, density=False, label="search with newbies average rating")
    axs[1, 1].set(xlabel="rating")

    for ax in axs.flat:
        ax.grid()
        ax.legend(loc="upper left")

    plt.savefig(f"test {team_size}v{team_size} {min_influx}-{max_influx}.png")
    plt.show()


def test_player_generation(player_factory):
    searches = get_random_searches_list(player_factory, min_size=1000, max_size=1000, max_players=4)
    searches_by_size = TeamMatchMaker()._searches_by_size(searches)
    for i in range(5):
        print(f"{len(searches_by_size[i])} searches with {i} players")

    x = [search.average_rating for search in searches]
    n = [search.average_rating for search in searches if search.has_newbie()]
    bins = [-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
            1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
    fig, ax = plt.subplots()
    ax.hist(x, bins, density=False)
    ax.hist(n, bins, density=False)
    ax.grid(axis='x')
    plt.show()


def test_game_quality_for_2v2_example(player_factory):
    s = make_searches([2156, -32, 1084, 570], player_factory)
    team_a = CombinedSearch(*[s[0], s[1]])
    team_b = CombinedSearch(*[s[2], s[3]])
    game = TeamMatchMaker().assign_game_quality((team_a, team_b), 2, 900)
    capped_quality, _, _ = calculate_capped_game_quality((team_a, team_b))

    assert game.quality == capped_quality
    # This is just my lazy way of getting the game quality displayed in the output log
    assert game.quality == 0.0


def test_game_quality_for_2v2_example2(player_factory):
    s = make_searches([1156, 1108, 810, 1456], player_factory)
    team_a = CombinedSearch(*[s[0], s[1]])
    team_b = CombinedSearch(*[s[2], s[3]])
    game = TeamMatchMaker().assign_game_quality((team_a, team_b), 2, 900)

    assert game.quality == 0.0


def test_game_quality_for_4v4_example(player_factory):
    s = make_searches([1641, 1936, 1791, 2314, 1930, 2258, 1402, 2090], player_factory)
    team_a = CombinedSearch(*[s[0], s[1], s[2], s[3]])
    team_b = CombinedSearch(*[s[4], s[5], s[6], s[7]])
    game = TeamMatchMaker().assign_game_quality((team_a, team_b), 4, 1000)

    assert game.quality == 0.0


def test_game_quality(player_factory):
    s = make_searches([2420, 2022, 1820, 1777, 1342, 2141, 1937, 1275], player_factory)
    matchmaker = TeamMatchMaker()
    team_size = 4
    rating_peak = 1000
    matches, unmatched = matchmaker.find(s, team_size, rating_peak)

    game = matchmaker.assign_game_quality(matches[0], team_size, rating_peak)
    print()
    print(matches[0])

    assert game.quality == 0.0


def test_game_quality2v2(player_factory):
    s = make_searches([2156, 2108, 1010, 2456], player_factory)
    matchmaker = TeamMatchMaker()
    team_size = 2
    rating_peak = 1000
    matches, unmatched = matchmaker.find(s, team_size, rating_peak)

    game = matchmaker.assign_game_quality(matches[0], team_size, rating_peak)
    print()
    print(matches[0])

    assert game.quality == 0.0
