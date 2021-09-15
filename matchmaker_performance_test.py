import logging
import random
import statistics
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
    ratings = []
    for team in match:
        for search in team.get_original_searches():
            ratings.append(search.average_rating)

    rating_disparity = abs(match[0].cumulative_rating - match[1].cumulative_rating)
    fairness = 1 - (rating_disparity / config.MAXIMUM_RATING_IMBALANCE) ** 2
    deviation = statistics.pstdev(ratings)
    uniformity = 1 - (deviation / config.MAXIMUM_RATING_DEVIATION) ** 2

    quality = (fairness + uniformity) / 2
    return quality, rating_disparity, deviation


def calculate_capped_game_quality(match: Match):
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
        # With this distribution the number of players is 1.4 * len(list)
        num_players = min(int(random.paretovariate(2.0)), max_players)

        players = []
        for i in range(num_players):
            if random.random() < 0.05:
                mean = random.triangular(-500, 1300, 0)
                ladder_games = random.randint(0, config.NEWBIE_MIN_GAMES)
            else:
                mean = min(max(random.gauss(1000, 400), 0), 2300)
                ladder_games = random.randint(config.NEWBIE_MIN_GAMES + 1, 200)
            player = player_factory(
                mean=int(mean),
                deviation=33,
                ladder_games=ladder_games,
                name=f"p{i}",
            )
            players.append(player)
        searches.append(Search(players))
    return searches


def test_matchmaker(caplog, player_factory):
    # Disable debug logging for performance
    caplog.set_level(logging.INFO)
    print()

    matchmaker = TeamMatchMaker()
    rating_disparities = []
    deviations = []
    skill_differences = []
    queue_len_before_pop = []
    created_games = []
    queue_len_after_pop = []
    wait_time = []
    newbie_wait_time = []
    queue = []
    search_ratings = []
    search_newbie_ratings = []
    team_size = 2
    influx = 2
    iterations = int(20000 / influx)
    for i in range(iterations):
        queue.extend(get_random_searches_list(player_factory, 0, influx, team_size))
        queue_len_before_pop.append(sum(len(search.players) for search in queue))

        matches, unmatched = matchmaker.find(queue, team_size)

        created_games.append(len(matches))
        queue_len_after_pop.append(sum(len(search.players) for search in unmatched))
        for search in unmatched:
            search.register_failed_matching_attempt()
        for match in matches:
            quality_without_bonuses, rating_disparity, deviation = calculate_game_quality(match)
            rating_disparities.append(rating_disparity)
            deviations.append(deviation)
            ratings = [search.average_rating for team in match for search in team.get_original_searches()]
            ratings.sort()
            min_rating = ratings[0]
            max_rating = ratings.pop()
            skill_differences.append(max_rating - min_rating)
            if rating_disparity > 1200 or any(search.has_newbie() and search.failed_matching_attempts > 15 for team in match for search in team.get_original_searches()):
                print(f"{repr(match[0].get_original_searches())} tot. rating: {match[0].cumulative_rating} vs "
                      f"{repr(match[1].get_original_searches())} tot. rating: {match[1].cumulative_rating} "
                      f"Quality: {quality_without_bonuses}")
            wait_time.extend(search.failed_matching_attempts
                                  for team in match for search in team.get_original_searches())
            newbie_wait_time.extend(search.failed_matching_attempts
                                  for team in match for search in team.get_original_searches() if search.has_newbie())
            search_ratings.extend(search.average_rating
                                  for team in match for search in team.get_original_searches())
            search_newbie_ratings.extend(search.average_rating
                                  for team in match for search in team.get_original_searches() if search.has_newbie())

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
    print(f"{avg_rating_disparity:.2f},{med_rating_disparity:.2f},{rating_disparity_percentile:.2f},{rating_disparity_90_percentile:.2f},{max_rating_disparity:.2f}")
    print(f"{avg_deviations:.2f},{med_deviations:.2f},{deviations_percentile:.2f},{deviations_90_percentile:.2f},{max_deviations:.2f}")
    print(f"{avg_skill_difference:.2f},{med_skill_difference:.2f},{skill_difference_percentile:.2f},{skill_difference_90_percentile:.2f},{max_skill_difference:.2f}")
    print(f"{avg_length:.2f},{med_length:.2f},{length_percentile:.2f},{length_90_percentile:.2f},{max_length:.2f}")
    print(f"{avg_wait_time:.1f},{med_wait_time:.1f},{wait_time_percentile:.1f},{wait_time_90_percentile:.1f},{max_wait_time:.1f}")
    print(f"{newbie_avg_wait_time:.1f},{newbie_med_wait_time:.1f},{newbie_wait_time_percentile:.1f},{newbie_wait_time_90_percentile:.1f},{newbie_max_wait_time:.1f}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"{team_size}v{team_size} influx: 0-{influx} iterations: {iterations}\n "
                 f"time bonus: {config.TIME_BONUS} max: {config.MAXIMUM_TIME_BONUS} "
                 f"newbie bonus: {config.NEWBIE_TIME_BONUS} max: {config.MAXIMUM_NEWBIE_TIME_BONUS}\n"
                 f"min quality: {config.MINIMUM_GAME_QUALITY}, max imbalance: {config.MAXIMUM_RATING_IMBALANCE}, max deviation: {config.MAXIMUM_RATING_DEVIATION}")
    axs[0, 1].scatter(search_ratings, wait_time, label='wait time', marker="x")
    axs[0, 1].scatter(search_newbie_ratings, newbie_wait_time, label='newbie wait time', marker="x")
    axs[0, 1].set_ylim((0, 100))
    axs[0, 1].set(xlabel="rating")
    rating_disparities.sort()
    deviations.sort()
    skill_differences.sort()
    wait_time.sort()
    newbie_wait_time.sort()
    axs[0, 0].plot(rating_disparities, label='rating disparity')
    axs[0, 0].plot(deviations, label='rating deviation')
    axs[0, 0].plot(skill_differences, label='skill differences')
    axs[0, 0].set_ylim((0, 2000))
    axs[0, 0].set(xlabel="game number")
    axs[1, 0].plot(wait_time, label='wait time')
    axs[1, 0].plot(newbie_wait_time, label='newbie wait time')
    axs[1, 0].text(0.9, 0.78, f"average: {avg_wait_time:.1f}\nmedian: {med_wait_time}\nnewbie average: {newbie_avg_wait_time:.1f}\nnewbie median: {newbie_med_wait_time}",
                   horizontalalignment='right', transform=axs[1, 0].transAxes)
    axs[1, 0].set_ylim((0, 200))
    axs[1, 0].set(xlabel="search number")
    bins = [-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
            1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
    axs[1, 1].hist(search_ratings, bins, density=False, label="search average rating")
    axs[1, 1].hist(search_newbie_ratings, bins, density=False, label="search with newbies average rating")
    axs[1, 1].set(xlabel="rating")

    for ax in axs.flat:
        ax.grid()
        ax.legend(loc="upper left")

    plt.savefig(f"test {team_size}v{team_size} 0-{influx}.png")
    plt.show()


def test_player_generation(player_factory):
    searches = get_random_searches_list(player_factory, min_size=1000, max_size=1000, max_players=4)
    searches_by_size = TeamMatchMaker()._searches_by_size(searches)
    for i in range(5):
        print(f"{len(searches_by_size[i])} searches with {i} players")

    x = [search.average_rating for search in searches]
    n = [search.average_rating for search in searches if search.has_newbie()]
    bins = [-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
    fig, ax = plt.subplots()
    ax.hist(x, bins, density=False)
    ax.hist(n, bins, density=False)
    ax.grid(axis='x')
    plt.show()


def test_game_quality_for_2v2_example(player_factory):
    s = make_searches([2156, -32, 1084, 570], player_factory)
    team_a = CombinedSearch(*[s[0], s[1]])
    team_b = CombinedSearch(*[s[2], s[3]])
    game = TeamMatchMaker().assign_game_quality((team_a, team_b), 2)

    assert game.quality == calculate_capped_game_quality((team_a, team_b))
    assert game.quality == 0.0


def test_game_quality_for_2v2_example2(player_factory):
    s = make_searches([900, 800, 2000, 1300], player_factory)
    team_a = CombinedSearch(*[s[0], s[1]])
    team_b = CombinedSearch(*[s[2], s[3]])
    game = TeamMatchMaker().assign_game_quality((team_a, team_b), 2)

    assert game.quality == 0.0


def test_game_quality_for_4v4_example(player_factory):
    s = make_searches([100, 100, 4000, 4000, 4000, 4000, 100, 100], player_factory)
    team_a = CombinedSearch(*[s[0], s[1], s[2], s[3]])
    team_b = CombinedSearch(*[s[4], s[5], s[6], s[7]])
    game = TeamMatchMaker().assign_game_quality((team_a, team_b), 4)

    assert game.quality == 0.0
