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
    fairness = 1 - (rating_disparity / config.MAXIMUM_RATING_IMBALANCE)
    deviation = statistics.pstdev(ratings)
    uniformity = 1 - (deviation / config.MAXIMUM_RATING_DEVIATION)

    quality = fairness * uniformity
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
    print()

    matchmaker = TeamMatchMaker()
    qualities = []
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
    team_size = 4
    for i in range(2000):
        queue.extend(get_random_searches_list(player_factory, 0, 4, team_size))
        queue_len_before_pop.append(sum(len(search.players) for search in queue))

        matches, unmatched = matchmaker.find(queue, team_size)

        created_games.append(len(matches))
        queue_len_after_pop.append(sum(len(search.players) for search in unmatched))
        for search in unmatched:
            search.register_failed_matching_attempt()
        for match in matches:
            quality_without_bonuses, rating_disparity, deviation = calculate_game_quality(match)
            qualities.append(quality_without_bonuses)
            rating_disparities.append(rating_disparity)
            deviations.append(deviation)
            ratings = [search.average_rating for team in match for search in team.get_original_searches()]
            ratings.sort()
            min_rating = ratings[0]
            max_rating = ratings.pop()
            skill_differences.append(max_rating - min_rating)
            if any(team.failed_matching_attempts > 40 for team in match):
                print(f"{repr(match[0].get_original_searches())} tot. rating: {match[0].cumulative_rating} vs \n"
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

    wait_time_90_percentile = numpy.percentile(wait_time, 90)
    max_wait_time = max(wait_time)
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
    skill_difference_90_percentile = numpy.percentile(skill_differences, 90)
    max_skill_difference = max(skill_differences)
    avg_skill_difference = statistics.mean(skill_differences)
    med_skill_difference = statistics.median(skill_differences)

    print()
    print(f"quality was between {worst_quality:.3f} and {best_quality:.3f} "
          f"with average {avg_quality:.2f} and 75th percentile {quality_percentile:.2f}")
    print(f"rating disparity was on average {avg_rating_disparity:.2f}, median {med_rating_disparity:.2f}, "
          f"90th percentile {rating_disparity_90_percentile:.2f} and max {max_rating_disparity}")
    print(f"rating deviation was on average {avg_deviations:.2f}, median {med_deviations:.2f}, "
          f"90th percentile {deviations_90_percentile:.2f} and max {max_deviations:.2f}")
    print(f"skill difference was on average {avg_skill_difference:.2f}, median {med_skill_difference:.2f}, "
          f"90th percentile {skill_difference_90_percentile:.2f} and max {max_skill_difference:.2f}")
    print(f"number of unmatched players was between {min_length} and {max_length} "
          f"with average {avg_length:.2f} and median {med_length}")
    print(f"matched {len(wait_time)} searches total")
    print(f"wait time was on average {avg_wait_time:.2f}, median {med_wait_time}, "
          f"90th percentile {wait_time_90_percentile} and max {max_wait_time} cycles")
    print(f"matched {len(newbie_wait_time)} newbie searches")
    print(f"newbie wait time was on average {newbie_avg_wait_time:.2f}, median {newbie_med_wait_time}, "
          f"90th percentile {newbie_wait_time_90_percentile} and max {newbie_max_wait_time} cycles")
    print()
    print(f"{worst_quality:.2f},{best_quality:.2f},{avg_quality:.2f},{med_quality:.2f},{quality_percentile:.2f}")
    print(f" ,{max_rating_disparity:.2f},{avg_rating_disparity:.2f},{med_rating_disparity:.2f},{rating_disparity_90_percentile:.2f}")
    print(f" ,{max_deviations:.2f},{avg_deviations:.2f},{med_deviations:.2f},{deviations_90_percentile:.2f}")
    print(f"{min_length:.2f},{max_length:.2f},{avg_length:.2f},{med_length:.2f},")
    print(f" ,{max_wait_time:.2f},{avg_wait_time:.2f},{med_wait_time:.2f},{wait_time_90_percentile:.2f}")
    print(f" ,{newbie_max_wait_time:.2f},{newbie_avg_wait_time:.2f},{newbie_med_wait_time:.2f},{newbie_wait_time_90_percentile:.2f}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs[0, 1].scatter(search_ratings, wait_time, label='wait time', marker="x")
    axs[0, 1].scatter(search_newbie_ratings, newbie_wait_time, label='newbie wait time', marker="x")
    rating_disparities.sort()
    deviations.sort()
    skill_differences.sort()
    wait_time.sort()
    newbie_wait_time.sort()
    axs[0, 0].plot(rating_disparities, label='rating disparity')
    axs[0, 0].plot(deviations, label='rating deviation')
    axs[0, 0].plot(skill_differences, label='skill differences')
    axs[1, 0].plot(wait_time, label='wait time')
    axs[1, 0].plot(newbie_wait_time, label='newbie wait time')
    bins = [-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
            1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
    axs[1, 1].hist(search_ratings, bins, density=False, label="average search rating")
    axs[1, 1].hist(search_newbie_ratings, bins, density=False, label="average search rating with newbies")

    for ax in axs.flat:
        ax.grid()
        ax.legend()

    plt.savefig("diagrams.png")
    plt.show()


def test_player_generation(player_factory):
    searches = get_random_searches_list(player_factory, min_size=1000, max_size=1000, max_players=4)
    searches_by_size = TeamMatchMaker()._searches_by_size(searches)
    for i in range(5):
        print(f"{len(searches_by_size[i])} searches with {i} players")

    x = [search.average_rating for search in searches]
    bins = [-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100]
    fig, ax = plt.subplots()
    ax.hist(x, bins, density=False)
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
