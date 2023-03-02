from Simulator.player import player
from Simulator.pool import pool
from Simulator.champion import champion
from Simulator import champion as c_object
from Simulator.item_stats import trait_items, starting_items
from Simulator.origin_class_stats import origin_class
from Simulator.observation import Observation
from Simulator.step_function import Step_Function
from Models.MCTS_keras import Batch_MCTSAgent
import config
import numpy as np
from AI_interface import DataWorker
import Simulator.utils as utils
from Simulator.item_stats import item_builds, uncraftable_items


def setup(player_num=0) -> player:
    """Creates fresh player and pool"""
    base_pool = pool()
    player1 = player(base_pool, player_num)
    return player1

def chosen_test():
    p1 = setup()
    p1.gold = 1000
    p1.max_units = 4
    p1.buy_champion(champion('leesin', chosen='duelist'))
    assert p1.chosen == 'duelist'
    p1.move_bench_to_board(0, 0, 0)
    assert p1.team_tiers['duelist'] == 1
    p2 = setup()
    p2.gold = 1000
    p2.max_units = 4
    p1.buy_champion(champion('leesin'))
    p1.buy_champion(champion('leesin'))
    assert p1.board[0][0].chosen == 'duelist'

def end_of_turn_actions_test():
    p1 = setup()
    p1.gold = 1000
    p1.max_units = 3
    for _ in range(8):
        p1.buy_champion(champion('leesin'))
    p1.move_bench_to_board(0, 0, 0)
    p1.buy_champion(champion('nami'))
    p1.move_bench_to_board(0, 1, 0)
    p1.add_to_item_bench('duelists_zeal')
    p1.move_item(0, 1, 0)
    p1.end_turn_actions()
    assert p1.bench[1] is None
    assert p1.bench[2] is not None
    assert p1.team_tiers['duelist'] == 1

def championDuplicatorTest():
    p1 = setup()
    p1.gold = 1000
    p1. max_units = 10
    p1.buy_champion(champion('leesin'))
    for x in range(4):
        p1.add_to_item_bench('champion_duplicator')
    p1.move_item(0, 0, -1)
    assert p1.item_bench[0] is None
    assert p1.bench[1].name == 'leesin'
    p1.move_bench_to_board(0, 0, 0)
    p1.move_item(1, 0, 0)
    assert p1.board[0][0].stars == 2
    assert p1.gold == 995
    p1.buy_champion(champion('jax'))
    p1.move_bench_to_board(0, 1, 0)
    p1.move_item(2, 1, 0)
    assert p1.bench[0].name == 'jax'
    p1.buy_champion(champion('nami'))
    p1.buy_champion(champion('aphelios'))
    p1.buy_champion(champion('vayne'))
    p1.buy_champion(champion('vi'))
    p1.buy_champion(champion('warwick'))
    p1.buy_champion(champion('teemo'))
    p1.buy_champion(champion('thresh'))
    p1.buy_champion(champion('talon'))
    p1.move_item(3, 3, -1)
    assert p1.item_bench[3] == 'champion_duplicator'
    for x in range(8):
        p1.sell_from_bench(x)
        assert p1.bench[x] is None

def magneticRemoverTest():
    p1 = setup()
    p1.gold = 1000
    p1.max_units = 10
    p1.buy_champion(champion('leesin'))
    p1.buy_champion(champion('jax'))
    p1.move_bench_to_board(0, 0, 0)
    p1.add_to_item_bench('magnetic_remover')
    p1.add_to_item_bench('magnetic_remover')
    p1.add_to_item_bench('mages_cap')
    for x in range(5):
        p1.add_to_item_bench('deathblade')
    for x in range(2, 5):
        p1.move_item(x, 0, 0)
    for x in range(5, 8):
        p1.move_item(x, 1, -1)
    assert p1.team_composition['mage'] != 0
    p1.move_item(0, 0, 0)
    p1.move_item(1, 1, -1)
    assert p1.team_composition['mage'] == 0
    assert p1.board[0][0].items == []
    assert p1.bench[1].items == []

def reforgerTest():
    p1 = setup()
    p1.gold = 1000
    p1.max_units = 10
    for x in range(3):
        p1.add_to_item_bench('reforger')
    p1.buy_champion(champion('leesin'))
    p1.buy_champion(champion('jax'))
    p1.buy_champion(champion('nami'))
    p1.add_to_item_bench('sunfire_cape')
    p1.add_to_item_bench('redemption')
    p1.add_to_item_bench('bf_sword')
    p1.add_to_item_bench('spatula')
    p1.add_to_item_bench('elderwood_heirloom')
    p1.add_to_item_bench('thieves_gloves')
    p1.move_bench_to_board(0, 0, 0)
    p1.move_item(3, 0, 0)
    p1.move_item(4, 0, 0)
    p1.move_item(5, 0, 0)
    p1.move_item(0, 0, 0)
    assert len(p1.board[0][0].items) == 0
    assert p1.item_bench[0] is None
    p1.move_item(6, 1, -1)
    p1.move_item(7, 1, -1)
    p1.move_item(8, 2, -1)
    p1.move_item(1, 1, -1)
    p1.move_item(2, 2, -1)
    test1 = False
    test2 = False
    test3 = False
    test4 = False
    for x in range(9):
        if p1.item_bench[x] == 'reforger':
            test1 = True
        if p1.item_bench[x] == 'spatula':
            test2 = True
        if p1.item_bench[x] in list(trait_items.values()):
            test3 = True
        if p1.item_bench[x] in starting_items:
            test4 = True
    assert not test1
    assert test2
    assert test3
    assert test4

def thiefsGloveCombatTest():
    p1 = setup()
    p1.gold = 1000
    p1.max_units = 1
    p2 = setup()
    p2.gold = 1000
    p2.max_units = 1
    p1.buy_champion(champion('nami'))
    p2.buy_champion(champion('nami'))
    p1.add_to_item_bench('thieves_gloves')
    p2.add_to_item_bench('thieves_gloves')
    p1.move_bench_to_board(0, 0, 0)
    p2.move_bench_to_board(0, 0, 0)
    p1.move_item(0, 0, 0)
    p2.move_item(0, 0, 0)
    p1.add_to_item_bench('deathblade')
    p1.move_item(0, 0, 0)
    assert p1.item_bench[0] == 'deathblade'
    c_object.run(c_object.champion, p1, p2)
    assert p1.board[0][0].items[0] == 'thieves_gloves'

def thiefsGlovesTest():
    p1 = setup()
    p1.gold = 1000
    p1.max_units = 1
    p1.buy_champion(champion('azir'))
    p1.buy_champion(champion('garen'))
    p1.add_to_item_bench('thieves_gloves')
    p1.move_bench_to_board(0, 0, 0)
    p1.move_item(0, 0, 0)
    assert p1.board[0][0].items[0] == 'thieves_gloves'
    for x in range(3):
        p1.start_round(x)
    p1.move_board_to_board(0, 0, 6, 3)
    p1.start_round(3)
    p1.move_board_to_bench(6, 3)
    p1.start_round(4)
    p1.sell_from_bench(0)
    p1.buy_champion(champion('azir'))
    p1.move_item(0, 0, -1)
    p1.start_round(5)

def kaynTests():
    p1 = setup()
    p2 = setup(1)
    p1.gold = 500
    p2.gold = 500
    p1.max_units = 10
    p2.max_units = 10
    p1.buy_champion(champion('kayn'))
    p1.move_bench_to_board(0, 0, 0)
    for x in range(3):
        p1.start_round(x)
        p2.start_round(x)
        p2.buy_champion(champion('kayn'))
        p2.move_bench_to_board(0, x, 0)
    assert p1.kayn_transformed,  'Kayn should transform after his third round in combat'
    assert not p2.kayn_transformed
    assert p1.item_bench[0] == 'kayn_shadowassassin'
    assert p1.item_bench[1] == 'kayn_rhast'
    p2.start_round(3)
    assert p2.kayn_transformed
    p1.move_item(0, 0, 0)
    assert p2.item_bench[0] == 'kayn_shadowassassin'
    assert p2.item_bench[1] == 'kayn_rhast'
    for x in range(7):
        for y in range(4):
            if p2.board[x][y]:
                p2.move_item(1, x, y)
                break
    assert p1.kayn_form == 'kayn_shadowassassin'
    assert p2.kayn_form == 'kayn_rhast'
    p1.buy_champion(champion('kayn'))
    assert p1.bench[0].kayn_form == 'kayn_shadowassassin'
    for x in range(10):
        assert not p1.item_bench[x]

def level2Champion():
    """Creates 3 Zileans, there should be 1 2* Zilean on bench"""
    p1 = setup()
    p1.gold = 100000
    p1.max_units = 10
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[0].stars == 2, "champion should be 2*"
    for x in range(1, 9):
        assert p1.bench[x] is None, "these slot should be empty"
    for x in p1.board:
        for y in x:
            assert y is None, "the board should be empty"


def level3Champion():
    """Creates 9 Zileans, there should be 1 3* Zilean on bench"""
    p1 = setup()
    p1.gold = 100000
    p1.max_units = 1000
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[0].stars == 2
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[1].stars == 2
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[0].stars == 3, "champion should be 3*"
    for x in range(1, 9):
        assert p1.bench[x] is None, "this slot should be empty"
    for x in p1.board:
        for y in x:
            assert y is None, "the board should be empty"


def levelChampFromField():
    """buy third copy while 1 copy on field"""
    p1 = setup()
    p1.gold = 100000
    p1.max_units = 1000
    p1.buy_champion(champion("zilean"))
    p1.buy_champion(champion("zilean"))
    p1.move_bench_to_board(1, 0, 0)
    p1.buy_champion(champion("zilean"))
    for x in p1.bench:
        assert x is None, "bench should be empty"
    assert p1.board[0][0].stars == 2, "the unit placed on the field should be 2*"


# Please expand on this test or add additional tests here.
# I am sure there are some bugs with the level cutoffs for example
# Like I do not think I am hitting level 3 on the correct round without buying any exp
def buyExp():
    p1 = setup()
    p1.level_up()
    lvl = p1.level
    while p1.level < p1.max_level:
        p1.exp = p1.level_costs[p1.level + 1]
        p1.level_up()
        lvl += 1
        assert lvl == p1.level


def spamExp():
    """buys tons of experience"""
    p1 = setup()
    p1.gold = 100000
    for _ in range(1000):
        p1.buy_exp()
    assert p1.level == p1.max_level, "I should be max level"
    assert p1.exp == 0, "I should not have been able to buy experience after hitting max lvl"


def incomeTest1():
    """first test for gold income"""
    p1 = setup()
    p1.gold = 15
    p1.gold_income(5)
    assert p1.gold == 21, f"Interest calculation is messy, gold should be 21, it is {p1.gold}"


def incomeTest2():
    """Check for income cap"""
    p1 = setup()
    p1.gold = 1000
    p1.gold_income(5)
    assert p1.gold == 1010, f"Interest calculation is messy, gold should be 1010, it is {p1.gold}"


def incomeTest3():
    """Checks win streak gold"""
    p1 = setup()
    p1.gold = 0
    p1.win_streak = 0
    p1.gold_income(5)
    assert p1.gold == 5, f"Interest calculation is messy, gold should be 5, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 1
    p1.gold_income(5)
    assert p1.gold == 5, f"Interest calculation is messy, gold should be 5, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 2
    p1.gold_income(5)
    assert p1.gold == 6, f"Interest calculation is messy, gold should be 6, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 3
    p1.gold_income(5)
    assert p1.gold == 6, f"Interest calculation is messy, gold should be 6, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 4
    p1.gold_income(5)
    assert p1.gold == 7, f"Interest calculation is messy, gold should be 7, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 5
    p1.gold_income(5)
    assert p1.gold == 8, f"Interest calculation is messy, gold should be 8, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 500
    p1.gold_income(5)
    assert p1.gold == 8, f"Interest calculation is messy, gold should be 8, it is {p1.gold}"


def incomeTest4():
    """Checks loss streak gold"""
    p1 = setup()
    p1.gold = 0
    p1.loss_streak = 0
    p1.gold_income(5)
    assert p1.gold == 5, f"Interest calculation is messy, gold should be 5, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 1
    p1.gold_income(5)
    assert p1.gold == 5, f"Interest calculation is messy, gold should be 5, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 2
    p1.gold_income(5)
    assert p1.gold == 6, f"Interest calculation is messy, gold should be 6, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 3
    p1.gold_income(5)
    assert p1.gold == 6, f"Interest calculation is messy, gold should be 6, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 4
    p1.gold_income(5)
    assert p1.gold == 7, f"Interest calculation is messy, gold should be 7, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 5
    p1.gold_income(5)
    assert p1.gold == 8, f"Interest calculation is messy, gold should be 8, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 500
    p1.gold_income(5)
    assert p1.gold == 8, f"Interest calculation is messy, gold should be 8, it is {p1.gold}"


def list_of_tests():
    """tests all test cases"""
    chosen_test()
    end_of_turn_actions_test()

    championDuplicatorTest()
    magneticRemoverTest()
    reforgerTest()

    thiefsGloveCombatTest()
    thiefsGlovesTest()

    kaynTests()

    level2Champion()
    level3Champion()
    levelChampFromField()

    buyExp()
    spamExp()

    # Problem: Interest gets calculated after base income is added
    incomeTest1()
    # Problem: Interest rate not capped
    incomeTest2()
    incomeTest3()
    incomeTest4()

    # I would like to go over move commands again before writing test code for that
    pass


def mask_test():
    game_observation = Observation()
    base_pool = pool()
    p1 = player(base_pool, 0)
    step_function = Step_Function(base_pool, game_observation)
    p1.level = 3
    step_function.generate_shops({p1.player_num: p1})
    p1.gold = 1000
    p1.max_units = p1.level
    p1.add_to_item_bench('sunfire_cape')
    p1.add_to_item_bench('sparring_gloves')
    p1.add_to_item_bench('redemption')
    # p1.add_to_item_bench('thieves_gloves')
    # p1.add_to_item_bench('sparring_gloves')
    p1.add_to_item_bench('bf_sword')
    p1.add_to_item_bench('bf_sword')
    p1.add_to_item_bench('bf_sword')
    p1.add_to_item_bench('bf_sword')
    p1.add_to_item_bench('bf_sword')
    p1.add_to_item_bench('bf_sword')
    p1.add_to_item_bench('bf_sword')
    p1.buy_champion(champion('tahmkench'))
    # print(utils.item_binary_encode(list(item_builds.keys()).index('thieves_gloves') + 1 + len(uncraftable_items))) # == TG
    # print(utils.item_binary_encode(list(uncraftable_items).index('sparring_gloves') + 1)) # == Normal Gloves
    p1.move_bench_to_board(0, 0, 0)
    # p1.buy_champion(champion('nunu'))
    # p1.move_bench_to_board(0, 4, 0)
    # p1.buy_champion(champion('maokai'))
    # p1.move_bench_to_board(0, 1, 0)
    # p1.buy_champion(champion('sylas'))
    # p1.move_bench_to_board(0, 2, 0)
    # p1.buy_champion(champion('ashe'))
    # p1.move_bench_to_board(0, 1, 3)
    # p1.buy_champion(champion('aphelios'))
    # p1.move_bench_to_board(0, 3, 3)
    # p1.buy_champion(champion('lissandra'))
    # p1.move_bench_to_board(0, 0, 2)
    # print(step_function.shops[p1.player_num])
    # step_function.batch_shop(0,p1,game_observation)
    # step_function.batch_shop(1,p1,game_observation)
    # step_function.batch_shop(2,p1,game_observation)
    # step_function.batch_shop(3,p1,game_observation)
    # step_function.batch_shop(4,p1,game_observation)
    # step_function.shops[p1.player_num] = base_pool.sample(p1, 5)
    # # print(step_function.shops[p1.player_num])
    # step_function.batch_shop(0,p1,game_observation)
    # step_function.batch_shop(1,p1,game_observation)
    # step_function.batch_shop(2,p1,game_observation)
    # step_function.batch_shop(3,p1,game_observation)
    # step_function.batch_shop(4,p1,game_observation)
    # print(p1.benchStr())
    p1.gold = 0
    # for x in range(7): 
    #         for y in range(4):
    #             if p1.board[x][y]:
    #                 print("x: " + str(x) + " y: " + str(y) + ": " + p1.board[x][y].name)

    obs = game_observation.get_lobo_observation(p1, step_function.shops[p1.player_num], {p1.player_num: p1})
    v1 = [np.ones(config.ACTION_DIM[0])]
    v2 = [np.ones(config.ACTION_DIM[1])]
    v3 = [np.ones(config.ACTION_DIM[2])]

    # print(p1.bench[0].name, p1.bench[1].name, p1.bench[2].name, p1.bench[3].name, 
    # p1.bench[4].name, p1.bench[5].name, p1.bench[6].name, p1.bench[7].name)
    actions = Batch_MCTSAgent.encode_action_to_str(None, v1, v2, v3, obs)
    print("t0 ACTIONS AVAILABLE", list(zip(*actions))[0])

    # one_hot = DataWorker.decode_action_to_one_hot("3_0_0")
    # _, obs = step_function.single_step_action_controller(one_hot, p1, {p1.player_num: p1}, p1.player_num, {p1.player_num: game_observation})

    # actions = Batch_MCTSAgent.encode_action_to_str(None, v1, v2, v3, obs)
    # # print("t1 ACTIONS AVAILABLE", list(zip(*actions))[0])

    # one_hot = DataWorker.decode_action_to_one_hot("3_0_1")
    # _, obs = step_function.single_step_action_controller(one_hot, p1, {p1.player_num: p1}, p1.player_num, {p1.player_num: game_observation})
    # actions = Batch_MCTSAgent.encode_action_to_str(None, v1, v2, v3, obs)
    # # print("t2 ACTIONS AVAILABLE", list(zip(*actions))[0])
    # # p1.add_to_item_bench('bf_sword')
    # # p1.add_to_item_bench('bf_sword')
    # # p1.add_to_item_bench('bf_sword')

    # one_hot = DataWorker.decode_action_to_one_hot("3_0_2")
    # _, obs = step_function.single_step_action_controller(one_hot, p1, {p1.player_num: p1}, p1.player_num, {p1.player_num: game_observation})

    # actions = Batch_MCTSAgent.encode_action_to_str(None, v1, v2, v3, obs)
    # # print("t3 ACTIONS AVAILABLE", list(zip(*actions))[0])

    # one_hot = DataWorker.decode_action_to_one_hot("2_0_37")
    # _, obs = step_function.single_step_action_controller(one_hot, p1, {p1.player_num: p1}, p1.player_num, {p1.player_num: game_observation})
    # actions = Batch_MCTSAgent.encode_action_to_str(None, v1, v2, v3, obs)
    # print("t4 ACTIONS AVAILABLE", list(zip(*actions))[0])

    # actions = Batch_MCTSAgent.encode_action_to_str(None, v1, v2, v3, obs)

    # print(p1.board)
    # print(p1.bench[0].items)
    # print(one_hot)
    # target_1 = np.argmax(one_hot[6:43])
    # one_hot[target_1 + 6] = 0
    # target_2 = np.argmax(one_hot[6:44])
    # swap_loc_from = min(target_1, target_2)
    # swap_loc_to = max(target_1, target_2)

    # print(swap_loc_from, swap_loc_to)

    # x1, y1 = utils.dcord_to_2dcord(swap_loc_from)
    # print(x1, y1)




    # print(p1.bench[0].name)
    
    # obs = game_observation.get_lobo_observation(p1, step_function.shops[p1.player_num], {p1.player_num: p1})
    # actions = Batch_MCTSAgent.encode_action_to_str(None, v1, v2, v3, obs)
    # print("ACTIONS AVAILABLE", list(zip(*actions))[0])

    # one_hot = DataWorker.decode_action_to_one_hot("2_31_37")
    # step_function.single_step_action_controller(one_hot, p1, {p1.player_num: p1}, p1.player_num, {p1.player_num: game_observation})

    # # print(p1.bench[0].name)
    
    # obs = game_observation.get_lobo_observation(p1, step_function.shops[p1.player_num], {p1.player_num: p1})
    # actions = Batch_MCTSAgent.encode_action_to_str(None, v1, v2, v3, obs)
    # print("ACTIONS AVAILABLE", list(zip(*actions))[0])


    # print(step_function.shops[p1.player_num])
