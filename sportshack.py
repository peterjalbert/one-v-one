
import pandas
import random
import numpy as np
import string

def handle_data(path):
    nba = pandas.read_csv(path)
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x.lower())    
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x.split(', '))    
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x[1] + ' ' + x[0] if len(x) == 2 else None)   
    return nba

def find_avg_defensive_distance(defender_name,offensive_name,nba_data):
  
    defender = nba_data[nba_data['CLOSEST_DEFENDER'] == defender_name]
    
    defendervsoffense = defender[defender['player_name'] == offensive_name]
    avg = np.mean((defender['CLOSE_DEF_DIST']))
    
    if len(defendervsoffense) == 0:
        std = np.std(defender['CLOSE_DEF_DIST'])
        avgd = avg
    else:
        average = np.mean((defendervsoffense['CLOSE_DEF_DIST']))
        std = .7*np.std((defendervsoffense['CLOSE_DEF_DIST'])) + .3*np.std(defender['CLOSE_DEF_DIST'])
        avgd = .7*average + .3*avg
    
    
    actual_distance = random.normalvariate(avgd,std)    
    return actual_distance


def find_avg_percentage_made_given_distance_from_defender(defender_name,offensive_name,nba_data,average_distance):

    offender = nba_data[nba_data['player_name'] == offensive_name ]
    offender_shots_given_distance = offender[offender['CLOSE_DEF_DIST'] >= average_distance+.5]
    offender_shooting_percentage = np.mean((offender_shots_given_distance['FGM']))
    offender_shots_given_dist_defender = offender_shots_given_distance[offender_shots_given_distance['CLOSEST_DEFENDER'] == defender_name]
   
    if len(offender_shots_given_dist_defender) == 0:
        offender_shooting_with_defender = 0
        percentage =  offender_shooting_percentage
        std = np.std(offender_shots_given_distance['FGM'])
    else: 
        offender_shooting_with_defender = np.mean((offender_shots_given_dist_defender['FGM'])) 
        percentage =  .3*offender_shooting_percentage + .7*offender_shooting_with_defender
        std = .3*np.std(offender_shots_given_distance['FGM']) + .7*np.std((offender_shots_given_dist_defender['FGM']))
   
    actual_percentage = random.normalvariate(percentage,std)
    
    return actual_percentage    
    
def find_avg_percentage_made_given_distance_from_basket(defender_name,offensive_name,nba_data,distance_from_basket):
    offender = nba_data[nba_data['player_name'] == offensive_name ]
    offender_shots_given_distance_to_hoop = offender[offender['SHOT_DIST'] <= distance_from_basket+2] 
    offender_shots_given_distance_to_hoop = offender_shots_given_distance_to_hoop[offender_shots_given_distance_to_hoop['SHOT_DIST'] >= distance_from_basket -2]
    offender_shots_given_dist_defender = offender_shots_given_distance_to_hoop[offender_shots_given_distance_to_hoop['CLOSEST_DEFENDER'] == defender_name]
    offender_shooting_percentage_distance = np.mean(offender_shots_given_distance_to_hoop['FGM'])  
    print offender_shooting_percentage_distance
    print offender_shots_given_dist_defender
    print offender_shots_given_distance_to_hoop
    if len(offender_shots_given_dist_defender) == 0:
        percentage = offender_shooting_percentage_distance
        std = np.std(offender_shots_given_distance_to_hoop['FGM'])
    else:
        offender_shooting_with_defender = np.mean(offender_shots_given_dist_defender['FGM'])
        percentage = .3*offender_shooting_percentage_distance + .7*offender_shooting_with_defender          
        std = .3*((np.std(offender_shots_given_distance_to_hoop['FGM']))) + .7*(np.std(offender_shots_given_dist_defender['FGM']))
        
    
    actual_percentage = random.normalvariate(percentage,std)
    return actual_percentage

def make_or_miss(dist_defender_percentage, dist_hoop_percentage_w_defender):
    shot_percentage = dist_defender_percentage + dist_hoop_percentage_w_defender
    a = random.random()
    if a <= shot_percentage:
        return True
    else:
        return False 
        
def find_avg_shot_distance(defender_name,offensive_name,nba_data):
    offender = nba_data[nba_data['player_name'] == offensive_name]
    offender_avg_shot_distance = np.mean((offender['SHOT_DIST']))
    a = offender[offender['CLOSEST_DEFENDER'] == defender_name]
    offender_avg_shot_dist_defense = np.mean((a['SHOT_DIST']))
    if len(a) == 0:
        distance = offender_avg_shot_distance
    else:
        distance = .3*offender_avg_shot_distance + .7*offender_avg_shot_dist_defense
    std = np.std(offender['SHOT_DIST'])
    actual_distance = random.normalvariate(distance,std)
    return actual_distance

def get_players(data):
    names = []
    for item in data["player_name"]:
        if item not in names:
            names.append(item)
    return names


def run_simulation(player,opponent,data, limit):
    game_log = ""
    nba_data = data
    coin = random.choice([0,1])
    if coin == 0:
        offense = opponent
        defense = player
    else:
        offense = player
        defense = opponent
    player_score = 0
    opponent_score = 0
    while player_score < limit and opponent_score < limit:
        position_offense = find_avg_shot_distance(defense,offense,nba_data)
        position_defense = find_avg_defensive_distance(defense,offense,nba_data)
        prob_1 = find_avg_percentage_made_given_distance_from_defender(defense,offense,nba_data,position_defense)
        prob_2 = find_avg_percentage_made_given_distance_from_basket(defense,offense,nba_data,position_offense)
        
        points = 0
        b = make_or_miss(prob_1,prob_2)
        if b == True:
            if position_offense >= 23.75:
                points = 3
            else:
                points = 2
                
            if offense == player:
                player_score += points
            else:
                opponent_score += points
        
        game_log += offense + " scored " + str(points) + " points!"  
        game_log += "\n"
        game_log += "SCORE::: " + player + ": " + str(player_score) + ', ' + opponent + ": " + str(opponent_score)
        game_log += "\n"
        
        temp = offense
        offense = defense
        defense = temp
    
    if player_score >= limit:
        game_log += player + " Wins!"
        game_log += "\n"
    else:
        game_log += opponent + " Wins!"
        game_log += "\n"

    return game_log, player_score, opponent_score

