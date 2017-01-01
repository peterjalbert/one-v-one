from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
import pandas as pd
import random
import numpy as np
import string
import plotly.plotly as plotly
import plotly.graph_objs as go
import plotly.tools as tls
tls.set_credentials_file(username='abhious', api_key='1Tdwg7pmZUvqlMJhNEgD')
plotly.sign_in(username='abhious',api_key='1Tdwg7pmZUvqlMJhNEgD')
pd.options.mode.chained_assignment = None

def handle_data_simple(path):
    nba = pd.read_csv(path)
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x.lower())    
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x.split(', '))    
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x[1] + ' ' + x[0] if len(x) == 2 else None)   
    return nba

def capitalize_player_name(player):
	temp = player.split()
	for i in range(len(temp)):
		temp[i] = temp[i][0].upper() + temp[i][1:]
	temp = ' '.join(temp)
	return temp

def get_playerlist():
    data = handle_data_simple("/Users/peterjalbert/Documents/one-v-one/shotlogs.csv")
    defenderset = set(data['CLOSEST_DEFENDER'])
    playerset = set(data['player_name'])
    playlist = defenderset.intersection(playerset)
    player_list = [x for x in playlist if x is not None]
    player_list.sort(key = lambda b: b.split()[1])
    player_list = [capitalize_player_name(player) for player in player_list]
    return player_list

def handle_data(path):
    nba = pd.read_csv(path)
    
    
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x.lower())    
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x.split(', '))    
    nba['CLOSEST_DEFENDER'] = nba['CLOSEST_DEFENDER'].apply(lambda x: x[1] + ' ' + x[0] if len(x) == 2 else None)
    for item in range(len(nba['SHOT_CLOCK'])):
        if np.isnan(nba['SHOT_CLOCK'][item]):
            nba['SHOT_CLOCK'][item] = float(nba['GAME_CLOCK'][item][3:])            
    drops = ['GAME_ID','MATCHUP','LOCATION','W','FINAL_MARGIN','SHOT_NUMBER','PERIOD','PTS','GAME_CLOCK','SHOT_RESULT']
    nba.drop(drops,axis=1,inplace=True)
    nba[nba.CLOSEST_DEFENDER.isin(nba.player_name)]
    le = LabelEncoder()
    
    nbacopy = nba.copy()
    nba['CLOSEST_DEFENDER'] = le.fit_transform(nba['CLOSEST_DEFENDER'])
    nba['player_name'] = le.transform(nba['player_name'])
    transformers = {}
    
    
    for header in list(nba.columns.values):
        mms = MinMaxScaler(feature_range=(0,1))
        mms.fit(nba[header])
        transformers[header] = mms
        
    return nbacopy,transformers,le

def scaler(transformers,data):
    datacopy = data.copy()
    for header in list(data.columns.values):
        datacopy[header] = transformers[header].transform(datacopy[header])
    return datacopy

def offensiveShotDistModel(op,data,transformers,le):
    datacopy = data.copy()
    offense = datacopy.loc[datacopy['player_name']==op] #get only the offensive players' attempts at offense
    offensetarget = offense['SHOT_DIST']
    features = ['SHOT_CLOCK','DRIBBLES','CLOSEST_DEFENDER']
    offense = offense[features]
    offense['CLOSEST_DEFENDER'] = le.transform(offense['CLOSEST_DEFENDER'])
    
    
    offense_scaled = scaler(transformers,offense)
    mlpr = RandomForestRegressor()
    mlpr.fit(offense_scaled,offensetarget)
    return mlpr

def defensivePositionModel(dp,data,transformers,le):
    datacopy = data.copy()
    defense = datacopy.loc[datacopy['CLOSEST_DEFENDER']==dp] #get only the offensive players' attempts at offense
    defensetarget = defense['CLOSE_DEF_DIST']
    features = ['SHOT_CLOCK','DRIBBLES','player_name','SHOT_DIST']
    defense = defense[features]
    defense['player_name'] = le.transform(defense['player_name'])
    
    
    defense_scaled = scaler(transformers,defense)
    mlpr = RandomForestRegressor()
    mlpr.fit(defense_scaled,defensetarget)
    return mlpr

def shotclock(player,data):
    playerdata = data.loc[data['player_name']==player]
    scdata = playerdata['SHOT_CLOCK']
    parameters = norm.fit(scdata)
    sc = np.random.normal(loc = parameters[0], scale = parameters[1])
    return sc

def dribbleModel(op,data,transformers,le):
    datacopy = data.copy()
    dribble = datacopy.loc[datacopy['player_name'] == op]   #get only the offensive players' attempts at offense
    dribbletarget = dribble['DRIBBLES']
    
    features = ['CLOSEST_DEFENDER']
    dribble = dribble[features]
    dribble['CLOSEST_DEFENDER'] = le.transform(dribble['CLOSEST_DEFENDER'])
    
    dribble_scaled = scaler(transformers,dribble)
    
    mlpr = RandomForestRegressor()
    
    mlpr.fit(dribble_scaled,dribbletarget)
    return mlpr

def xy_position(shot_dist):
    theta = 180*(random.random())
    theta = np.radians(theta)

    loc_x = (shot_dist*np.cos(theta))*10
    loc_y = (shot_dist*np.sin(theta))*10
    
    return loc_x,loc_y

def shotMadeModel(op,data,transformers,le):
    datacopy = data.copy()
    shot = datacopy.loc[datacopy['player_name']==op] #get only the offensive players' attempts at offense
    shottarget = shot['FGM']
    features = ['CLOSEST_DEFENDER','DRIBBLES','SHOT_CLOCK','SHOT_DIST','CLOSE_DEF_DIST']
    shot = shot[features]
    shot['CLOSEST_DEFENDER'] = le.transform(shot['CLOSEST_DEFENDER'])
    
    
    
    shot_scaled = scaler(transformers,shot)
    
    mlpc = GradientBoostingClassifier()
    mlpc.fit(shot_scaled,shottarget)
    return mlpc

def possession(dp,op,transformers,le,data):
    sc = (np.array((shotclock(op,data))))
    dpm = defensivePositionModel(dp,data,transformers,le)
    dm = dribbleModel(op,data,transformers,le)
    osdm = offensiveShotDistModel(op,data,transformers,le)
    
    smm = shotMadeModel(op,data,transformers,le)
    d_first_encode = le.transform(np.array([dp]))
    d_second_encode = transformers['CLOSEST_DEFENDER'].transform(d_first_encode)
    o_first_encode = le.transform(np.array([op]))
    o_second_encode = transformers['CLOSEST_DEFENDER'].transform(o_first_encode)
    dribbles = transformers['DRIBBLES'].transform(np.round(dm.predict(np.array([d_second_encode]).reshape(1,1))))
    shotDistance = (osdm.predict(np.array([sc,dribbles,d_second_encode]).reshape(1,3)))
    shotDistanceCopy = shotDistance.copy()
    defensivePosition = transformers['CLOSE_DEF_DIST'].transform(dpm.predict(np.array([sc,dribbles,o_second_encode,shotDistance])))
    loc_x,loc_y = xy_position(shotDistanceCopy)
    shotDistance = transformers['SHOT_DIST'].transform(shotDistance)
    
    shotMadeProbability = smm.predict_proba(np.array([d_second_encode,dribbles,sc,shotDistance,defensivePosition]).reshape(1,5))[0,1]

    flag = random.random()
    smf = True
    if flag <= shotMadeProbability:
        smf = True
    else:
        smf = False
    return smf, transformers['DRIBBLES'].inverse_transform(dribbles), loc_x,loc_y

def simulation(path,pointcap,player_1,player_2,totalgames):
    data,mms, le = handle_data(path)
    player_1_fgs = pd.DataFrame(columns = ['shotmade','loc_x', 'loc_y','shotdist','dribbles','fgev','points','game'])
    player_2_fgs = pd.DataFrame(columns = ['shotmade','loc_x', 'loc_y','shotdist','dribbles','fgev','points','game'])
    pointdiff = []
    wins = 0
    for game in range(1,totalgames+1):
        
        
        player_1_points = 0
        player_1_dribbles = []
        player_1_shotdist = []
        
        player_2_points = 0
        
        player_2_shotdist = []
        player_2_dribbles = []
        coin = random.randint(0,2) #coin flip
        if coin == 1:
            offense = player_1
            defense = player_2
        else:
            offense = player_2
            defense = player_1
        
        while player_1_points < pointcap and player_2_points < pointcap:
            smf, dribbles, loc_x, loc_y= possession(defense,offense,mms,le,data)
             #field goal expected value
            inbounds= True
            fgev = 0
            if (abs(loc_x) >= 250) == True:
                
                inbounds = False
            elif (abs(loc_y) >= 422.5) == True:
                inbounds = False
            elif (loc_y <= -47.5) == True:
                inbounds = False#out of bounds
            if (np.linalg.norm(np.array([loc_x,loc_y])) > 237.5 or (abs(loc_x) > 220)) and (inbounds==True):
                fgev = 3
            elif inbounds==True:
                fgev = 2
            
            if (smf==True) and (inbounds==True):
                if offense == player_1:
                    player_1_points += fgev
                    shotdist = np.linalg.norm(np.array([loc_x,loc_y]))/10
                    player_1_fgs = player_1_fgs.append(pd.DataFrame([[1,loc_x,loc_y,shotdist,dribbles,fgev,player_1_points,game]],columns = ['shotmade','loc_x', 'loc_y','shotdist','dribbles','fgev','points','game']),ignore_index=True)
                    #print(defense + ' ' + "can not stop " + offense + ' as he buries a ' + str(fgev) + ' pt. jumper!')
                    offense = player_2
                    defense = player_1
                    
                else:
                    player_2_points += fgev
                    shotdist = np.linalg.norm(np.array([loc_x,loc_y]))/10
                    player_2_fgs = player_2_fgs.append(pd.DataFrame([[1,loc_x,loc_y,shotdist,dribbles,fgev,player_2_points,game]],columns = ['shotmade','loc_x', 'loc_y','shotdist','dribbles','fgev','points','game']),ignore_index=True)
                    
                    #print(defense + ' ' + "can not stop " + offense + ' as he buries a ' + str(fgev) + ' pt. jumper!')
                    offense = player_1
                    defense = player_2
                    
            elif (smf==False) and (inbounds==True):
                if offense == player_1:
                    player_1_points += 0
                    shotdist = np.linalg.norm(np.array([loc_x,loc_y]))/10
                    player_1_fgs = player_1_fgs.append(pd.DataFrame([[0,loc_x,loc_y,shotdist,dribbles,fgev,player_1_points,game]],columns = ['shotmade','loc_x', 'loc_y','shotdist','dribbles','fgev','points','game']),ignore_index=True)
                
                    #print('Missed opportunity by ' + offense)
                    offense = player_2
                    defense = player_1
                    
                else:
                    player_2_points += 0
                    shotdist = np.linalg.norm(np.array([loc_x,loc_y]))/10
                    player_2_fgs = player_2_fgs.append(pd.DataFrame([[0,loc_x,loc_y,shotdist,dribbles,fgev,player_2_points,game]],columns = ['shotmade','loc_x', 'loc_y','shotdist','dribbles','fgev','points','game']),ignore_index=True)
                    #print('Missed opportunity by ' + offense)
                    offense = player_1
                    defense = player_2
            elif (inbounds==False):
                if offense == player_1:
                    
                    #print(offense + ' steps out of bounds.')
                    offense = player_2
                    defense = player_1
                else:
                    #print(offense + ' steps out of bounds.')
                    offense = player_1
                    defense = player_2
                
                
            #print (player_1 + ' ' + str(player_1_points))
            #print (player_2 + ' ' + str(player_2_points))
        
        if player_1_points > player_2_points:
            print(player_1 + ' Wins!')
            wins += 1
        else:
            print(player_2 + ' Wins!')
            
        print(player_1 + ': ' + str(player_1_points))
        print(player_2 + ': ' + str(player_2_points))
        pointdiff.append(abs(player_1_points-player_2_points))
    
    winpercentage = wins/float(totalgames)
    tg = ('Total Games: ' + str(totalgames))
    wp1 = (player_1 + ' Win Percentage: ' + str(winpercentage))
    wp2 = (player_2 + ' Win Percentage: ' + str(1-winpercentage))
    pntd = ('Avg. Point Differential: ' +str(np.mean(pointdiff)))
    
    
    return player_1_points, player_2_points, player_2_fgs, player_1_fgs, tg, wp1, wp2, pntd 
    
    
#fp = "/Users/peterjalbert/Documents/one-v-one/shot_logs.csv"
#print simulation(fp,21,'stephen curry','kyrie irving',100)

def plotSimulation(player_1,player_2,player_1_fgs,player_2_fgs):
    data = []
    color = 'rgb(0,0,0)'
    court_shapes = []
    outer_lines_shapes = {
        'type' : 'rect',
        'xref' : 'x',
        'yref' :'y',
        'x0':-250,
        'y0':-47.5,
        'x1':250,
        'y1':422.5,
        'line' : {
          'color':color,
          'width':2
      },
    }
    court_shapes.append(outer_lines_shapes)
    hoop_shape = {
          'type':'circle',
          'xref':'x',
          'yref':'y',
          'x0':7.5,
          'y0':7.5,
          'x1':-7.5,
          'y1':-7.5,
          'line': {
            'color':color,
            'width':2
          },
    }
    court_shapes.append(hoop_shape)
    backboard_shape = {
          'type':'rect',
          'xref':'x',
          'yref':'y',
          'x0':-30,
          'y0':-7.5,
          'x1':30,
          'y1':-6.5,
          'line': {
            'color':color,
            'width':2
          },
          'fillcolor' : color,
    }
    court_shapes.append(backboard_shape)
    outer_three_sec_shape = {
          'type':'rect',
          'xref':'x',
          'yref':'y',
          'x0':-80,
          'y0':-47.5,
          'x1':80,
          'y1':143.5,
          'line': {
            'color':color,
            'width':2
          },
    }
    court_shapes.append(outer_three_sec_shape)
    inner_three_sec_shape = {
          'type':'rect',
          'xref':'x',
          'yref':'y',
          'x0':-60,
          'y0':-47.5,
          'x1':60,
          'y1':143.5,
          'line': {
            'color':color,
            'width':2
          },
    }
    court_shapes.append(inner_three_sec_shape)
    left_line_shape = {
          'type':'line',
          'xref':'x',
          'yref':'y',
          'x0':-220,
          'y0':-47.5,
          'x1':-220,
          'y1':92.5,
          'line': {
            'color':color,
            'width':2
          },
    }
    court_shapes.append(left_line_shape)
    right_line_shape = {
          'type':'line',
          'xref':'x',
          'yref':'y',
          'x0':220,
          'y0':-47.5,
          'x1':220,
          'y1':92.5,
          'line': {
            'color':color,
            'width':2
          },
    }
    court_shapes.append(right_line_shape)
    three_point_arc_shape = {
          'type':'path',
          'xref':'x',
          'yref':'y',
          'path' : 'M -220 92.5 C -70 300, 70 300, 220 92.5',
          'line': {
            'color':color,
            'width':2
          },
    }
    court_shapes.append(three_point_arc_shape)
    center_circle_shape = {
          'type':'circle',
          'xref':'x',
          'yref':'y',
          'x0':60,
          'y0':482.5,
          'x1':-60,
          'y1':362.5,
          'line': {
            'color':color,
            'width':2
          },
    }
    court_shapes.append(center_circle_shape)
    res_circle_shape = {
          'type':'circle',
          'xref':'x',
          'yref':'y',
          'x0':20,
          'y0':442.5,
          'x1':-20,
          'y1':402.5,
          'line': {
            'color':color,
            'width':2
          },
    }
    court_shapes.append(res_circle_shape)
    free_throw_circle_shape = {
          'type':'circle',
          'xref':'x',
          'yref':'y',
          'x0':60,
          'y0':200,
          'x1':-60,
          'y1':80,
          'line': {
            'color':color,
            'width':2
          },
    }
    court_shapes.append(free_throw_circle_shape)
    res_area_shape = {
          'type':'circle',
          'xref':'x',
          'yref':'y',
          'x0':40,
          'y0':40,
          'x1':-40,
          'y1':-40,
          'line': {
            'color':color,
            'width':2,
            'dash':'dot'
          },
    }
    for player in [player_1,player_2]:
        if player == player_1:
            dataset = player_1_fgs
        else:
            dataset = player_2_fgs
            
        for i in range(0,2):
            if i == 0:
                name = 'Missed Shot'
                
            else:
                name = 'Made Shot'
                
            colors = random.sample(range(0,256),3)
            trace = go.Scatter(
                x = dataset[dataset['shotmade'] == i]['loc_x'],
                y = dataset[dataset['shotmade'] == i]['loc_y'],
                mode = 'markers',
                name = player + ': ' + name,
                marker = {'size': 5, 'color':'rgb'+'('+str(colors[0])+str(colors[1])+str(colors[2])
                },
                )
            data.append(trace)
    layout = dict(title = player_1 + ' vs. ' + player_2,showlegend=True,
               xaxis = dict(showgrid=False,range =[-300,300]),
               yaxis = dict(showgrid=False, range =[-100,500]),
               height=600,
               width=650,
               shapes=court_shapes)
    fig = dict(data=data,layout=layout)
    plot_url = plotly.plot(fig,filename= 'Shot Chart: '  + player_1 + ' and ' + player_2, auto_open = False)
    return tls.get_embed(plot_url)      
#plotSimulation(player_1,player_2,player_1_fgs,player_2_fgs)

def plotGameTimeLine(player_1,player_2,player_1_fgs,player_2_fgs):
    games = int(max(player_2_fgs['game']))
    
    y0 = []
    y1 = []
    x = list(range(1,games+1))
    for game in range(1,games+1):
        sub1 = player_1_fgs.loc[player_1_fgs['game']==game]
        sub2 = player_2_fgs.loc[player_2_fgs['game']==game]
        points_1 = np.amax(sub1['points'])
        points_2 = np.amax(sub2['points'])
        y0.append(points_1)
        y1.append(points_2)
    trace0 = go.Bar(
    x=x,
    y= y0,
    name= player_1,
    marker=dict(
        color='rgb(49,130,189)'
        )
    )
    trace1 = go.Bar(
    x=x,
    y= y1,
    name= player_2,
    marker=dict(
        color='rgb(204,204,204)',
        )
    )

    data = [trace0,trace1]
    layout = dict(title = player_1 + ' vs. ' + player_2 ,
              xaxis = dict(title = 'Games'),
              yaxis = dict(title = 'Points'),
              )
    fig = dict(data=data,layout=layout)
    plot_url = plotly.plot(fig, 'Results: ' + player_1 + ' and ' + player_2, auto_open = False )
    return tls.get_embed(plot_url)
#plotGameTimeLine(player_1,player_2,player_1_fgs,player_2_fgs)

def plotShotDistBreakDown(player_1,player_2,player_1_fgs,player_2_fgs):
    x0 = player_1_fgs['shotdist']
    x1 = player_2_fgs['shotdist']
    trace0 = go.Histogram(x = x0, opacity = .75, histnorm='probability', name = player_1)
    trace1 = go.Histogram(x = x1, opacity = .75, histnorm='probability', name = player_2)
    data = [trace0,trace1]
    layout = go.Layout(barmode= 'overlay', title= 'Shot Distance Breakdown', xaxis = dict(title = 'Shot Distance'), yaxis = dict(title = 'Probability'))
    fig = dict(data=data,layout=layout)
    plot_url = plotly.plot(fig, filename = 'Shot Distance Breakdown: ' + player_1 + ' and ' + player_2, auto_open = False )
    return tls.get_embed(plot_url)
#plotShotDistBreakDown(player_1,player_2,player_1_fgs,player_2_fgs)

def plotDribbleBreakdown(player_1,player_2,player_1_fgs,player_2_fgs):
    x0 = player_1_fgs['dribbles']
    x1 = player_2_fgs['dribbles']
    trace0 = go.Histogram(x = x0, opacity = .75, histnorm='probability', name = player_1)
    trace1 = go.Histogram(x = x1, opacity = .75, histnorm='probability', name = player_2)
    data = [trace0,trace1]
    layout = go.Layout(barmode= 'overlay', title= 'Dribble Breakdown', xaxis = dict(title = 'Dribbles'), yaxis = dict(title = 'Probability'))
    fig = dict(data=data,layout=layout)
    plot_url = plotly.plot(fig, filename = 'Dribble Breakdown: ' + player_1 + ' and ' + player_2, auto_open = False)
    return tls.get_embed(plot_url)
#plotDribbleBreakdown(player_1,player_2,player_1_fgs,player_2_fgs)

def plotFGPercentage(player_1,player_2,player_1_fgs,player_2_fgs):
    x0 = player_1_fgs['shotmade']
    x1 = player_2_fgs['shotmade']
    player_1_made = np.mean(x0)
    player_1_missed = 1-player_1_made
    player_2_made = np.mean(x1)
    player_2_missed = 1-player_2_made
    x = ['% Made' , '% Missed']
    trace0 = go.Bar(x = x, y = [player_1_made,player_1_missed], name = player_1)
    trace1 = go.Bar(x = x, y = [player_2_made,player_2_missed], name = player_2)
    data = [trace0,trace1]
    layout = go.Layout(barmode = 'group', title = 'FG % Breakdown')
    fig = go.Figure(data=data,layout=layout)
    plot_url = plotly.plot(fig, filename = 'FG % Breakdown: ' + player_1 + ' and ' + player_2, auto_open = False)
    return tls.get_embed(plot_url)
#plotFGPercentage(player_1,player_2,player_1_fgs,player_2_fgs)

def finalsimulation(path,player_1,player_2):
    player_1_points, player_2_points, player_2_fgs, player_1_fgs, tg, wp1, wp2, pntd = simulation(path,21,player_1,player_2, 5) # returns dataframe of player_2_game,player_1_game, total games(string),
    # win percentage for player 1(string) and player 2(string), and point differential for games(string)
    htmlShotChart = plotSimulation(player_1,player_2,player_1_fgs,player_2_fgs) #html code for the shot chart
    htmlGameTimeLine = plotGameTimeLine(player_1,player_2,player_1_fgs,player_2_fgs) #html chart for the points/game 
    htmlShotDistBreakDown = plotShotDistBreakDown(player_1,player_2,player_1_fgs,player_2_fgs) #html chart for shot distribution
    htmlDribbleBreakdown = plotDribbleBreakdown(player_1,player_2,player_1_fgs,player_2_fgs) #html chart for dribble breakdown
    htmlFGPercentage = plotFGPercentage(player_1,player_2,player_1_fgs,player_2_fgs) #html chart for fg percentage
    return player_1_points, player_2_points, tg, wp1, wp2, pntd, htmlShotChart, htmlGameTimeLine, htmlShotDistBreakDown, htmlDribbleBreakdown, htmlFGPercentage

#this last function is the one to run
#as an input it takes the file path which is commented below
#you will probably have to change it on your computer or put it on a database (SQL or something so it doesn't have to pull from your computer everytime)
#i'm also running python 3.6 so idk you will need that and also pip install plotly
#you might have to change the interface for how it finds the names in the data:
#right now you have it something like this where the name is "Stephen Curry". I've made it so that it can only accept 'stephen curry'. All uppercase and all
#dashes and hyphens and apostrophes are deleted so someone like "J.R. Smith" becomes jr smith. 

#good luck and thank you


    
                
        



    
    
    
