from flask import Flask, render_template, json, request, redirect
import sportshack

app = Flask(__name__)


@app.route("/")
def main():
    player_list = sportshack.get_playerlist()
    print player_list
    return render_template("index.html", player_list = player_list)


@app.route('/game', methods = ['POST'])
def game():
    selected_player = request.form['player'].encode('ascii', 'ignore').lower()
    selected_opponent = request.form['opponent'].encode('ascii', 'ignore').lower()
    player_score = 0
    opponent_score = 0
    player_score, opponent_score, totalgames, player_wp, opponent_wp, pointdifference, htmlShotChart, htmlGameTimeLine, htmlShotDistBreakdown, htmlDribbleBreakdown, htmlFg = sportshack.finalsimulation("./shotlogs.csv", selected_player, selected_opponent)
    selected_player, selected_opponent = sportshack.capitalize_player_name(selected_player), sportshack.capitalize_player_name(selected_opponent)
    return render_template("game.html", selected_player=selected_player, selected_opponent=selected_opponent, player_score = player_score, opponent_score = opponent_score, totalgames = totalgames, player_wp = player_wp, opponent_wp = opponent_wp, pointdifference = pointdifference, htmlShotChart = htmlShotChart, htmlGameTimeLine = htmlGameTimeLine, htmlShotDistBreakdown = htmlShotDistBreakdown, htmlDribbleBreakdown = htmlDribbleBreakdown, htmlFg = htmlFg)

if __name__ == "__main__":
	app.run()
