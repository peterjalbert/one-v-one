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
    nba_data = sportshack.handle_data("./shot_logs.csv")
    selected_player = request.form['player'].encode('ascii', 'ignore').lower()
    selected_opponent = request.form['opponent'].encode('ascii', 'ignore').lower()
    print selected_player
    print selected_opponent
    player_score = 0
    opponent_score = 0
    game_text, player_score, opponent_score = sportshack.run_simulation(selected_player, selected_opponent, nba_data, 15)
    selected_player, selected_opponent = sportshack.capitalize_player_name(selected_player), sportshack.capitalize_player_name(selected_opponent)
    return render_template("game.html", selected_player=selected_player, selected_opponent=selected_opponent, player_score = player_score, opponent_score = opponent_score, game_text = game_text)

if __name__ == "__main__":
	app.run()
