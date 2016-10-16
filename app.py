from flask import Flask, render_template, json, request, redirect
import sportshack 

app = Flask(__name__)

def capitalize_player_name(player):
	temp = player.split()
	for i in range(len(temp)):
		temp[i] = temp[i][0].upper() + temp[i][1:]
	temp = ' '.join(temp)
	return temp


@app.route("/")
def main():
	return render_template("index.html")


@app.route('/game', methods = ['POST'])
def game():
	nba_data = sportshack.handle_data("./shot_logs.csv")
	selected_player = request.form['player'].encode('ascii', 'ignore')
	selected_opponent = request.form['opponent'].encode('ascii', 'ignore')
	player_score = 0
	opponent_score = 0
	game_text, player_score, opponent_score = sportshack.run_simulation(selected_player, selected_opponent, nba_data, 30)
	selected_player, selected_opponent = capitalize_player_name(selected_player), capitalize_player_name(selected_opponent)
	return render_template("game.html", selected_player=selected_player, selected_opponent=selected_opponent, player_score = player_score, opponent_score = opponent_score, game_text = game_text)

if __name__ == "__main__":
	app.run()
