import flask
from sql_and_docs_chat import init_model
app = flask.Flask(__name__)

model = init_model()
@app.route("/", methods=["POST"])
def starting_url():
    json_data = flask.request.json
    user_input = json_data["user_input"]
    answer = model.query(user_input)
    return "asnwer: " + str(answer)

app.run(host="0.0.0.0", port=8100)