# coding='utf-8'
from flask import Flask, render_template, request, jsonify
import utils
import time
import threading


# define heartbeat detection function
def heartbeat():
  print(time.strftime("%Y-%m-%d %H:%M:%S - heartbeat", time.localtime(time.time())))
  timer = threading.Timer(60, heartbeat)
  timer.start()
timer = threading.Timer(60, heartbeat)
timer.start()

app = Flask(__name__, static_url_path='/static')


@app.route('/message', methods=["POST"])
# receive the user message from the front end and make response
def reply():

  # request user message
  req_msg = request.form['msg']

  # produce sentence to response to user
  res_msg = utils.predict(req_msg)
  res_msg = res_msg.strip()
  
  # if generated response is empty, manually create a response
  if res_msg == '':
    res_msg = '跟我聊个天吧！' # "Chat with me!"

  return jsonify({'text':res_msg})  # return the result in json format

# use a chat user interface as the front end
@app.route('/')
def index():
  return render_template('index.html')

# start the app
if __name__ == '__main__':
  app.run(debug=True)