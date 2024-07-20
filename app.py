import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import flask
import time

# Comment out this line if running on a CPU
#keras.mixed_precision.set_global_policy("mixed_float16")

# Initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

LLM = "gpt2_base_en"

def load_model():
    '''Load the pre-trained LLM'''
    global model
    model = keras.models.load_model('ShakeGPT.keras')


def prepare_text(text):
    '''This function:
        - Prepares text
    '''

    # return the processed text
    return text

@app.route("/answer/", methods=["POST"])
def answer():
    '''This function:
        - Checks that the request method is POST (enabling us to send arbitrary data to the endpoint)
        #- Checks that an image has been passed into the `files` attribute
        #- Reads in the data in PIL format
        #- Preprocesses it
        #- Passes it to our network
        #- Loops over results and adds them to the predictions list
        - Returns the reponse to the client in JSON format
    '''
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        question = flask.request.form['content']

        start = time.time()
        output = model.generate(question, max_length=200)
        end = time.time()
        time_taken = f"{end - start:.2f}s"
        data["time"] = time_taken

        # Remove question from reply
        data["answer"] = output.strip(question)
        
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model
# and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask, starting server..."
           "please wait until the server has fully started"))
    load_model()
    app.run()