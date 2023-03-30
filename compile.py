import json
from flask import Flask, request, render_template
import pickle
import numpy as np
import cv2
import keras 
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences 

app = Flask(__name__)

model = keras.models.load_model('my_model_maxlen90v6.h5')
word_index = {
    '{': 1,
 '}': 2,
 'small-title': 3,
 'text': 4,
 'btn-orange': 5,
 'quadruple': 6,
 'btn-inactive': 7,
 'row': 8,
 'double': 9,
 '<start>': 10,
 'header': 11,
 '<end>': 12,
 'single': 13}
tokenizer = Tokenizer()
tokenizer.word_index = word_index
print(tokenizer.word_index )

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
# generate caption for an image
def predict_caption(model, image, max_length):
    # add start tag for generation process
    sequence = [10]
    html = "<start>"
    img = []
    img.append(image)
    img = np.array(img)
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        if i == 0:
            sequence = [10]
        print(sequence)
        pad = pad_sequences([sequence], maxlen = max_length)
        
        yhat = model.predict([img, pad], verbose=0)
        yhat = np.argmax(yhat)
        sequence.append(yhat)
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        
        if word is None:
            break
        # append word as input for generating next word
        html += " " + word
        # stop if we reach end tag
        if word == '<end>':
            break
      
    return html

def resize_img(png_file_path):
    # Load the image


# Resize the image to 200x200
    resized = cv2.resize(png_file_path, (200, 200))

# Normalize the pixel values to the range [0, 1]
    normalized = resized / 255.0
    return normalized


# DSL to HTML
DEFAULT_DSL_MAPPING_FILEPATH = {
    "opening-tag": "{",
    "closing-tag": "}",
    "body": "<html>\n  <header>\n    <meta charset=\"utf-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n   <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css\" integrity=\"sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3\" crossorigin=\"anonymous\">\n<style>\n.header{margin:20px 0}nav ul.nav-pills li{background-color:#333;border-radius:4px;margin-right:10px}.col-lg-3{width:24%;margin-right:1.333333%}.col-lg-6{width:49%;margin-right:2%}.col-lg-12,.col-lg-3,.col-lg-6{margin-bottom:20px;border-radius:6px;background-color:#f5f5f5;padding:20px}.row .col-lg-3:last-child,.row .col-lg-6:last-child{margin-right:0}footer{padding:20px 0;text-align:center;border-top:1px solid #bbb}\n</style>\n    <title>Scaffold</title>\n  </header>\n  <body>\n    <main class=\"container\">\n      {}\n      <footer class=\"footer\">\n        <p>&copy; Footer</p>\n      </footer>\n    </main>\n    <script src=\"js/jquery.min.js\"></script>\n    <script src=\"js/bootstrap.min.js\"></script>\n  </body>\n</html>\n",
    "header": "<div class=\"header clearfix\">\n  <nav>\n    <ul class=\"nav nav-pills pull-left\">\n      {}\n    </ul>\n  </nav>\n</div>\n",
    "btn-active": "<li class=\"active\"><a href=\"#\">Text</a></li>\n",
    "btn-inactive": "<li><a href=\"#\">Text</a></li>\n",
    "row": "<div class=\"row\">{}</div>\n",
    "single": "<div class=\"col-lg-12\">\n{}\n</div>\n",
    "double": "<div class=\"col-lg-6\">\n{}\n</div>\n",
    "quadruple": "<div class=\"col-lg-3\">\n{}\n</div>\n",
    "": "<a class=\"btn btn-success\" href=\"#\" role=\"button\">Text</a>\n",
    "btn-orange": "<a class=\"btn btn-warning\" href=\"#\" role=\"button\">Text</a>\n",
    "btn-red": "<a class=\"btn btn-danger\" href=\"#\" role=\"button\">Text</a>",
    "big-title": "<h2>Text</h2>",
    "small-title": "<h4>Text</h4>",
    "text": "<p>-----Text-----</p>\n"
}


class Node:
    def __init__(self, key, parent_node, content_holder):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.content_holder = content_holder

    def add_childNode(self, child):
        self.children.append(child)

    def show(self):
        print(self.key)
        for child in self.children:
            child.show()

    def render(self, mapping, rendering_function=None):
        content = ""
        for child in self.children:
            content += child.render(mapping, rendering_function)

        value = mapping[self.key]
        if rendering_function is not None:
            value = rendering_function(self.key, value)

        if len(self.children) != 0:
            value = value.replace(self.content_holder, content)
        return value

    def getParent(self):
        return self.parent


def add_child(node, currentNode):
    currentNode.add_childNode(node)
    return currentNode


def Compilee(dsl):
    MasterNode = Node("body", None, "{}")
    currentNode = Node("body", None, "{}")
    numopenTage = 0
    for tk in dsl:
        if (tk == ''):
            continue
        if (tk == ","):
            continue

        if (tk == '{'):
            MasterNode = currentNode

        elif tk == '}':
            MasterNode = MasterNode.getParent()
        else:

            element = Node(tk, MasterNode, "{}")
            currentNode = element
            MasterNode.add_childNode(element)

    return MasterNode.render(DEFAULT_DSL_MAPPING_FILEPATH, rendering_function=None)


@app.route('/')
def home():
    return render_template('Rm.html')

@app.route('/predicts', methods=['POST'])
def predicts():
    file = request.files['myFile'].read()
    image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_UNCHANGED)
    resized_image = resize_img(image)
    predict = predict_caption(model, resized_image, 90)
    # code to process the uploaded file
    predict = predict.replace("<start>", "")
    predict= predict.replace("<end>", "")
    predict= predict.replace(",", "")
    predict = predict.split(" ")
    predict = Compilee(predict)
    
    
    return render_template('Rm.html', prediction_text=predict)





if __name__ == "__main__":
    app.run(debug=True)
